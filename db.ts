import { existsSync, readFileSync, unlinkSync } from "node:fs";
import { dirname } from "node:path";
import Database from "better-sqlite3";
import { load as loadVec } from "sqlite-vec";
import { getRagDir, legacyIndexFile, ensureDir } from "./store.ts";
import { resolveActiveDbPath } from "./index-manager.ts";
import * as repo from "./repository.ts";
import { loadConfig } from "./config.ts";
import { VECTOR_DIM } from "./constants.ts";
import { checkIndexCompatibility } from "./index-manager.ts";

export interface Chunk {
  id: string;
  file: string;
  content: string;
  lineStart: number;
  lineEnd: number;
  hash: string;
  indexed: string;
  tokens: number;
  pageStart?: number | null;
  pageEnd?: number | null;
  section?: string | null;
}

interface FileEntry {
  hash: string;
  chunks: number;
  indexed: string;
  size: number;
  embedded: boolean;
}

export interface IndexMeta {
  chunks: Chunk[];
  files: Record<string, FileEntry>;
  lastBuild: string;
  embeddingModel?: string;
}

export interface IndexStats {
  totalChunks: number;
  totalFiles: number;
  totalTokens: number;
  embeddedCount: number;
  lastBuild: string;
  embeddingModel: string;
  embeddingFingerprint?: string;
  processingFingerprint?: string;
  embeddingDimensions?: number;
  needsRebuild?: boolean;
  rebuildReason?: string;
}

export class RagDatabase {
  private static _instance: Database.Database | null = null;
  private constructor() {}

  static get instance(): Database.Database {
    // If a caller closed the singleton Database object directly (db.close())
    // without going through closeDbConn(), `_instance` would otherwise keep
    // handing out a dead connection. Detect that and reopen.
    if (RagDatabase._instance && !RagDatabase._instance.open) {
      RagDatabase._instance = null;
    }
    return RagDatabase._instance ??= RagDatabase.open();
  }

  static get isOpen(): boolean { return RagDatabase._instance !== null; }

  static close(): void {
    const db = RagDatabase._instance;
    RagDatabase._instance = null;
    try {
      db?.close();
    } catch (err) {
      process.stderr.write(`[rag] closeDb() failed: ${(err as Error).message}\n`);
    }
  }

  static open(ragDir?: string): Database.Database {
    const dir = ragDir ?? getRagDir();
    ensureDir(dir);
    const path = resolveActiveDbPath(dir);
    const isNew = !existsSync(path);
    const db = new Database(path);
    db.pragma("journal_mode = WAL");
    db.pragma("foreign_keys = ON");
    loadVec(db);
    const dim = isNew ? loadConfig().embedding.dimensions : (repo.detectVectorDimensions(db) ?? VECTOR_DIM);
    repo.initSchema(db, dim);

    const legacyPath = legacyIndexFile(dir);
    if (existsSync(legacyPath)) {
      if (repo.countChunksTotal(db) === 0) {
        migrateFromJson(db, legacyPath);
      }
    }

    return db;
  }

  static getFreshDbConn(ragDir?: string): Database.Database & Disposable {
    const db = RagDatabase.open(ragDir);
    return Object.assign(db, {
      [Symbol.dispose]: () => db.close(),
    });
  }
}

export const getDbConn   = () => RagDatabase.instance;
export const closeDbConn = () => { RagDatabase.close(); };

/**
 * Returns a brand-new, throwaway DB connection. **Bypasses the singleton** —
 * the caller is responsible for closing it. Use `getDbConn()` for normal access.
 */
export const getFreshDbConn = (dir?: string) => RagDatabase.getFreshDbConn(dir);

/** @deprecated Use getDbConn(). Alias for callers that still import openDb. */
export const openDb = getDbConn;
/** @deprecated Use getDbConn(). Alias for callers that still import getDb. */
export const getDb = getDbConn;

export { float32ToBuffer } from "./repository.ts";

export function initSchema(db: Database.Database, dimensions?: number) {
  repo.initSchema(db, dimensions);
}

/** Open a database at an explicit path (staging indexes). Caller closes it. */
export function openDbAt(path: string, dimensions: number): Database.Database & Disposable {
  ensureDir(dirname(path));
  const db = new Database(path);
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
  loadVec(db);
  repo.initSchema(db, dimensions);
  return Object.assign(db, { [Symbol.dispose]: () => db.close() });
}

function migrateFromJson(db: Database.Database, jsonPath: string): void {
  let data: IndexMeta;
  try {
    data = JSON.parse(readFileSync(jsonPath, "utf-8"));
  } catch { return; }

  if (!data.chunks || data.chunks.length === 0) {
    try { unlinkSync(jsonPath); } catch {}
    return;
  }

  const tx = db.transaction(() => {
    for (const c of data.chunks) {
      repo.insertChunk(db, {
        id: c.id, filePath: c.file, content: c.content,
        lineStart: c.lineStart, lineEnd: c.lineEnd,
        hash: c.hash, indexedAt: c.indexed, tokens: c.tokens,
      });
    }

    for (const [fp, info] of Object.entries(data.files || {})) {
      repo.replaceFile(db, fp, info.hash, info.chunks, info.indexed, info.size, info.embedded);
    }

    if (data.lastBuild) {
      repo.setMetadata(db, repo.MetadataKey.LastBuild, data.lastBuild);
    }
    if (data.embeddingModel) {
      repo.setMetadata(db, repo.MetadataKey.EmbeddingModel, data.embeddingModel);
    }
  });

  tx();
  try { unlinkSync(jsonPath); } catch {}
}

export function getIndexStats(db?: Database.Database): IndexStats {
  const dbConn = db ?? getDbConn();
  const { totalChunks, totalTokens } = repo.getChunkStats(dbConn);
  const compat = checkIndexCompatibility(dbConn, loadConfig());
  const dim = repo.detectVectorDimensions(dbConn);

  return {
    totalChunks: totalChunks,
    totalFiles: repo.countFiles(dbConn),
    totalTokens: totalTokens,
    embeddedCount: repo.getEmbeddedCount(dbConn),
    lastBuild: repo.getMetadata(dbConn, repo.MetadataKey.LastBuild) ?? "",
    embeddingModel: repo.getMetadata(dbConn, repo.MetadataKey.EmbeddingModel) ?? "",
    embeddingFingerprint: repo.getMetadata(dbConn, repo.MetadataKey.EmbeddingFingerprint),
    processingFingerprint: repo.getMetadata(dbConn, repo.MetadataKey.ProcessingFingerprint),
    embeddingDimensions: dim,
    needsRebuild: compat.ok ? false : true,
    rebuildReason: compat.ok ? undefined : compat.reason,
  };
}

/** No-op shim — JSON-era callers (and tests) compile against this. SQLite
 *  writes are committed by indexFiles' transactions; there is no separate
 *  save step. Kept on the public surface to avoid breaking external imports. */
export function saveIndex(_index: IndexMeta) { /* writes are transactional in indexFiles */ }

export function loadIndex(): IndexMeta {
  const db = getDbConn();
  const chunks = repo.getAllChunks(db) as Chunk[];

  const filesRaw = repo.listFiles(db);
  const files: IndexMeta["files"] = {};
  for (const f of filesRaw) {
    files[f.path] = { hash: f.hash, chunks: f.chunks, indexed: f.indexed, size: f.size, embedded: !!f.embedded };
  }

  return {
    chunks, files,
    lastBuild: repo.getMetadata(db, repo.MetadataKey.LastBuild) ?? "",
    embeddingModel: repo.getMetadata(db, repo.MetadataKey.EmbeddingModel),
  };
}

export function getEmbeddedCount(): number {
  const db = getDbConn();
  return repo.getEmbeddedCount(db);
}

export function getIndexedFiles(): repo.FileRow[] {
  return repo.listFiles(getDbConn());
}

export function listIndexedFilePaths(): string[] {
  return repo.listFilePaths(getDbConn());
}

export function pruneIndexedFile(filePath: string): void {
  repo.deleteIndexedFile(getDbConn(), filePath);
}

export function markFileUnembedded(filePath: string): void {
  repo.setFileEmbedded(getDbConn(), filePath, false);
}

/** Wipe chunks, vectors, files, and reset build metadata. Does not close the connection. */
export function clearIndex(db?: Database.Database): void {
  const dbConn = db ?? getDbConn();
  repo.clearAllVectors(dbConn);
  repo.setMetadata(dbConn, repo.MetadataKey.LastBuild, "");
  repo.setMetadata(dbConn, repo.MetadataKey.EmbeddingModel, "");
}
