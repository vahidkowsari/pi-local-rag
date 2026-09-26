import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type Database from "better-sqlite3";
import { getRagDir, dbFile, activeManifestFile, ensureDir } from "./store.ts";
import * as repo from "./repository.ts";
import type { RagConfig } from "./config.ts";
import {
  embeddingFingerprintFromConfig,
  embeddingFingerprintsEqual,
  fingerprintsEqual,
  indexIdFor,
  parseEmbeddingFingerprint,
  parseProcessingFingerprint,
  processingFingerprintFromConfig,
  serializeFingerprint,
  type EmbeddingFingerprint,
  type ProcessingFingerprint,
} from "./fingerprint.ts";

export interface ActiveManifest {
  version: 1;
  indexId: string;
  relativeDbPath: string;
  embeddingFingerprint: string;
  processingFingerprint: string;
  createdAt: string;
}

export type CompatResult =
  | { ok: true; embedding: EmbeddingFingerprint; processing: ProcessingFingerprint; empty: boolean }
  | { ok: false; needsRebuild: true; reason: string };

export class IndexIncompatibleError extends Error {
  readonly needsRebuild = true;
  readonly reason: string;
  constructor(reason: string) {
    super(reason);
    this.name = "IndexIncompatibleError";
    this.reason = reason;
  }
}

export function newGeneration(): string {
  return `${Date.now().toString(36)}-${randomBytes(4).toString("hex")}`;
}

let writeChain: Promise<unknown> = Promise.resolve();

export function withIndexWriteLock<T>(fn: () => Promise<T>): Promise<T> {
  const run = writeChain.then(fn, fn);
  writeChain = run.then(() => undefined, () => undefined);
  return run;
}

export function readActiveManifest(ragDir = getRagDir()): ActiveManifest | undefined {
  const path = activeManifestFile(ragDir);
  if (!existsSync(path)) return undefined;
  try {
    const raw = JSON.parse(readFileSync(path, "utf-8")) as ActiveManifest;
    if (raw.version !== 1 || !raw.indexId || !raw.relativeDbPath) return undefined;
    return raw;
  } catch {
    return undefined;
  }
}

export function resolveActiveDbPath(ragDir = getRagDir()): string {
  ensureDir(ragDir);
  const manifest = readActiveManifest(ragDir);
  if (manifest) {
    const abs = join(ragDir, manifest.relativeDbPath);
    if (existsSync(abs)) return abs;
  }
  return dbFile(ragDir);
}

export function publishActiveManifest(ragDir: string, manifest: ActiveManifest): void {
  const dest = activeManifestFile(ragDir);
  const tmp = dest + ".tmp";
  writeFileSync(tmp, JSON.stringify(manifest, null, 2));
  renameSync(tmp, dest);
}

export function checkIndexCompatibility(db: Database.Database, config: RagConfig): CompatResult {
  const desiredEmb = embeddingFingerprintFromConfig(config.embedding);
  const desiredProc = processingFingerprintFromConfig(config);
  const chunks = repo.countChunksTotal(db);
  const storedEmb = parseEmbeddingFingerprint(repo.getMetadata(db, repo.MetadataKey.EmbeddingFingerprint));
  const storedProc = parseProcessingFingerprint(repo.getMetadata(db, repo.MetadataKey.ProcessingFingerprint));
  const tableDim = repo.detectVectorDimensions(db);

  if (tableDim !== undefined && tableDim !== desiredEmb.dimensions) {
    return {
      ok: false, needsRebuild: true,
      reason: `Vector table dimension ${tableDim} does not match requested ${desiredEmb.dimensions}. Rebuild required.`,
    };
  }
  if (chunks === 0) {
    if (storedEmb && !embeddingFingerprintsEqual(storedEmb, desiredEmb)) {
      return {
        ok: false, needsRebuild: true,
        reason: `Index embedding contract is ${storedEmb.provider}/${storedEmb.model}/${storedEmb.dimensions}; config requests ${desiredEmb.provider}/${desiredEmb.model}/${desiredEmb.dimensions}. Rebuild required.`,
      };
    }
    if (storedProc && !fingerprintsEqual(storedProc, desiredProc)) {
      return { ok: false, needsRebuild: true, reason: "Document processing fingerprint changed (parser/chunker). Rebuild required; file hashes from the old processor will not be reused." };
    }
    return { ok: true, embedding: desiredEmb, processing: desiredProc, empty: true };
  }
  if (!storedEmb || !storedProc) {
    return { ok: false, needsRebuild: true, reason: "Existing index has no fingerprint. Run /rag rebuild --force to rebuild under the current embedding contract." };
  }
  if (tableDim !== undefined && tableDim !== storedEmb.dimensions) {
    return { ok: false, needsRebuild: true, reason: `Vector table dimension ${tableDim} does not match fingerprint ${storedEmb.dimensions}.` };
  }
  if (!embeddingFingerprintsEqual(storedEmb, desiredEmb)) {
    return { ok: false, needsRebuild: true, reason: `Index embedding contract is ${storedEmb.provider}/${storedEmb.model}/${storedEmb.dimensions}; config requests ${desiredEmb.provider}/${desiredEmb.model}/${desiredEmb.dimensions}. Rebuild required.` };
  }
  if (!fingerprintsEqual(storedProc, desiredProc)) {
    return { ok: false, needsRebuild: true, reason: "Document processing fingerprint changed (parser/chunker). Rebuild required; file hashes from the old processor will not be reused." };
  }
  return { ok: true, embedding: storedEmb, processing: storedProc, empty: false };
}

export function stampFingerprints(db: Database.Database, config: RagConfig): void {
  const emb = embeddingFingerprintFromConfig(config.embedding);
  const proc = processingFingerprintFromConfig(config);
  repo.setMetadata(db, repo.MetadataKey.EmbeddingFingerprint, serializeFingerprint(emb));
  repo.setMetadata(db, repo.MetadataKey.ProcessingFingerprint, serializeFingerprint(proc));
  repo.setMetadata(db, repo.MetadataKey.EmbeddingDimensions, String(emb.dimensions));
  repo.setMetadata(db, repo.MetadataKey.EmbeddingModel, emb.model);
}

export interface StagingSpec {
  indexId: string;
  generation: string;
  dbPath: string;
  relativeDbPath: string;
}

export function stagingDbPath(config: RagConfig, ragDir = getRagDir(), generation = newGeneration()): StagingSpec {
  const emb = embeddingFingerprintFromConfig(config.embedding);
  const proc = processingFingerprintFromConfig(config);
  const indexId = indexIdFor(emb, proc);
  const relativeDbPath = join("staging", generation, "rag.db");
  return { indexId, generation, dbPath: join(ragDir, relativeDbPath), relativeDbPath };
}

/**
 * Create a unique unpublished staging directory. Never deletes a published
 * index — previous generations stay on disk until the user removes them.
 */
export function prepareStagingDir(config: RagConfig, ragDir = getRagDir()): StagingSpec {
  const spec = stagingDbPath(config, ragDir);
  mkdirSync(dirname(spec.dbPath), { recursive: true });
  return spec;
}

/** Move a finished staging tree into indexes/ and point active.json at it. */
export function finalizeStaging(
  ragDir: string,
  spec: StagingSpec,
  config: RagConfig,
): ActiveManifest {
  const destRelDir = join("indexes", spec.indexId, spec.generation);
  const destDir = join(ragDir, destRelDir);
  mkdirSync(join(ragDir, "indexes", spec.indexId), { recursive: true });
  const stagingDir = dirname(spec.dbPath);
  if (existsSync(destDir)) {
    throw new Error(`Refusing to overwrite existing index generation at ${destDir}`);
  }
  renameSync(stagingDir, destDir);
  const manifest: ActiveManifest = {
    version: 1,
    indexId: spec.indexId,
    relativeDbPath: join(destRelDir, "rag.db"),
    embeddingFingerprint: serializeFingerprint(embeddingFingerprintFromConfig(config.embedding)),
    processingFingerprint: serializeFingerprint(processingFingerprintFromConfig(config)),
    createdAt: new Date().toISOString(),
  };
  publishActiveManifest(ragDir, manifest);
  return manifest;
}

export function checkpointAndClose(db: Database.Database): void {
  try { db.pragma("wal_checkpoint(TRUNCATE)"); } catch { /* ignore */ }
  db.close();
}
