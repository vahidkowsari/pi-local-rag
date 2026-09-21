import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type Database from "better-sqlite3";
import { getRagDir, dbFile, indexDbFile, activeManifestFile, ensureDir } from "./store.ts";
import * as repo from "./repository.ts";
import type { RagConfig } from "./config.ts";
import {
  embeddingFingerprintFromConfig,
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

  if (chunks === 0) {
    return { ok: true, embedding: desiredEmb, processing: desiredProc, empty: true };
  }
  if (!storedEmb || !storedProc) {
    return { ok: false, needsRebuild: true, reason: "Existing index has no fingerprint. Run /rag rebuild --force to rebuild under the current embedding contract." };
  }
  if (tableDim !== undefined && tableDim !== storedEmb.dimensions) {
    return { ok: false, needsRebuild: true, reason: `Vector table dimension ${tableDim} does not match fingerprint ${storedEmb.dimensions}.` };
  }
  if (!fingerprintsEqual(storedEmb, desiredEmb)) {
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

export function stagingDbPath(config: RagConfig, ragDir = getRagDir()): { indexId: string; dbPath: string; relativeDbPath: string } {
  const emb = embeddingFingerprintFromConfig(config.embedding);
  const proc = processingFingerprintFromConfig(config);
  const indexId = indexIdFor(emb, proc);
  const dbPath = indexDbFile(ragDir, indexId);
  return { indexId, dbPath, relativeDbPath: join("indexes", indexId, "rag.db") };
}

export function prepareStagingDir(config: RagConfig, ragDir = getRagDir()): { indexId: string; dbPath: string; relativeDbPath: string } {
  const spec = stagingDbPath(config, ragDir);
  const active = readActiveManifest(ragDir);
  if (active?.indexId === spec.indexId) {
    // Same contract as the live index: callers should rebuild in place, not stage.
    return spec;
  }
  if (existsSync(spec.dbPath)) {
    rmSync(dirname(spec.dbPath), { recursive: true, force: true });
  }
  mkdirSync(dirname(spec.dbPath), { recursive: true });
  return spec;
}

export function checkpointAndClose(db: Database.Database): void {
  try { db.pragma("wal_checkpoint(TRUNCATE)"); } catch { /* ignore */ }
  db.close();
}
