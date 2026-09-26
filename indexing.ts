import { basename } from "node:path";
import type Database from "better-sqlite3";
import { getDbConn, closeDbConn, openDbAt, type IndexStats } from "./db.ts";
import { VECTOR_DIM, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS } from "./constants.ts";
import { chunkBlocks, estimateTokens, sha256, type TextChunk } from "./chunking.ts";
import { extractBlocks } from "./parsing.ts";
import * as repo from "./repository.ts";
import { loadConfig, type RagConfig } from "./config.ts";
import { createEmbeddingProvider, embeddingProviderForIndex } from "./providers/embedding/factory.ts";
import {
  checkIndexCompatibility, stampFingerprints, withIndexWriteLock,
  prepareStagingDir, finalizeStaging, checkpointAndClose,
} from "./index-manager.ts";
import { isAbortError, throwIfAborted } from "./abort.ts";
import { getRagDir } from "./store.ts";
import { getModelSpec } from "./provider-config.ts";

export interface ProgressCallbacks {
  onFile?: (current: number, total: number, filename: string, skipped: number) => void;
  onChunk?: (fileChunk: number, totalChunks: number, filename: string) => void;
  /** Fires after each cross-file embed micro-batch completes. `done` is the
   *  number of chunks embedded so far across all files; `total` is the grand
   *  total. Used by the TUI to render live embedding progress instead of
   *  freezing at "Rebuilding 100%". */
  onEmbed?: (done: number, total: number) => void;
  onSave?: () => void;
}

export function isIndexStale(index: IndexStats, maxAgeMs = 24 * 60 * 60 * 1000): boolean {
  if (!index.lastBuild) return false;
  return Date.now() - new Date(index.lastBuild).getTime() > maxAgeMs;
}

/** Local indexes keep 24h refresh. Cloud indexes refresh only when cloudAutoRefresh is on. */
export function shouldAutoRefresh(config: RagConfig, stats: IndexStats): boolean {
  if (!isIndexStale(stats)) return false;
  let isLocal = false;
  try {
    isLocal = getModelSpec(config.embedding.provider, "embedding", config.embedding.model).type === "transformers";
  } catch {
    // Config validation reports the detailed provider error. Do not silently
    // enable a cloud refresh when the registry cannot be resolved.
    isLocal = false;
  }
  if (!isLocal && config.cloudAutoRefresh !== true) return false;
  return true;
}

const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

let _suppressStderr = false;

function stderrProgress(msg: string) {
  if (_suppressStderr) return;
  process.stderr.write(`\r\x1b[2K${msg}`);
}

interface FileWork {
  fp: string;
  hash: string;
  size: number;
  rawChunks: { content: string; lineStart: number; lineEnd: number; hash: string; pageStart?: number | null; pageEnd?: number | null; section?: string | null; chunkIndex?: number }[];
  _vectors?: number[][];
}

export interface IndexFilesResult {
  indexed: number;
  chunks: number;
  skipped: number;
  failed: number;
  errors: string[];
  durationMs: number;
}

function isValidVector(v: number[] | undefined, dim = VECTOR_DIM): v is number[] {
  if (!v || v.length !== dim) return false;
  let normSq = 0;
  for (const x of v) {
    if (!Number.isFinite(x)) return false;
    normSq += x * x;
  }
  return normSq > 0;
}

function fileVectorsReady(fw: FileWork, dim: number): boolean {
  if (fw.rawChunks.length === 0) return true;
  const vectors = fw._vectors;
  if (!vectors || vectors.length !== fw.rawChunks.length) return false;
  return vectors.every(v => isValidVector(v, dim));
}

function replaceFileIndex(database: Database.Database, fw: FileWork, indexedAt: string): number {
  return database.transaction(() => {
    repo.deleteVectorsForFile(database, fw.fp);
    repo.deleteChunksForFile(database, fw.fp);
    let n = 0;
    const vectors = fw._vectors;
    for (let j = 0; j < fw.rawChunks.length; j++) {
      const c = fw.rawChunks[j];
      const chunkResult = repo.insertChunk(database, {
        id: `${sha256(fw.fp)}-${c.chunkIndex ?? j}`,
        filePath: fw.fp, content: c.content,
        lineStart: c.lineStart, lineEnd: c.lineEnd, hash: c.hash,
        indexedAt, tokens: estimateTokens(c.content),
        pageStart: c.pageStart ?? null,
        pageEnd: c.pageEnd ?? null,
        section: c.section ?? null,
        chunkIndex: c.chunkIndex ?? j,
      });
      repo.insertVector(database, Number(chunkResult.lastInsertRowid), vectors![j]);
      n++;
    }
    repo.upsertFile(database, fw.fp, fw.hash, fw.rawChunks.length, indexedAt, fw.size, true, {
      documentId: sha256(fw.fp),
      title: basename(fw.fp),
    });
    return n;
  })();
}

export async function indexFiles(
  paths: string[],
  progress?: ProgressCallbacks,
  _db?: Database.Database,
  force?: boolean,
  signal?: AbortSignal,
): Promise<IndexFilesResult> {
  return withIndexWriteLock(() => indexFilesUnlocked(paths, progress, _db, force, signal));
}

export async function rebuildWithSwitch(
  paths: string[],
  progress?: ProgressCallbacks,
  force?: boolean,
  droppedFiles?: string[],
  signal?: AbortSignal,
): Promise<IndexFilesResult> {
  return withIndexWriteLock(async () => {
    const config = loadConfig();
    const ragDir = getRagDir();
    const live = getDbConn();
    const compat = checkIndexCompatibility(live, config);
    const useStaging = !!force || !compat.ok;
    if (!useStaging) {
      const result = await indexFilesUnlocked(paths, progress, live, force, signal);
      throwIfAborted(signal);
      if (result.failed === 0 && droppedFiles?.length) {
        for (const f of droppedFiles) repo.deleteIndexedFile(live, f);
      }
      return result;
    }
    const spec = prepareStagingDir(config, ragDir);
    const staging = openDbAt(spec.dbPath, config.embedding.dimensions);
    try {
      const result = await indexFilesUnlocked(paths, progress, staging, true, signal);
      if (result.failed > 0) {
        checkpointAndClose(staging);
        return result;
      }
      const chunks = repo.countChunksTotal(staging);
      const vecs = repo.getEmbeddedCount(staging);
      if (chunks > 0 && vecs !== chunks) {
        checkpointAndClose(staging);
        return {
          ...result,
          failed: result.failed + 1,
          errors: [...result.errors, `vector coverage ${vecs}/${chunks}; refusing to publish staging`],
        };
      }
      throwIfAborted(signal);
      stampFingerprints(staging, config);
      checkpointAndClose(staging);
      throwIfAborted(signal);
      finalizeStaging(ragDir, spec, config);
      closeDbConn();
      return result;
    } catch (err) {
      try { staging.close(); } catch { /* ignore */ }
      throw err;
    }
  });
}

async function indexFilesUnlocked(
  paths: string[],
  progress?: ProgressCallbacks,
  _db?: Database.Database,
  force?: boolean,
  signal?: AbortSignal,
): Promise<IndexFilesResult> {
  const hadCallbacks = !!progress;
  if (hadCallbacks) _suppressStderr = true;
  const database = _db ?? getDbConn();
  const startMs = Date.now();
  const total = paths.length;
  const empty = (): IndexFilesResult => ({
    indexed: 0, chunks: 0, skipped: 0, failed: 0, errors: [], durationMs: Date.now() - startMs,
  });

  try {
    throwIfAborted(signal);
    if (total === 0) return empty();

    const config = loadConfig();
    const compat = checkIndexCompatibility(database, config);
    if (!compat.ok) {
      return {
        indexed: 0, chunks: 0, skipped: 0, failed: paths.length,
        errors: [compat.reason], durationMs: Date.now() - startMs,
      };
    }
    const provider = compat.empty
      ? createEmbeddingProvider(config)
      : embeddingProviderForIndex(database, config);

    // Phase 1: parallel read + chunk; DB ops on main thread
    const CONCURRENCY = 32;
    const YIELD_INTERVAL = 64;

    interface ReadResult { fp: string; hash: string; size: number; raw: TextChunk[] }

    const readQueue: ReadResult[] = [];
    let readQueueDone = false;
    const readErrors: string[] = [];
    let resolveRead: (() => void) | null = null;
    const notifyRead = () => { resolveRead?.(); resolveRead = null; };
    const waitRead = () => new Promise<void>(r => { resolveRead = r; });

    const workerCount = Math.min(CONCURRENCY, paths.length);
    let pathsIdx = 0;
    let producersDone = 0;
    const producers: Promise<void>[] = [];
    for (let w = 0; w < workerCount; w++) {
      producers.push((async () => {
        while (true) {
          throwIfAborted(signal);
          const i = pathsIdx++;
          if (i >= paths.length) { producersDone++; if (producersDone >= workerCount) { readQueueDone = true; notifyRead(); } return; }
          try {
            const parsed = await extractBlocks(paths[i]);
            const raw = chunkBlocks(parsed.blocks, {
              targetTokens: CHUNK_TARGET_TOKENS,
              maxTokens: CHUNK_MAX_TOKENS,
              overlapTokens: CHUNK_OVERLAP_TOKENS,
            });
            readQueue.push({ fp: paths[i], hash: parsed.hash, size: parsed.size, raw });
            notifyRead();
          } catch (err) {
            if (isAbortError(err)) throw err;
            const msg = err instanceof Error ? err.message : String(err);
            readErrors.push(`${paths[i]}: ${msg}`);
            stderrProgress(`[${i + 1}/${total}] ERROR ${basename(paths[i])}: ${msg}`);
            notifyRead();
          }
        }
      })());
    }
    const producersDoneP = Promise.all(producers).catch(err => {
      readQueueDone = true;
      notifyRead();
      throw err;
    });

    const toIndex: FileWork[] = [];
    let skipped = 0;
    let processedCount = 0;
    let nextYieldAt = 0;

    const drainReads = () => {
      while (readQueue.length > 0) {
        const r = readQueue.shift()!;
        processedCount++;
        const name = basename(r.fp);

        const existing = repo.getFile(database, r.fp);
        if (!force && existing?.hash === r.hash && existing?.embedded) {
          skipped++;
          progress?.onFile?.(processedCount, total, name, skipped);
          continue;
        }

        // Do not delete the live rows here. Old chunks/vectors stay searchable
        // until a validated replacement commits in replaceFileIndex.

        const rawChunks = r.raw.map((c, idx) => ({
          ...c,
          hash: sha256(c.content),
          chunkIndex: c.chunkIndex ?? idx,
        }));
        stderrProgress(`[${processedCount}/${total}] chunked ${name} (${rawChunks.length} chunks)`);
        progress?.onFile?.(processedCount, total, name, skipped);

        toIndex.push({ fp: r.fp, hash: r.hash, size: r.size, rawChunks });
      }
    };

    const maybeYield = async () => {
      if (processedCount >= nextYieldAt) {
        nextYieldAt = processedCount + YIELD_INTERVAL;
        await yield_();
      }
    };

    while (!readQueueDone || readQueue.length > 0) {
      throwIfAborted(signal);
      drainReads();
      if (!readQueueDone) await waitRead();
      await maybeYield();
    }
    drainReads();
    await producersDoneP;
    await yield_();
    throwIfAborted(signal);

    // Phase 2: embed in cross-file groups
    const EMBED_GROUP_TARGET = 256;
    const groupChunks: { fw: FileWork; ci: number }[] = [];
    let globalChunkIdx = 0;
    const totalChunks = toIndex.reduce((s, f) => s + f.rawChunks.length, 0);

    const flushGroup = async () => {
      if (groupChunks.length === 0) return;
      const texts = groupChunks.map(g => g.fw.rawChunks[g.ci].content);
      stderrProgress(`Embedding ${globalChunkIdx - groupChunks.length + 1}…${globalChunkIdx}/${totalChunks} chunks`);
      throwIfAborted(signal);
      const vectors = await provider.embedDocuments(texts, { signal });
      throwIfAborted(signal);
      if (!Array.isArray(vectors) || vectors.length !== texts.length) {
        throw new Error(`embedBatch returned ${Array.isArray(vectors) ? vectors.length : "non-array"} vectors for ${texts.length} texts`);
      }
      for (let vi = 0; vi < groupChunks.length; vi++) {
        const g = groupChunks[vi];
        g.fw._vectors ??= new Array(g.fw.rawChunks.length);
        g.fw._vectors[g.ci] = vectors[vi];
      }
      progress?.onEmbed?.(globalChunkIdx, totalChunks);
      groupChunks.length = 0;
      // Yield so the TUI can render the progress update before the next batch.
      await yield_();
    };

    try {
      for (const fw of toIndex) {
        for (let j = 0; j < fw.rawChunks.length; j++) {
          groupChunks.push({ fw, ci: j });
          globalChunkIdx++;
          if (groupChunks.length >= EMBED_GROUP_TARGET) await flushGroup();
        }
      }
      await flushGroup();
      throwIfAborted(signal);
    } catch (err) {
      if (isAbortError(err)) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      return {
        indexed: 0,
        chunks: 0,
        skipped,
        failed: toIndex.length + readErrors.length,
        errors: [...readErrors, ...toIndex.map(fw => `${fw.fp}: ${msg}`)],
        durationMs: Date.now() - startMs,
      };
    }

    throwIfAborted(signal);

    // Phase 3: replace each file in its own transaction only after vectors validate.
    let chunked = 0;
    let indexed = 0;
    const errors: string[] = [];
    const indexedAt = new Date().toISOString();
    for (const fw of toIndex) {
      throwIfAborted(signal);
      if (!fileVectorsReady(fw, provider.dimensions)) {
        errors.push(`${fw.fp}: missing or invalid embedding vectors; keeping previous index`);
        continue;
      }
      try {
        chunked += replaceFileIndex(database, fw, indexedAt);
        indexed++;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        errors.push(`${fw.fp}: ${msg}`);
      }
    }

    throwIfAborted(signal);
    if (!hadCallbacks) process.stderr.write(`\r\x1b[2K`);
    progress?.onSave?.();
    if (indexed > 0 || (toIndex.length === 0 && errors.length === 0)) {
      repo.setMetadata(database, repo.MetadataKey.LastBuild, new Date().toISOString());
      repo.setMetadata(database, repo.MetadataKey.EmbeddingModel, provider.model);
      stampFingerprints(database, config);
    }

    return {
      indexed, chunks: chunked, skipped,
      failed: errors.length + readErrors.length,
      errors: [...readErrors, ...errors],
      durationMs: Date.now() - startMs,
    };
  } finally {
    if (hadCallbacks) _suppressStderr = false;
  }
}
