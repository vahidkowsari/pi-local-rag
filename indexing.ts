import { basename } from "node:path";
import { EMBEDDING_MODEL } from "./constants.ts";
import { loadIndex, saveIndex, type IndexMeta } from "./index-store.ts";
import { extractText, chunkText, sha256 } from "./chunking.ts";
import { embedBatch } from "./embed.ts";

/** True when the index's `lastBuild` timestamp is older than `maxAgeMs` (default 24 h). */
export function isIndexStale(index: IndexMeta, maxAgeMs = 24 * 60 * 60 * 1000): boolean {
  if (!index.lastBuild) return false;
  return Date.now() - new Date(index.lastBuild).getTime() > maxAgeMs;
}

export interface ProgressCallbacks {
  onFile?: (current: number, total: number, filename: string, skipped: number) => void;
  onChunk?: (fileChunk: number, totalChunks: number, filename: string) => void;
  onSave?: () => void;
}

/** Yield to the event loop so the TUI can re-render between heavy operations */
export const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

/** Write overwriting progress line to stderr (visible in terminal even during tool calls) */
export function stderrProgress(msg: string) {
  process.stderr.write(`\r\x1b[2K${msg}`);
}

// Bounded concurrency for Phase 1 (read + chunk + hash). Embedding stays
// strictly sequential — the ONNX pipeline is single-threaded internally so
// parallelizing the embed step would just thrash the runtime.
const READ_CONCURRENCY = 32;

type PendingChunk = { content: string; lineStart: number; lineEnd: number; hash: string };
type Pending =
  | { fp: string; name: string; skip: true }
  | { fp: string; name: string; hash: string; size: number; rawChunks: PendingChunk[] };

export async function indexFiles(
  paths: string[],
  progress?: ProgressCallbacks,
): Promise<{ indexed: number; chunks: number; skipped: number; durationMs: number }> {
  const startMs = Date.now();
  if (!paths.length) return { indexed: 0, chunks: 0, skipped: 0, durationMs: 0 };

  const index = loadIndex();
  const total = paths.length;

  // ── Phase 1: parallel read + chunk + hash (bounded to READ_CONCURRENCY)
  // Pure I/O + CPU; no shared mutable state. Skip-check happens here too so
  // we don't spend chunking work on unchanged files.
  const pending: Pending[] = new Array(paths.length);
  let cursor = 0;
  let readDone = 0;

  async function readOne(i: number): Promise<Pending> {
    const fp = paths[i];
    const name = basename(fp);
    try {
      const { text, hash, size } = await extractText(fp);
      if (index.files[fp]?.hash === hash && index.files[fp]?.embedded) {
        return { fp, name, skip: true };
      }
      const rawChunks: PendingChunk[] = chunkText(text).map(c => ({
        content: c.content,
        lineStart: c.lineStart,
        lineEnd: c.lineEnd,
        hash: sha256(c.content),
      }));
      return { fp, name, hash, size, rawChunks };
    } catch {
      // Treat unreadable / parse-failing files as skipped rather than aborting.
      return { fp, name, skip: true };
    }
  }

  async function readWorker() {
    while (cursor < paths.length) {
      const i = cursor++;
      pending[i] = await readOne(i);
      readDone++;
      if (readDone % 64 === 0) {
        stderrProgress(`reading ${readDone}/${total}…`);
        await yield_();
      }
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(READ_CONCURRENCY, paths.length) }, readWorker),
  );

  // ── Phase 2: sequential embed + index update
  let indexed = 0, chunked = 0, skipped = 0;
  for (let i = 0; i < pending.length; i++) {
    const p = pending[i];
    const pct = Math.round(((i + 1) / total) * 100);

    if ("skip" in p) {
      skipped++;
      stderrProgress(`[${i + 1}/${total}] ${pct}% skipped ${p.name}`);
      progress?.onFile?.(i + 1, total, p.name, skipped);
      if ((i + 1) % 64 === 0) await yield_();
      continue;
    }

    index.chunks = index.chunks.filter(c => c.file !== p.fp);

    stderrProgress(`[${i + 1}/${total}] ${pct}% embedding ${p.name} (${p.rawChunks.length} chunks)`);
    progress?.onFile?.(i + 1, total, p.name, skipped);
    await yield_();

    const vectors = await embedBatch(
      p.rawChunks.map(c => c.content),
      (ci) => {
        stderrProgress(`[${i + 1}/${total}] ${pct}% ${p.name} — chunk ${ci}/${p.rawChunks.length}`);
        progress?.onChunk?.(ci, p.rawChunks.length, p.name);
      },
    );

    const fileId = sha256(p.fp);
    const nowIso = new Date().toISOString();
    for (let j = 0; j < p.rawChunks.length; j++) {
      const chunk = p.rawChunks[j];
      index.chunks.push({
        id: `${fileId}-${chunk.lineStart}`,
        file: p.fp,
        content: chunk.content,
        lineStart: chunk.lineStart,
        lineEnd: chunk.lineEnd,
        hash: chunk.hash,
        indexed: nowIso,
        tokens: Math.ceil(chunk.content.length / 4),
        vector: vectors[j],
      });
      chunked++;
    }

    index.files[p.fp] = { hash: p.hash, chunks: p.rawChunks.length, indexed: nowIso, size: p.size, embedded: true };
    indexed++;
  }

  // Clear stderr progress line
  process.stderr.write(`\r\x1b[2K`);

  progress?.onSave?.();
  index.lastBuild = new Date().toISOString();
  index.embeddingModel = EMBEDDING_MODEL;
  saveIndex(index);
  return { indexed, chunks: chunked, skipped, durationMs: Date.now() - startMs };
}
