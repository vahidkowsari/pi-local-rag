import type Database from "better-sqlite3";
import { getDbConn, type Chunk } from "./db.ts";
import * as repo from "./repository.ts";
import { embeddingProviderForIndex } from "./providers/embedding/factory.ts";
import { CANDIDATE_TOP_K_MAX, loadConfig, type RagConfig } from "./config.ts";
import { checkIndexCompatibility, IndexIncompatibleError } from "./index-manager.ts";
import { isAbortError, throwIfAborted } from "./abort.ts";
import { HttpError } from "./providers/http.ts";

export interface ScoredChunk {
  chunk: Chunk;
  bm25: number;
  vector: number;
  hybrid: number;
  degraded?: string;
}

export interface HybridSearchOptions {
  signal?: AbortSignal;
  config?: RagConfig;
}

export interface HybridSearchResult {
  hits: ScoredChunk[];
  /** Set when query embedding failed and the request fell back to BM25. Independent of hits.length. */
  degraded?: string;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function normalize(scores: number[]): number[] {
  const max = Math.max(...scores);
  const min = Math.min(...scores);
  const range = max - min;
  if (range === 0) return scores.map(() => 0);
  return scores.map(s => (s - min) / range);
}

/**
 * Map FTS5 `bm25()` raw scores onto [0, 1] internal relevance.
 * SQLite FTS5 BM25 is lower (more negative) for better matches, so this
 * inverts the range. Equal scores (including a single candidate) map to 1
 * so a lone hit is not filtered out as hybrid=0.
 */
export function bm25ToRelevance(rawScores: number[]): number[] {
  if (rawScores.length === 0) return [];
  const max = Math.max(...rawScores);
  const min = Math.min(...rawScores);
  const range = max - min;
  if (range === 0) return rawScores.map(() => 1);
  return rawScores.map(s => (max - s) / range);
}

function l2ToCosine(l2Dist: number): number {
  return 1 - (l2Dist * l2Dist) / 2;
}

/**
 * Hybrid search using SQLite FTS5 (BM25) + sqlite-vec (vector).
 */
function isTransientEmbedError(err: unknown): boolean {
  if (isAbortError(err)) return false;
  if (err instanceof IndexIncompatibleError) return false;
  if (err instanceof HttpError) return err.retryable || err.status === 0;
  const msg = err instanceof Error ? err.message : String(err);
  return /network|ECONN|ETIMEDOUT|ENOTFOUND|fetch|timeout|503|502|429|model down/i.test(msg);
}

export async function hybridSearchDetailed(
  query: string,
  limit = 10,
  alpha = 0.4,
  _db?: Database.Database,
  opts?: HybridSearchOptions,
): Promise<HybridSearchResult> {
  const database = _db ?? getDbConn();
  const config = opts?.config ?? loadConfig();
  throwIfAborted(opts?.signal);

  if (!repo.hasAnyChunks(database)) return { hits: [] };

  const compat = checkIndexCompatibility(database, config);
  if (!compat.ok) throw new IndexIncompatibleError(compat.reason);

  const capped = Math.min(CANDIDATE_TOP_K_MAX, Math.max(1, limit));
  const ftsQuery = query.split(/\s+/).map(t => `"${t.replace(/"/g, '""')}"`).join(" ");
  const ftsLimit = Math.min(CANDIDATE_TOP_K_MAX, Math.max(capped * 20, 200));
  const ftsResults = repo.searchFts(database, ftsQuery, ftsLimit);

  let vecResults: repo.VecMatch[] = [];
  let embedDegraded: string | undefined;
  const embeddedCount = repo.getEmbeddedCount(database);
  if (embeddedCount > 0) {
    try {
      throwIfAborted(opts?.signal);
      const provider = embeddingProviderForIndex(database, config);
      const queryVec = await provider.embedQuery(query, { signal: opts?.signal });
      const vecLimit = Math.min(CANDIDATE_TOP_K_MAX, Math.max(capped * 10, 100));
      vecResults = repo.searchVectors(database, queryVec, vecLimit);
    } catch (err) {
      if (isAbortError(err)) throw err;
      if (err instanceof IndexIncompatibleError) throw err;
      if (!isTransientEmbedError(err)) throw err;
      embedDegraded = `query embedding failed, BM25 only: ${err instanceof Error ? err.message : String(err)}`;
    }
  }

  const ftsRowIds = new Set(ftsResults.map(r => r.rowid));
  const vecRowIds = new Set(vecResults.map(r => r.rowid));
  const allRowIds: Set<number> = new Set([...ftsRowIds, ...vecRowIds]);

  if (allRowIds.size === 0) return { hits: [], degraded: embedDegraded };

  const chunks = repo.getChunksByRowids(database, Array.from(allRowIds));

  const chunkMap = new Map<number, typeof chunks[0]>();
  for (const c of chunks) chunkMap.set(c.rowid, c);

  const bm25Scores = ftsResults.map(r => r.bm25_score);
  const hasBm25 = bm25Scores.length > 0;
  const distances = vecResults.map(r => r.distance);
  const hasVectors = distances.length > 0;

  // Normalize BM25: FTS5 bm25() is lower-is-better; invert onto [0, 1].
  const bm25NormMap = new Map<number, number>();
  if (hasBm25) {
    const relevance = bm25ToRelevance(bm25Scores);
    for (let i = 0; i < ftsResults.length; i++) {
      bm25NormMap.set(ftsResults[i].rowid, relevance[i]);
    }
  }

  // Normalize distances → cosine → min-max
  const vecNormMap = new Map<number, number>();
  if (hasVectors) {
    for (const r of vecResults) {
      vecNormMap.set(r.rowid, l2ToCosine(r.distance));
    }
    const cosines = Array.from(vecNormMap.values());
    const cosMax = Math.max(...cosines);
    const cosMin = Math.min(...cosines);
    const cosRange = cosMax - cosMin;
    if (cosRange > 0) {
      const normalized = new Map<number, number>();
      for (const [rowid, cos] of vecNormMap) {
        normalized.set(rowid, (cos - cosMin) / cosRange);
      }
      vecNormMap.clear();
      for (const [k, v] of normalized) vecNormMap.set(k, v);
    } else {
      for (const k of vecNormMap.keys()) vecNormMap.set(k, 1);
    }
  }

  // Build scored results
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);
  const scored: ScoredChunk[] = [];

  for (const rowid of allRowIds) {
    const c = chunkMap.get(rowid);
    if (!c) continue;

    const bm25Norm = bm25NormMap.get(rowid) ?? 0;
    const vecNorm = vecNormMap.get(rowid) ?? 0;

      let bm25Final = bm25Norm;
      // Boost when the first meaningful query term appears in the file path.
      // Guard on terms[0]: an empty/short query makes includes("") always true,
      // which would spuriously boost every result.
      if (terms[0] && c.file_path.toLowerCase().includes(terms[0])) {
        bm25Final = Math.min(1, bm25Final * 1.5);
      }

    const hybrid = hasVectors
      ? alpha * bm25Final + (1 - alpha) * vecNorm
      : bm25Final;

    scored.push({
      chunk: {
        id: c.id, file: c.file_path, content: c.chunk_content,
        lineStart: c.line_start, lineEnd: c.line_end,
        hash: c.chunk_hash, indexed: c.indexed_at, tokens: c.tokens,
        pageStart: c.page_start ?? null,
        pageEnd: c.page_end ?? null,
        section: c.section ?? null,
        chunkIndex: c.chunk_index ?? 0,
      },
      bm25: bm25Final, vector: vecNorm, hybrid,
      degraded: embedDegraded,
    });
  }

  const hits = scored
    .filter(s => s.hybrid > 0)
    .sort((a, b) => b.hybrid - a.hybrid)
    .slice(0, limit);
  return { hits, degraded: embedDegraded };
}

export async function hybridSearch(
  query: string,
  limit = 10,
  alpha = 0.4,
  _db?: Database.Database,
  opts?: HybridSearchOptions,
): Promise<ScoredChunk[]> {
  return (await hybridSearchDetailed(query, limit, alpha, _db, opts)).hits;
}
