import type Database from "better-sqlite3";
import { hybridSearch, type ScoredChunk } from "./search.ts";
import { loadConfig, type RagConfig } from "./config.ts";
import { createReranker } from "./providers/reranker/factory.ts";
import { NoneReranker } from "./providers/reranker/none.ts";
import { getDbConn } from "./db.ts";

export interface RetrievedChunk extends ScoredChunk {
  rerank?: number;
  degraded?: string;
}

export interface RetrieveOptions {
  limit?: number;
  alpha?: number;
  db?: Database.Database;
  config?: RagConfig;
  signal?: AbortSignal;
  /** Expand recall only when a real reranker is enabled. */
  candidateTopK?: number;
}

export async function retrieve(query: string, opts: RetrieveOptions = {}): Promise<RetrievedChunk[]> {
  const config = opts.config ?? loadConfig();
  const limit = opts.limit ?? 10;
  const alpha = opts.alpha ?? config.ragAlpha;
  const db = opts.db ?? getDbConn();
  const reranker = (() => {
    try { return createReranker(config); }
    catch { return new NoneReranker(); }
  })();
  const expand = reranker.id !== "none";
  const recall = expand
    ? Math.max(limit, Math.min(opts.candidateTopK ?? config.candidateTopK, 200))
    : limit;

  let hits: ScoredChunk[];
  try {
    hits = await hybridSearch(query, recall, alpha, db);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (opts.signal?.aborted) throw err;
    // Query embedding temporary failure: BM25-only is handled inside hybridSearch
    // when vectors are absent. If embedQuery itself throws, surface it.
    throw new Error(`Query embedding failed: ${msg}`);
  }

  if (!expand) {
    return hits.slice(0, limit);
  }
  if (hits.length === 0) return [];

  try {
    if (opts.signal?.aborted) {
      const err = new Error("Retrieval cancelled");
      err.name = "AbortError";
      throw err;
    }
    const ranked = await reranker.rerank(
      query,
      hits.map(h => ({ id: h.chunk.id, text: h.chunk.content })),
      { topK: limit, signal: opts.signal },
    );
    const byId = new Map(hits.map(h => [h.chunk.id, h]));
    const out: RetrievedChunk[] = [];
    for (const r of ranked) {
      const src = byId.get(r.id);
      if (!src) continue;
      out.push({ ...src, rerank: r.rerankScore });
    }
    return out.slice(0, limit);
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    const reason = err instanceof Error ? err.message : String(err);
    return hits.slice(0, limit).map(h => ({ ...h, degraded: `rerank failed: ${reason}` }));
  }
}
