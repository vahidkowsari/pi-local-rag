import type Database from "better-sqlite3";
import { hybridSearchDetailed, type ScoredChunk } from "./search.ts";
import { CANDIDATE_TOP_K_MAX, loadConfig, type RagConfig } from "./config.ts";
import { configuredRerankerIsEnabled, createReranker } from "./providers/reranker/factory.ts";
import { NoneReranker } from "./providers/reranker/none.ts";
import type { Reranker } from "./providers/reranker/types.ts";
import { getDbConn } from "./db.ts";
import { isAbortError, throwIfAborted } from "./abort.ts";
import { IndexIncompatibleError } from "./index-manager.ts";

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
  /** Expand recall only when a real reranker is enabled, unless expandCandidates is set. */
  candidateTopK?: number;
  /** Recall candidateTopK even when the reranker is none (eval / debugging). */
  expandCandidates?: boolean;
}

export type RetrieveMethod = "hybrid" | "bm25-fallback" | "rerank" | "rerank-fallback";

export interface RetrieveBundle {
  hits: RetrievedChunk[];
  candidates: RetrievedChunk[];
  degraded?: string;
  method: RetrieveMethod;
}

function capLimit(n: number): number {
  if (!Number.isFinite(n) || n < 1) return 1;
  return Math.min(CANDIDATE_TOP_K_MAX, Math.floor(n));
}

export async function retrieveWithCandidates(query: string, opts: RetrieveOptions = {}): Promise<RetrieveBundle> {
  throwIfAborted(opts.signal);
  const config = opts.config ?? loadConfig();
  const limit = capLimit(opts.limit ?? 10);
  const alpha = opts.alpha ?? config.ragAlpha;
  const db = opts.db ?? getDbConn();

  let rerankDegraded: string | undefined;
  let reranker: Reranker = new NoneReranker();
  let expand = false;
  if (configuredRerankerIsEnabled(config)) {
    try {
      reranker = createReranker(config);
      expand = reranker.id !== "none";
    } catch (err) {
      rerankDegraded = `reranker unavailable: ${err instanceof Error ? err.message : String(err)}`;
      reranker = new NoneReranker();
      expand = false;
    }
  }

  const candidateCap = capLimit(opts.candidateTopK ?? config.candidateTopK);
  const recall = expand || opts.expandCandidates ? candidateCap : limit;

  let searchHits: ScoredChunk[];
  let embedDegraded: string | undefined;
  try {
    const searched = await hybridSearchDetailed(query, recall, alpha, db, { signal: opts.signal, config });
    searchHits = searched.hits;
    embedDegraded = searched.degraded;
  } catch (err) {
    if (isAbortError(err) || err instanceof IndexIncompatibleError) throw err;
    throw err;
  }
  const mark = (h: RetrievedChunk): RetrievedChunk => {
    const reason = [embedDegraded, rerankDegraded, h.degraded].filter(Boolean).join("; ");
    return reason ? { ...h, degraded: reason } : h;
  };

  throwIfAborted(opts.signal, "Retrieval cancelled");
  const candidates = searchHits.map(h => mark({ ...h }));
  const joinDegraded = (...parts: Array<string | undefined>) => parts.filter(Boolean).join("; ") || undefined;
  const fallbackMethod = (): RetrieveMethod => (embedDegraded || rerankDegraded ? "bm25-fallback" : "hybrid");

  if (!expand) {
    return {
      hits: candidates.slice(0, limit),
      candidates,
      degraded: joinDegraded(embedDegraded, rerankDegraded),
      method: fallbackMethod(),
    };
  }
  if (searchHits.length === 0) {
    return {
      hits: [],
      candidates,
      degraded: joinDegraded(embedDegraded, rerankDegraded),
      method: embedDegraded ? "bm25-fallback" : "rerank",
    };
  }

  try {
    throwIfAborted(opts.signal, "Retrieval cancelled");
    const ranked = await reranker.rerank(
      query,
      searchHits.map(h => ({ id: h.chunk.id, text: h.chunk.content })),
      { topK: limit, signal: opts.signal },
    );
    const byId = new Map(searchHits.map(h => [h.chunk.id, h]));
    const out: RetrievedChunk[] = [];
    for (const r of ranked) {
      const src = byId.get(r.id);
      if (!src) continue;
      out.push(mark({ ...src, rerank: r.rerankScore }));
    }
    return {
      hits: out.slice(0, limit),
      candidates,
      degraded: joinDegraded(embedDegraded),
      method: embedDegraded ? "bm25-fallback" : "rerank",
    };
  } catch (err) {
    if (isAbortError(err)) throw err;
    const reason = err instanceof Error ? err.message : String(err);
    const degraded = joinDegraded(embedDegraded, `rerank failed: ${reason}`);
    return {
      hits: candidates.slice(0, limit).map(h => ({ ...h, degraded })),
      candidates,
      degraded,
      method: "rerank-fallback",
    };
  }
}

export async function retrieve(query: string, opts: RetrieveOptions = {}): Promise<RetrievedChunk[]> {
  return (await retrieveWithCandidates(query, opts)).hits;
}
