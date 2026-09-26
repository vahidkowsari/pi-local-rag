import { postJson } from "../http.ts";
import { DEFAULT_VOYAGE_RERANK_MODEL } from "../../config.ts";
import type { RerankHit, RerankInput, RerankOptions, Reranker } from "./types.ts";

const DEFAULT_BASE_URL = "https://api.voyageai.com/v1";
const MAX_DOCS = 1000;
const MAX_QUERY_TOKENS = 8_000;
const MAX_QUERY_PLUS_DOC_TOKENS = 32_000;
const MAX_TOTAL_TOKENS = 600_000;

function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 2));
}

interface VoyageRerankItem {
  index: number;
  relevance_score: number;
}

interface VoyageRerankResponse {
  data?: VoyageRerankItem[];
}

export class VoyageReranker implements Reranker {
  readonly id: string;
  readonly model: string;
  private readonly apiKey: string;
  private readonly endpoint: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  requestCount = 0;

  constructor(opts: { id?: string; baseUrl?: string; model?: string; apiKey: string; timeoutMs?: number; maxRetries?: number }) {
    if (!opts.apiKey) throw new Error("An API key is required for the Voyage reranker");
    this.id = opts.id ?? "voyage";
    this.model = opts.model || DEFAULT_VOYAGE_RERANK_MODEL;
    this.apiKey = opts.apiKey;
    this.endpoint = `${(opts.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "")}/rerank`;
    this.timeoutMs = opts.timeoutMs ?? 30_000;
    this.maxRetries = opts.maxRetries ?? 3;
  }

  async rerank(query: string, candidates: RerankInput[], opts?: RerankOptions): Promise<RerankHit[]> {
    if (candidates.length === 0) return [];
    if (candidates.length > MAX_DOCS) {
      throw new Error(`Voyage rerank accepts at most ${MAX_DOCS} documents (got ${candidates.length})`);
    }
    const qTokens = estimateTokens(query);
    if (qTokens > MAX_QUERY_TOKENS) {
      throw new Error(`Rerank query exceeds ${MAX_QUERY_TOKENS} token limit (estimated ${qTokens}). Refusing to truncate.`);
    }
    let docSum = 0;
    for (const c of candidates) {
      const d = estimateTokens(c.text);
      docSum += d;
      if (qTokens + d > MAX_QUERY_PLUS_DOC_TOKENS) {
        throw new Error(`Rerank query+document exceeds ${MAX_QUERY_PLUS_DOC_TOKENS} token limit. Refusing to truncate.`);
      }
    }
    if (qTokens * candidates.length + docSum > MAX_TOTAL_TOKENS) {
      throw new Error(`Rerank total tokens exceed ${MAX_TOTAL_TOKENS}. Refusing to truncate.`);
    }
    this.requestCount++;
    const json = await postJson<VoyageRerankResponse>(this.endpoint, {
      query,
      documents: candidates.map(c => c.text),
      model: this.model,
      top_k: opts?.topK ?? candidates.length,
      truncation: false,
    }, {
      apiKey: this.apiKey,
      timeoutMs: this.timeoutMs,
      maxRetries: this.maxRetries,
      signal: opts?.signal,
    });
    if (!Array.isArray(json.data)) throw new Error("Voyage rerank response is missing data[]");
    const seen = new Set<number>();
    const out: RerankHit[] = [];
    for (const item of json.data) {
      if (!Number.isInteger(item.index) || item.index < 0 || item.index >= candidates.length) {
        throw new Error(`Voyage rerank index ${item.index} is out of range`);
      }
      if (seen.has(item.index)) throw new Error(`Voyage rerank index ${item.index} is duplicated`);
      if (!Number.isFinite(item.relevance_score)) throw new Error(`Voyage rerank score at ${item.index} is invalid`);
      seen.add(item.index);
      out.push({ id: candidates[item.index].id, rerankScore: item.relevance_score });
    }
    return out;
  }
}
