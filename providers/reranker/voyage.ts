import { postJson } from "../http.ts";
import { DEFAULT_VOYAGE_RERANK_MODEL } from "../../config.ts";
import type { RerankHit, RerankInput, RerankOptions, Reranker } from "./types.ts";

const RERANK_URL = "https://api.voyageai.com/v1/rerank";

interface VoyageRerankItem {
  index: number;
  relevance_score: number;
}

interface VoyageRerankResponse {
  data?: VoyageRerankItem[];
}

export class VoyageReranker implements Reranker {
  readonly id = "voyage";
  readonly model: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  requestCount = 0;

  constructor(opts: { model?: string; apiKey: string; timeoutMs?: number; maxRetries?: number }) {
    if (!opts.apiKey) throw new Error("VOYAGE_API_KEY is required for the voyage reranker");
    this.model = opts.model || DEFAULT_VOYAGE_RERANK_MODEL;
    this.apiKey = opts.apiKey;
    this.timeoutMs = opts.timeoutMs ?? 30_000;
    this.maxRetries = opts.maxRetries ?? 3;
  }

  async rerank(query: string, candidates: RerankInput[], opts?: RerankOptions): Promise<RerankHit[]> {
    if (candidates.length === 0) return [];
    this.requestCount++;
    const json = await postJson<VoyageRerankResponse>(RERANK_URL, {
      query,
      documents: candidates.map(c => c.text),
      model: this.model,
      top_k: opts?.topK ?? candidates.length,
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
