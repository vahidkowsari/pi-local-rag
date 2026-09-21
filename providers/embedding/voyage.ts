import { postJson } from "../http.ts";
import { assertValidVectors } from "./validate.ts";
import type { EmbedBatchOptions, EmbeddingProvider } from "./types.ts";
import { DEFAULT_VOYAGE_EMBED_DIM, DEFAULT_VOYAGE_EMBED_MODEL } from "../../config.ts";

const EMBED_URL = "https://api.voyageai.com/v1/embeddings";
const MAX_ITEMS = 1000;
const MAX_INPUT_TOKENS = 32_000;
const MAX_BATCH_TOKENS = 100_000;

export function estimateTokens(text: string): number {
  // Conservative upper bound without the Voyage tokenizer: ~2 chars/token.
  return Math.max(1, Math.ceil(text.length / 2));
}

export interface VoyageEmbeddingOptions {
  model?: string;
  dimensions?: number;
  apiKey: string;
  timeoutMs?: number;
  maxRetries?: number;
}

interface VoyageEmbeddingItem {
  object: string;
  embedding: number[];
  index: number;
}

interface VoyageEmbeddingResponse {
  object?: string;
  data?: VoyageEmbeddingItem[];
  model?: string;
  usage?: { total_tokens?: number };
}

function mapByIndex(data: VoyageEmbeddingItem[], expected: number): number[][] {
  const seen = new Set<number>();
  const out: number[][] = new Array(expected);
  for (const item of data) {
    if (!Number.isInteger(item.index) || item.index < 0 || item.index >= expected) {
      throw new Error(`Voyage embedding index ${item.index} is out of range for ${expected} inputs`);
    }
    if (seen.has(item.index)) throw new Error(`Voyage embedding index ${item.index} is duplicated`);
    if (!Array.isArray(item.embedding)) throw new Error(`Voyage embedding ${item.index} is missing`);
    seen.add(item.index);
    out[item.index] = item.embedding;
  }
  for (let i = 0; i < expected; i++) {
    if (!out[i]) throw new Error(`Voyage embedding response missing index ${i}`);
  }
  return out;
}

export class VoyageEmbeddingProvider implements EmbeddingProvider {
  readonly id = "voyage";
  readonly model: string;
  readonly dimensions: number;
  private readonly apiKey: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  requestCount = 0;

  constructor(opts: VoyageEmbeddingOptions) {
    if (!opts.apiKey) throw new Error("VOYAGE_API_KEY is required for the voyage embedding provider");
    this.model = opts.model ?? DEFAULT_VOYAGE_EMBED_MODEL;
    this.dimensions = opts.dimensions ?? DEFAULT_VOYAGE_EMBED_DIM;
    this.apiKey = opts.apiKey;
    this.timeoutMs = opts.timeoutMs ?? 30_000;
    this.maxRetries = opts.maxRetries ?? 3;
  }

  async embedQuery(text: string, opts?: { signal?: AbortSignal }): Promise<number[]> {
    const [v] = await this.embedRole([text], "query", opts);
    return v;
  }

  async embedDocuments(texts: string[], opts?: EmbedBatchOptions): Promise<number[][]> {
    return this.embedRole(texts, "document", opts);
  }

  private async embedRole(
    texts: string[],
    inputType: "query" | "document",
    opts?: EmbedBatchOptions,
  ): Promise<number[][]> {
    if (texts.length === 0) return [];
    for (const t of texts) {
      const tokens = estimateTokens(t);
      if (tokens > MAX_INPUT_TOKENS) {
        throw new Error(`Input exceeds voyage-4-lite 32k token limit (estimated ${tokens} tokens). Refusing to truncate.`);
      }
    }

    const out: number[][] = new Array(texts.length);
    let done = 0;
    for (const batch of this.planBatches(texts)) {
      if (opts?.signal?.aborted) {
        const err = new Error("Embedding cancelled");
        err.name = "AbortError";
        throw err;
      }
      const vectors = await this.requestBatch(batch.texts, inputType, opts?.signal);
      for (let i = 0; i < batch.texts.length; i++) out[batch.offset + i] = vectors[i];
      done += batch.texts.length;
      opts?.onProgress?.(done, texts.length);
    }
    return assertValidVectors(out, texts.length, this.dimensions);
  }

  private planBatches(texts: string[]): { offset: number; texts: string[] }[] {
    const batches: { offset: number; texts: string[] }[] = [];
    let offset = 0;
    while (offset < texts.length) {
      const chunk: string[] = [];
      let tokens = 0;
      while (offset + chunk.length < texts.length && chunk.length < MAX_ITEMS) {
        const next = texts[offset + chunk.length];
        const t = estimateTokens(next);
        if (chunk.length > 0 && tokens + t > MAX_BATCH_TOKENS) break;
        chunk.push(next);
        tokens += t;
      }
      batches.push({ offset, texts: chunk });
      offset += chunk.length;
    }
    return batches;
  }

  private async requestBatch(
    texts: string[],
    inputType: "query" | "document",
    signal?: AbortSignal,
  ): Promise<number[][]> {
    this.requestCount++;
    const body = {
      input: texts,
      model: this.model,
      input_type: inputType,
      truncation: false,
      output_dtype: "float",
      output_dimension: this.dimensions,
    };
    const json = await postJson<VoyageEmbeddingResponse>(EMBED_URL, body, {
      apiKey: this.apiKey,
      timeoutMs: this.timeoutMs,
      maxRetries: this.maxRetries,
      signal,
    });
    if (!Array.isArray(json.data)) throw new Error("Voyage embedding response is missing data[]");
    return mapByIndex(json.data, texts.length);
  }
}
