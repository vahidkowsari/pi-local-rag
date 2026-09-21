import { EMBEDDING_MODEL, VECTOR_DIM } from "../../constants.ts";
import { assertValidVectors } from "./validate.ts";
import type { EmbedBatchOptions, EmbeddingProvider } from "./types.ts";

const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

export const LOCAL_BATCH_SIZE = 64;

function throwIfAborted(signal?: AbortSignal) {
  if (!signal?.aborted) return;
  const err = new Error("Embedding cancelled");
  err.name = "AbortError";
  throw err;
}

export class LocalEmbeddingProvider implements EmbeddingProvider {
  readonly id = "local";
  readonly model: string;
  readonly dimensions: number;
  private pipeline: any = null;

  constructor(opts: { model?: string; dimensions?: number } = {}) {
    this.model = opts.model ?? EMBEDDING_MODEL;
    this.dimensions = opts.dimensions ?? VECTOR_DIM;
  }

  private async getEmbedder() {
    if (this.pipeline) return this.pipeline;
    const { pipeline } = await import("@xenova/transformers");
    this.pipeline = await pipeline("feature-extraction", this.model);
    return this.pipeline;
  }

  async embedQuery(text: string, opts?: { signal?: AbortSignal }): Promise<number[]> {
    const [v] = await this.embedDocuments([text], { signal: opts?.signal });
    return v;
  }

  async embedDocuments(texts: string[], opts?: EmbedBatchOptions): Promise<number[][]> {
    if (texts.length === 0) return [];
    throwIfAborted(opts?.signal);
    const embedder = await this.getEmbedder();
    const results: number[][] = new Array(texts.length);

    for (let start = 0; start < texts.length; start += LOCAL_BATCH_SIZE) {
      throwIfAborted(opts?.signal);
      const batch = texts.slice(start, start + LOCAL_BATCH_SIZE);
      const output = await embedder(batch, { pooling: "mean", normalize: true });
      const flat = output.data as Float32Array;
      const dim = flat.length / batch.length;
      for (let j = 0; j < batch.length; j++) {
        results[start + j] = Array.from(flat.subarray(j * dim, (j + 1) * dim));
      }
      opts?.onProgress?.(Math.min(start + batch.length, texts.length), texts.length);
      await yield_();
    }

    return assertValidVectors(results, texts.length, this.dimensions);
  }
}

let _default: LocalEmbeddingProvider | null = null;

export function getLocalEmbeddingProvider(): LocalEmbeddingProvider {
  return _default ??= new LocalEmbeddingProvider();
}

/** Test-only: drop the cached pipeline so a new provider can be constructed. */
export function resetLocalEmbeddingProvider(): void {
  _default = null;
}
