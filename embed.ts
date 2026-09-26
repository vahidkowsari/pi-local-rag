import { getLocalEmbeddingProvider } from "./providers/embedding/local.ts";

export { LOCAL_BATCH_SIZE as BATCH_SIZE } from "./providers/embedding/local.ts";

/** Compatibility facade: query embedding via the local MiniLM provider. */
export async function embed(text: string, opts?: { signal?: AbortSignal }): Promise<number[]> {
  return getLocalEmbeddingProvider().embedQuery(text, opts);
}

/**
 * Compatibility facade: document batch embedding via the local MiniLM provider.
 * ONNX inference is a blocking forward pass; we still yield between batches
 * so the TUI can paint.
 */
export async function embedBatch(
  texts: string[],
  onProgress?: (i: number, total: number) => void,
  opts?: { signal?: AbortSignal },
): Promise<number[][]> {
  return getLocalEmbeddingProvider().embedDocuments(texts, { onProgress, signal: opts?.signal });
}
