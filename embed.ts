import { EMBEDDING_MODEL } from "./constants.ts";

let _pipeline: any = null;

async function getEmbedder() {
  if (_pipeline) return _pipeline;
  const { pipeline } = await import("@xenova/transformers");
  _pipeline = await pipeline("feature-extraction", EMBEDDING_MODEL);
  return _pipeline;
}

export async function embed(text: string): Promise<number[]> {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}

/**
 * Yield to the event loop so the TUI can render progress updates.
 * ONNX inference is synchronous from the event loop's perspective;
 * without this, the UI freezes during embedding.
 */
const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

/** Default batch size for a single ONNX forward pass. */
export const BATCH_SIZE = 64;

/**
 * Embed `texts` using true batched ONNX inference.
 *
 * The model is called once per batch of up to `BATCH_SIZE` texts rather than
 * once per text, giving a ~BATCH_SIZE× speedup on CPU.  The output Tensor has
 * dims [batchSize, VECTOR_DIM]; we slice it into per-text arrays.
 *
 * `onProgress` is fired after each batch with the cumulative count so the TUI
 * can render a smooth progress bar (same contract as before).
 */
export async function embedBatch(
  texts: string[],
  onProgress?: (i: number, total: number) => void,
): Promise<number[][]> {
  if (texts.length === 0) return [];
  const embedder = await getEmbedder();
  const results: number[][] = new Array(texts.length);

  for (let start = 0; start < texts.length; start += BATCH_SIZE) {
    const batch = texts.slice(start, start + BATCH_SIZE);
    // Pass the whole batch in a single forward pass — the model returns a
    // Tensor with dims [batchSize, VECTOR_DIM].
    const output = await embedder(batch, { pooling: "mean", normalize: true });
    const flat = output.data as Float32Array;
    const dim = flat.length / batch.length; // should equal VECTOR_DIM (384)

    for (let j = 0; j < batch.length; j++) {
      results[start + j] = Array.from(flat.subarray(j * dim, (j + 1) * dim));
    }

    onProgress?.(Math.min(start + batch.length, texts.length), texts.length);
    // Yield after each batch so the TUI can re-render before the next pass.
    await yield_();
  }

  return results;
}
