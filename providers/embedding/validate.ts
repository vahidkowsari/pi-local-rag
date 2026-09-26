/** L2-normalize in place-friendly copy so sqlite-vec cosine-via-L2 stays valid. */
export function l2Normalize(v: number[]): number[] {
  let normSq = 0;
  for (const x of v) normSq += x * x;
  const norm = Math.sqrt(normSq);
  if (norm === 0) throw new Error("embedding vector has zero norm");
  if (Math.abs(norm - 1) < 1e-5) return v;
  return v.map(x => x / norm);
}

export function assertValidVectors(
  vectors: number[][],
  expectedCount: number,
  dimensions: number,
): number[][] {
  if (vectors.length !== expectedCount) {
    throw new Error(`expected ${expectedCount} embedding vectors, got ${vectors.length}`);
  }
  return vectors.map((v, i) => {
    if (!v || v.length !== dimensions) {
      throw new Error(`embedding ${i} has dimension ${v?.length ?? 0}, expected ${dimensions}`);
    }
    for (const x of v) {
      if (!Number.isFinite(x)) throw new Error(`embedding ${i} contains a non-finite value`);
    }
    return l2Normalize(v);
  });
}
