import { describe, it, expect, vi, afterEach } from "vitest";
import { LocalEmbeddingProvider } from "../providers/embedding/local.ts";
import { embed, embedBatch } from "../embed.ts";
import { EMBEDDING_MODEL, VECTOR_DIM } from "../constants.ts";

vi.mock("@xenova/transformers", () => ({
  pipeline: vi.fn().mockResolvedValue(
    vi.fn().mockImplementation(async (texts: string | string[]) => {
      const batch = Array.isArray(texts) ? texts : [texts];
      const DIM = 384;
      const flat = new Float32Array(batch.length * DIM).fill(0.1);
      return { data: flat };
    }),
  ),
}));

afterEach(async () => {
  const { closeDbConn } = await import("../db.ts");
  closeDbConn();
});

describe("LocalEmbeddingProvider", () => {
  it("exposes id/model/dimensions matching the MiniLM contract", () => {
    const p = new LocalEmbeddingProvider();
    expect(p.id).toBe("local");
    expect(p.model).toBe(EMBEDDING_MODEL);
    expect(p.dimensions).toBe(VECTOR_DIM);
  });

  it("embedDocuments([]) returns [] without calling the model", async () => {
    const p = new LocalEmbeddingProvider();
    await expect(p.embedDocuments([])).resolves.toEqual([]);
  });

  it("embedQuery and embedDocuments return 384-dim finite non-zero vectors", async () => {
    const p = new LocalEmbeddingProvider();
    const q = await p.embedQuery("hello world");
    expect(q).toHaveLength(384);
    expect(q.every(Number.isFinite)).toBe(true);
    const docs = await p.embedDocuments(["aaa", "bbb"]);
    expect(docs).toHaveLength(2);
    expect(docs[0]).toHaveLength(384);
  });

  it("reports batch progress and honors abort", async () => {
    const p = new LocalEmbeddingProvider();
    const seen: number[] = [];
    await p.embedDocuments(["one", "two"], { onProgress: (d, t) => seen.push(d / t) });
    expect(seen.length).toBeGreaterThan(0);
    expect(seen[seen.length - 1]).toBe(1);

    const ac = new AbortController();
    ac.abort();
    await expect(p.embedDocuments(["x"], { signal: ac.signal })).rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("embed.ts facade", () => {
  it("forwards embed to embedQuery and embedBatch to embedDocuments", async () => {
    const q = await embed("facade query");
    const d = await embedBatch(["facade doc"]);
    expect(q).toHaveLength(384);
    expect(d).toHaveLength(1);
    expect(d[0]).toHaveLength(384);
  });
});
