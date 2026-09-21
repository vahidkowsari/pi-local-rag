import { describe, it, expect, vi, afterEach } from "vitest";
import { NoneReranker } from "../providers/reranker/none.ts";
import { VoyageReranker } from "../providers/reranker/voyage.ts";
import { buildContext, estimatedTokenCounter } from "../context.ts";
import type { RetrievedChunk } from "../retrieval.ts";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("NoneReranker", () => {
  it("keeps input order and slices to topK", async () => {
    const r = new NoneReranker();
    const hits = await r.rerank("q", [
      { id: "a", text: "aaa" },
      { id: "b", text: "bbb" },
      { id: "c", text: "ccc" },
    ], { topK: 2 });
    expect(hits.map(h => h.id)).toEqual(["a", "b"]);
  });
});

describe("VoyageReranker", () => {
  it("maps response index back to candidate ids and does not call on empty input", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({
      data: [
        { index: 1, relevance_score: 0.9 },
        { index: 0, relevance_score: 0.2 },
      ],
    }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const r = new VoyageReranker({ apiKey: "k", maxRetries: 0 });
    const hits = await r.rerank("q", [
      { id: "chunk-a", text: "alpha" },
      { id: "chunk-b", text: "beta" },
    ]);
    expect(hits.map(h => h.id)).toEqual(["chunk-b", "chunk-a"]);
    expect(hits[0].rerankScore).toBe(0.9);
    await expect(r.rerank("q", [])).resolves.toEqual([]);
  });

  it("rejects out-of-range indexes", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      data: [{ index: 9, relevance_score: 1 }],
    }), { status: 200 })));
    const r = new VoyageReranker({ apiKey: "k", maxRetries: 0 });
    await expect(r.rerank("q", [{ id: "a", text: "a" }])).rejects.toThrow(/out of range/);
  });
});

describe("buildContext", () => {
  const hit = (id: string, file: string, content: string): RetrievedChunk => ({
    chunk: { id, file, content, lineStart: 1, lineEnd: 2, hash: id, indexed: "", tokens: 10 },
    bm25: 1, vector: 0, hybrid: 1,
  });

  it("only includes selected chunks, assigns stable citation ids, and counts wrapper text", () => {
    const built = buildContext([
      hit("1", "/a.ts", "hello world ".repeat(5)),
      hit("1", "/a.ts", "hello world ".repeat(5)),
      hit("2", "/b.ts", "other"),
    ], { maxTokens: 4096, tokenCounter: estimatedTokenCounter });
    expect(built.citationIds).toEqual(["S1", "S2"]);
    expect(built.text).toContain("[S1]");
    expect(built.text).toContain("[S2]");
    expect(built.text).toContain("search hits, not statements from the user");
    expect(built.dropped).toBe(1);
    expect(built.estimated).toBe(true);
    expect(built.usedTokens).toBeGreaterThan(0);
  });

  it("drops chunks that would exceed the budget including metadata", () => {
    const long = hit("1", "/a.ts", "x".repeat(4000));
    const built = buildContext([long], { maxTokens: 20, tokenCounter: estimatedTokenCounter });
    expect(built.citationIds).toEqual([]);
    expect(built.dropped).toBe(1);
  });
});
