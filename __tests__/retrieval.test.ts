import { describe, it, expect, vi, afterEach } from "vitest";
import { NoneReranker } from "../providers/reranker/none.ts";
import { VoyageReranker } from "../providers/reranker/voyage.ts";
import { buildContext, estimatedTokenCounter } from "../context.ts";
import { retrieveWithCandidates, type RetrievedChunk } from "../retrieval.ts";
import { defaultConfig } from "../config.ts";

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
    const firstCall = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    const body = JSON.parse(String(firstCall[1].body));
    expect(body.truncation).toBe(false);
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

describe("retrieveWithCandidates", () => {
  it("records rerank-fallback when the reranker HTTP call fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("synthetic rerank unavailable", { status: 400 })));
    process.env.VOYAGE_API_KEY = "synthetic";
    const v = new Array(384).fill(0);
    v[0] = 1;
    const { LocalEmbeddingProvider } = await import("../providers/embedding/local.ts");
    const embedQuery = vi.spyOn(LocalEmbeddingProvider.prototype, "embedQuery").mockResolvedValue(v);
    const cfg = defaultConfig();
    cfg.reranker = { provider: "voyage", model: "rerank-2.5-lite" };
    cfg.http = { timeoutMs: 1000, maxRetries: 0 };
    const Database = (await import("better-sqlite3")).default;
    const { load: loadVec } = await import("sqlite-vec");
    const { initSchema } = await import("../repository.ts");
    const { stampFingerprints } = await import("../index-manager.ts");
    const db = new Database(":memory:");
    loadVec(db);
    initSchema(db);
    stampFingerprints(db, defaultConfig());
    const repo = await import("../repository.ts");
    const r = repo.insertChunk(db, {
      id: "c1", filePath: "/a.md", content: "alpha evidence", lineStart: 1, lineEnd: 1,
      hash: "h", indexedAt: new Date().toISOString(), tokens: 4,
    });
    repo.insertVector(db, Number(r.lastInsertRowid), v);
    const bundle = await retrieveWithCandidates("alpha", { db, config: cfg, limit: 5, candidateTopK: 10 });
    expect(bundle.method).toBe("rerank-fallback");
    expect(bundle.degraded).toMatch(/rerank failed/i);
    expect(bundle.hits.length).toBeGreaterThan(0);
    db.close();
    embedQuery.mockRestore();
    delete process.env.VOYAGE_API_KEY;
  });

  it("keeps embedding failure when FTS has zero hits", async () => {
    const v = new Array(384).fill(0);
    v[0] = 1;
    const { LocalEmbeddingProvider } = await import("../providers/embedding/local.ts");
    const embedQuery = vi.spyOn(LocalEmbeddingProvider.prototype, "embedQuery")
      .mockRejectedValue(new TypeError("synthetic network outage"));
    const Database = (await import("better-sqlite3")).default;
    const { load: loadVec } = await import("sqlite-vec");
    const { initSchema } = await import("../repository.ts");
    const { stampFingerprints } = await import("../index-manager.ts");
    const db = new Database(":memory:");
    loadVec(db);
    initSchema(db);
    stampFingerprints(db, defaultConfig());
    const repo = await import("../repository.ts");
    const r = repo.insertChunk(db, {
      id: "c1", filePath: "/a.md", content: "alpha evidence", lineStart: 1, lineEnd: 1,
      hash: "h", indexedAt: new Date().toISOString(), tokens: 4,
    });
    repo.insertVector(db, Number(r.lastInsertRowid), v);
    const bundle = await retrieveWithCandidates("no_matching_term_987654", {
      db, config: defaultConfig(), limit: 5, candidateTopK: 10,
    });
    expect(bundle.hits).toEqual([]);
    expect(bundle.candidates).toEqual([]);
    expect(bundle.degraded).toMatch(/query embedding failed/i);
    expect(bundle.method).toBe("bm25-fallback");
    db.close();
    embedQuery.mockRestore();
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

  it("does not merge distinct pages that share content", () => {
    const a: RetrievedChunk = {
      chunk: { id: "p1", file: "paper.pdf", content: "same", lineStart: 0, lineEnd: 0, hash: "h", indexed: "", tokens: 1, pageStart: 1, pageEnd: 1, chunkIndex: 0 },
      bm25: 1, vector: 0, hybrid: 1,
    };
    const b: RetrievedChunk = {
      chunk: { id: "p2", file: "paper.pdf", content: "same", lineStart: 0, lineEnd: 0, hash: "h", indexed: "", tokens: 1, pageStart: 2, pageEnd: 2, chunkIndex: 1 },
      bm25: 1, vector: 0, hybrid: 1,
    };
    const built = buildContext([a, b], { maxTokens: 4096, tokenCounter: estimatedTokenCounter });
    expect(built.citationIds).toEqual(["S1", "S2"]);
    expect(built.text).toContain("page 1");
    expect(built.text).toContain("page 2");
  });

  it("includes a retrieval degraded note in the header", () => {
    const h = hit("1", "/a.ts", "hello");
    h.degraded = "query embedding failed, BM25 only: network";
    const built = buildContext([h], { maxTokens: 4096, tokenCounter: estimatedTokenCounter });
    expect(built.text).toContain("Retrieval note:");
    expect(built.text).toContain("BM25 only");
  });

  it("drops chunks that would exceed the budget including metadata", () => {
    const long = hit("1", "/a.ts", "x".repeat(4000));
    const built = buildContext([long], { maxTokens: 20, tokenCounter: estimatedTokenCounter });
    expect(built.citationIds).toEqual([]);
    expect(built.dropped).toBe(1);
  });
});
