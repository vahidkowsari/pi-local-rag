import { describe, it, expect, afterEach, vi } from "vitest";
import { VoyageEmbeddingProvider } from "../providers/embedding/voyage.ts";
import { HttpError } from "../providers/http.ts";

function floatVec(dim = 1024, fill = 0.02): number[] {
  return Array.from({ length: dim }, () => fill);
}

function jsonResponse(body: unknown, init: { status?: number; headers?: Record<string, string> } = {}) {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
  });
}

describe("VoyageEmbeddingProvider HTTP contract", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("sends input_type=query for embedQuery and input_type=document for embedDocuments", async () => {
    const bodies: unknown[] = [];
    vi.stubGlobal("fetch", vi.fn(async (_url: string, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)));
      const n = (JSON.parse(String(init?.body)) as { input: string[] }).input.length;
      return jsonResponse({
        data: Array.from({ length: n }, (_, i) => ({ object: "embedding", index: i, embedding: floatVec() })),
      });
    }));
    const p = new VoyageEmbeddingProvider({ apiKey: "test-key", timeoutMs: 5000, maxRetries: 0 });
    await p.embedQuery("q1");
    await p.embedDocuments(["d1", "d2"]);
    expect(bodies[0]).toMatchObject({ input: ["q1"], input_type: "query", truncation: false, output_dtype: "float", output_dimension: 1024, model: "voyage-4-lite" });
    expect(bodies[1]).toMatchObject({ input: ["d1", "d2"], input_type: "document" });
    const auth = (vi.mocked(fetch).mock.calls[0][1] as RequestInit).headers as Record<string, string>;
    expect(auth.Authorization).toBe("Bearer test-key");
  });

  it("maps out-of-order response indices back to the original inputs", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({
      data: [
        { object: "embedding", index: 1, embedding: floatVec(1024, 0.03) },
        { object: "embedding", index: 0, embedding: floatVec(1024, 0.01) },
      ],
    })));
    const p = new VoyageEmbeddingProvider({ apiKey: "k", maxRetries: 0 });
    const out = await p.embedDocuments(["a", "b"]);
    expect(out[0][0]).toBeCloseTo(out[0][1]); // normalized
    expect(out).toHaveLength(2);
  });

  it("rejects missing, duplicate, and out-of-range indices", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({
      data: [{ object: "embedding", index: 0, embedding: floatVec() }],
    })));
    const p = new VoyageEmbeddingProvider({ apiKey: "k", maxRetries: 0 });
    await expect(p.embedDocuments(["a", "b"])).rejects.toThrow(/missing index/);

    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({
      data: [
        { object: "embedding", index: 0, embedding: floatVec() },
        { object: "embedding", index: 0, embedding: floatVec() },
      ],
    })));
    await expect(p.embedDocuments(["a", "b"])).rejects.toThrow(/duplicated/);
  });

  it("rejects non-finite values and wrong dimensions", async () => {
    const nanVec = floatVec();
    nanVec[2] = Number.NaN;
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({
      data: [{ object: "embedding", index: 0, embedding: nanVec }],
    })));
    const p = new VoyageEmbeddingProvider({ apiKey: "k", maxRetries: 0 });
    await expect(p.embedQuery("x")).rejects.toThrow(/non-finite/);

    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({
      data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
    })));
    await expect(p.embedQuery("x")).rejects.toThrow(/dimension/);
  });

  it("does not retry 400/401/403", async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ error: "nope" }, { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);
    const p = new VoyageEmbeddingProvider({ apiKey: "k", maxRetries: 3, timeoutMs: 2000 });
    await expect(p.embedQuery("x")).rejects.toBeInstanceOf(HttpError);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("retries 429 honoring Retry-After, then succeeds", async () => {
    vi.useFakeTimers();
    let n = 0;
    vi.stubGlobal("fetch", vi.fn(async () => {
      n++;
      if (n === 1) return jsonResponse({ error: "slow down" }, { status: 429, headers: { "retry-after": "0" } });
      return jsonResponse({ data: [{ object: "embedding", index: 0, embedding: floatVec() }] });
    }));
    const p = new VoyageEmbeddingProvider({ apiKey: "k", maxRetries: 2, timeoutMs: 2000 });
    const pending = p.embedQuery("x");
    await vi.runAllTimersAsync();
    const v = await pending;
    expect(v).toHaveLength(1024);
    expect(n).toBe(2);
  });

  it("does not call the API for an empty document list", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const p = new VoyageEmbeddingProvider({ apiKey: "k" });
    await expect(p.embedDocuments([])).resolves.toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("honors abort before the request", async () => {
    const ac = new AbortController();
    ac.abort();
    const p = new VoyageEmbeddingProvider({ apiKey: "k" });
    await expect(p.embedQuery("x", { signal: ac.signal })).rejects.toMatchObject({ name: "AbortError" });
  });

  it("refuses an oversized single input instead of truncating", async () => {
    const p = new VoyageEmbeddingProvider({ apiKey: "k" });
    const huge = "a".repeat(70_000);
    await expect(p.embedDocuments([huge])).rejects.toThrow(/Refusing to truncate/);
  });
});
