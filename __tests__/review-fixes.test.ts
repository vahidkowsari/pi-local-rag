import { describe, it, expect, beforeAll, afterAll, afterEach, vi } from "vitest";
import { mkdtempSync, writeFileSync, rmSync, realpathSync, existsSync, mkdirSync, readFileSync, renameSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const DIM = 384;
function unitVec(seed = 1): number[] {
  const v = new Array(DIM).fill(0);
  v[0] = seed;
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map(x => x / n);
}

vi.mock("@xenova/transformers", () => ({
  pipeline: vi.fn().mockResolvedValue(
    vi.fn().mockImplementation(async (texts: string | string[]) => {
      const batch = Array.isArray(texts) ? texts : [texts];
      const flat = new Float32Array(batch.length * DIM).fill(0.1);
      return { data: flat };
    }),
  ),
}));

describe("Astra review regressions", () => {
  let ragDir: string;
  let proj: string;
  let savedCwd: string;
  let savedRagDir: string | undefined;
  let savedKey: string | undefined;
  let mod: typeof import("../index.ts");
  let embedDocuments: ReturnType<typeof vi.spyOn>;
  let embedQuery: ReturnType<typeof vi.spyOn>;

  beforeAll(async () => {
    ragDir = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-rev-")));
    proj = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-rev-proj-")));
    savedCwd = process.cwd();
    savedRagDir = process.env.PI_RAG_DIR;
    savedKey = process.env.VOYAGE_API_KEY;
    process.env.PI_RAG_DIR = ragDir;
    delete process.env.VOYAGE_API_KEY;
    process.chdir(proj);
    vi.resetModules();
    mod = await import("../index.ts");
    const { LocalEmbeddingProvider } = await import("../providers/embedding/local.ts");
    embedDocuments = vi.spyOn(LocalEmbeddingProvider.prototype, "embedDocuments");
    embedDocuments.mockImplementation(async (texts: string[]) => texts.map(() => unitVec(1)));
    embedQuery = vi.spyOn(LocalEmbeddingProvider.prototype, "embedQuery");
    embedQuery.mockImplementation(async () => unitVec(1));
  });

  afterEach(async () => {
    embedDocuments.mockReset();
    embedDocuments.mockImplementation(async (texts: string[]) => texts.map(() => unitVec(1)));
    embedQuery.mockReset();
    embedQuery.mockImplementation(async () => unitVec(1));
    writeFileSync(join(ragDir, "config.json"), JSON.stringify(mod.defaultConfig()));
    process.env.PI_RAG_DIR = ragDir;
    delete process.env.VOYAGE_API_KEY;
    const { closeDbConn } = await import("../db.ts");
    closeDbConn();
    vi.unstubAllGlobals();
  });

  afterAll(() => {
    process.chdir(savedCwd);
    rmSync(ragDir, { recursive: true, force: true });
    rmSync(proj, { recursive: true, force: true });
    if (savedRagDir !== undefined) process.env.PI_RAG_DIR = savedRagDir;
    else delete process.env.PI_RAG_DIR;
    if (savedKey !== undefined) process.env.VOYAGE_API_KEY = savedKey;
    else delete process.env.VOYAGE_API_KEY;
  });

  function seedFile(name: string, content: string): string {
    const fp = join(proj, name);
    writeFileSync(fp, content);
    return fp;
  }

  it("R1: parse failure does not publish a new empty active index", async () => {
    const keep = seedFile("keep-r1.md", "alpha evidence stays here ".repeat(4));
    await mod.indexFiles([keep]);
    const before = mod.getIndexStats().totalChunks;
    expect(before).toBeGreaterThan(0);
    const result = await mod.rebuildWithSwitch([join(proj, "missing-r1.md")], undefined, true);
    expect(result.failed).toBeGreaterThan(0);
    expect(result.errors.some(e => /missing-r1/.test(e))).toBe(true);
    expect(mod.getIndexStats().totalChunks).toBe(before);
  });

  it("R2: failed force rebuild does not prune dropped files from the live index", async () => {
    const keep = seedFile("keep-r2.md", "keep marker original ".repeat(4));
    const gone = seedFile("gone-r2.md", "gone marker original ".repeat(4));
    await mod.indexFiles([keep, gone]);
    rmSync(gone);
    embedDocuments.mockImplementation(async () => { throw new Error("model down"); });
    const result = await mod.rebuildWithSwitch([keep], undefined, true, [gone]);
    expect(result.failed).toBeGreaterThan(0);
    const db = mod.getDbConn();
    const { listFilePaths } = await import("../repository.ts");
    expect(listFilePaths(db).some(p => p.endsWith("gone-r2.md"))).toBe(true);
    expect(listFilePaths(db).some(p => p.endsWith("keep-r2.md"))).toBe(true);
  });

  it("R3: cloudAutoRefresh=false skips cloud document refresh on auto-inject", async () => {
    process.env.VOYAGE_API_KEY = "synthetic-key";
    const isolated = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-cloud-")));
    process.env.PI_RAG_DIR = isolated;
    const { closeDbConn } = await import("../db.ts");
    closeDbConn();
    const cfg = mod.defaultConfig();
    cfg.embedding = { provider: "voyage", model: "voyage-4-lite", dimensions: 1024 };
    cfg.cloudAutoRefresh = false;
    cfg.trackedPaths = [proj];
    writeFileSync(join(isolated, "config.json"), JSON.stringify(cfg));
    const fp = seedFile("cloud.md", "alpha new synthetic cloud document ".repeat(4));
    const db = mod.getDbConn();
    const repo = await import("../repository.ts");
    const r = repo.insertChunk(db, {
      id: "c1", filePath: fp, content: "alpha evidence", lineStart: 1, lineEnd: 1,
      hash: "h", indexedAt: "2020-01-01T00:00:00Z", tokens: 8,
    });
    const vec = new Float32Array(1024).fill(0);
    vec[0] = 1;
    repo.insertVector(db, Number(r.lastInsertRowid), Array.from(vec));
    repo.upsertFile(db, fp, "h", 1, "2020-01-01T00:00:00Z", 10, true);
    mod.stampFingerprints(db, cfg);
    repo.setMetadata(db, repo.MetadataKey.LastBuild, "2020-01-01T00:00:00Z");
    const calls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as { input_type?: string; input?: unknown };
      calls.push(body.input_type ?? "unknown");
      const n = Array.isArray(body.input) ? body.input.length : 1;
      return new Response(JSON.stringify({
        data: Array.from({ length: n }, (_, i) => ({ index: i, embedding: Array.from(vec) })),
      }), { status: 200 });
    }));
    const hooks: Record<string, (e: { prompt: string }) => Promise<unknown>> = {};
    mod.default({
      on: (n: string, f: (e: { prompt: string }) => Promise<unknown>) => { hooks[n] = f; },
      registerTool: () => {},
      registerCommand: () => {},
    } as never);
    await hooks.before_agent_start({ prompt: "alpha" });
    expect(calls.includes("document")).toBe(false);
    closeDbConn();
    rmSync(isolated, { recursive: true, force: true });
  });

  it("R5: query on an unfingerprinted index does not assume the local model", async () => {
    const fp = seedFile("legacy.md", "alpha evidence for review ".repeat(4));
    await mod.indexFiles([fp]);
    const db = mod.getDbConn();
    const repo = await import("../repository.ts");
    repo.setMetadata(db, repo.MetadataKey.EmbeddingFingerprint, "");
    repo.setMetadata(db, repo.MetadataKey.ProcessingFingerprint, "");
    const compat = mod.checkIndexCompatibility(db, mod.defaultConfig());
    expect(compat.ok).toBe(false);
    await expect(mod.hybridSearch("alpha", 10, 0.4, db)).rejects.toMatchObject({ name: "IndexIncompatibleError" });
    mod.stampFingerprints(db, mod.defaultConfig());
  });

  it("R6: empty index with a 384-d table is incompatible with 1024-d config", async () => {
    const db = mod.getDbConn();
    const other = mod.defaultConfig();
    other.embedding = { provider: "voyage", model: "voyage-4-lite", dimensions: 1024 };
    const r = mod.checkIndexCompatibility(db, other);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toMatch(/dimension/i);
  });

  it("R7: preparing staging for an old contract does not delete the previous generation", async () => {
    const cfg = mod.defaultConfig();
    const a = mod.prepareStagingDir(cfg, ragDir);
    mkdirSync(join(a.dbPath, ".."), { recursive: true });
    writeFileSync(a.dbPath, "old-a");
    const bCfg = { ...cfg, embedding: { ...cfg.embedding, model: "other-minilm" } };
    const b = mod.prepareStagingDir(bCfg, ragDir);
    writeFileSync(b.dbPath, "b");
    const a2 = mod.prepareStagingDir(cfg, ragDir);
    expect(existsSync(a.dbPath)).toBe(true);
    expect(a2.dbPath).not.toBe(a.dbPath);
  });

  it("R8: an already-aborted signal does not call query embedding", async () => {
    const fp = seedFile("abort.md", "alpha evidence abort ".repeat(4));
    await mod.indexFiles([fp]);
    embedQuery.mockClear();
    const ac = new AbortController();
    ac.abort();
    await expect(mod.retrieve("alpha", { signal: ac.signal })).rejects.toMatchObject({ name: "AbortError" });
    expect(embedQuery).toHaveBeenCalledTimes(0);
  });

  it("R10: transient query embedding failure falls back to BM25 and marks degraded", async () => {
    const fp = seedFile("bm25.md", "alpha unique bm25 fallback marker ".repeat(4));
    await mod.indexFiles([fp]);
    embedQuery.mockImplementation(async () => { throw new Error("synthetic network outage"); });
    const hits = await mod.retrieve("alpha unique bm25");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].degraded).toMatch(/BM25 only/i);
  });

  it("R11: context and retrieve keep page/section/id", async () => {
    const hits = [{
      chunk: {
        id: "doc-2", file: "paper.pdf", content: "method text",
        lineStart: 0, lineEnd: 0, hash: "h", indexed: "", tokens: 4,
        pageStart: 2, pageEnd: 2, section: "Method", chunkIndex: 1,
      },
      bm25: 1, vector: 1, hybrid: 1,
    }];
    const built = mod.buildContext(hits, { maxTokens: 4096 });
    expect(built.text).toContain("page 2");
    expect(built.text).toContain("Method");
    expect(built.text).toContain("id=doc-2");
    expect(built.text).not.toMatch(/lines 0-0/);
  });

  it("R16: the same local contract reuses one provider instance", async () => {
    const a = mod.createEmbeddingProvider(mod.defaultConfig());
    const b = mod.createEmbeddingProvider(mod.defaultConfig());
    expect(a).toBe(b);
  });

  it("F3: abort during the last embed batch does not write or publish", async () => {
    const fp = seedFile("abort-mid.md", "original evidence stays ".repeat(4));
    await mod.indexFiles([fp]);
    const before = (mod.getDbConn().prepare("SELECT chunk_content FROM chunks").all() as Array<{ chunk_content: string }>)
      .map(r => r.chunk_content).join("\n");
    expect(before).toContain("original evidence stays");
    writeFileSync(fp, "replacement evidence should not land ".repeat(4));
    const ac = new AbortController();
    embedDocuments.mockImplementation(async (texts: string[]) => {
      ac.abort();
      return texts.map(() => unitVec(1));
    });
    await expect(mod.rebuildWithSwitch([fp], undefined, true, [], ac.signal)).rejects.toMatchObject({ name: "AbortError" });
    const after = (mod.getDbConn().prepare("SELECT chunk_content FROM chunks").all() as Array<{ chunk_content: string }>)
      .map(r => r.chunk_content).join("\n");
    expect(after).toContain("original evidence stays");
    expect(after).not.toContain("replacement evidence should not land");
  });

  it("F4: rag_index does not overwrite a broken config.json", async () => {
    writeFileSync(join(ragDir, "config.json"), "{BROKEN");
    const tools = new Map<string, { execute: (id: string, params: Record<string, unknown>) => Promise<{ content: Array<{ text: string }> }> }>();
    mod.default({
      on: () => {},
      registerCommand: () => {},
      registerTool: (t: { name: string; execute: (id: string, params: Record<string, unknown>) => Promise<{ content: Array<{ text: string }> }> }) => { tools.set(t.name, t); },
    } as never);
    const added = seedFile("new-broken.md", "added evidence");
    const out = await tools.get("rag_index")!.execute("probe", { path: added });
    expect(out.content[0].text).toMatch(/invalid/i);
    expect(readFileSync(join(ragDir, "config.json"), "utf-8")).toBe("{BROKEN");
    const q = await tools.get("rag_query")!.execute("probe", { query: "added" });
    expect(q.content[0].text).toMatch(/invalid/i);
  });

  it("F6: a truly empty 384-d schema is incompatible with 1024-d config", async () => {
    const isolated = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-empty-")));
    process.env.PI_RAG_DIR = isolated;
    const { closeDbConn } = await import("../db.ts");
    closeDbConn();
    const db = mod.getDbConn();
    expect((db.prepare("SELECT COUNT(*) as c FROM chunks").get() as { c: number }).c).toBe(0);
    const other = mod.defaultConfig();
    other.embedding = { provider: "voyage", model: "voyage-4-lite", dimensions: 1024 };
    const r = mod.checkIndexCompatibility(db, other);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toMatch(/dimension/i);
    closeDbConn();
    rmSync(isolated, { recursive: true, force: true });
  });

  it("F7: Voyage HTTP timeout/retry changes produce a new provider", async () => {
    process.env.VOYAGE_API_KEY = "synthetic-offline-key";
    mod.resetEmbeddingProviderCache();
    const cfg = mod.defaultConfig();
    cfg.embedding = { provider: "voyage", model: "voyage-4-lite", dimensions: 1024 };
    cfg.http = { timeoutMs: 30_000, maxRetries: 3 };
    const a = mod.createEmbeddingProvider(cfg) as unknown as { timeoutMs: number; maxRetries: number };
    const b = mod.createEmbeddingProvider({ ...cfg, http: { timeoutMs: 1000, maxRetries: 0 } }) as unknown as { timeoutMs: number; maxRetries: number };
    expect(a).not.toBe(b);
    expect(b.timeoutMs).toBe(1000);
    expect(b.maxRetries).toBe(0);
    delete process.env.VOYAGE_API_KEY;
    mod.resetEmbeddingProviderCache();
  });

  it("T1: a missing tracked root does not publish an empty index", async () => {
    const corpus = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-mount-")));
    const keep = join(corpus, "keep.md");
    const gone = join(corpus, "gone.md");
    writeFileSync(keep, "alpha evidence");
    writeFileSync(gone, "beta evidence");
    const cfg = mod.defaultConfig();
    cfg.trackedPaths = [corpus];
    writeFileSync(join(ragDir, "config.json"), JSON.stringify(cfg));
    await mod.indexFiles([keep, gone]);
    const before = mod.getIndexStats().totalChunks;
    expect(before).toBeGreaterThan(0);
    const parked = `${corpus}-unmounted`;
    renameSync(corpus, parked);
    let handler: ((args: string, ctx: { ui: Record<string, unknown> }) => Promise<void>) | undefined;
    const notes: string[] = [];
    mod.default({
      on: () => {},
      registerTool: () => {},
      registerCommand: (_n: string, c: { handler: (args: string, ctx: { ui: Record<string, unknown> }) => Promise<void> }) => { handler = c.handler; },
    } as never);
    const ctx = {
      ui: {
        notify: (m: string) => { notes.push(m); },
        setStatus: () => {},
        setWidget: () => {},
        theme: { bold: (s: string) => s, fg: (_c: string, s: string) => s },
      },
    };
    await handler!("rebuild --force", ctx);
    expect(notes.some(m => /unavailable|refusing/i.test(m))).toBe(true);
    const { closeDbConn } = await import("../db.ts");
    closeDbConn();
    expect(mod.getIndexStats().totalChunks).toBe(before);
    renameSync(parked, corpus);
    rmSync(corpus, { recursive: true, force: true });
  });

  it("T3: /rag search refuses a broken config.json", async () => {
    const fp = seedFile("search-broken.md", "alpha evidence for search ".repeat(4));
    await mod.indexFiles([fp]);
    writeFileSync(join(ragDir, "config.json"), "{BROKEN");
    let handler: ((args: string, ctx: { ui: Record<string, unknown> }) => Promise<void>) | undefined;
    const notes: Array<{ m: string; l?: string }> = [];
    const widgets: unknown[] = [];
    mod.default({
      on: () => {},
      registerTool: () => {},
      registerCommand: (_n: string, c: { handler: (args: string, ctx: { ui: Record<string, unknown> }) => Promise<void> }) => { handler = c.handler; },
    } as never);
    const ctx = {
      ui: {
        notify: (m: string, l?: string) => { notes.push({ m, l }); },
        setStatus: () => {},
        setWidget: (_n: string, v: unknown) => { widgets.push(v); },
        theme: { bold: (s: string) => s, fg: (_c: string, s: string) => s },
      },
    };
    await handler!("search alpha", ctx);
    expect(notes.some(n => /invalid/i.test(n.m))).toBe(true);
    expect(widgets.length).toBe(0);
  });
});
