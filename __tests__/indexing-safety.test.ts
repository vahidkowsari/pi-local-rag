/**
 * A3: index replacement must not drop live rows until embeddings validate
 * and a short transaction commits. Unchanged files skip embedding.
 */
import { describe, it, expect, beforeAll, afterAll, afterEach, vi } from "vitest";
import { mkdtempSync, writeFileSync, rmSync, realpathSync } from "node:fs";
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

describe("indexFiles replacement safety", () => {
  let ragDir: string;
  let proj: string;
  let savedCwd: string;
  let savedRagDir: string | undefined;
  let mod: typeof import("../index.ts");
  let embedDocuments: ReturnType<typeof vi.spyOn>;

  beforeAll(async () => {
    ragDir = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-safe-")));
    proj = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-safe-proj-")));
    savedCwd = process.cwd();
    savedRagDir = process.env.PI_RAG_DIR;
    process.env.PI_RAG_DIR = ragDir;
    process.chdir(proj);
    vi.resetModules();
    mod = await import("../index.ts");
    const { LocalEmbeddingProvider } = await import("../providers/embedding/local.ts");
    embedDocuments = vi.spyOn(LocalEmbeddingProvider.prototype, "embedDocuments");
    embedDocuments.mockImplementation(async (texts: string[]) => texts.map(() => unitVec(1)));
  });

  afterEach(async () => {
    embedDocuments.mockReset();
    embedDocuments.mockImplementation(async (texts: string[]) => texts.map(() => unitVec(1)));
    const { closeDbConn } = await import("../db.ts");
    closeDbConn();
  });

  afterAll(() => {
    process.chdir(savedCwd);
    rmSync(ragDir, { recursive: true, force: true });
    rmSync(proj, { recursive: true, force: true });
    if (savedRagDir !== undefined) process.env.PI_RAG_DIR = savedRagDir;
    else delete process.env.PI_RAG_DIR;
  });

  function chunkCount(filePath: string): number {
    const db = mod.getFreshDbConn();
    try {
      return (db.prepare("SELECT COUNT(*) as c FROM chunks WHERE file_path = ?").get(filePath) as { c: number }).c;
    } finally {
      db.close();
    }
  }

  function fileRow(filePath: string): { hash: string; embedded: number; chunks: number } | undefined {
    const db = mod.getFreshDbConn();
    try {
      return db.prepare("SELECT hash, embedded, chunks FROM files WHERE path = ?").get(filePath) as
        | { hash: string; embedded: number; chunks: number }
        | undefined;
    } finally {
      db.close();
    }
  }

  function chunkContents(filePath: string): string[] {
    const db = mod.getFreshDbConn();
    try {
      return (db.prepare("SELECT chunk_content FROM chunks WHERE file_path = ? ORDER BY line_start").all(filePath) as Array<{ chunk_content: string }>)
        .map(r => r.chunk_content);
    } finally {
      db.close();
    }
  }

  async function searchable(query: string): Promise<string[]> {
    const results = await mod.hybridSearch(query, 10, 1.0);
    return results.map(r => r.chunk.content);
  }

  it("unchanged file is skipped with zero embedDocuments calls", async () => {
    const fp = join(proj, "stable.ts");
    writeFileSync(fp, "export const stableMarkerAlpha = 1;\n");
    await mod.indexFiles([fp]);
    embedDocuments.mockClear();
    const r = await mod.indexFiles([fp]);
    expect(r.skipped).toBe(1);
    expect(r.indexed).toBe(0);
    expect(embedDocuments).toHaveBeenCalledTimes(0);
  });

  it("modifying one file re-embeds only that file", async () => {
    const a = join(proj, "keep.ts");
    const b = join(proj, "change.ts");
    writeFileSync(a, "export const keepMarkerOriginal = 1;\n");
    writeFileSync(b, "export const changeMarkerOriginal = 1;\n");
    await mod.indexFiles([a, b]);
    writeFileSync(b, "export const changeMarkerUpdated = 2;\n");
    embedDocuments.mockClear();
    const r = await mod.indexFiles([a, b]);
    expect(r.indexed).toBe(1);
    expect(r.skipped).toBe(1);
    expect(embedDocuments).toHaveBeenCalledTimes(1);
    const embeddedTexts = (embedDocuments.mock.calls[0][0] as string[]).join("\n");
    expect(embeddedTexts).toContain("changeMarkerUpdated");
    expect(embeddedTexts).not.toContain("keepMarkerOriginal");
    expect(chunkContents(a).join("\n")).toContain("keepMarkerOriginal");
    expect(chunkContents(b).join("\n")).toContain("changeMarkerUpdated");
  });

  it("model exception during embed keeps the previous index searchable", async () => {
    const fp = join(proj, "live.ts");
    writeFileSync(fp, "export const liveMarkerOriginal = 1;\n");
    await mod.indexFiles([fp]);
    expect(chunkCount(fp)).toBeGreaterThan(0);
    writeFileSync(fp, "export const liveMarkerBroken = 2;\n");
    embedDocuments.mockImplementation(async () => { throw new Error("model down"); });
    const r = await mod.indexFiles([fp]);
    expect(r.indexed).toBe(0);
    expect(r.failed).toBe(1);
    expect(fileRow(fp)?.embedded).toBe(1);
    expect(chunkContents(fp).join("\n")).toContain("liveMarkerOriginal");
    embedDocuments.mockImplementation(async (texts: string[]) => texts.map(() => unitVec(1)));
    const hits = await searchable("liveMarkerOriginal");
    expect(hits.some(c => c.includes("liveMarkerOriginal"))).toBe(true);
    expect(hits.some(c => c.includes("liveMarkerBroken"))).toBe(false);
  });

  it("second embed batch failure keeps the previous index", async () => {
    const fp = join(proj, "batch.ts");
    writeFileSync(fp, "export const batchMarkerOriginal = 1;\n");
    await mod.indexFiles([fp]);
    const files = [fp];
    writeFileSync(fp, "export const batchMarkerNew = 2;\n");
    for (let i = 0; i < 300; i++) {
      const extra = join(proj, `batch-extra-${i}.ts`);
      writeFileSync(extra, `export const batchExtra${i} = ${"word ".repeat(20)};\n`);
      files.push(extra);
    }
    let calls = 0;
    embedDocuments.mockImplementation(async (texts: string[]) => {
      calls++;
      if (calls >= 2) throw new Error("second batch failed");
      return texts.map(() => unitVec(1));
    });
    const r = await mod.indexFiles(files);
    expect(calls).toBeGreaterThanOrEqual(2);
    expect(r.indexed).toBe(0);
    expect(r.failed).toBeGreaterThan(0);
    expect(chunkContents(fp).join("\n")).toContain("batchMarkerOriginal");
    expect(chunkContents(fp).join("\n")).not.toContain("batchMarkerNew");
  });

  it("wrong vector dimension does not mark the file embedded with mixed rows", async () => {
    const fp = join(proj, "dim.ts");
    writeFileSync(fp, "export const dimMarkerOriginal = 1;\n");
    await mod.indexFiles([fp]);
    writeFileSync(fp, "export const dimMarkerNew = 2;\n");
    embedDocuments.mockImplementation(async (texts: string[]) => texts.map(() => [0.1, 0.2]));
    const r = await mod.indexFiles([fp]);
    expect(r.indexed).toBe(0);
    expect(r.failed).toBe(1);
    expect(fileRow(fp)?.embedded).toBe(1);
    expect(chunkContents(fp).join("\n")).toContain("dimMarkerOriginal");
    expect(chunkContents(fp).join("\n")).not.toContain("dimMarkerNew");
  });

  it("NaN / missing vectors keep the previous index", async () => {
    const fp = join(proj, "nan.ts");
    writeFileSync(fp, "export const nanMarkerOriginal = 1;\n");
    await mod.indexFiles([fp]);
    writeFileSync(fp, "export const nanMarkerNew = 2;\n");
    embedDocuments.mockImplementation(async (texts: string[]) => texts.map(() => {
      const v = unitVec(1);
      v[3] = Number.NaN;
      return v;
    }));
    const r = await mod.indexFiles([fp]);
    expect(r.indexed).toBe(0);
    expect(r.failed).toBe(1);
    expect(chunkContents(fp).join("\n")).toContain("nanMarkerOriginal");
  });

  it("transaction write error keeps previous chunks, FTS, vectors, and file row", async () => {
    const fp = join(proj, "tx.ts");
    writeFileSync(fp, "export const txMarkerOriginal = 1;\n");
    await mod.indexFiles([fp]);
    writeFileSync(fp, "export const txMarkerNew = 2;\n");
    const db = mod.getDbConn();
    const orig = db.transaction.bind(db);
    db.transaction = ((fn: () => unknown) => {
      return () => { throw new Error("disk full"); };
    }) as unknown as typeof db.transaction;
    try {
      const r = await mod.indexFiles([fp]);
      expect(r.indexed).toBe(0);
      expect(r.failed).toBe(1);
    } finally {
      db.transaction = orig;
    }
    expect(chunkContents(fp).join("\n")).toContain("txMarkerOriginal");
    expect(fileRow(fp)?.embedded).toBe(1);
    const hits = await searchable("txMarkerOriginal");
    expect(hits.some(c => c.includes("txMarkerOriginal"))).toBe(true);
  });

  it("force rebuild failure keeps the original index; retry succeeds", async () => {
    const fp = join(proj, "force.ts");
    writeFileSync(fp, "export const forceMarkerOriginal = 1;\n");
    await mod.indexFiles([fp]);
    writeFileSync(fp, "export const forceMarkerNew = 2;\n");
    embedDocuments.mockImplementation(async () => { throw new Error("model down"); });
    const failed = await mod.indexFiles([fp], undefined, undefined, true);
    expect(failed.indexed).toBe(0);
    expect(failed.failed).toBe(1);
    expect(chunkContents(fp).join("\n")).toContain("forceMarkerOriginal");
    embedDocuments.mockImplementation(async (texts: string[]) => texts.map(() => unitVec(1)));
    const ok = await mod.indexFiles([fp], undefined, undefined, true);
    expect(ok.indexed).toBe(1);
    expect(chunkContents(fp).join("\n")).toContain("forceMarkerNew");
    expect(chunkContents(fp).join("\n")).not.toContain("forceMarkerOriginal");
  });
});
