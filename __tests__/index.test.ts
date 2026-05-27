/**
 * Combined test suite for pi-local-rag.
 *
 * Layout mirrors upstream forks (theli-ua, kallewoof) so future fork-sync work
 * can be ported in unchanged. Each top-level `describe(...)` groups tests for
 * one area of the module.
 */
import { describe, it, expect, beforeEach, afterEach, beforeAll, afterAll, vi } from "vitest";
import {
  mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, existsSync, realpathSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, relative, basename } from "node:path";
import { fileURLToPath } from "node:url";
import ignore from "ignore";
import Database from "better-sqlite3";
import { load as loadVec } from "sqlite-vec";

// Mock @xenova/transformers so search/embed tests don't load the ~23 MB ONNX
// model. The mocked pipeline handles both single-string and batched-array
// inputs (commit 849e485 fix).
vi.mock("@xenova/transformers", () => ({
  pipeline: vi.fn().mockResolvedValue(
    vi.fn().mockImplementation(async (texts: string | string[]) => {
      // Mirror the real Xenova/transformers batch API: always return a single
      // Tensor-like object whose `data` is a flat Float32Array of
      // [batchSize × dim].  Single-string input is treated as batchSize=1.
      const batch = Array.isArray(texts) ? texts : [texts];
      const DIM = 384;
      const flat = new Float32Array(batch.length * DIM).fill(0.1);
      return { data: flat };
    })
  ),
}));

const __dirname = dirname(fileURLToPath(import.meta.url));
const SAMPLE_PDF = readFileSync(join(__dirname, "fixtures", "sample.pdf"));
const SAMPLE_IMAGE_PDF = readFileSync(join(__dirname, "fixtures", "sample-image.pdf"));

// Imports that don't depend on env-time state can be static.
import {
  chunkText,
  cosineSimilarity,
  normalize,
  DEFAULT_TEXT_EXTS,
  normalizeExt,
  resolveExtensions,
  collectFiles,
  collectFromTracked,
  isExcludedByConfig,
  extractText,
  hybridSearch,
  embed,
  sha256,
  initSchema,
  getOcrTooling,
  isSparsePdfText,
} from "../index.ts";

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Create an in-memory SQLite DB with the RAG schema, pre-populated with chunks.
 *  Used by hybridSearch tests — avoids the per-file rag.db write overhead and
 *  isolates each test. Pulled verbatim from kallewoof@849e485 tests. */
function createTestDb(chunks: Array<{
  id?: string; file?: string; content: string; lineStart?: number; lineEnd?: number;
  vector?: number[];
}>): Database.Database {
  const db = new Database(":memory:");
  db.pragma("journal_mode = WAL");
  loadVec(db);
  initSchema(db);

  const insChunk = db.prepare(`
    INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insVec = db.prepare(
    "INSERT INTO chunks_vec(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)",
  );
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    const result = insChunk.run(
      c.id ?? `chunk-${i}`,
      c.file ?? "/src/file.ts",
      c.content,
      c.lineStart ?? 1,
      c.lineEnd ?? 10,
      sha256(c.content),
      new Date().toISOString(),
      Math.ceil(c.content.length / 4),
    );
    if (c.vector) {
      const f = new Float32Array(c.vector);
      insVec.run(Number(result.lastInsertRowid), Buffer.from(f.buffer, f.byteOffset, f.byteLength));
    }
  }
  return db;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

async function buildMinimalDocx(text: string): Promise<Buffer> {
  const { default: JSZip } = await import("jszip");
  const zip = new JSZip();
  zip.file(
    "[Content_Types].xml",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>`,
  );
  zip.folder("_rels")!.file(
    ".rels",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>`,
  );
  zip.folder("word")!.file(
    "document.xml",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>${text}</w:t></w:r></w:p>
  </w:body>
</w:document>`,
  );
  return await zip.generateAsync({ type: "nodebuffer" });
}

function chunkFixture(file: string, content: string, lineStart = 1) {
  return {
    id: `${file}-${lineStart}`,
    file,
    content,
    lineStart,
    lineEnd: lineStart + content.split("\n").length - 1,
    hash: "abc",
    indexed: new Date().toISOString(),
    tokens: Math.ceil(content.length / 4),
  };
}

// ─── chunkText ──────────────────────────────────────────────────────────────

describe("chunkText", () => {
  it("short text under threshold returns a single chunk starting at line 1", () => {
    const text = "line one\nline two\nline three";
    const chunks = chunkText(text);
    expect(chunks.length).toBe(1);
    expect(chunks[0].lineStart).toBe(1);
    expect(chunks[0].lineEnd).toBe(3);
    expect(chunks[0].content).toBe(text);
  });

  it("text just under 20 chars after trimming is dropped", () => {
    expect(chunkText("tiny").length).toBe(0);
  });

  it("respects maxLines and produces consecutive line ranges", () => {
    const lines = Array.from({ length: 120 }, (_, i) => `line ${i + 1} content`);
    const chunks = chunkText(lines.join("\n"), 50);
    expect(chunks.length).toBeGreaterThanOrEqual(2);
    expect(chunks[0].lineStart).toBe(1);
    for (let i = 1; i < chunks.length; i++) {
      expect(chunks[i].lineStart, "consecutive chunks should be contiguous")
        .toBe(chunks[i - 1].lineEnd + 1);
    }
  });

  it("prefers breaking at blank lines near the window end", () => {
    const lines = Array.from({ length: 80 }, (_, i) => (i === 44 ? "" : `content line ${i + 1}`));
    const chunks = chunkText(lines.join("\n"), 50);
    expect(chunks[0].lineEnd).toBe(45);
  });

  it("does not lose lines across the boundary", () => {
    const lines = Array.from({ length: 200 }, (_, i) => `data ${i}`);
    const chunks = chunkText(lines.join("\n"), 50);
    expect(chunks[chunks.length - 1].lineEnd).toBe(200);
  });
});

// ─── math: cosineSimilarity, normalize ──────────────────────────────────────

describe("cosineSimilarity", () => {
  it("identical vectors = 1", () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBe(1);
  });
  it("orthogonal vectors = 0", () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBe(0);
  });
  it("opposite vectors = -1", () => {
    expect(cosineSimilarity([1, 2, 3], [-1, -2, -3])).toBe(-1);
  });
  it("scale-invariant", () => {
    expect(Math.abs(cosineSimilarity([1, 2, 3], [2, 4, 6]) - 1)).toBeLessThan(1e-9);
  });
  it("mismatched lengths returns 0", () => {
    expect(cosineSimilarity([1, 2], [1, 2, 3])).toBe(0);
  });
  it("zero vector returns 0 (no divide-by-zero)", () => {
    expect(cosineSimilarity([0, 0, 0], [1, 2, 3])).toBe(0);
  });
});

describe("normalize", () => {
  it("maps to [0,1] preserving order", () => {
    const out = normalize([10, 0, 5]);
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(0);
    expect(out[2]).toBe(0.5);
  });
  it("all-equal input returns all zeros", () => {
    expect(normalize([3, 3, 3])).toEqual([0, 0, 0]);
  });
  it("single value returns [0]", () => {
    expect(normalize([7])).toEqual([0]);
  });
});

// ─── extensions ─────────────────────────────────────────────────────────────

describe("normalizeExt", () => {
  it("adds leading dot and lowercases", () => {
    expect(normalizeExt("cs")).toBe(".cs");
    expect(normalizeExt(".CS")).toBe(".cs");
    expect(normalizeExt("  .TeX  ")).toBe(".tex");
    expect(normalizeExt("")).toBe("");
    expect(normalizeExt("   ")).toBe("");
  });
});

describe("resolveExtensions", () => {
  it("returns the default set when no overrides", () => {
    const exts = resolveExtensions({ extraExtensions: [], excludeExtensions: [] });
    for (const e of DEFAULT_TEXT_EXTS) expect(exts.has(e), `default ${e} missing`).toBe(true);
    expect(exts.size).toBe(DEFAULT_TEXT_EXTS.length);
  });
  it("default set covers common languages including the ones from issue #9", () => {
    const exts = resolveExtensions({ extraExtensions: [], excludeExtensions: [] });
    for (const e of [".cs", ".tsx", ".jsx", ".kt", ".swift", ".rb", ".php", ".lua", ".vue", ".svelte"]) {
      expect(exts.has(e), `expected default set to include ${e}`).toBe(true);
    }
  });
  it("extraExtensions are added and normalized", () => {
    const exts = resolveExtensions({ extraExtensions: ["tex", ".ZIG", " .nix "], excludeExtensions: [] });
    expect(exts.has(".tex")).toBe(true);
    expect(exts.has(".zig")).toBe(true);
    expect(exts.has(".nix")).toBe(true);
  });
  it("excludeExtensions remove from the default set", () => {
    const exts = resolveExtensions({ extraExtensions: [], excludeExtensions: [".md", "JSON"] });
    expect(exts.has(".md")).toBe(false);
    expect(exts.has(".json")).toBe(false);
    expect(exts.has(".ts")).toBe(true);
  });
  it("empty/whitespace entries are ignored", () => {
    const baseline = resolveExtensions({ extraExtensions: [], excludeExtensions: [] }).size;
    const exts = resolveExtensions({ extraExtensions: ["", "   "], excludeExtensions: ["", "  "] });
    expect(exts.size).toBe(baseline);
  });
});

// ─── collectFiles ───────────────────────────────────────────────────────────

describe("collectFiles", () => {
  let tmp: string;
  beforeEach(() => { tmp = mkdtempSync(join(tmpdir(), "pi-rag-test-")); });
  afterEach(() => { rmSync(tmp, { recursive: true, force: true }); });

  it("walks dir, applies extension allowlist, skips node_modules and dotdirs", () => {
    writeFileSync(join(tmp, "a.ts"), "export const a = 1;");
    writeFileSync(join(tmp, "b.md"), "# heading");
    writeFileSync(join(tmp, "c.bin"), Buffer.from([0, 1, 2, 3]));
    writeFileSync(join(tmp, "image.png"), Buffer.alloc(10));
    mkdirSync(join(tmp, "node_modules"));
    writeFileSync(join(tmp, "node_modules", "skip.ts"), "// should not be indexed");
    mkdirSync(join(tmp, ".git"));
    writeFileSync(join(tmp, ".git", "config"), "x");
    mkdirSync(join(tmp, ".hidden"));
    writeFileSync(join(tmp, ".hidden", "secret.ts"), "// hidden");
    mkdirSync(join(tmp, "src"));
    writeFileSync(join(tmp, "src", "deep.py"), "print('hi')");
    writeFileSync(join(tmp, "huge.ts"), "x".repeat(500_001));

    const files = collectFiles(tmp).map(f => f.replace(tmp, "")).sort();
    expect(files).toContain("/a.ts");
    expect(files).toContain("/b.md");
    expect(files).toContain("/src/deep.py");
    expect(files.some(f => f.includes("node_modules"))).toBe(false);
    expect(files.some(f => f.includes(".git"))).toBe(false);
    expect(files.some(f => f.includes(".hidden"))).toBe(false);
    expect(files.some(f => f.endsWith(".bin") || f.endsWith(".png"))).toBe(false);
    expect(files.some(f => f.endsWith("huge.ts"))).toBe(false);
  });

  it("file path returns single entry when extension allowed", () => {
    const fp = join(tmp, "single.ts");
    writeFileSync(fp, "export {};");
    expect(collectFiles(fp)).toEqual([fp]);
  });

  it("file path returns empty when extension not allowed", () => {
    const fp = join(tmp, "data.bin");
    writeFileSync(fp, "x");
    expect(collectFiles(fp)).toEqual([]);
  });

  it("nonexistent path returns empty", () => {
    expect(collectFiles(join(tmpdir(), "definitely-not-here-xyz-12345"))).toEqual([]);
  });

  it("picks up .pdf and .docx even without being in TEXT_EXTS", () => {
    writeFileSync(join(tmp, "doc.pdf"), Buffer.from("%PDF-1.4 stub"));
    writeFileSync(join(tmp, "doc.docx"), Buffer.from("PK\x03\x04 stub"));
    writeFileSync(join(tmp, "a.ts"), "x");
    const files = collectFiles(tmp).map(f => f.replace(tmp, "")).sort();
    expect(files).toContain("/doc.pdf");
    expect(files).toContain("/doc.docx");
    expect(files).toContain("/a.ts");
  });

  it("9 MB PDF accepted, 500 KB text rejected", () => {
    writeFileSync(join(tmp, "big.pdf"), Buffer.alloc(9_000_000));
    writeFileSync(join(tmp, "big.txt"), "x".repeat(500_000));
    const files = collectFiles(tmp).map(f => f.replace(tmp, "")).sort();
    expect(files).toContain("/big.pdf");
    expect(files.some(f => f.endsWith("big.txt"))).toBe(false);
  });

  it("PDF over 10 MB cap is rejected", () => {
    writeFileSync(join(tmp, "huge.pdf"), Buffer.alloc(10_000_000));
    expect(collectFiles(tmp).length).toBe(0);
  });

  it("custom extension set is honored", () => {
    writeFileSync(join(tmp, "a.ts"), "x");
    writeFileSync(join(tmp, "b.cs"), "x");
    const files = collectFiles(tmp, new Set([".cs"]));
    expect(files.length).toBe(1);
    expect(files[0].endsWith("b.cs")).toBe(true);
  });

  it("excludePatterns filters a top-level file", () => {
    writeFileSync(join(tmp, "a.ts"), "x");
    writeFileSync(join(tmp, "b.ts"), "x");
    const files = collectFiles(tmp, undefined, ["b.ts"]).map(f => f.replace(tmp, ""));
    expect(files).not.toContain("/b.ts");
    expect(files).toContain("/a.ts");
  });

  it("excludePatterns filters a whole directory subtree", () => {
    writeFileSync(join(tmp, "a.ts"), "x");
    mkdirSync(join(tmp, "gen"));
    writeFileSync(join(tmp, "gen", "ignored.ts"), "x");
    const files = collectFiles(tmp, undefined, ["gen/"]).map(f => f.replace(tmp, ""));
    expect(files.some(f => f.includes("/gen/"))).toBe(false);
    expect(files).toContain("/a.ts");
  });

  it("extension glob exclude", () => {
    writeFileSync(join(tmp, "page.html"), "<p>x</p>");
    writeFileSync(join(tmp, "a.ts"), "x");
    const files = collectFiles(tmp, undefined, ["*.html"]).map(f => f.replace(tmp, ""));
    expect(files.some(f => f.endsWith(".html"))).toBe(false);
    expect(files.some(f => f.endsWith(".ts"))).toBe(true);
  });
});

// ─── collectFromTracked + isExcludedByConfig ────────────────────────────────

describe("collectFromTracked", () => {
  it("walks every tracked path, dedupes overlaps", () => {
    const a = mkdtempSync(join(tmpdir(), "rag-track-a-"));
    const b = mkdtempSync(join(tmpdir(), "rag-track-b-"));
    try {
      writeFileSync(join(a, "x.ts"), "x");
      writeFileSync(join(b, "y.ts"), "y");
      const cfg = {
        ragEnabled: true, ragTopK: 5, ragScoreThreshold: 0.1, ragAlpha: 0.4,
        extraExtensions: [], excludeExtensions: [],
        trackedPaths: [a, b, a],
        excludePatterns: [],
      };
      const files = collectFromTracked(cfg);
      expect(files.filter(f => f.endsWith("x.ts")).length).toBe(1);
      expect(files.some(f => f.endsWith("x.ts"))).toBe(true);
      expect(files.some(f => f.endsWith("y.ts"))).toBe(true);
    } finally {
      rmSync(a, { recursive: true, force: true });
      rmSync(b, { recursive: true, force: true });
    }
  });

  it("silently skips non-existent tracked paths", () => {
    const a = mkdtempSync(join(tmpdir(), "rag-track-a-"));
    try {
      writeFileSync(join(a, "x.ts"), "x");
      const cfg = {
        ragEnabled: true, ragTopK: 5, ragScoreThreshold: 0.1, ragAlpha: 0.4,
        extraExtensions: [], excludeExtensions: [],
        trackedPaths: [a, "/definitely/not/a/real/dir-xyz-123"],
        excludePatterns: [],
      };
      expect(collectFromTracked(cfg).length).toBe(1);
    } finally {
      rmSync(a, { recursive: true, force: true });
    }
  });

  it("applies excludePatterns per tracked root", () => {
    const root = mkdtempSync(join(tmpdir(), "rag-track-"));
    try {
      writeFileSync(join(root, "a.ts"), "x");
      mkdirSync(join(root, "gen"));
      writeFileSync(join(root, "gen", "ignored.ts"), "x");
      const cfg = {
        ragEnabled: true, ragTopK: 5, ragScoreThreshold: 0.1, ragAlpha: 0.4,
        extraExtensions: [], excludeExtensions: [],
        trackedPaths: [root],
        excludePatterns: ["gen/"],
      };
      const files = collectFromTracked(cfg);
      expect(files.some(f => f.includes("/gen/"))).toBe(false);
      expect(files.some(f => f.endsWith("a.ts"))).toBe(true);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

describe("isExcludedByConfig", () => {
  it("false when no patterns", () => {
    expect(isExcludedByConfig("/repo/a.ts", ["/repo"], [])).toBe(false);
  });
  it("matches a file relative to a root", () => {
    expect(isExcludedByConfig("/repo/gen/x.ts", ["/repo"], ["gen/"])).toBe(true);
    expect(isExcludedByConfig("/repo/src/x.ts", ["/repo"], ["gen/"])).toBe(false);
  });
  it("tries all roots; returns true if any matches", () => {
    expect(isExcludedByConfig("/repo-b/gen/x.ts", ["/repo-a", "/repo-b"], ["gen/"])).toBe(true);
  });
  it("file outside every root is not excluded", () => {
    expect(isExcludedByConfig("/elsewhere/a.ts", ["/repo"], ["*.ts"])).toBe(false);
  });
});

// ─── collectFilesAsync / collectFromTrackedAsync ────────────────────────────

describe("collectFilesAsync", () => {
  it("walks a tree like the sync version (extension allowlist, skip dirs, size caps)", async () => {
    const root = mkdtempSync(join(tmpdir(), "rag-async-walk-"));
    try {
      writeFileSync(join(root, "a.ts"), "x");
      writeFileSync(join(root, "b.md"), "x");
      writeFileSync(join(root, "huge.ts"), "x".repeat(500_001));
      mkdirSync(join(root, "node_modules"));
      writeFileSync(join(root, "node_modules", "skip.ts"), "x");
      mkdirSync(join(root, "src"));
      writeFileSync(join(root, "src", "deep.py"), "x");

      const { collectFilesAsync } = await import("../index.ts");
      const files = (await collectFilesAsync(root)).map(f => f.replace(root, "")).sort();
      expect(files).toContain("/a.ts");
      expect(files).toContain("/b.md");
      expect(files).toContain("/src/deep.py");
      expect(files.some(f => f.includes("node_modules"))).toBe(false);
      expect(files.some(f => f.endsWith("huge.ts"))).toBe(false);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("excludePatterns work the same as in the sync collectFiles", async () => {
    const root = mkdtempSync(join(tmpdir(), "rag-async-excl-"));
    try {
      writeFileSync(join(root, "page.html"), "<p>x</p>");
      writeFileSync(join(root, "a.ts"), "x");
      const { collectFilesAsync } = await import("../index.ts");
      const files = (await collectFilesAsync(root, undefined, ["*.html"])).map(f => f.replace(root, ""));
      expect(files.some(f => f.endsWith(".html"))).toBe(false);
      expect(files.some(f => f.endsWith(".ts"))).toBe(true);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

// ─── indexFiles force flag ──────────────────────────────────────────────────

describe("indexFiles --force", () => {
  let tmp: string;
  let savedRagDir: string | undefined;
  let mod: typeof import("../index.ts");

  beforeAll(async () => {
    tmp = realpathSync(mkdtempSync(join(tmpdir(), "rag-force-")));
    savedRagDir = process.env.PI_RAG_DIR;
    process.env.PI_RAG_DIR = tmp;
    vi.resetModules();
    mod = await import("../index.ts");
  });

  afterAll(() => {
    rmSync(tmp, { recursive: true, force: true });
    if (savedRagDir !== undefined) process.env.PI_RAG_DIR = savedRagDir;
    else delete process.env.PI_RAG_DIR;
  });

  it("second pass: skips unchanged files by default; re-embeds them when force=true", async () => {
    const proj = mkdtempSync(join(tmpdir(), "rag-force-proj-"));
    try {
      const fp = join(proj, "stable.ts");
      writeFileSync(fp, "export const stable = 1;\n");

      // First pass: file is fresh, gets indexed.
      const r1 = await mod.indexFiles([fp]);
      expect(r1.indexed).toBe(1);
      expect(r1.skipped).toBe(0);

      // Second pass without force: hash matches → file should be skipped.
      const r2 = await mod.indexFiles([fp]);
      expect(r2.skipped).toBe(1);
      expect(r2.indexed).toBe(0);

      // Third pass with force=true: re-embeds the file even though the hash
      // hasn't changed (commit 8432a15 / theli-ua).
      const r3 = await mod.indexFiles([fp], undefined, undefined, true);
      expect(r3.indexed).toBe(1);
      expect(r3.skipped).toBe(0);
    } finally {
      rmSync(proj, { recursive: true, force: true });
    }
  });
});

// ─── extractText (plain / PDF / DOCX / HTML) ────────────────────────────────

describe("extractText", () => {
  let tmp: string;
  beforeEach(() => { tmp = mkdtempSync(join(tmpdir(), "rag-extract-")); });
  afterEach(() => { rmSync(tmp, { recursive: true, force: true }); });

  it("reads plain text files as utf-8", async () => {
    const fp = join(tmp, "a.txt");
    writeFileSync(fp, "hello world");
    const { text, hash, size } = await extractText(fp);
    expect(text).toBe("hello world");
    expect(hash).toMatch(/^[0-9a-f]{12}$/);
    expect(size).toBe(11);
  });

  it("extracts text from a .pdf", async () => {
    const fp = join(tmp, "a.pdf");
    writeFileSync(fp, SAMPLE_PDF);
    const { text, hash, size } = await extractText(fp);
    expect(text).toContain("RagPdfMarker");
    expect(hash).toMatch(/^[0-9a-f]{12}$/);
    expect(size).toBe(SAMPLE_PDF.length);
  });

  it("extracts text from a .docx", async () => {
    const fp = join(tmp, "a.docx");
    writeFileSync(fp, await buildMinimalDocx("RagDocxMarker"));
    const { text } = await extractText(fp);
    expect(text).toContain("RagDocxMarker");
  });

  it("silences pdfjs Warning/Info console output during PDF parse", async () => {
    const fp = join(tmp, "loud.pdf");
    writeFileSync(fp, SAMPLE_PDF);
    const leaked: string[] = [];
    const origLog = console.log;
    console.log = (...args: unknown[]) => {
      const first = args[0];
      if (typeof first === "string" && /^(Warning|Info|Deprecated API usage):/.test(first)) {
        leaked.push(first);
      }
    };
    try {
      const r = await extractText(fp);
      expect(r.text).toContain("RagPdfMarker");
    } finally {
      console.log = origLog;
    }
    expect(leaked.length).toBe(0);
  });

  it("hash is stable across reads of the same binary file (skip-on-rebuild)", async () => {
    const fp = join(tmp, "stable.pdf");
    writeFileSync(fp, SAMPLE_PDF);
    const a = await extractText(fp);
    const b = await extractText(fp);
    expect(a.hash).toBe(b.hash);
  });
});

describe("extractText HTML", () => {
  let tmp: string;
  beforeEach(() => { tmp = mkdtempSync(join(tmpdir(), "rag-html-")); });
  afterEach(() => { rmSync(tmp, { recursive: true, force: true }); });

  it("converts simple HTML to markdown", async () => {
    const fp = join(tmp, "simple.html");
    writeFileSync(fp, "<p>Hello <strong>world</strong></p>");
    const { text } = await extractText(fp);
    expect(text).toContain("Hello");
    expect(text).toContain("world");
    expect(text).not.toContain("<p>");
    expect(text).not.toContain("<strong>");
  });

  it("removes script and style blocks", async () => {
    const fp = join(tmp, "no-script.html");
    writeFileSync(fp, "<p>Before</p><script>alert('xss')</script><style>.x{}</style><p>After</p>");
    const { text } = await extractText(fp);
    expect(text).toContain("Before");
    expect(text).toContain("After");
    expect(text).not.toContain("alert");
    expect(text).not.toContain(".x{}");
  });

  it("removes nav and footer elements", async () => {
    const fp = join(tmp, "no-nav.html");
    writeFileSync(fp, "<nav>Home | About</nav><p>Content</p><footer>Copyright</footer>");
    const { text } = await extractText(fp);
    expect(text).toContain("Content");
    expect(text).not.toContain("Home | About");
    expect(text).not.toContain("Copyright");
  });

  it("converts headings to atx style", async () => {
    const fp = join(tmp, "headings.html");
    writeFileSync(fp, "<h1>Title</h1><h2>Subtitle</h2><p>Body</p>");
    const { text } = await extractText(fp);
    expect(text).toContain("# Title");
    expect(text).toContain("## Subtitle");
    expect(text).toContain("Body");
  });

  it("fences code blocks", async () => {
    const fp = join(tmp, "code.html");
    writeFileSync(fp, '<pre><code class="lang-cs">var x = 1;</code></pre>');
    const { text } = await extractText(fp);
    expect(text).toContain("```");
    expect(text).toContain("var x = 1;");
  });

  it("converts lists to markdown", async () => {
    const fp = join(tmp, "lists.html");
    writeFileSync(fp, "<ul><li>One</li><li>Two</li></ul>");
    const { text } = await extractText(fp);
    expect(text).toContain("One");
    expect(text).toContain("Two");
    expect(text).not.toContain("<li>");
  });

  it("hashes the raw HTML, not the markdown", async () => {
    const fp = join(tmp, "hash-test.html");
    writeFileSync(fp, "<p>Content</p>");
    const { hash, text } = await extractText(fp);
    expect(hash).toMatch(/^[0-9a-f]{12}$/);
    expect(text).not.toContain("<p>");
  });

  it("handles real-world Unity doc HTML structure", async () => {
    const fp = join(tmp, "unity-doc.html");
    const html = `<!DOCTYPE html><html><head><script>var x = 1;</script></head>
<body><nav>Navigation</nav><div class="content"><h1>Add textures to the camera history</h1>
<p>To add your own texture to the <strong>camera</strong> history.</p>
<pre><code>public class Example : CameraHistoryItem { }</code></pre>
<ul><li>Step one</li><li>Step two</li></ul>
</div><footer>Copyright</footer></body></html>`;
    writeFileSync(fp, html);
    const { text } = await extractText(fp);
    expect(text).toContain("# Add textures to the camera history");
    expect(text).toContain("public class Example : CameraHistoryItem { }");
    expect(text).toContain("Step one");
    expect(text).toContain("Step two");
    expect(text).not.toContain("<script>");
    expect(text).not.toContain("var x");
    expect(text).not.toContain("Navigation");
    expect(text).not.toContain("Copyright");
  });

  it("also handles .htm extension", async () => {
    const fp = join(tmp, "page.htm");
    writeFileSync(fp, "<h1>Title</h1><p>Body</p>");
    const { text } = await extractText(fp);
    expect(text).toContain("# Title");
    expect(text).toContain("Body");
  });

  it("produces much smaller output than raw HTML for Unity docs", async () => {
    const fp = join(tmp, "big.html");
    const html = "<script>" + "x".repeat(5000) + "</script>"
      + "<style>" + "y".repeat(3000) + "</style>"
      + "<nav>" + "z".repeat(2000) + "</nav>"
      + "<p>Actual content here about framebuffer fetch</p>"
      + "<footer>" + "w".repeat(1000) + "</footer>";
    writeFileSync(fp, html);
    const { text } = await extractText(fp);
    expect(text.length).toBeLessThan(html.length / 2);
    expect(text).toContain("Actual content here about framebuffer fetch");
    expect(text).not.toContain("x".repeat(100));
  });
});

// ─── OCR fallback ────────────────────────────────────────────────────────────
// Tests ported verbatim from kallewoof@a5e2b96.

describe("isSparsePdfText", () => {
  it("empty text → sparse", () => {
    expect(isSparsePdfText("", 1)).toBe(true);
  });
  it("just below 50 chars/page → sparse", () => {
    expect(isSparsePdfText("x".repeat(49), 1)).toBe(true);
  });
  it("at the 50-char threshold → not sparse", () => {
    expect(isSparsePdfText("x".repeat(50), 1)).toBe(false);
  });
  it("scales with page count", () => {
    expect(isSparsePdfText("x".repeat(150), 3)).toBe(false);
    expect(isSparsePdfText("x".repeat(149), 3)).toBe(true);
  });
  it("numpages of 0 is treated as 1", () => {
    expect(isSparsePdfText("x".repeat(49), 0)).toBe(true);
    expect(isSparsePdfText("x".repeat(50), 0)).toBe(false);
  });
});

describe("getOcrTooling", () => {
  it("returns a stable shape", () => {
    const r = getOcrTooling();
    if (r.available) {
      expect(typeof r.langs).toBe("string");
      expect(r.langs.length).toBeGreaterThan(0);
    } else {
      expect(r).toEqual({ available: false });
    }
  });
  it("is cached across calls", () => {
    expect(getOcrTooling()).toBe(getOcrTooling());
  });
});

const ocrTools = getOcrTooling();
describe.skipIf(!ocrTools.available)("OCR end-to-end", () => {
  let tmp: string;
  beforeEach(() => { tmp = mkdtempSync(join(tmpdir(), "rag-ocr-test-")); });
  afterEach(() => { rmSync(tmp, { recursive: true, force: true }); });

  it("OCRs an image-only PDF and returns the rendered text", async () => {
    const fp = join(tmp, "image.pdf");
    writeFileSync(fp, SAMPLE_IMAGE_PDF);
    const { text } = await extractText(fp);
    expect(text).toMatch(/OcrMarker/);
  }, 60_000);
});

// ─── hybridSearch (FTS5 BM25 + sqlite-vec) ──────────────────────────────────
// Tests ported from kallewoof@849e485 — populate an in-memory DB via
// createTestDb() and pass it to hybridSearch's optional _db arg.

describe("hybridSearch (BM25 via FTS5, no vectors)", () => {
  it("empty index → []", async () => {
    const db = createTestDb([]);
    const results = await hybridSearch("query", { chunks: [], files: {}, lastBuild: "" }, 10, 0.4, db);
    db.close();
    expect(results).toEqual([]);
  });

  it("returns scored result for matching content", async () => {
    const db = createTestDb([
      { content: "function authenticate(user, password) { return checkCredentials(user, password); }" },
      { content: "function renderTemplate(html) { return sanitize(html); }" },
    ]);
    const results = await hybridSearch("authenticate", { chunks: [], files: {}, lastBuild: "" }, 10, 1.0, db);
    db.close();
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0].chunk.content).toContain("authenticate");
  });

  it("non-matching query → no results", async () => {
    const db = createTestDb([{ content: "function computeSquareRoot(n) { return Math.sqrt(n); }" }]);
    const results = await hybridSearch("unrelated query term xyz", { chunks: [], files: {}, lastBuild: "" }, 10, 1.0, db);
    db.close();
    const nonZero = results.filter(r => r.hybrid > 0);
    expect(nonZero.length).toBe(0);
  });

  it("exact phrase match scores higher than partial match", async () => {
    const db = createTestDb([
      { content: "function handle user authentication: validate token from request" },
      { content: "function handle request: process data from input" },
    ]);
    const results = await hybridSearch("user authentication", { chunks: [], files: {}, lastBuild: "" }, 10, 1.0, db);
    db.close();
    const first = results[0]?.chunk.content ?? "";
    expect(first).toContain("authentication");
  });

  it("respects limit parameter", async () => {
    const chunks = Array.from({ length: 10 }, (_, i) => ({
      content: `function processItem${i}(value) { return transform(value); }`,
    }));
    const db = createTestDb(chunks);
    const results = await hybridSearch("function process", { chunks: [], files: {}, lastBuild: "" }, 3, 1.0, db);
    db.close();
    expect(results.length).toBeLessThanOrEqual(3);
  });

  it("result shape has bm25, vector, hybrid, chunk fields", async () => {
    const db = createTestDb([{ content: "export function calculateTotal(items) { return items.reduce((a, b) => a + b, 0); }" }]);
    const results = await hybridSearch("calculate total", { chunks: [], files: {}, lastBuild: "" }, 10, 1.0, db);
    db.close();
    if (results.length > 0) {
      expect(results[0]).toHaveProperty("bm25");
      expect(results[0]).toHaveProperty("vector");
      expect(results[0]).toHaveProperty("hybrid");
      expect(results[0]).toHaveProperty("chunk");
    }
  });

  it("filename boost: first query term matching filename scores higher", async () => {
    const db = createTestDb([
      { file: "/src/auth module", content: "export function login for user verification" },
      { file: "/src/render module", content: "export function display for user rendering" },
    ]);
    const results = await hybridSearch("auth user", { chunks: [], files: {}, lastBuild: "" }, 10, 1.0, db);
    db.close();
    expect(results[0]?.chunk.file).toContain("auth");
  });
});

describe("hybridSearch with vectors", () => {
  const vec = (seed: number) => Array.from({ length: 384 }, (_, i) => (i === seed ? 1 : 0));

  it("uses vector scores when chunks have embeddings", async () => {
    const db = createTestDb([
      { content: "handle user login with password verification and auth", vector: vec(0) },
      { content: "render the homepage template with context data", vector: vec(1) },
    ]);
    const results = await hybridSearch("login", { chunks: [], files: {}, lastBuild: "" }, 10, 0.5, db);
    db.close();
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toHaveProperty("bm25");
    expect(results[0]).toHaveProperty("vector");
    expect(results[0]).toHaveProperty("hybrid");
  });

  it("hybrid score is blend of bm25 and vector when alpha=0.5", async () => {
    const db = createTestDb([
      { content: "authenticate user credentials and verify identity", vector: vec(0) },
      { content: "logout session token and destroy active session", vector: vec(1) },
    ]);
    const results = await hybridSearch("authenticate", { chunks: [], files: {}, lastBuild: "" }, 10, 0.5, db);
    db.close();
    expect(results.length).toBeGreaterThan(0);
    const r = results[0];
    const expectedHybrid = 0.5 * r.bm25 + 0.5 * r.vector;
    expect(r.hybrid).toBeCloseTo(expectedHybrid, 5);
  });

  it("falls back to pure bm25 when no chunks have valid vectors", async () => {
    const db = createTestDb([
      { content: "process payment amount through payment gateway charge" },
      { content: "refund order through payment gateway refund" },
    ]);
    const results = await hybridSearch("payment", { chunks: [], files: {}, lastBuild: "" }, 10, 0.5, db);
    db.close();
    if (results.length > 0) {
      expect(results[0].hybrid).toBe(results[0].bm25);
    }
  });
});

// ─── /rag find glob matching ────────────────────────────────────────────────

describe("/rag find glob matching", () => {
  function findMatches(indexedFiles: string[], glob: string, cwd: string): string[] {
    const ig = ignore().add([glob]);
    const matches: string[] = [];
    for (const fp of indexedFiles) {
      const rel = relative(cwd, fp);
      const candidate = rel && !rel.startsWith("..") ? rel : basename(fp);
      if (ig.ignores(candidate)) matches.push(fp);
    }
    return matches.sort();
  }

  it("matches by extension glob (*.ts)", () => {
    const files = ["/repo/src/a.ts", "/repo/src/b.js", "/repo/test/c.ts", "/repo/README.md"];
    expect(findMatches(files, "*.ts", "/repo")).toEqual(["/repo/src/a.ts", "/repo/test/c.ts"]);
  });
  it("matches by basename prefix (page*)", () => {
    const files = ["/repo/page1.html", "/repo/page2.html", "/repo/about.html"];
    expect(findMatches(files, "page*", "/repo")).toEqual(["/repo/page1.html", "/repo/page2.html"]);
  });
  it("matches a directory subtree (src/)", () => {
    const files = ["/repo/src/a.ts", "/repo/src/inner/b.ts", "/repo/test/c.ts"];
    const m = findMatches(files, "src", "/repo");
    expect(m).toContain("/repo/src/a.ts");
    expect(m).toContain("/repo/src/inner/b.ts");
    expect(m).not.toContain("/repo/test/c.ts");
  });
  it("returns empty when nothing matches", () => {
    expect(findMatches(["/repo/a.ts", "/repo/b.md"], "*.py", "/repo")).toEqual([]);
  });
  it("falls back to basename for files outside cwd", () => {
    expect(findMatches(["/elsewhere/notes.md", "/repo/src/a.ts"], "notes.md", "/repo")).toEqual(["/elsewhere/notes.md"]);
  });
  it("exact filename glob", () => {
    const m = findMatches(["/repo/src/foo.js", "/repo/lib/foo.js", "/repo/src/bar.js"], "foo.js", "/repo");
    expect(m).toContain("/repo/src/foo.js");
    expect(m).toContain("/repo/lib/foo.js");
    expect(m).not.toContain("/repo/src/bar.js");
  });
});

// ─── embed + hybrid (real ONNX pipeline; opt-out via SKIP_EMBEDDING_TESTS) ──

// (Real-ONNX embed + vector-path hybridSearch tests live in __tests__/embedding.test.ts —
// that file deliberately doesn't mock @xenova/transformers, matching the
// fork's split between index.test.ts (mocked) and embedding.test.ts (real).)

// ─── Storage: loadConfig / saveConfig / loadIndex / saveIndex / ensureDir ───
//
// These tests mutate process.env.PI_RAG_DIR / PI_RAG_LEGACY_DIR before
// importing index.ts, which means they need a fresh module instance (the
// env vars are read into module-top-level `const`s). `vi.resetModules()`
// invalidates the cached graph so the dynamic import re-evaluates.

describe("Storage (loadConfig/saveConfig/loadIndex/saveIndex/ensureDir)", () => {
  let ragDir: string;
  let legacyDir: string;
  // Bound at beforeAll-time via fresh module import.
  let loadConfig: typeof import("../index.ts").loadConfig;
  let saveConfig: typeof import("../index.ts").saveConfig;
  let loadIndex: typeof import("../index.ts").loadIndex;
  let saveIndex: typeof import("../index.ts").saveIndex;

  beforeAll(async () => {
    ragDir = mkdtempSync(join(tmpdir(), "pi-rag-storage-"));
    legacyDir = mkdtempSync(join(tmpdir(), "pi-lens-legacy-"));
    process.env.PI_RAG_DIR = ragDir;
    process.env.PI_RAG_LEGACY_DIR = legacyDir;
    rmSync(ragDir, { recursive: true, force: true });
    rmSync(legacyDir, { recursive: true, force: true });

    vi.resetModules();
    const mod = await import("../index.ts");
    ({ loadConfig, saveConfig, loadIndex, saveIndex } = mod);
  });

  afterAll(() => {
    rmSync(ragDir, { recursive: true, force: true });
    rmSync(legacyDir, { recursive: true, force: true });
    delete process.env.PI_RAG_DIR;
    delete process.env.PI_RAG_LEGACY_DIR;
  });

  it("loadConfig: returns defaults when no config file exists", () => {
    const cfg = loadConfig();
    expect(cfg.ragEnabled).toBe(true);
    expect(cfg.ragTopK).toBe(5);
    expect(cfg.ragScoreThreshold).toBe(0.1);
    expect(cfg.ragAlpha).toBe(0.4);
    expect(cfg.extraExtensions).toEqual([]);
    expect(cfg.excludeExtensions).toEqual([]);
    expect(cfg.trackedPaths).toEqual([]);
    expect(cfg.excludePatterns).toEqual([]);
  });

  it("saveConfig / loadConfig round-trip persists every field", () => {
    const written = {
      ragEnabled: false,
      ragTopK: 12,
      ragScoreThreshold: 0.25,
      ragAlpha: 0.7,
      extraExtensions: [".cs", ".tex"],
      excludeExtensions: [".md"],
      trackedPaths: ["/tmp/proj-a", "/tmp/proj-b"],
      excludePatterns: ["*.log", "node_modules/"],
    };
    saveConfig(written);
    expect(loadConfig()).toEqual(written);
    expect(existsSync(join(ragDir, "config.json"))).toBe(true);
    const raw = JSON.parse(readFileSync(join(ragDir, "config.json"), "utf-8"));
    expect(raw).toEqual(written);
  });

  it("loadConfig: merges saved partial config over defaults", () => {
    mkdirSync(ragDir, { recursive: true });
    writeFileSync(join(ragDir, "config.json"), JSON.stringify({ ragTopK: 99 }));
    const cfg = loadConfig();
    expect(cfg.ragTopK).toBe(99);
    expect(cfg.ragEnabled).toBe(true);
    expect(cfg.ragAlpha).toBe(0.4);
  });

  it("loadConfig: malformed JSON falls back to defaults instead of throwing", () => {
    writeFileSync(join(ragDir, "config.json"), "{not valid json");
    const cfg = loadConfig();
    expect(cfg.ragEnabled).toBe(true);
    expect(cfg.ragTopK).toBe(5);
  });

  it("loadIndex: empty/missing index returns an empty IndexMeta shell", () => {
    rmSync(join(ragDir, "index.json"), { force: true });
    const idx = loadIndex();
    expect(idx.chunks).toEqual([]);
    expect(idx.files).toEqual({});
    expect(idx.lastBuild).toBe("");
  });

  it("loadIndex: reconstructs IndexMeta from chunks + files + metadata in the DB", async () => {
    // Insert directly via the same openDb the rest of the system uses, then
    // read back through loadIndex. This replaces the old saveIndex/loadIndex
    // round-trip — saveIndex is now a no-op (writes are transactional in
    // indexFiles); loadIndex hydrates the legacy IndexMeta shape from SQLite.
    const mod = await import("../index.ts");
    const db = mod.openDb();
    try {
      const r = db.prepare(`
        INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `).run("abc-1", "/some/file.ts", "export const x = 1;", 1, 1, "deadbeef", "2026-05-15T00:00:00Z", 6);
      const vec = new Float32Array(384).fill(0.1);
      db.prepare("INSERT INTO chunks_vec(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)").run(
        Number(r.lastInsertRowid),
        Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength),
      );
      db.prepare(`
        INSERT OR REPLACE INTO files(path, hash, chunks, indexed, size, embedded)
        VALUES (?, ?, ?, ?, ?, ?)
      `).run("/some/file.ts", "deadbeef", 1, "2026-05-15T00:00:00Z", 19, 1);
      db.prepare("INSERT OR REPLACE INTO metadata(key, value) VALUES ('last_build', ?)").run("2026-05-15T00:00:00Z");
      db.prepare("INSERT OR REPLACE INTO metadata(key, value) VALUES ('embedding_model', ?)").run("Xenova/all-MiniLM-L6-v2");
    } finally {
      db.close();
    }

    const read = mod.loadIndex();
    expect(read.lastBuild).toBe("2026-05-15T00:00:00Z");
    expect(read.embeddingModel).toBe("Xenova/all-MiniLM-L6-v2");
    expect(read.chunks).toHaveLength(1);
    expect(read.chunks[0].id).toBe("abc-1");
    expect(read.chunks[0].file).toBe("/some/file.ts");
    expect(read.chunks[0].content).toBe("export const x = 1;");
    expect(read.files["/some/file.ts"]).toEqual({
      hash: "deadbeef", chunks: 1, indexed: "2026-05-15T00:00:00Z", size: 19, embedded: true,
    });
  });

  it("saveIndex is a no-op (writes happen via indexFiles transactions)", () => {
    // Documented behavior — saveIndex used to write index.json. Under SQLite,
    // chunk + file writes are committed by indexFiles. The export exists for
    // back-compat with callers that still call it.
    expect(() => saveIndex({ chunks: [], files: {}, lastBuild: "" })).not.toThrow();
  });
});

// ─── getRagDir: walk-up resolution + project vs global store ────────────────

describe("getRagDir (per-project store resolution)", () => {
  let fakeHome: string;
  let projectRoot: string;
  let savedCwd: string;
  let savedHome: string | undefined;
  let savedRagDir: string | undefined;
  let getRagDir: typeof import("../index.ts").getRagDir;
  let GLOBAL_RAG_DIR: typeof import("../index.ts").GLOBAL_RAG_DIR;

  // macOS resolves /var/folders/... → /private/var/folders/... through symlink.
  // mkdtempSync returns one form; process.cwd() after chdir returns the realpath.
  // Use the resolved form everywhere for stable comparisons.
  const resolveTmp = (p: string) => realpathSync(p);

  beforeAll(async () => {
    fakeHome = resolveTmp(mkdtempSync(join(tmpdir(), "pi-rag-home-")));
    projectRoot = resolveTmp(mkdtempSync(join(tmpdir(), "pi-rag-proj-")));
    savedCwd = process.cwd();
    savedHome = process.env.HOME;
    savedRagDir = process.env.PI_RAG_DIR;
    process.env.HOME = fakeHome;
    delete process.env.PI_RAG_DIR;

    vi.resetModules();
    ({ getRagDir, GLOBAL_RAG_DIR } = await import("../index.ts"));
  });

  afterAll(() => {
    process.chdir(savedCwd);
    rmSync(fakeHome, { recursive: true, force: true });
    rmSync(projectRoot, { recursive: true, force: true });
    if (savedHome !== undefined) process.env.HOME = savedHome; else delete process.env.HOME;
    if (savedRagDir !== undefined) process.env.PI_RAG_DIR = savedRagDir;
  });

  it("$PI_RAG_DIR override wins over everything", () => {
    const override = resolveTmp(mkdtempSync(join(tmpdir(), "pi-rag-override-")));
    process.env.PI_RAG_DIR = override;
    try {
      expect(getRagDir()).toBe(override);
    } finally {
      delete process.env.PI_RAG_DIR;
      rmSync(override, { recursive: true, force: true });
    }
  });

  it("returns ${cwd}/.pi/rag when one exists at cwd", () => {
    const projectStore = join(projectRoot, ".pi", "rag");
    mkdirSync(projectStore, { recursive: true });
    process.chdir(projectRoot);
    expect(getRagDir()).toBe(projectStore);
  });

  it("walks up to find a parent .pi/rag", () => {
    const sub = join(projectRoot, "src", "deep");
    mkdirSync(sub, { recursive: true });
    process.chdir(sub);
    expect(getRagDir()).toBe(join(projectRoot, ".pi", "rag"));
  });

  it("falls back to the global ~/.pi/rag when no project store is in scope", () => {
    const isolated = resolveTmp(mkdtempSync(join(tmpdir(), "pi-rag-iso-")));
    try {
      process.chdir(isolated);
      const got = getRagDir();
      expect(got).toBe(GLOBAL_RAG_DIR());
      expect(got.startsWith(fakeHome)).toBe(true);
    } finally {
      rmSync(isolated, { recursive: true, force: true });
    }
  });

  it("createIfMissing: anchors a new project store at cwd", () => {
    const fresh = resolveTmp(mkdtempSync(join(tmpdir(), "pi-rag-fresh-")));
    try {
      process.chdir(fresh);
      const got = getRagDir({ createIfMissing: true });
      expect(got).toBe(join(fresh, ".pi", "rag"));
      expect(existsSync(got)).toBe(true);
    } finally {
      rmSync(fresh, { recursive: true, force: true });
    }
  });
});

// The legacy ~/.pi/lens → ~/.pi/rag migration now only fires when the
// home-dir global store is in play (PI_RAG_DIR override skips it). To
// exercise it we have to fake HOME rather than use PI_RAG_DIR.
describe("ensureDir: legacy ~/.pi/lens → ~/.pi/rag migration (global store only)", () => {
  let fakeHome: string;
  let cwdSandbox: string;
  let savedCwd: string;
  let savedHome: string | undefined;
  let savedRagDir: string | undefined;
  let savedLegacyDir: string | undefined;
  let loadIndex: typeof import("../index.ts").loadIndex;

  beforeAll(async () => {
    fakeHome = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-home-")));
    // chdir to an isolated path so walk-up doesn't discover the real user's
    // ~/.pi/rag (which exists outside our fake HOME).
    cwdSandbox = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-cwd-")));
    savedCwd = process.cwd();
    savedHome = process.env.HOME;
    savedRagDir = process.env.PI_RAG_DIR;
    savedLegacyDir = process.env.PI_RAG_LEGACY_DIR;
    process.env.HOME = fakeHome;
    delete process.env.PI_RAG_DIR;
    delete process.env.PI_RAG_LEGACY_DIR;
    process.chdir(cwdSandbox);

    // Populate the legacy dir at the fake home so migration has work to do.
    // Under SQLite this exercises two migration paths in sequence:
    //   1. ensureDir() renames ~/.pi/lens → ~/.pi/rag (dir rename)
    //   2. openDb() spots the moved index.json inside ~/.pi/rag and imports
    //      its contents into rag.db, then unlinks the JSON file.
    // migrateFromJson skips the import when chunks.length === 0, so the
    // fixture has at least one chunk so we can observe the round-trip.
    const legacy = join(fakeHome, ".pi", "lens");
    mkdirSync(legacy, { recursive: true });
    writeFileSync(join(legacy, "index.json"), JSON.stringify({
      chunks: [{
        id: "legacy-1",
        file: "/legacy/file.ts",
        content: "from-legacy-payload",
        lineStart: 1, lineEnd: 1,
        hash: "abc",
        indexed: "2026-05-01T00:00:00Z",
        tokens: 5,
      }],
      files: {},
      lastBuild: "from-legacy",
    }));

    vi.resetModules();
    ({ loadIndex } = await import("../index.ts"));
  });

  afterAll(() => {
    process.chdir(savedCwd);
    rmSync(fakeHome, { recursive: true, force: true });
    rmSync(cwdSandbox, { recursive: true, force: true });
    if (savedHome !== undefined) process.env.HOME = savedHome; else delete process.env.HOME;
    if (savedRagDir !== undefined) process.env.PI_RAG_DIR = savedRagDir;
    if (savedLegacyDir !== undefined) process.env.PI_RAG_LEGACY_DIR = savedLegacyDir;
  });

  it("renames ~/.pi/lens → ~/.pi/rag, imports legacy index.json into rag.db, then deletes the JSON", () => {
    const idx = loadIndex();
    expect(idx.lastBuild).toBe("from-legacy");
    expect(idx.chunks).toHaveLength(1);
    expect(idx.chunks[0].content).toBe("from-legacy-payload");
    expect(existsSync(join(fakeHome, ".pi", "rag"))).toBe(true);
    expect(existsSync(join(fakeHome, ".pi", "lens"))).toBe(false);
    // index.json should have been consumed by openDb's auto-migration.
    expect(existsSync(join(fakeHome, ".pi", "rag", "index.json"))).toBe(false);
    // The DB itself should exist now.
    expect(existsSync(join(fakeHome, ".pi", "rag", "rag.db"))).toBe(true);
  });
});

// ─── isIndexStale ───────────────────────────────────────────────────────────

import { isIndexStale } from "../index.ts";

describe("isIndexStale", () => {
  const DAY_MS = 24 * 60 * 60 * 1000;
  const fresh = () => new Date(Date.now() - 60_000).toISOString();
  const stale = () => new Date(Date.now() - DAY_MS - 1_000).toISOString();

  it("returns false when lastBuild is empty", () => {
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: "" })).toBe(false);
  });

  it("returns false when index was built recently", () => {
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: fresh() })).toBe(false);
  });

  it("returns true when lastBuild is more than 24 h ago", () => {
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: stale() })).toBe(true);
  });

  it("respects a custom maxAgeMs", () => {
    const tenMinAgo = new Date(Date.now() - 10 * 60 * 1_000).toISOString();
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: tenMinAgo }, 5 * 60 * 1_000)).toBe(true);
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: tenMinAgo }, 15 * 60 * 1_000)).toBe(false);
  });
});

// ─── before_agent_start auto-refresh ────────────────────────────────────────
//
// Exercises the 24h auto-refresh path. Uses vi.doMock to swap the embedder for
// a fast stub (the real ONNX model is ~23 MB and slow to load). PI_RAG_DIR
// pins storage to a throwaway dir so the hook can mutate the on-disk index
// without touching the developer's real ~/.pi/rag.

describe("before_agent_start: 24h auto-refresh", () => {
  const DAY_MS = 24 * 60 * 60 * 1000;
  let ragDir: string;
  let cwdSandbox: string;
  let savedCwd: string;
  let savedRagDir: string | undefined;
  let mod: typeof import("../index.ts");
  let extensionFactory: typeof import("../index.ts").default;

  function makePi() {
    let hookFn: ((event: any, ctx: any) => Promise<any>) | undefined;
    const pi = {
      on: (event: string, fn: any) => { if (event === "before_agent_start") hookFn = fn; },
      registerCommand: () => {},
      registerTool: () => {},
      sendMessage: () => {},
    };
    const fire = (event = { prompt: "hello world", systemPrompt: "" }) => hookFn!(event, {});
    return { pi, fire };
  }

  /** Write a single chunk + file row + lastBuild directly into the DB. */
  function seedIndex(opts: { filePath: string; lastBuild: string; fileHash?: string }) {
    const db = mod.openDb();
    try {
      // Clear so each test starts clean.
      db.exec(`DELETE FROM chunks_vec; DELETE FROM chunks; DELETE FROM files; DELETE FROM metadata;`);
      const r = db.prepare(`
        INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `).run("test-1", opts.filePath, "const x = 1;", 1, 1, "abc", opts.lastBuild, 5);
      const vec = new Float32Array(384).fill(0.1);
      db.prepare("INSERT INTO chunks_vec(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)").run(
        Number(r.lastInsertRowid),
        Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength),
      );
      db.prepare(`
        INSERT OR REPLACE INTO files(path, hash, chunks, indexed, size, embedded)
        VALUES (?, ?, ?, ?, ?, ?)
      `).run(opts.filePath, opts.fileHash ?? "old", 1, opts.lastBuild, 10, 1);
      db.prepare("INSERT OR REPLACE INTO metadata(key, value) VALUES ('last_build', ?)").run(opts.lastBuild);
    } finally {
      db.close();
    }
  }

  function readLastBuild(): string {
    const db = mod.openDb();
    try {
      const row = db.prepare("SELECT value FROM metadata WHERE key='last_build'").get() as { value?: string } | undefined;
      return row?.value ?? "";
    } finally {
      db.close();
    }
  }

  beforeAll(async () => {
    ragDir = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-refresh-")));
    cwdSandbox = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-refresh-cwd-")));
    savedCwd = process.cwd();
    savedRagDir = process.env.PI_RAG_DIR;
    process.env.PI_RAG_DIR = ragDir;
    process.chdir(cwdSandbox);

    vi.resetModules();
    mod = await import("../index.ts");
    extensionFactory = mod.default;
  });

  afterAll(() => {
    process.chdir(savedCwd);
    rmSync(ragDir, { recursive: true, force: true });
    rmSync(cwdSandbox, { recursive: true, force: true });
    if (savedRagDir !== undefined) process.env.PI_RAG_DIR = savedRagDir;
    else delete process.env.PI_RAG_DIR;
  });

  it("does not update last_build when index is fresh", async () => {
    const freshBuild = new Date(Date.now() - 60_000).toISOString();
    seedIndex({ filePath: "/some/file.ts", lastBuild: freshBuild });
    const { pi, fire } = makePi();
    extensionFactory(pi as any);
    await fire();
    expect(readLastBuild()).toBe(freshBuild);
  });

  it("updates last_build when index is stale and files exist on disk", async () => {
    const testFile = join(cwdSandbox, "sample.ts");
    writeFileSync(testFile, "export const answer = 42;\n");
    const staleBuild = new Date(Date.now() - DAY_MS - 1_000).toISOString();
    seedIndex({ filePath: testFile, lastBuild: staleBuild });
    const { pi, fire } = makePi();
    extensionFactory(pi as any);
    await fire();
    expect(new Date(readLastBuild()).getTime()).toBeGreaterThan(new Date(staleBuild).getTime());
  });

  it("does not update last_build when stale but all referenced files are gone", async () => {
    const staleBuild = new Date(Date.now() - DAY_MS - 1_000).toISOString();
    const missingFile = join(cwdSandbox, "deleted.ts");
    seedIndex({ filePath: missingFile, lastBuild: staleBuild });
    const { pi, fire } = makePi();
    extensionFactory(pi as any);
    await fire();
    expect(readLastBuild()).toBe(staleBuild);
  });
});
