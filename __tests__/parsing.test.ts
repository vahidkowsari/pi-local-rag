import { describe, it, expect } from "vitest";
import { mkdtempSync, writeFileSync, rmSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { chunkBlocks, estimateTokens } from "../chunking.ts";
import { extractBlocks } from "../parsing.ts";

const SAMPLE_PDF = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "fixtures", "sample.pdf"));

describe("chunkBlocks", () => {
  it("carries PDF page ranges and assigns unique chunkIndex values", () => {
    const chunks = chunkBlocks([
      { text: "alpha word ".repeat(120), section: "Intro", pageStart: 1, pageEnd: 1 },
      { text: "UNIQUE_PAGE_TWO_MARKER word ".repeat(120), section: "Method", pageStart: 2, pageEnd: 2 },
      { text: "omega word ".repeat(120), section: "End", pageStart: 3, pageEnd: 3 },
    ], { targetTokens: 80, maxTokens: 120, overlapTokens: 10 });
    expect(chunks.length).toBeGreaterThanOrEqual(2);
    const ids = new Set(chunks.map(c => c.chunkIndex));
    expect(ids.size).toBe(chunks.length);
    const hit = chunks.find(c => c.content.includes("UNIQUE_PAGE_TWO_MARKER"));
    expect(hit).toBeDefined();
    expect(hit!.pageStart).toBe(2);
    expect(hit!.pageEnd).toBe(2);
  });

  it("splits a single oversized paragraph and does not emit an overlap-only tail", () => {
    const text = Array.from({ length: 800 }, (_, i) => `token${i}`).join(" ");
    const chunks = chunkBlocks(
      [{ text, section: null, pageStart: null, pageEnd: null, lineStart: 1, lineEnd: 1 }],
      { targetTokens: 180, maxTokens: 240, overlapTokens: 30 },
    );
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) expect(estimateTokens(c.content)).toBeLessThanOrEqual(240);
    const last = chunks[chunks.length - 1];
    const prev = chunks[chunks.length - 2];
    expect(prev.content.endsWith(last.content)).toBe(false);
  });

  it("keeps CJK text under the max token budget using the CJK-aware estimate", () => {
    const text = "汉字".repeat(400);
    const chunks = chunkBlocks(
      [{ text, section: null, pageStart: null, pageEnd: null }],
      { targetTokens: 180, maxTokens: 240, overlapTokens: 20 },
    );
    for (const c of chunks) {
      expect(estimateTokens(c.content)).toBeLessThanOrEqual(240);
    }
  });

  it("keeps a short first paragraph plus a near-max second paragraph under maxTokens", () => {
    const text = "a".repeat(400) + "\n\n" + "b".repeat(960);
    const chunks = chunkBlocks(
      [{ text, section: null, pageStart: null, pageEnd: null, lineStart: 1, lineEnd: 3 }],
      { targetTokens: 180, maxTokens: 240, overlapTokens: 30 },
    );
    expect(chunks.length).toBeGreaterThanOrEqual(2);
    for (const c of chunks) expect(estimateTokens(c.content)).toBeLessThanOrEqual(240);
  });

  it("keeps identical text from distinct PDF pages as separate chunks", () => {
    const same = "Important repeated evidence";
    const chunks = chunkBlocks([
      { text: same, section: null, pageStart: 1, pageEnd: 1 },
      { text: same, section: null, pageStart: 2, pageEnd: 2 },
    ]);
    expect(chunks.length).toBe(2);
    expect(chunks[0].pageStart).toBe(1);
    expect(chunks[1].pageStart).toBe(2);
    expect(chunks[1].content).toContain("Important repeated evidence");
  });

  it("keeps a short suffix that is new content in a different section", () => {
    const chunks = chunkBlocks([
      { text: "alpha beta gamma delta epsilon", section: "A", pageStart: null, pageEnd: null, lineStart: 1, lineEnd: 1 },
      { text: "epsilon", section: "B", pageStart: null, pageEnd: null, lineStart: 3, lineEnd: 3 },
    ], { targetTokens: 8, maxTokens: 40, overlapTokens: 0 });
    expect(chunks.some(c => c.section === "B" && c.content.includes("epsilon"))).toBe(true);
  });

  it("does not invent PDF page numbers for markdown", () => {
    const chunks = chunkBlocks([
      { text: "# Title\n\nmarkdown body that is long enough to keep ".repeat(8), section: "Title", pageStart: null, pageEnd: null },
    ]);
    expect(chunks[0].pageStart).toBeNull();
  });
});

describe("extractBlocks markdown sections", () => {
  it("keeps short heading sections instead of dropping them", async () => {
    const dir = mkdtempSync(join(tmpdir(), "rag-md-"));
    const mdPath = join(dir, "note.md");
    writeFileSync(mdPath, "# Short\n42\n\n# Long\n" + "full evidence ".repeat(8) + "\n");
    try {
      const parsed = await extractBlocks(mdPath);
      expect(parsed.blocks.some(b => b.text.includes("42"))).toBe(true);
      expect(parsed.blocks.some(b => b.section === "Short")).toBe(true);
      expect(parsed.blocks.some(b => b.section === "Long")).toBe(true);
      const short = parsed.blocks.find(b => b.section === "Short")!;
      expect(short.lineStart).toBe(1);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("extractBlocks PDF pages", () => {
  it("records 1-based physical pages for a real PDF and never invents pages for markdown", async () => {
    const dir = mkdtempSync(join(tmpdir(), "rag-pdf-"));
    const pdfPath = join(dir, "sample.pdf");
    const mdPath = join(dir, "note.md");
    writeFileSync(pdfPath, SAMPLE_PDF);
    writeFileSync(mdPath, "# Heading\n\nMarkdown body without any PDF pages.\n");
    try {
      const pdf = await extractBlocks(pdfPath);
      expect(pdf.blocks.length).toBeGreaterThan(0);
      expect(pdf.blocks.every(b => b.pageStart === null || b.pageStart >= 1)).toBe(true);
      const md = await extractBlocks(mdPath);
      expect(md.blocks.every(b => b.pageStart === null)).toBe(true);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
