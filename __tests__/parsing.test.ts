import { describe, it, expect } from "vitest";
import { mkdtempSync, writeFileSync, rmSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { chunkBlocks } from "../chunking.ts";
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

  it("does not invent PDF page numbers for markdown", () => {
    const chunks = chunkBlocks([
      { text: "# Title\n\nmarkdown body that is long enough to keep ".repeat(8), section: "Title", pageStart: null, pageEnd: null },
    ]);
    expect(chunks[0].pageStart).toBeNull();
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
