import { extname, basename } from "node:path";
import { readFileSync, readdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { extractText, getOcrTooling, isSparsePdfText, sha256 } from "./chunking.ts";

export interface SourceBlock {
  text: string;
  section: string | null;
  /** 1-based PDF physical page. null for non-PDF. */
  pageStart: number | null;
  pageEnd: number | null;
  /** 1-based original file lines. null when unknown (typical for PDF). */
  lineStart?: number | null;
  lineEnd?: number | null;
}

export interface ParsedDocument {
  blocks: SourceBlock[];
  hash: string;
  size: number;
}

const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

/**
 * Structured parse. PDF pages are recorded at extraction time (physical page,
 * 1-based). Markdown/text never invent PDF page numbers. Title/author/DOI are
 * not guessed from the filename.
 */
export async function extractBlocks(fp: string): Promise<ParsedDocument> {
  const ext = extname(fp).toLowerCase();
  if (ext === ".pdf") return parsePdf(fp);
  const { text, hash, size } = await extractText(fp);
  return { blocks: textToBlocks(text, ext), hash, size };
}

function textToBlocks(text: string, ext: string): SourceBlock[] {
  if (!text.trim()) return [];
  const converted = ext === ".html" || ext === ".htm" || ext === ".docx";
  if (converted) {
    return [{ text, section: null, pageStart: null, pageEnd: null, lineStart: null, lineEnd: null }];
  }
  const lines = text.split("\n");
  if (ext === ".md" || ext === ".mdx") {
    const starts: { line: number; section: string | null }[] = [];
    if (!/^#{1,6} /.test(lines[0] ?? "")) starts.push({ line: 1, section: null });
    for (let i = 0; i < lines.length; i++) {
      const m = lines[i].match(/^(#{1,6}) (.*)$/);
      if (m) starts.push({ line: i + 1, section: m[2].trim() });
    }
    const blocks: SourceBlock[] = [];
    for (let i = 0; i < starts.length; i++) {
      const from = starts[i].line;
      const to = i + 1 < starts.length ? starts[i + 1].line - 1 : lines.length;
      const raw = lines.slice(from - 1, to).join("\n");
      if (!raw.trim()) continue;
      blocks.push({
        text: raw,
        section: starts[i].section,
        pageStart: null,
        pageEnd: null,
        lineStart: from,
        lineEnd: to,
      });
    }
    if (blocks.length) return blocks;
  }
  return [{
    text, section: null, pageStart: null, pageEnd: null,
    lineStart: 1, lineEnd: Math.max(1, lines.length),
  }];
}

async function parsePdf(fp: string): Promise<ParsedDocument> {
  const buf = readFileSync(fp);
  const { default: pdf } = await import("pdf-parse/lib/pdf-parse.js");
  const pages: SourceBlock[] = [];
  let pageNo = 0;
  let data: { text: string; numpages?: number };
  try {
    data = await pdf(buf, {
      pagerender: async (pageData: { getTextContent: () => Promise<{ items: Array<{ str?: string }> }> }) => {
        pageNo += 1;
        const tc = await pageData.getTextContent();
        const text = tc.items.map(it => it.str ?? "").join(" ").replace(/\s+/g, " ").trim();
        pages.push({ text, section: null, pageStart: pageNo, pageEnd: pageNo });
        return text;
      },
    });
  } catch {
    data = await pdf(buf);
  }
  let blocks = pages.filter(p => p.text.length > 0);
  if (blocks.length === 0 && data.text?.trim()) {
    blocks = [{ text: data.text, section: null, pageStart: 1, pageEnd: data.numpages ?? 1 }];
  }
  if (isSparsePdfText(blocks.map(b => b.text).join("\n"), data.numpages ?? 1)) {
    const tools = getOcrTooling();
    if (tools.available) {
      const ocrPages = await ocrPdfPages(buf, tools.langs, basename(fp));
      if (ocrPages.length) blocks = ocrPages;
    }
  }
  return { blocks, hash: sha256(buf.toString("binary")), size: buf.length };
}

async function ocrPdfPages(buf: Buffer, langs: string, label: string): Promise<SourceBlock[]> {
  const dir = mkdtempSync(join(tmpdir(), "rag-ocr-"));
  try {
    const pdfPath = join(dir, "in.pdf");
    writeFileSync(pdfPath, buf);
    const render = spawnSync("pdftoppm", ["-png", "-r", "200", pdfPath, join(dir, "p")], { encoding: "utf-8" });
    if (render.status !== 0) return [];
    const files = readdirSync(dir).filter(f => f.startsWith("p-") && f.endsWith(".png")).sort();
    const blocks: SourceBlock[] = [];
    for (let i = 0; i < files.length; i++) {
      process.stderr.write(`\r\x1b[2K[OCR ${i + 1}/${files.length}] ${label}`);
      await yield_();
      const r = spawnSync("tesseract", [join(dir, files[i]), "-", "-l", langs], {
        encoding: "utf-8", timeout: 60_000, maxBuffer: 16 * 1024 * 1024,
      });
      const text = r.status === 0 ? (r.stdout ?? "") : "";
      blocks.push({ text, section: null, pageStart: i + 1, pageEnd: i + 1 });
    }
    process.stderr.write(`\r\x1b[2K`);
    return blocks;
  } finally {
    try { rmSync(dir, { recursive: true, force: true }); } catch {}
  }
}
