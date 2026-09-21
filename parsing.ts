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
  if (ext === ".md" || ext === ".mdx") {
    const parts = text.split(/^(?=#{1,6} )/m).filter(p => p.trim().length > 20);
    if (parts.length) {
      return parts.map(p => {
        const m = p.match(/^(#{1,6}) (.*)$/m);
        return { text: p, section: m ? m[2].trim() : null, pageStart: null, pageEnd: null };
      });
    }
  }
  return [{ text, section: null, pageStart: null, pageEnd: null }];
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
