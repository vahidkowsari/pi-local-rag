import { readFileSync, readdirSync, statSync, mkdtempSync, rmSync, writeFileSync, promises as fsPromises } from "node:fs";
import { extname, basename, join, relative } from "node:path";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import ignore from "ignore";
import {
  BINARY_DOC_EXTS, TEXT_MAX_BYTES, BINARY_DOC_MAX_BYTES, SKIP_DIRS,
  CHUNK_TARGET_TOKENS, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS,
} from "./constants.ts";
import { loadConfig, resolveExtensions, type RagConfig } from "./config.ts";
import type { SourceBlock } from "./parsing.ts";

const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

function stderrProgress(msg: string) { process.stderr.write(`\r\x1b[2K${msg}`); }

export function sha256(data: string): string {
  return createHash("sha256").update(data).digest("hex").slice(0, 12);
}

export interface TextChunk {
  content: string;
  lineStart: number;
  lineEnd: number;
  pageStart?: number | null;
  pageEnd?: number | null;
  section?: string | null;
  chunkIndex?: number;
}

export function chunkText(text: string, maxLines = 50): TextChunk[] {
  const lines = text.split("\n");
  const chunks: { content: string; lineStart: number; lineEnd: number }[] = [];
  let i = 0;
  while (i < lines.length) {
    let end = Math.min(i + maxLines, lines.length);
    for (let j = end - 1; j > i + 10 && j > end - 15; j--) {
      if (lines[j]?.trim() === "") { end = j + 1; break; }
    }
    const chunk = lines.slice(i, end).join("\n");
    if (chunk.trim().length > 20) {
      chunks.push({ content: chunk, lineStart: i + 1, lineEnd: end });
    }
    i = end;
  }
  return chunks;
}

function isCjk(ch: string): boolean {
  const c = ch.codePointAt(0) ?? 0;
  return (
    (c >= 0x3400 && c <= 0x9fff) ||
    (c >= 0xf900 && c <= 0xfaff) ||
    (c >= 0x3040 && c <= 0x30ff) ||
    (c >= 0xac00 && c <= 0xd7af)
  );
}

/** Conservative estimate: CJK ≈ 1 token/char, other ≈ 4 chars/token. Not a model tokenizer. */
export function estimateTokens(text: string): number {
  let cjk = 0;
  let other = 0;
  for (const ch of text) {
    if (isCjk(ch)) cjk++;
    else other++;
  }
  return Math.max(1, cjk + Math.ceil(other / 4));
}

function hardSplitByTokens(text: string, maxTokens: number): string[] {
  const out: string[] = [];
  let start = 0;
  while (start < text.length) {
    let lo = 1;
    let hi = Math.min(text.length - start, Math.max(8, maxTokens * 4));
    let best = 1;
    while (lo <= hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (estimateTokens(text.slice(start, start + mid)) <= maxTokens) {
        best = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    out.push(text.slice(start, start + best));
    start += best;
  }
  return out.length ? out : [text];
}

function splitOversized(text: string, maxTokens: number): string[] {
  if (estimateTokens(text) <= maxTokens) return [text];
  const sentences = text.split(/(?<=[。！？.!?])\s+|\n+/).filter(s => s.length > 0);
  const out: string[] = [];
  let buf = "";
  const pushBuf = () => {
    if (!buf) return;
    if (estimateTokens(buf) <= maxTokens) out.push(buf);
    else out.push(...hardSplitByTokens(buf, maxTokens));
    buf = "";
  };
  for (const s of sentences) {
    if (estimateTokens(s) > maxTokens) {
      pushBuf();
      out.push(...hardSplitByTokens(s, maxTokens));
      continue;
    }
    if (!buf) { buf = s; continue; }
    const joined = buf + " " + s;
    if (estimateTokens(joined) > maxTokens) {
      pushBuf();
      buf = s;
    } else {
      buf = joined;
    }
  }
  pushBuf();
  return out.length ? out : hardSplitByTokens(text, maxTokens);
}

function paragraphSpans(block: SourceBlock): { text: string; lineStart: number | null; lineEnd: number | null }[] {
  const original = block.text;
  const known = block.lineStart != null && block.lineStart > 0;
  const spans: { text: string; lineStart: number | null; lineEnd: number | null }[] = [];
  const parts = original.split(/\n{2,}/);
  let cursor = 0;
  for (const part of parts) {
    const start = original.indexOf(part, cursor);
    const end = start + part.length;
    const piece = part.trim();
    if (piece) {
      if (known) {
        const before = original.slice(0, start);
        const lineStart = (block.lineStart as number) + (before.split("\n").length - 1);
        const lineEnd = lineStart + piece.split("\n").length - 1;
        spans.push({ text: piece, lineStart, lineEnd });
      } else {
        spans.push({ text: piece, lineStart: null, lineEnd: null });
      }
    }
    cursor = end;
  }
  return spans;
}

/**
 * Token-aware chunker over structured blocks. PDF page range is the union of
 * the source blocks that contributed text. IDs are assigned by callers using
 * document + chunkIndex.
 */
export function chunkBlocks(
  blocks: SourceBlock[],
  opts: { targetTokens?: number; maxTokens?: number; overlapTokens?: number } = {},
): TextChunk[] {
  const target = opts.targetTokens ?? CHUNK_TARGET_TOKENS;
  const max = opts.maxTokens ?? CHUNK_MAX_TOKENS;
  const overlap = opts.overlapTokens ?? CHUNK_OVERLAP_TOKENS;
  const chunks: TextChunk[] = [];
  let buf = "";
  let addedSinceFlush = "";
  let lineStart = 0;
  let lineEnd = 0;
  let pageStart: number | null = null;
  let pageEnd: number | null = null;
  let section: string | null = null;

  const pushChunk = (content: string) => {
    chunks.push({
      content,
      lineStart: lineStart > 0 ? lineStart : 0,
      lineEnd: lineEnd > 0 ? lineEnd : 0,
      pageStart,
      pageEnd,
      section,
      chunkIndex: chunks.length,
    });
  };

  const emit = (content: string) => {
    const trimmed = content.trimEnd();
    if (!trimmed) return;
    if (estimateTokens(trimmed) > max) {
      for (const part of hardSplitByTokens(trimmed, max)) pushChunk(part);
      return;
    }
    pushChunk(trimmed);
  };

  const resetBuf = () => {
    buf = "";
    addedSinceFlush = "";
    lineStart = 0;
    lineEnd = 0;
    pageStart = null;
    pageEnd = null;
  };

  const flush = (force = false) => {
    if (!buf.trim()) return;
    if (!force && estimateTokens(buf) < target) return;
    if (addedSinceFlush.trim()) emit(buf);
    if (overlap > 0 && addedSinceFlush.trim()) {
      const keep = buf.slice(-Math.max(1, overlap * 2));
      buf = keep;
      addedSinceFlush = "";
      if (lineEnd > 0) lineStart = Math.max(1, lineEnd - keep.split("\n").length + 1);
    } else {
      resetBuf();
    }
  };

  const joinPiece = (base: string, piece: string) => (base ? base + "\n\n" + piece : piece);

  const appendPiece = (
    piece: string,
    src: { lineStart: number | null; lineEnd: number | null; pageStart: number | null; pageEnd: number | null; section: string | null },
  ) => {
    if (buf && estimateTokens(joinPiece(buf, piece)) > max) {
      flush(true);
      while (buf && estimateTokens(joinPiece(buf, piece)) > max) {
        if (buf.length <= 1) { resetBuf(); break; }
        buf = buf.slice(Math.ceil(buf.length / 2));
      }
    }
    if (!buf) {
      lineStart = src.lineStart ?? 0;
      pageStart = src.pageStart;
      section = src.section;
    }
    buf = joinPiece(buf, piece);
    addedSinceFlush = addedSinceFlush ? addedSinceFlush + "\n\n" + piece : piece;
    if (src.lineEnd != null && src.lineEnd > 0) lineEnd = src.lineEnd;
    if (src.pageStart != null) pageStart = pageStart ?? src.pageStart;
    if (src.pageEnd != null) pageEnd = src.pageEnd;
    if (estimateTokens(buf) > max) {
      emit(buf);
      resetBuf();
      return;
    }
    if (estimateTokens(buf) >= target) flush(true);
  };

  for (const block of blocks) {
    if (!block.text.trim()) continue;
    if (buf && block.pageStart != null && pageEnd != null && block.pageStart > pageEnd) {
      flush(true);
      resetBuf();
    }
    for (const span of paragraphSpans(block)) {
      const pieces = splitOversized(span.text, max);
      for (const piece of pieces) {
        appendPiece(piece, {
          lineStart: span.lineStart,
          lineEnd: span.lineEnd,
          pageStart: block.pageStart,
          pageEnd: block.pageEnd,
          section: block.section,
        });
      }
    }
  }
  flush(true);
  return chunks.map((c, i) => ({ ...c, chunkIndex: i }));
}

export function collectFiles(
  dirPath: string,
  exts?: Set<string>,
  excludePatterns: string[] = [],
  errors?: string[],
): string[] {
  const allowed = exts ?? resolveExtensions(loadConfig());
  const ig = excludePatterns.length ? ignore().add(excludePatterns) : null;
  const files: string[] = [];
  const root = dirPath;

  function acceptable(fp: string, size: number): boolean {
    const ext = extname(fp).toLowerCase();
    if (allowed.has(ext)) return size < TEXT_MAX_BYTES;
    if (BINARY_DOC_EXTS.has(ext)) return size < BINARY_DOC_MAX_BYTES;
    return false;
  }

  function isExcluded(absPath: string): boolean {
    if (!ig) return false;
    const rel = relative(root, absPath);
    if (!rel || rel.startsWith("..")) return false;
    return ig.ignores(rel);
  }

  try {
    const stat = statSync(dirPath);
    if (stat.isFile()) {
      if (!acceptable(dirPath, stat.size)) return [];
      if (ig && ig.ignores(basename(dirPath))) return [];
      return [dirPath];
    }
  } catch (err) {
    errors?.push(formatPathError(dirPath, err));
    return [];
  }

  function walk(dir: string) {
    let entries: import("node:fs").Dirent[];
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch (err) {
      errors?.push(formatPathError(dir, err));
      return;
    }
    for (const entry of entries) {
      const fp = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
        if (isExcluded(fp)) continue;
        walk(fp);
      } else {
        const ext = extname(entry.name).toLowerCase();
        if (!allowed.has(ext) && !BINARY_DOC_EXTS.has(ext)) continue;
        if (isExcluded(fp)) continue;
        try {
          if (acceptable(fp, statSync(fp).size)) files.push(fp);
        } catch (err) {
          errors?.push(formatPathError(fp, err));
        }
      }
    }
  }
  walk(root);
  return files;
}

function formatPathError(path: string, err: unknown): string {
  const code = (err as { code?: string }).code;
  const msg = err instanceof Error ? err.message : String(err);
  if (code === "ENOENT") return `${path}: path does not exist`;
  if (code === "EACCES" || code === "EPERM") return `${path}: not readable`;
  return `${path}: ${msg}`;
}

export interface TrackedScan {
  files: string[];
  errors: string[];
  unavailableRoots: string[];
}



export function collectFromTrackedDetailed(cfg: Pick<RagConfig, "trackedPaths" | "excludePatterns">): TrackedScan {
  const files = new Set<string>();
  const errors: string[] = [];
  const unavailableRoots: string[] = [];
  for (const p of cfg.trackedPaths) {
    const walkErrors: string[] = [];
    const found = collectFiles(p, undefined, cfg.excludePatterns, walkErrors);
    if (walkErrors.length) {
      unavailableRoots.push(p);
      errors.push(...walkErrors);
    }
    for (const f of found) files.add(f);
  }
  return { files: [...files], errors, unavailableRoots };
}

export function collectFromTracked(cfg: Pick<RagConfig, "trackedPaths" | "excludePatterns">): string[] {
  return collectFromTrackedDetailed(cfg).files;
}

/**
 * Async variant of collectFiles that uses fs.promises and yields to the event
 * loop between directories. Required for /rag rebuild on large trackedPaths
 * (45k+ files) — the synchronous walk pegs the event loop long enough that
 * the TUI freezes before reaching the embed phase. Adapted from
 * theli-ua/pi-local-rag@8432a15.
 */
export async function collectFilesAsync(
  dirPath: string,
  exts?: Set<string>,
  excludePatterns: string[] = [],
  errors?: string[],
): Promise<string[]> {
  const allowed = exts ?? resolveExtensions(loadConfig());
  const ig = excludePatterns.length ? ignore().add(excludePatterns) : null;
  const files: string[] = [];
  const root = dirPath;

  function acceptable(fp: string, size: number): boolean {
    const ext = extname(fp).toLowerCase();
    if (allowed.has(ext)) return size < TEXT_MAX_BYTES;
    if (BINARY_DOC_EXTS.has(ext)) return size < BINARY_DOC_MAX_BYTES;
    return false;
  }

  function isExcluded(absPath: string): boolean {
    if (!ig) return false;
    const rel = relative(root, absPath);
    if (!rel || rel.startsWith("..")) return false;
    return ig.ignores(rel);
  }

  try {
    const st = await fsPromises.stat(dirPath);
    if (st.isFile()) {
      if (!acceptable(dirPath, st.size)) return [];
      if (ig && ig.ignores(basename(dirPath))) return [];
      return [dirPath];
    }
  } catch (err) {
    errors?.push(formatPathError(dirPath, err));
    return [];
  }

  async function walk(dir: string): Promise<void> {
    let entries: import("node:fs").Dirent[];
    try {
      entries = await fsPromises.readdir(dir, { withFileTypes: true });
    } catch (err) {
      errors?.push(formatPathError(dir, err));
      return;
    }
    for (const entry of entries) {
      const fp = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
        if (isExcluded(fp)) continue;
        await walk(fp);
      } else {
        const ext = extname(entry.name).toLowerCase();
        if (!allowed.has(ext) && !BINARY_DOC_EXTS.has(ext)) continue;
        if (isExcluded(fp)) continue;
        try {
          const st = await fsPromises.stat(fp);
          if (acceptable(fp, st.size)) files.push(fp);
        } catch (err) {
          errors?.push(formatPathError(fp, err));
        }
      }
    }
    // Yield between directories so the event loop can process UI updates.
    await yield_();
  }

  await walk(root);
  return files;
}

export async function collectFromTrackedDetailedAsync(
  cfg: Pick<RagConfig, "trackedPaths" | "excludePatterns">,
): Promise<TrackedScan> {
  const files = new Set<string>();
  const errors: string[] = [];
  const unavailableRoots: string[] = [];
  for (const p of cfg.trackedPaths) {
    const walkErrors: string[] = [];
    const found = await collectFilesAsync(p, undefined, cfg.excludePatterns, walkErrors);
    if (walkErrors.length) {
      unavailableRoots.push(p);
      errors.push(...walkErrors);
    }
    for (const f of found) files.add(f);
  }
  return { files: [...files], errors, unavailableRoots };
}

export async function collectFromTrackedAsync(cfg: Pick<RagConfig, "trackedPaths" | "excludePatterns">): Promise<string[]> {
  return (await collectFromTrackedDetailedAsync(cfg)).files;
}

/** Returns true if `file` is matched by `excludePatterns` relative to any of `roots`. */
export function isExcludedByConfig(file: string, roots: string[], excludePatterns: string[]): boolean {
  if (!excludePatterns.length) return false;
  const ig = ignore().add(excludePatterns);
  for (const root of roots) {
    const rel = relative(root, file);
    if (!rel || rel.startsWith("..")) continue;
    if (ig.ignores(rel)) return true;
  }
  return false;
}

// pdfjs (bundled inside pdf-parse) routes warnings through console.log with a
// "Warning: " prefix. On real-world PDFs this fires thousands of times per
// document ("Ran out of space in font private use area", missing glyphs, …).
// The font warnings come from pdf.worker.js, which is a separate webpack
// bundle whose verbosity is not externally configurable (its setVerbosityLevel
// export exists only as a placeholder at the outer module level). Filtering
// console.log for the known pdfjs prefixes is the only reliable approach.
const PDFJS_LOG_PREFIX = /^(Warning|Info|Deprecated API usage):/;
async function withPdfjsSilenced<T>(fn: () => Promise<T>): Promise<T> {
  const origLog = console.log;
  console.log = (...args: unknown[]) => {
    const first = args[0];
    if (typeof first === "string" && PDFJS_LOG_PREFIX.test(first)) return;
    origLog(...args);
  };
  try {
    return await fn();
  } finally {
    console.log = origLog;
  }
}

// ─── OCR fallback for image-based PDFs ───────────────────────────────────────

type OcrTooling = { available: false } | { available: true; langs: string };
let _ocrTooling: OcrTooling | undefined;
let _ocrUnavailableLogged = false;

/** One-shot probe for system pdftoppm + tesseract. Caches the result. */
export function getOcrTooling(): OcrTooling {
  if (_ocrTooling) return _ocrTooling;
  const pdftoppm = spawnSync("pdftoppm", ["-v"]);
  const tess = spawnSync("tesseract", ["--list-langs"], { encoding: "utf-8" });
  if (pdftoppm.error || tess.error) return (_ocrTooling = { available: false });
  // tesseract prints langs on stderr in some builds, stdout in others.
  const out = `${tess.stdout || ""}\n${tess.stderr || ""}`;
  const have = new Set(out.split(/\r?\n/).map(s => s.trim()).filter(Boolean));
  const wanted = ["jpn", "eng"].filter(l => have.has(l));
  if (!wanted.length) return (_ocrTooling = { available: false });
  return (_ocrTooling = { available: true, langs: wanted.join("+") });
}

/** Render `buf` to PNGs via pdftoppm, OCR each page via tesseract, return concatenated text. */
async function ocrPdf(buf: Buffer, langs: string, label: string): Promise<string> {
  const MAX_PAGES = 200;
  const PER_PAGE_TIMEOUT_MS = 60_000;
  const dir = mkdtempSync(join(tmpdir(), "rag-ocr-"));
  try {
    const pdfPath = join(dir, "in.pdf");
    writeFileSync(pdfPath, buf);
    const render = spawnSync("pdftoppm", ["-png", "-r", "200", pdfPath, join(dir, "p")], { encoding: "utf-8" });
    if (render.status !== 0) return "";
    const pages = readdirSync(dir).filter(f => f.startsWith("p-") && f.endsWith(".png")).sort();
    const total = Math.min(pages.length, MAX_PAGES);
    if (pages.length > MAX_PAGES) {
      process.stderr.write(`\r\x1b[2K[rag] OCR ${label}: ${pages.length} pages, capping at ${MAX_PAGES}\n`);
    }
    const out: string[] = [];
    for (let i = 0; i < total; i++) {
      stderrProgress(`[OCR ${i + 1}/${total}] ${label}`);
      await yield_();
      const r = spawnSync("tesseract", [join(dir, pages[i]), "-", "-l", langs], {
        encoding: "utf-8",
        timeout: PER_PAGE_TIMEOUT_MS,
        maxBuffer: 16 * 1024 * 1024,
      });
      out.push(r.status === 0 ? (r.stdout ?? "") : "");
    }
    process.stderr.write(`\r\x1b[2K`);
    return out.join("\n\n");
  } finally {
    try { rmSync(dir, { recursive: true, force: true }); } catch {}
  }
}

/** True if `text` looks too sparse for `numpages` to be the real content of the document. */
export function isSparsePdfText(text: string, numpages: number): boolean {
  return text.trim().length < 50 * Math.max(1, numpages);
}

/**
 * Read and decode a file into UTF-8 text. PDF and DOCX are routed through
 * extraction libraries; everything else is read as plain UTF-8. Hash is
 * computed over the raw bytes for binaries (so the source file's identity
 * drives skip-on-rebuild) and over the decoded text for plain text files.
 */
export async function extractText(fp: string): Promise<{ text: string; hash: string; size: number }> {
  const ext = extname(fp).toLowerCase();
  if (ext === ".pdf") {
    const buf = readFileSync(fp);
    const { default: pdf } = await import("pdf-parse/lib/pdf-parse.js");
    const data = await withPdfjsSilenced(() => pdf(buf));
    let text = data.text;
    if (isSparsePdfText(text, data.numpages ?? 1)) {
      const tools = getOcrTooling();
      if (tools.available) {
        const ocr = await ocrPdf(buf, tools.langs, basename(fp));
        if (ocr.trim().length > text.trim().length) text = ocr;
      } else if (!_ocrUnavailableLogged) {
        _ocrUnavailableLogged = true;
        process.stderr.write(
          `\r\x1b[2K[rag] OCR unavailable: install pdftoppm + tesseract (with jpn/eng traineddata) to index image PDFs\n`
        );
      }
    }
    return { text, hash: sha256(buf.toString("binary")), size: buf.length };
  }
  if (ext === ".docx") {
    const buf = readFileSync(fp);
    const { default: mammoth } = await import("mammoth");
    const { value } = await mammoth.extractRawText({ buffer: buf });
    return { text: value, hash: sha256(buf.toString("binary")), size: buf.length };
  }
  if (ext === ".html" || ext === ".htm") {
    const { default: TurndownService } = await import("turndown");
    const raw = readFileSync(fp, "utf-8");
    const td = new TurndownService({
      headingStyle: "atx",
      codeBlockStyle: "fenced",
      blankReplacement: (_content, node) => node.tagName === "BR" ? "\n" : "",
    });
    td.remove(["script", "style"]);
    td.remove(["nav", "footer"]);
    const text = td.turndown(raw);
    return { text, hash: sha256(raw), size: raw.length };
  }
  const text = readFileSync(fp, "utf-8");
  return { text, hash: sha256(text), size: text.length };
}
