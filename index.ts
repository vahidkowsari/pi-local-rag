/**
 * pi-local-rag — Local RAG Pipeline
 * 
 * Index local files → chunk → store → retrieve. AI consults YOUR knowledge before hallucinating.
 * Zero cloud dependency. Embeddings via Ollama (local) or keyword fallback.
 * 
 * /lens index <path>     → index a file or directory
 * /lens search <query>   → search indexed content
 * /lens status           → show index stats
 * /lens rebuild          → rebuild entire index
 * /lens clear            → clear index
 * /lens context <query>  → generate context.md snippet for injection
 * 
 * Tools: lens_index, lens_query, lens_status
 */
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { join, extname, basename } from "node:path";
import { homedir } from "node:os";
import { createHash } from "node:crypto";

const RAG_DIR = join(homedir(), ".pi", "lens");
const INDEX_FILE = join(RAG_DIR, "index.json");
const RST = "\x1b[0m", B = "\x1b[1m", D = "\x1b[2m";
const GREEN = "\x1b[32m", YELLOW = "\x1b[33m", CYAN = "\x1b[36m", RED = "\x1b[31m";

const TEXT_EXTS = new Set([
  ".md", ".txt", ".ts", ".js", ".py", ".rs", ".go", ".java", ".c", ".cpp", ".h",
  ".css", ".html", ".json", ".yaml", ".yml", ".toml", ".xml", ".csv", ".sh",
  ".sql", ".graphql", ".proto", ".env", ".gitignore", ".dockerfile",
]);

const SKIP_DIRS = new Set(["node_modules", ".git", ".next", "dist", "build", "__pycache__", ".venv", "venv", ".cache"]);

interface Chunk {
  id: string;
  file: string;
  content: string;
  lineStart: number;
  lineEnd: number;
  hash: string;
  indexed: string;
  tokens: number;
}

interface IndexMeta {
  chunks: Chunk[];
  files: Record<string, { hash: string; chunks: number; indexed: string; size: number }>;
  lastBuild: string;
}

function ensureDir() {
  if (!existsSync(RAG_DIR)) mkdirSync(RAG_DIR, { recursive: true });
}

function loadIndex(): IndexMeta {
  ensureDir();
  if (!existsSync(INDEX_FILE)) return { chunks: [], files: {}, lastBuild: "" };
  try {
    const data = JSON.parse(readFileSync(INDEX_FILE, "utf-8"));
    return {
      chunks: Array.isArray(data.chunks) ? data.chunks : [],
      files: data.files && typeof data.files === "object" ? data.files : {},
      lastBuild: data.lastBuild ?? "",
    };
  } catch { return { chunks: [], files: {}, lastBuild: "" }; }
}

function saveIndex(index: IndexMeta) {
  ensureDir();
  writeFileSync(INDEX_FILE, JSON.stringify(index, null, 2));
}

function sha256(data: string): string {
  return createHash("sha256").update(data).digest("hex").slice(0, 12);
}

function chunkText(text: string, maxLines = 50): { content: string; lineStart: number; lineEnd: number }[] {
  const lines = text.split("\n");
  const chunks: { content: string; lineStart: number; lineEnd: number }[] = [];

  let i = 0;
  while (i < lines.length) {
    // Try to break at a natural blank-line boundary near the end of the window
    let end = Math.min(i + maxLines, lines.length);
    for (let j = end - 1; j > i + 10 && j > end - 15; j--) {
      if (lines[j]?.trim() === "") { end = j + 1; break; }
    }
    const chunk = lines.slice(i, end).join("\n");
    if (chunk.trim().length > 20) {
      chunks.push({ content: chunk, lineStart: i + 1, lineEnd: end });
    }
    i = end; // advance past this chunk; no off-by-one with += maxLines
  }
  return chunks;
}

function collectFiles(dirPath: string, maxFiles = 500): string[] {
  const files: string[] = [];
  function walk(dir: string) {
    if (files.length >= maxFiles) return;
    try {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (files.length >= maxFiles) return;
        if (entry.isDirectory()) {
          if (!SKIP_DIRS.has(entry.name) && !entry.name.startsWith(".")) {
            walk(join(dir, entry.name));
          }
        } else if (TEXT_EXTS.has(extname(entry.name).toLowerCase())) {
          const fp = join(dir, entry.name);
          try {
            const stat = statSync(fp);
            if (stat.size < 500_000) files.push(fp); // Skip files > 500KB
          } catch {}
        }
      }
    } catch {}
  }

  try {
    const stat = statSync(dirPath);
    // Single file: apply the same extension + size guards as the directory walker
    if (stat.isFile()) {
      if (!TEXT_EXTS.has(extname(dirPath).toLowerCase())) return [];
      if (stat.size >= 500_000) return [];
      return [dirPath];
    }
  } catch { return []; }
  walk(dirPath);
  return files;
}

function indexFiles(paths: string[]): { indexed: number; chunks: number; skipped: number } {
  const index = loadIndex();
  let indexed = 0, chunked = 0, skipped = 0;
  
  for (const fp of paths) {
    try {
      const content = readFileSync(fp, "utf-8");
      const hash = sha256(content);
      
      // Skip if unchanged
      if (index.files[fp]?.hash === hash) { skipped++; continue; }
      
      // Remove old chunks for this file
      index.chunks = index.chunks.filter(c => c.file !== fp);
      
      // Chunk and add
      const chunks = chunkText(content);
      for (const chunk of chunks) {
        index.chunks.push({
          id: `${sha256(fp)}-${chunk.lineStart}`,
          file: fp,
          content: chunk.content,
          lineStart: chunk.lineStart,
          lineEnd: chunk.lineEnd,
          hash: sha256(chunk.content),
          indexed: new Date().toISOString(),
          tokens: Math.ceil(chunk.content.length / 4),
        });
        chunked++;
      }
      
      index.files[fp] = { hash, chunks: chunks.length, indexed: new Date().toISOString(), size: content.length };
      indexed++;
    } catch { skipped++; }
  }
  
  index.lastBuild = new Date().toISOString();
  saveIndex(index);
  return { indexed, chunks: chunked, skipped };
}

// BM25-style keyword search (no embeddings needed)
function searchChunks(query: string, index: IndexMeta, limit = 10): Chunk[] {
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);
  if (!terms.length) return [];

  // Pre-compute IDF per term once (avoids O(n²) re-scan inside the map)
  const idfMap = new Map<string, number>();
  for (const term of terms) {
    const docsWithTerm = index.chunks.filter(c => c.content.toLowerCase().includes(term)).length;
    idfMap.set(term, Math.log(1 + index.chunks.length / (1 + docsWithTerm)));
  }
  const queryLower = query.toLowerCase();

  const scored = index.chunks.map(chunk => {
    const lower = chunk.content.toLowerCase();
    let score = 0;
    for (const term of terms) {
      const count = (lower.match(new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), "g")) || []).length;
      if (count > 0) {
        const tf = Math.log(1 + count);
        score += tf * idfMap.get(term)!;
      }
    }
    // Boost for exact phrase match
    if (lower.includes(queryLower)) score *= 2;
    // Boost for filename match
    if (chunk.file.toLowerCase().includes(terms[0])) score *= 1.5;

    return { chunk, score };
  });

  return scored
    .filter(s => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(s => s.chunk);
}

export default function (pi: ExtensionAPI) {
  ensureDir();

  pi.registerCommand("lens", {
    description: "pi-local-rag pipeline: /lens index|search|status|rebuild|clear|context",
    handler: async (args, ctx) => {
      const parts = (args || "").trim().split(/\s+/);
      const cmd = parts[0] || "status";

      if (cmd === "index") {
        const path = parts[1] || ".";
        if (!existsSync(path)) return `${RED}Path not found:${RST} ${path}`;
        const files = collectFiles(path);
        if (!files.length) return `${YELLOW}No indexable files found in:${RST} ${path}`;
        const result = indexFiles(files);
        return `${GREEN}✅ Indexed:${RST} ${result.indexed} files, ${result.chunks} chunks (${result.skipped} skipped/unchanged)`;
      }

      if (cmd === "search") {
        const query = parts.slice(1).join(" ");
        if (!query) return `${YELLOW}Usage:${RST} /lens search <query>`;
        const index = loadIndex();
        const results = searchChunks(query, index);
        if (!results.length) return `${YELLOW}No results for:${RST} ${query}`;
        let out = `${B}${CYAN}🔍 ${results.length} results for "${query}"${RST}\n\n`;
        for (const r of results) {
          out += `${GREEN}${basename(r.file)}${RST}:${r.lineStart}-${r.lineEnd} ${D}(${r.tokens} tokens)${RST}\n`;
          const preview = r.content.split("\n").slice(0, 3).join("\n");
          out += `${D}${preview.slice(0, 200)}${RST}\n\n`;
        }
        return out;
      }

      if (cmd === "context") {
        const query = parts.slice(1).join(" ");
        if (!query) return `${YELLOW}Usage:${RST} /lens context <query>`;
        const index = loadIndex();
        const results = searchChunks(query, index, 5);
        if (!results.length) return `${YELLOW}No relevant context found for:${RST} ${query}`;
        let context = `# Relevant Context for: ${query}\n\n`;
        for (const r of results) {
          context += `## ${basename(r.file)} (lines ${r.lineStart}-${r.lineEnd})\n\`\`\`\n${r.content.slice(0, 500)}\n\`\`\`\n\n`;
        }
        return context;
      }

      if (cmd === "rebuild") {
        const index = loadIndex();
        const allFiles = Object.keys(index.files);
        if (!allFiles.length) return `${YELLOW}No files in index. Run /lens index <path> first.${RST}`;
        // Prune deleted files without clearing hashes of surviving files
        const existingFiles = allFiles.filter(f => existsSync(f));
        const deletedFiles = allFiles.filter(f => !existsSync(f));
        for (const f of deletedFiles) {
          index.chunks = index.chunks.filter(c => c.file !== f);
          delete index.files[f];
        }
        saveIndex(index); // hashes intact so unchanged files will be skipped
        const result = indexFiles(existingFiles);
        return `${GREEN}✅ Rebuilt:${RST} pruned ${deletedFiles.length} deleted, re-indexed ${result.indexed} changed, ${result.skipped} unchanged (${result.chunks} new chunks)`;
      }

      if (cmd === "clear") {
        saveIndex({ chunks: [], files: {}, lastBuild: "" });
        return `${GREEN}✅ Index cleared.${RST}`;
      }

      // Default: status
      const index = loadIndex();
      const fileCount = Object.keys(index.files).length;
      const totalTokens = index.chunks.reduce((sum, c) => sum + c.tokens, 0);
      let out = `${B}${CYAN}🔍 pi-local-rag Index Status${RST}\n\n`;
      out += `  Files indexed: ${GREEN}${fileCount}${RST}\n`;
      out += `  Chunks: ${GREEN}${index.chunks.length}${RST}\n`;
      out += `  Total tokens: ${GREEN}${totalTokens.toLocaleString()}${RST}\n`;
      out += `  Last build: ${index.lastBuild || "never"}\n`;
      out += `  Storage: ${D}${RAG_DIR}${RST}\n`;
      if (fileCount) {
        out += `\n  ${B}Top file types:${RST}\n`;
        const byExt: Record<string, number> = {};
        for (const f of Object.keys(index.files)) byExt[extname(f)] = (byExt[extname(f)] || 0) + 1;
        for (const [ext, count] of Object.entries(byExt).sort((a, b) => b[1] - a[1]).slice(0, 8)) {
          out += `    ${ext}: ${count}\n`;
        }
      }
      return out;
    }
  });

  pi.registerTool({
    name: "lens_index",
    description: "Index a file or directory into the local pi-local-rag pipeline. Chunks text files, stores for BM25 keyword search.",
    parameters: Type.Object({
      path: Type.String({ description: "File or directory path to index" }),
    }),
    execute: async (_toolCallId, params) => {
      let text: string;
      if (!existsSync(params.path)) text = `Path not found: ${params.path}`;
      else {
        const files = collectFiles(params.path);
        if (!files.length) text = `No indexable text files found in: ${params.path}`;
        else {
          const result = indexFiles(files);
          text = `Indexed ${result.indexed} files (${result.chunks} chunks). ${result.skipped} unchanged.`;
        }
      }
      return { content: [{ type: "text" as const, text }] };
    }
  });

  pi.registerTool({
    name: "lens_query",
    description: "Search the local pi-local-rag index using BM25 keyword matching. Returns relevant chunks from indexed files with file paths and line numbers.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 10)" })),
    }),
    execute: async (_toolCallId, params) => {
      const index = loadIndex();
      let text: string;
      if (!index.chunks.length) text = "pi-local-rag index is empty. Run lens_index first."
      else {
        const results = searchChunks(params.query, index, params.limit || 10);
        if (!results.length) text = `No results for: ${params.query}`;
        else text = JSON.stringify(results.map(r => ({
          file: r.file, lines: `${r.lineStart}-${r.lineEnd}`,
          tokens: r.tokens, preview: r.content.slice(0, 300)
        })), null, 2);
      }
      return { content: [{ type: "text" as const, text }] };
    }
  });

  pi.registerTool({
    name: "lens_status",
    description: "Show pi-local-rag index statistics: file count, chunk count, total tokens, last build time.",
    parameters: Type.Object({}),
    execute: async (_toolCallId) => {
      const index = loadIndex();
      const text = JSON.stringify({
        files: Object.keys(index.files).length,
        chunks: index.chunks.length,
        totalTokens: index.chunks.reduce((s, c) => s + c.tokens, 0),
        lastBuild: index.lastBuild || "never",
        storagePath: RAG_DIR, // ~/.pi/lens
      }, null, 2);
      return { content: [{ type: "text" as const, text }] };
    }
  });
}
