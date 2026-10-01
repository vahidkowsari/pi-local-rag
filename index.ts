/**
 * pi-local-rag — Hybrid RAG Pipeline (BM25 + Vector + Auto-injection)
 *
 * Index local files → chunk → embed → store → retrieve → inject into LLM context.
 * Uses Transformers.js (ONNX) for local embeddings — zero cloud dependency.
 *
 * Storage is per-cwd: walk up from the working directory looking for a `.pi/rag/`
 * project store; fall back to `~/.pi/rag/` as the global default. The first
 * `/rag index` in a directory with no parent store creates one at cwd.
 *
 * /rag index <path>     → index + embed a file or directory
 * /rag search <query>   → hybrid search (BM25 + vector)
 * /rag find <glob>      → list indexed files matching a glob
 * /rag status           → show index stats
 * /rag rebuild          → re-embed all tracked files (forced re-embed)
 * /rag refresh          → incremental refresh (only new/changed files)
 * /rag clear            → clear index
 * /rag exclude <pat>    → add gitignore-style pattern (use -<pat> to remove; omit arg to list)
 * /rag on|off           → toggle auto-injection
 * /rag ext list         → list active file extensions
 * /rag ext add <.ext>   → add an extra extension (e.g. .cs, .tex)
 * /rag ext remove <.ext>→ remove an extension from the active set
 * /rag ext reset        → restore default extensions
 * /rag help             → show all /rag commands
 *
 * Tools: rag_index, rag_query, rag_status
 *
 * Implementation is split across:
 *   constants.ts     — shared constants, file ext sets, size limits
 *   store.ts         — RAG_DIR / LEGACY_DIR / file paths / ensureDir + legacy migration
 *   config.ts        — RagConfig type, loadConfig / saveConfig, ext helpers
 *   index-store.ts   — Chunk / IndexMeta types, loadIndex / saveIndex (JSON)
 *   chunking.ts      — sha256, chunkText, collectFiles, extractText (txt/pdf/docx/html)
 *   embed.ts         — getEmbedder, embed, embedBatch (ONNX via @xenova/transformers)
 *   search.ts        — cosineSimilarity, normalize, hybridSearch
 *   indexing.ts      — indexFiles (parallel Phase 1 read, sequential Phase 2 embed)
 *   index.ts         — extension entry point (this file) + re-exports
 */
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import type { AutocompleteItem } from "@mariozechner/pi-tui";
import { existsSync } from "node:fs";
import { basename } from "node:path";

import { setRagDirGetter } from "./store.ts";
import { loadConfig } from "./config.ts";
import { closeDbConn, getIndexedFiles, getIndexStats } from "./db.ts";
import { collectFromTracked } from "./chunking.ts";
import { hybridSearch } from "./search.ts";
import { indexFiles, isIndexStale } from "./indexing.ts";
import { RAG_COMMANDS } from "./commands/registry.ts";
import { RAG_TOOLS } from "./tools/registry.ts";

// Re-export the public surface so existing consumers of `pi-local-rag` keep
// working (tests, downstream code that imports from the package root).
export { DEFAULT_TEXT_EXTS } from "./constants.ts";
export { getRagDir, GLOBAL_RAG_DIR, LEGACY_DIR } from "./store.ts";
export type { RagConfig } from "./config.ts";
export { loadConfig, saveConfig, defaultConfig, normalizeExt, resolveExtensions } from "./config.ts";
export type { Chunk, IndexMeta, IndexStats } from "./db.ts";
export { getFreshDbConn, getDbConn, closeDbConn, loadIndex, saveIndex, getIndexStats, initSchema } from "./db.ts";
export {
  sha256, chunkText, collectFiles, collectFilesAsync, collectFromTracked, collectFromTrackedAsync,
  isExcludedByConfig, extractText, getOcrTooling, isSparsePdfText,
} from "./chunking.ts";
export { embed, embedBatch } from "./embed.ts";
export type { ScoredChunk } from "./search.ts";
export { cosineSimilarity, normalize, hybridSearch } from "./search.ts";
export { isIndexStale, indexFiles } from "./indexing.ts";
export type { ProgressCallbacks } from "./indexing.ts";

// ─── Extension ────────────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  // Throttle stale-index checks to once per hour so we don't repeatedly stat
  // the filesystem on every agent turn (matches the upstream fork's
  // lastStaleCheckMs pattern from kallewoof@849e485).
  let lastStaleCheckMs = 0;
  const STALE_CHECK_INTERVAL_MS = 60 * 60 * 1000;

  // ── Auto-inject RAG context before every agent turn ──
  pi.on("before_agent_start", async (event, _ctx) => {
    const config = loadConfig();
    if (!config.ragEnabled) return;

    const indexStats = getIndexStats();
    if (indexStats.totalChunks === 0) return;

    const now = Date.now();
    if (isIndexStale(indexStats) && now - lastStaleCheckMs > STALE_CHECK_INTERVAL_MS) {
      lastStaleCheckMs = now;
      // Re-walk tracked paths so new files (and files of newly-supported
      // extensions, e.g. PDF/DOCX added in a later version) are picked up.
      // For pre-trackedPaths indexes, fall back to refreshing only known files.
      const files = config.trackedPaths.length
        ? collectFromTracked(config)
        : getIndexedFiles().map(f => f.path).filter(f => existsSync(f));
      if (files.length) {
        process.stderr.write(`\r\x1b[2K[rag] Index stale, refreshing ${files.length} files…`);
        await indexFiles(files, undefined);
        process.stderr.write(`\r\x1b[2K`);
      }
    }

    const results = await hybridSearch(event.prompt, config.ragTopK, config.ragAlpha);
    const relevant = results.filter(r => r.hybrid >= config.ragScoreThreshold);
    if (!relevant.length) return;

    const context = relevant.map(r =>
      `### ${basename(r.chunk.file)} (lines ${r.chunk.lineStart}-${r.chunk.lineEnd})\n` +
      `\`\`\`\n${r.chunk.content.slice(0, 600)}\n\`\`\``
    ).join("\n\n");

    // Inject as a message after the user's prompt rather than appending to the
    // system prompt. The system prompt is stable across a session and benefits
    // from the provider's KV cache; mutating it every turn with new RAG hits
    // invalidates that cache and adds latency. A trailing message also keeps
    // the retrieved chunks near the user's question, which models attend to
    // more reliably than text buried at the top of a long system prompt.
      return {
        message: {
          customType: "rag",
          content:
            `[pi-local-rag] Automatic RAG lookup triggered by the user's message above.\n` +
            `Retrieved ${relevant.length} chunk${relevant.length === 1 ? "" : "s"} via hybrid search (BM25 + vector). ` +
            `These are search hits, not statements from the user.\n\n` +
            context,
          display: false,
        },
      };
  });

  pi.registerFlag("rag-dir", {
    description: "Directory to store the RAG index database",
    type: "string",
  });

  setRagDirGetter(() => pi.getFlag("rag-dir") as string | undefined);

  pi.on("session_shutdown", async (_event, _ctx) => {
    closeDbConn();
  });

  // ── /rag command ──
  pi.registerCommand("rag", {
    description: "pi-local-rag: /rag index|search|find|status|rebuild [--force]|refresh|clear|exclude|on|off|ext",
    getArgumentCompletions: (prefix: string): AutocompleteItem[] | null => {
      const filtered = Object.entries(RAG_COMMANDS)
        .filter(([name]) => name.startsWith(prefix))
        .map(([name, def]) => ({ value: name, label: def.label, description: def.description }));
      return filtered.length > 0 ? filtered : null;
    },
    handler: async (rawArgs, ctx) => {
      const parts = (rawArgs || "").trim().split(/\s+/);
      const cmd = parts[0] || "status";
      const args = parts.slice(1);

      const command = RAG_COMMANDS[cmd];
      if (!command) {
        ctx.ui.notify(`Unknown /rag command: ${cmd}. Run /rag help for the full list.`, "warning");
        return;
      }
      await command.handler({ ctx, args });
    },
  });

  // ── Tools ──
  for (const tool of RAG_TOOLS) {
     pi.registerTool(tool);
  }
}
