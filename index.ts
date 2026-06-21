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
import { Type } from "@sinclair/typebox";
import { existsSync } from "node:fs";
import { resolve, extname, basename, relative } from "node:path";
import ignore from "ignore";

import { RST, B, D, GREEN, CYAN } from "./constants.ts";
import { getRagDir, GLOBAL_RAG_DIR } from "./store.ts";
import { loadConfig, saveConfig, normalizeExt, resolveExtensions } from "./config.ts";
import { openDb, loadIndex, getIndexedFiles, getEmbeddedCount, saveIndex, getIndexStats } from "./db.ts";
import { collectFiles, collectFromTracked, collectFromTrackedAsync, isExcludedByConfig } from "./chunking.ts";
import { hybridSearch } from "./search.ts";
import { indexFiles, isIndexStale } from "./indexing.ts";

// Re-export the public surface so existing consumers of `pi-local-rag` keep
// working (tests, downstream code that imports from the package root).
export { DEFAULT_TEXT_EXTS } from "./constants.ts";
export { getRagDir, GLOBAL_RAG_DIR, LEGACY_DIR } from "./store.ts";
export type { RagConfig } from "./config.ts";
export { loadConfig, saveConfig, defaultConfig, normalizeExt, resolveExtensions } from "./config.ts";
export type { Chunk, IndexMeta, IndexStats } from "./db.ts";
export { openDb, getDb, loadIndex, saveIndex, getIndexStats, initSchema, float32ToBuffer } from "./db.ts";
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

    const database = openDb();
    try {
      const indexStats = getIndexStats(database);
      if (indexStats.totalChunks === 0) return;

      const now = Date.now();
      if (isIndexStale(indexStats) && now - lastStaleCheckMs > STALE_CHECK_INTERVAL_MS) {
        lastStaleCheckMs = now;
        // Re-walk tracked paths so new files (and files of newly-supported
        // extensions, e.g. PDF/DOCX added in a later version) are picked up.
        // For pre-trackedPaths indexes, fall back to refreshing only known files.
        const files = config.trackedPaths.length
          ? collectFromTracked(config)
          : Object.keys(loadIndex().files).filter(f => existsSync(f));
        if (files.length) {
          process.stderr.write(`\r\x1b[2K[rag] Index stale, refreshing ${files.length} files…`);
          await indexFiles(files, undefined, database);
          process.stderr.write(`\r\x1b[2K`);
        }
      }

      const results = await hybridSearch(event.prompt, config.ragTopK, config.ragAlpha, database);
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
    } finally {
      database.close();
    }
  });

  // ── /rag command ──
  const RAG_SUBCOMMANDS: { value: string; label: string; description: string }[] = [
    { value: "index",    label: "index",    description: "Index a file or directory" },
    { value: "search",   label: "search",   description: "Search the index" },
    { value: "find",     label: "find",     description: "List indexed files matching a glob" },
    { value: "status",   label: "status",   description: "Show index statistics" },
    { value: "rebuild",  label: "rebuild",  description: "Re-embed tracked files (--force to skip hash check + wipe DB)" },
    { value: "refresh",  label: "refresh",  description: "Incremental refresh — new/changed files only" },
    { value: "clear",    label: "clear",    description: "Clear the index" },
    { value: "exclude",  label: "exclude",  description: "Manage gitignore-style exclude patterns" },
    { value: "ext",      label: "ext",      description: "Manage indexable file-extension allowlist" },
    { value: "on",       label: "on",       description: "Enable auto-injection" },
    { value: "off",      label: "off",      description: "Disable auto-injection" },
    { value: "help",     label: "help",     description: "Show all /rag commands" },
  ];

  pi.registerCommand("rag", {
    description: "pi-local-rag: /rag index|search|find|status|rebuild [--force]|refresh|clear|exclude|on|off|ext",
    getArgumentCompletions: (prefix: string): AutocompleteItem[] | null => {
      const filtered = RAG_SUBCOMMANDS
        .filter((s) => s.value.startsWith(prefix))
        .map((s) => ({ value: s.value, label: s.label, description: s.description }));
      return filtered.length > 0 ? filtered : null;
    },
    handler: async (args, ctx) => {
      const parts = (args || "").trim().split(/\s+/);
      const cmd = parts[0] || "status";

      // ── index ──
      if (cmd === "index") {
        const path = parts[1] || ".";
        if (!existsSync(path)) { ctx.ui.notify(`Path not found: ${path}`, "error"); return; }
        // Anchor a project-local store at cwd if there isn't one in scope yet.
        getRagDir({ createIfMissing: true });
        const config = loadConfig();
        const absPath = resolve(path);
        if (!config.trackedPaths.includes(absPath)) {
          config.trackedPaths.push(absPath);
          saveConfig(config);
        }
        const files = collectFiles(absPath, undefined, config.excludePatterns);
        if (!files.length) { ctx.ui.notify(`No indexable files found in: ${path}`, "warning"); return; }

        const total = files.length;
        ctx.ui.notify(`Found ${total} files to index`, "info");

        function progressBar(n: number, total: number, width = 24): string {
          const filled = Math.round((n / total) * width);
          return CYAN + "█".repeat(filled) + D + "░".repeat(width - filled) + RST;
        }

        const result = await indexFiles(files, {
          onFile(current, total, filename, skipped) {
            const pct = Math.round((current / total) * 100);
            const bar = progressBar(current, total);
            ctx.ui.setStatus("rag", `■ Indexing ${pct}% │ ${current}/${total} files │ ${skipped} unchanged`);
            ctx.ui.setWidget("rag", [
              `${B}${CYAN}Indexing${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
              `${D}file:    ${RST}${filename}`,
              `${D}done:    ${RST}${GREEN}${current - skipped} embedded${RST}  ${D}${skipped} unchanged${RST}`,
            ]);
          },
          onChunk(ci, total, filename) {
            ctx.ui.setStatus("rag", `■ Embedding ${filename} — chunk ${ci}/${total}`);
          },
          onSave() {
            ctx.ui.setStatus("rag", `■ Saving index...`);
          },
        });

        ctx.ui.setStatus("rag", undefined);
        ctx.ui.setWidget("rag", undefined);

        const secs = (result.durationMs / 1000).toFixed(1);
        const ragDir = getRagDir();
        const scope = ragDir === GLOBAL_RAG_DIR() ? "global" : "project";
        ctx.ui.notify(`✅ Indexed ${result.indexed} files (${result.chunks} chunks) · ${result.skipped} unchanged · ${secs}s · tracking ${config.trackedPaths.length} path(s) · ${scope} store`, "info");
        return;
      }

      // ── search ──
      if (cmd === "search") {
        const query = parts.slice(1).join(" ");
        if (!query) { ctx.ui.notify("Usage: /rag search <query>", "warning"); return; }
        const config = loadConfig();
        const results = await hybridSearch(query, 10, config.ragAlpha);
        if (!results.length) { ctx.ui.notify(`No results for: ${query}`, "warning"); return; }

        const th = ctx.ui.theme;
        const hasVectors = getEmbeddedCount() > 0;
        const lines: string[] = [
          th.bold(th.fg("accent", "🔍 ") + `${results.length} results for "${query}"`) +
            "  " + th.fg("dim", hasVectors ? "hybrid BM25+vector" : "BM25 only"),
          "",
        ];
        for (const r of results) {
          lines.push(
            th.fg("success", basename(r.chunk.file)) +
            th.fg("muted", `:${r.chunk.lineStart}-${r.chunk.lineEnd}`) +
            "  " + th.fg("dim", `score=${r.hybrid.toFixed(2)}`)
          );
          const preview = r.chunk.content.split("\n").slice(0, 3).join("\n");
          lines.push(th.fg("dim", preview.slice(0, 200)));
          lines.push("");
        }
        ctx.ui.setWidget("rag-search", lines);
        return;
      }

      // ── on/off toggle ──
      if (cmd === "on" || cmd === "off") {
        const config = loadConfig();
        config.ragEnabled = cmd === "on";
        saveConfig(config);
        ctx.ui.notify(cmd === "on" ? "RAG auto-injection enabled" : "RAG auto-injection disabled", "info");
        return;
      }

      // ── rebuild ──
      if (cmd === "rebuild") {
        // Parse --force flag from any position after "rebuild".
        const rebuildArgs = parts.slice(1);
        const force = rebuildArgs.includes("--force");

        const database = openDb();
        const config = loadConfig();
        try {
          const indexedRows = database.prepare("SELECT path FROM files").all() as Array<{ path: string }>;
          const indexedFileSet = new Set(indexedRows.map(f => f.path));

          // Walking tracked paths can stall the event loop on large trees
          // (45k+ files). Use the async variant + yield up-front so the user
          // gets immediate feedback before the heavy work begins.
          ctx.ui.notify("Scanning tracked paths...", "info");
          const trackedFiles = await collectFromTrackedAsync(config);

          // Union of currently-indexed files and files discovered by walking tracked paths.
          const targetSet = new Set<string>([...trackedFiles]);
          for (const f of indexedFileSet) {
            if (existsSync(f) && !isExcludedByConfig(f, config.trackedPaths, config.excludePatterns)) {
              targetSet.add(f);
            }
          }
          const targetFiles = [...targetSet];

          if (!targetFiles.length && !indexedFileSet.size) {
            ctx.ui.notify("No files to rebuild. Run /rag index <path> first.", "warning");
            return;
          }

          // Files in the index but no longer present (deleted, excluded, or untracked).
          const droppedFiles = [...indexedFileSet].filter(f => !targetSet.has(f));
          for (const f of droppedFiles) {
            database.prepare("DELETE FROM chunks_vec WHERE rowid IN (SELECT rowid FROM chunks WHERE file_path = ?)").run(f);
            database.prepare("DELETE FROM chunks WHERE file_path = ?").run(f);
            database.prepare("DELETE FROM files WHERE path = ?").run(f);
          }
          if (force) {
            // --force: wipe everything and rebuild the FTS index. indexFiles
            // will then insert fresh rows for every targetFile, bypassing the
            // skip-on-equal-hash check.
            database.exec("DELETE FROM chunks_vec; DELETE FROM chunks; DELETE FROM files;");
            database.exec("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')");
          } else {
            for (const f of targetFiles) {
              database.prepare("UPDATE files SET embedded = 0 WHERE path = ?").run(f);
            }
          }

          const newFiles = targetFiles.filter(f => !indexedFileSet.has(f));
          ctx.ui.notify(`Rebuilding ${targetFiles.length} files${force ? " (forced)" : ""}...`, "info");
          if (droppedFiles.length) ctx.ui.notify(`Pruned ${droppedFiles.length} files (deleted/excluded)`, "info");
          if (newFiles.length) ctx.ui.notify(`Discovered ${newFiles.length} new files`, "info");

          // Yield so the TUI can paint the "Rebuilding" message before
          // indexFiles starts hammering the event loop.
          await new Promise<void>(r => setTimeout(r, 0));

          function progressBar(n: number, total: number, width = 24): string {
            const filled = Math.round((n / total) * width);
            return CYAN + "█".repeat(filled) + D + "░".repeat(width - filled) + RST;
          }

          const result = await indexFiles(targetFiles, {
            onFile(current, total, filename, skipped) {
              const pct = Math.round((current / total) * 100);
              const bar = progressBar(current, total);
              ctx.ui.setStatus("rag", `■ Rebuilding ${pct}% │ ${current}/${total} │ ${skipped} unchanged`);
              ctx.ui.setWidget("rag", [
                `${B}${CYAN}Rebuilding${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
                `${D}file:    ${RST}${filename}`,
                `${D}done:    ${RST}${GREEN}${current - skipped} re-embedded${RST}  ${D}${skipped} unchanged${RST}`,
              ]);
            },
            onEmbed(done, total) {
              const pct = Math.round((done / total) * 100);
              const bar = progressBar(done, total);
              ctx.ui.setStatus("rag", `■ Embedding ${pct}% │ ${done}/${total} chunks`);
              ctx.ui.setWidget("rag", [
                `${B}${CYAN}Embedding${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
                `${D}chunks:  ${RST}${done}/${total}`,
              ]);
            },
            onChunk(ci, total, filename) {
              ctx.ui.setStatus("rag", `■ Embedding ${filename} — chunk ${ci}/${total}`);
            },
            onSave() {
              ctx.ui.setStatus("rag", `■ Saving index...`);
            },
          }, database, force);

          ctx.ui.setStatus("rag", undefined);
          ctx.ui.setWidget("rag", undefined);

          const secs = (result.durationMs / 1000).toFixed(1);
          ctx.ui.notify(`✅ Rebuilt: ${result.indexed} re-indexed · ${result.skipped} unchanged · ${droppedFiles.length} deleted · ${result.chunks} chunks · ${secs}s`, "info");
        } finally {
          database.close();
        }
        return;
      }

      // ── refresh (on-demand equivalent of the 24h auto-refresh) ──
      if (cmd === "refresh") {
        const config = loadConfig();
        const filesFromDb = getIndexedFiles()
        const files = config.trackedPaths.length
          ? collectFromTracked(config)
          : filesFromDb.map(f => f.path).filter(f => existsSync(f));
        if (!files.length) {
          ctx.ui.notify("No tracked files to refresh. Run /rag index <path> first.", "warning");
          return;
        }

        ctx.ui.notify(`Refreshing ${files.length} files...`, "info");

        function progressBar(n: number, total: number, width = 24): string {
          const filled = Math.round((n / total) * width);
          return CYAN + "█".repeat(filled) + D + "░".repeat(width - filled) + RST;
        }

        const result = await indexFiles(files, {
          onFile(current, total, filename, skipped) {
            const pct = Math.round((current / total) * 100);
            const bar = progressBar(current, total);
            ctx.ui.setStatus("rag", `■ Refreshing ${pct}% │ ${current}/${total} │ ${skipped} unchanged`);
            ctx.ui.setWidget("rag", [
              `${B}${CYAN}Refreshing${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
              `${D}file:    ${RST}${filename}`,
              `${D}done:    ${RST}${GREEN}${current - skipped} new/changed${RST}  ${D}${skipped} unchanged${RST}`,
            ]);
          },
          onChunk(ci, total, filename) {
            ctx.ui.setStatus("rag", `■ Embedding ${filename} — chunk ${ci}/${total}`);
          },
          onSave() {
            ctx.ui.setStatus("rag", `■ Saving index...`);
          },
        });

        ctx.ui.setStatus("rag", undefined);
        ctx.ui.setWidget("rag", undefined);

        const secs = (result.durationMs / 1000).toFixed(1);
        ctx.ui.notify(`✅ Refreshed ${result.indexed} new/changed · ${result.skipped} unchanged · ${result.chunks} chunks · ${secs}s`, "info");
        return;
      }

      // ── ext (configure file extensions) ──
      if (cmd === "ext") {
        const sub = (parts[1] || "list").toLowerCase();
        const config = loadConfig();

        if (sub === "list") {
          const th = ctx.ui.theme;
          const active = Array.from(resolveExtensions(config)).sort();
          const lines: string[] = [
            th.bold("Active file extensions") + "  " + th.fg("dim", `(${active.length})`),
            th.fg("muted", "  " + active.join(" ")),
          ];
          if (config.extraExtensions.length)
            lines.push("  " + th.fg("dim", "extra:   ") + th.fg("success", config.extraExtensions.join(" ")));
          if (config.excludeExtensions.length)
            lines.push("  " + th.fg("dim", "excluded:") + " " + th.fg("warning", config.excludeExtensions.join(" ")));
          lines.push("", th.fg("dim", "Edit via /rag ext add <.ext> / remove <.ext> / reset"));
          ctx.ui.setWidget("rag-ext", lines);
          return;
        }

        if (sub === "add") {
          const ext = normalizeExt(parts[2] || "");
          if (!ext) { ctx.ui.notify("Usage: /rag ext add <.ext>", "warning"); return; }
          config.excludeExtensions = config.excludeExtensions.filter(e => normalizeExt(e) !== ext);
          if (!config.extraExtensions.map(normalizeExt).includes(ext)) config.extraExtensions.push(ext);
          saveConfig(config);
          ctx.ui.notify(`Added ${ext} to indexable extensions. Run /rag index <path> to pick up matching files.`, "info");
          return;
        }

        if (sub === "remove" || sub === "rm") {
          const ext = normalizeExt(parts[2] || "");
          if (!ext) { ctx.ui.notify("Usage: /rag ext remove <.ext>", "warning"); return; }
          const wasExtra = config.extraExtensions.map(normalizeExt).includes(ext);
          config.extraExtensions = config.extraExtensions.filter(e => normalizeExt(e) !== ext);
          if (!wasExtra && !config.excludeExtensions.map(normalizeExt).includes(ext)) config.excludeExtensions.push(ext);
          saveConfig(config);
          ctx.ui.notify(`Removed ${ext} from indexable extensions.`, "info");
          return;
        }

        if (sub === "reset") {
          config.extraExtensions = [];
          config.excludeExtensions = [];
          saveConfig(config);
          ctx.ui.notify("Extension list reset to defaults.", "info");
          return;
        }

        ctx.ui.notify("Usage: /rag ext list|add <.ext>|remove <.ext>|reset", "warning");
        return;
      }

      // ── clear ──
      if (cmd === "clear") {
        saveIndex({ chunks: [], files: {}, lastBuild: "" });
        ctx.ui.notify("Index cleared.", "info");
        return;
      }

      // ── exclude ──
      if (cmd === "exclude") {
        const config = loadConfig();
        const expr = parts.slice(1).join(" ").trim();
        const th = ctx.ui.theme;

        if (!expr) {
          if (!config.excludePatterns.length) {
            ctx.ui.notify("No exclude patterns set. Add one with: /rag exclude <pattern>", "info");
            return;
          }
          const lines: string[] = [
            th.bold(`Exclude patterns (${config.excludePatterns.length})`),
            "",
          ];
          for (const p of config.excludePatterns) lines.push("  " + th.fg("muted", p));
          ctx.ui.setWidget("rag-exclude", lines);
          return;
        }

        if (expr.startsWith("-")) {
          const target = expr.slice(1);
          const before = config.excludePatterns.length;
          config.excludePatterns = config.excludePatterns.filter(p => p !== target);
          if (config.excludePatterns.length === before) {
            ctx.ui.notify(`Pattern not found: ${target}`, "warning");
            return;
          }
          saveConfig(config);
          ctx.ui.notify(`✅ Removed exclude: ${target} · ${config.excludePatterns.length} pattern(s) remain. Run /rag rebuild to re-apply.`, "info");
          return;
        }

        if (config.excludePatterns.includes(expr)) {
          ctx.ui.notify(`Already excluded: ${expr}`, "warning");
          return;
        }
        config.excludePatterns.push(expr);
        saveConfig(config);
        ctx.ui.notify(`✅ Added exclude: ${expr} · ${config.excludePatterns.length} pattern(s) total. Run /rag rebuild to re-apply.`, "info");
        return;
      }

      // ── find ──
      if (cmd === "find") {
        const glob = parts.slice(1).join(" ").trim();
        if (!glob) {
          ctx.ui.notify("Usage: /rag find <glob>   e.g. *.html, page*, foo.js, src/*.ts", "warning");
          return;
        }

        const files = getIndexedFiles();
        const cwd = process.cwd();
        const ig = ignore().add([glob]);

        const matches: string[] = [];
        for (const fp of files.map(f => f.path)) {
          const rel = relative(cwd, fp);
          const candidate = rel && !rel.startsWith("..") ? rel : basename(fp);
          if (ig.ignores(candidate)) matches.push(fp);
        }
        matches.sort();

        if (!matches.length) {
          ctx.ui.notify(`No indexed files match: ${glob}`, "warning");
          return;
        }
        const th = ctx.ui.theme;
        const lines: string[] = [
          th.bold(`🔍 ${matches.length} indexed file${matches.length === 1 ? "" : "s"} matching "${glob}"`),
          "",
        ];
        for (const fp of matches) lines.push(th.fg("success", fp));
        ctx.ui.setWidget("rag-find", lines);
        return;
      }

      // ── help ──
      if (cmd === "help") {
        const pad = (s: string, n: number) => s + " ".repeat(Math.max(0, n - s.length));
        const cmds: [string, string][] = [
          ["/rag index <path>",       "Index a file or directory (chunks, embeds, stores)"],
          ["/rag search <query>",     "Hybrid BM25 + vector search over the index"],
          ["/rag find <glob>",        "List indexed files matching a glob (e.g. *.ts, src/*)"],
          ["/rag status",             "Show index stats and active configuration"],
          ["/rag rebuild [--force]",  "Re-embed tracked files; --force wipes DB and bypasses hash skip"],
          ["/rag refresh",            "Incremental refresh — only new/changed files (also fires automatically every 24h)"],
          ["/rag clear",              "Delete all indexed chunks"],
          ["/rag exclude <pattern>",  "Add a gitignore-style exclude pattern (omit to list; -<pattern> to remove)"],
          ["/rag ext list|add|remove|reset", "Manage the indexable file-extension allowlist"],
          ["/rag on",                 "Enable automatic RAG injection before each agent turn"],
          ["/rag off",                "Disable automatic RAG injection"],
          ["/rag help",               "Show this help"],
        ];
        const COL = 36;
        const th = ctx.ui.theme;
        const lines: string[] = [th.bold("pi-local-rag commands"), ""];
        for (const [usage, desc] of cmds) {
          lines.push("  " + th.fg("success", pad(usage, COL)) + "  " + th.fg("dim", desc));
        }
        ctx.ui.setWidget("rag-help", lines);
        return;
      }

      // ── status (default) ──
      const indexStats = getIndexStats()
      const config = loadConfig();
      const fileCount = indexStats.totalFiles;
      const totalTokens = indexStats.totalTokens;
      const embeddedCount = indexStats.embeddedCount;
      const vectorCoverage = indexStats.totalChunks ? Math.round(embeddedCount / indexStats.totalChunks * 100) : 0;

      const th = ctx.ui.theme;
      const label = (k: string) => th.fg("dim", k.padEnd(18));
      const val = (v: string | number) => th.fg("success", String(v));
      const ragDir = getRagDir();
      const scope = ragDir === GLOBAL_RAG_DIR() ? "global" : "project";
      const lines: string[] = [
        th.bold("🔍 pi-local-rag"),
        "",
        "  " + label("Files indexed:")  + val(fileCount),
        "  " + label("Chunks:")         + val(indexStats.totalChunks),
        "  " + label("Vectors:")        + val(embeddedCount) + "  " + th.fg("dim", `(${vectorCoverage}% coverage)`),
        "  " + label("Total tokens:")   + val(totalTokens.toLocaleString()),
        "  " + label("Embedding model:") + th.fg("dim", indexStats.embeddingModel || "none"),
        "  " + label("Last build:")     + (indexStats.lastBuild || th.fg("dim", "never")),
        "  " + label("Storage:")        + th.fg("dim", `${ragDir} (${scope})`),
        "",
        "  " + label("RAG injection:")  +
          (config.ragEnabled ? th.fg("success", "enabled") : th.fg("warning", "disabled")) +
          th.fg("dim", `  topK=${config.ragTopK}  threshold=${config.ragScoreThreshold}  alpha=${config.ragAlpha}`),
      ];

      if (fileCount) {
        lines.push("", "  " + th.bold("File types:"));
        const files = getIndexedFiles()
        const byExt: Record<string, number> = {};
        for (const f of files.map(f => f.path)) byExt[extname(f)] = (byExt[extname(f)] || 0) + 1;
        for (const [ext, count] of Object.entries(byExt).sort((a, b) => b[1] - a[1]).slice(0, 8)) {
          lines.push("    " + th.fg("muted", ext) + "  " + th.fg("dim", String(count)));
        }
      }

      lines.push("", "  " + th.bold("Tracked paths:"));
      if (config.trackedPaths.length) {
        for (const p of config.trackedPaths) lines.push("    " + th.fg("muted", p));
      } else {
        lines.push("    " + th.fg("dim", "(none — run /rag index <path> to track)"));
      }

      lines.push("", "  " + th.bold("Exclude patterns:"));
      if (config.excludePatterns.length) {
        for (const p of config.excludePatterns) lines.push("    " + th.fg("muted", p));
      } else {
        lines.push("    " + th.fg("dim", "(none — add with /rag exclude <pattern>)"));
      }

      ctx.ui.setWidget("rag-status", lines);
    },
  });

  // ── Tools ──

  pi.registerTool({
    name: "rag_index",
    label: "RAG index",
    description: "Index a file or directory into the local pi-local-rag pipeline. Chunks text files (including PDF and DOCX), generates embeddings, stores for hybrid BM25+vector search.",
    parameters: Type.Object({
      path: Type.String({ description: "File or directory path to index" }),
    }),
    execute: async (_toolCallId, params) => {
      if (!existsSync(params.path)) return { content: [{ type: "text" as const, text: `Path not found: ${params.path}` }], details: undefined };
      // Anchor a project-local store at cwd if there isn't one in scope yet.
      getRagDir({ createIfMissing: true });
      const config = loadConfig();
      const absPath = resolve(params.path);
      if (!config.trackedPaths.includes(absPath)) {
        config.trackedPaths.push(absPath);
        saveConfig(config);
      }
      const files = collectFiles(absPath, undefined, config.excludePatterns);
      if (!files.length) return { content: [{ type: "text" as const, text: `No indexable files found in: ${params.path}` }], details: undefined };
      const result = await indexFiles(files, {});
      process.stderr.write(`\n`);
      return { content: [{ type: "text" as const, text: `Indexed ${result.indexed} files (${result.chunks} chunks, embeddings generated). ${result.skipped} unchanged. ${(result.durationMs / 1000).toFixed(1)}s` }], details: undefined };
    },
  });

  pi.registerTool({
    name: "rag_query",
    label: "RAG query",
    description: "Search the local pi-local-rag index using hybrid BM25+vector search. Returns relevant chunks with file paths, line numbers, and relevance scores.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 10)" })),
    }),
    execute: async (_toolCallId, params) => {
      const index = loadIndex();
      if (!index.chunks.length) return { content: [{ type: "text" as const, text: "pi-local-rag index is empty. Run rag_index first." }], details: undefined };
      const config = loadConfig();
      const results = await hybridSearch(params.query, params.limit ?? 10, config.ragAlpha);
      if (!results.length) return { content: [{ type: "text" as const, text: `No results for: ${params.query}` }], details: undefined };
      const text = JSON.stringify(results.map(r => ({
        file: r.chunk.file,
        lines: `${r.chunk.lineStart}-${r.chunk.lineEnd}`,
        tokens: r.chunk.tokens,
        scores: { bm25: r.bm25.toFixed(3), vector: r.vector.toFixed(3), hybrid: r.hybrid.toFixed(3) },
        preview: r.chunk.content.slice(0, 300),
      })), null, 2);
      return { content: [{ type: "text" as const, text }], details: undefined };
    },
  });

  pi.registerTool({
    name: "rag_status",
    label: "RAG status",
    description: "Show pi-local-rag index statistics: file count, chunk count, vector coverage, embedding model, RAG config.",
    parameters: Type.Object({}),
    execute: async (_toolCallId) => {
      const index = loadIndex();
      const config = loadConfig();
      const embeddedCount = index.chunks.filter(c => c.vector).length;
      const text = JSON.stringify({
        files: Object.keys(index.files).length,
        chunks: index.chunks.length,
        vectorsEmbedded: embeddedCount,
        vectorCoverage: index.chunks.length ? `${Math.round(embeddedCount / index.chunks.length * 100)}%` : "0%",
        embeddingModel: index.embeddingModel ?? "none",
        totalTokens: index.chunks.reduce((s, c) => s + c.tokens, 0),
        lastBuild: index.lastBuild || "never",
        ragConfig: config,
        storagePath: getRagDir(),
        storageScope: getRagDir() === GLOBAL_RAG_DIR() ? "global" : "project",
      }, null, 2);
      return { content: [{ type: "text" as const, text }], details: undefined };
    },
  });
}
