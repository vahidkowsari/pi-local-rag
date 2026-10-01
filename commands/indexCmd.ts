import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { getRagDir, GLOBAL_RAG_DIR } from "../store.ts";
import { loadConfig, saveConfig } from "../config.ts";
import { collectFiles } from "../chunking.ts";
import { indexFiles } from "../indexing.ts";
import { makeProgressCallbacks } from "../progress.ts";
import type { RagCommandHandler } from "./types.ts";

export const indexCommand: RagCommandHandler = async ({ ctx, args }) => {
  const path = args[0] || ".";
  if (!existsSync(path)) {
    ctx.ui.notify(`Path not found: ${path}`, "error");
    return;
  }
  // Anchor a project-local store at cwd if there isn't one in scope yet.
  getRagDir({ createIfMissing: true });
  const config = loadConfig();
  const absPath = resolve(path);
  if (!config.trackedPaths.includes(absPath)) {
    config.trackedPaths.push(absPath);
    saveConfig(config);
  }
  const files = collectFiles(absPath, undefined, config.excludePatterns);
  if (!files.length) {
    ctx.ui.notify(`No indexable files found in: ${path}`, "warning");
    return;
  }

  const total = files.length;
  ctx.ui.notify(`Found ${total} files to index`, "info");

  const result = await indexFiles(files, makeProgressCallbacks(ctx, "Indexing", "embedded"));
  ctx.ui.setStatus("rag", undefined);
  ctx.ui.setWidget("rag", undefined);

  const secs = (result.durationMs / 1000).toFixed(1);
  const ragDir = getRagDir();
  const scope = ragDir === GLOBAL_RAG_DIR() ? "global" : "project";
  ctx.ui.notify(`✅ Indexed ${result.indexed} files (${result.chunks} chunks) · ${result.skipped} unchanged · ${secs}s · tracking ${config.trackedPaths.length} path(s) · ${scope} store`, "info");
};
