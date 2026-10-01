import { existsSync } from "node:fs";
import { getDbConn } from "../db.ts";
import { loadConfig } from "../config.ts";
import { collectFromTrackedAsync, isExcludedByConfig } from "../chunking.ts";
import { indexFiles } from "../indexing.ts";
import { makeProgressCallbacks } from "../progress.ts";
import * as repo from "../repository.ts";
import type { RagCommandHandler } from "./types.ts";

export const rebuildCommand: RagCommandHandler = async ({ ctx, args }) => {
  // Parse --force flag from any position after "rebuild".
  const force = args.includes("--force");

  const database = getDbConn();
  const config = loadConfig();
  const indexedFileSet = new Set(repo.listFilePaths(database));

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
    repo.deleteVectorsForFile(database, f);
    repo.deleteChunksForFile(database, f);
    repo.deleteFile(database, f);
  }
  if (force) {
    // --force: wipe everything and rebuild the FTS index. indexFiles
    // will then insert fresh rows for every targetFile, bypassing the
    // skip-on-equal-hash check.
    repo.wipeIndex(database);
  } else {
    for (const f of targetFiles) {
      repo.setFileEmbedded(database, f, false);
    }
  }

  const newFiles = targetFiles.filter(f => !indexedFileSet.has(f));
  ctx.ui.notify(`Rebuilding ${targetFiles.length} files${force ? " (forced)" : ""}...`, "info");
  if (droppedFiles.length) {
    ctx.ui.notify(`Pruned ${droppedFiles.length} files (deleted/excluded)`, "info");
  }
  if (newFiles.length) {
    ctx.ui.notify(`Discovered ${newFiles.length} new files`, "info");
  }

  // Yield so the TUI can paint the "Rebuilding" message before
  // indexFiles starts hammering the event loop.
  await new Promise<void>(r => setTimeout(r, 0));

  const result = await indexFiles(targetFiles, makeProgressCallbacks(ctx, "Rebuilding", "re-embedded", true), database, force);
  ctx.ui.setStatus("rag", undefined);
  ctx.ui.setWidget("rag", undefined);

  const secs = (result.durationMs / 1000).toFixed(1);
  ctx.ui.notify(`✅ Rebuilt: ${result.indexed} re-indexed · ${result.skipped} unchanged · ${droppedFiles.length} deleted · ${result.chunks} chunks · ${secs}s`, "info");
};
