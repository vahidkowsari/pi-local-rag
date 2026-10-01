import { existsSync } from "node:fs";
import { loadConfig } from "../config.ts";
import { getIndexedFiles } from "../db.ts";
import { collectFromTracked } from "../chunking.ts";
import { indexFiles } from "../indexing.ts";
import { makeProgressCallbacks } from "../progress.ts";
import type { RagCommandHandler } from "./types.ts";

export const refreshCommand: RagCommandHandler = async ({ ctx }) => {
  const config = loadConfig();
  const filesFromDb = getIndexedFiles();
  const files = config.trackedPaths.length
    ? collectFromTracked(config)
    : filesFromDb.map(f => f.path).filter(f => existsSync(f));
  if (!files.length) {
    ctx.ui.notify("No tracked files to refresh. Run /rag index <path> first.", "warning");
    return;
  }

  ctx.ui.notify(`Refreshing ${files.length} files...`, "info");

  const result = await indexFiles(files, makeProgressCallbacks(ctx, "Refreshing", "new/changed"));
  ctx.ui.setStatus("rag", undefined);
  ctx.ui.setWidget("rag", undefined);

  const secs = (result.durationMs / 1000).toFixed(1);
  ctx.ui.notify(`✅ Refreshed ${result.indexed} new/changed · ${result.skipped} unchanged · ${result.chunks} chunks · ${secs}s`, "info");
};
