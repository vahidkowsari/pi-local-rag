import { extname } from "node:path";
import { loadConfig } from "../config.ts";
import { getIndexedFiles, getIndexStats } from "../db.ts";
import { getRagDir, GLOBAL_RAG_DIR } from "../store.ts";
import type { RagCommandHandler } from "./types.ts";
import { setInfoWidget } from "./widget.ts";

export const statusCommand: RagCommandHandler = ({ ctx }) => {
  const indexStats = getIndexStats();
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
    const files = getIndexedFiles();
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

  setInfoWidget(ctx, lines);
};
