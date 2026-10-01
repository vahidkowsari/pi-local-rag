import { basename } from "node:path";
import { loadConfig } from "../config.ts";
import { getEmbeddedCount } from "../db.ts";
import { hybridSearch } from "../search.ts";
import type { RagCommandHandler } from "./types.ts";
import { setInfoWidget } from "./widget.ts";

export const searchCommand: RagCommandHandler = async ({ ctx, args }) => {
  const query = args.join(" ");
  if (!query) {
    ctx.ui.notify("Usage: /rag search <query>", "warning");
    return;
  }
  const config = loadConfig();
  const results = await hybridSearch(query, 10, config.ragAlpha);
  if (!results.length) {
    ctx.ui.notify(`No results for: ${query}`, "warning");
    return;
  }

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
  setInfoWidget(ctx, lines);
};
