import { basename, relative } from "node:path";
import ignore from "ignore";
import { getIndexedFiles } from "../db.ts";
import type { RagCommandHandler } from "./types.ts";
import { setInfoWidget } from "./widget.ts";

export const findCommand: RagCommandHandler = ({ ctx, args }) => {
  const glob = args.join(" ").trim();
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
    if (ig.ignores(candidate)) {
      matches.push(fp);
    }
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
  setInfoWidget(ctx, lines);
};
