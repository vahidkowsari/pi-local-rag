import { loadConfig, saveConfig } from "../config.ts";
import type { RagCommandHandler } from "./types.ts";
import { setInfoWidget } from "./widget.ts";

export const excludeCommand: RagCommandHandler = ({ ctx, args }) => {
  const config = loadConfig();
  const expr = args.join(" ").trim();
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
    for (const p of config.excludePatterns) {
      lines.push("  " + th.fg("muted", p));
    }
    setInfoWidget(ctx, lines);
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
};
