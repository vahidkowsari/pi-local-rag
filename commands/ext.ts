import { loadConfig, saveConfig, normalizeExt, resolveExtensions } from "../config.ts";
import type { RagCommandHandler } from "./types.ts";
import { setInfoWidget } from "./widget.ts";

export const extCommand: RagCommandHandler = ({ ctx, args }) => {
  const sub = (args[0] || "list").toLowerCase();
  const config = loadConfig();

  if (sub === "list") {
    const th = ctx.ui.theme;
    const active = Array.from(resolveExtensions(config)).sort();
    const lines: string[] = [
      th.bold("Active file extensions") + "  " + th.fg("dim", `(${active.length})`),
      th.fg("muted", "  " + active.join(" ")),
    ];
    if (config.extraExtensions.length) {
      lines.push("  " + th.fg("dim", "extra:   ") + th.fg("success", config.extraExtensions.join(" ")));
    }
    if (config.excludeExtensions.length) {
      lines.push("  " + th.fg("dim", "excluded:") + " " + th.fg("warning", config.excludeExtensions.join(" ")));
    }
    lines.push("", th.fg("dim", "Edit via /rag ext add <.ext> / remove <.ext> / reset"));
    setInfoWidget(ctx, lines);
    return;
  }

  if (sub === "add") {
    const ext = normalizeExt(args[1] || "");
    if (!ext) {
      ctx.ui.notify("Usage: /rag ext add <.ext>", "warning");
      return;
    }
    config.excludeExtensions = config.excludeExtensions.filter(e => normalizeExt(e) !== ext);
    if (!config.extraExtensions.map(normalizeExt).includes(ext)) {
      config.extraExtensions.push(ext);
    }
    saveConfig(config);
    ctx.ui.notify(`Added ${ext} to indexable extensions. Run /rag index <path> to pick up matching files.`, "info");
    return;
  }

  if (sub === "remove" || sub === "rm") {
    const ext = normalizeExt(args[1] || "");
    if (!ext) {
      ctx.ui.notify("Usage: /rag ext remove <.ext>", "warning");
      return;
    }
    const wasExtra = config.extraExtensions.map(normalizeExt).includes(ext);
    config.extraExtensions = config.extraExtensions.filter(e => normalizeExt(e) !== ext);
    if (!wasExtra && !config.excludeExtensions.map(normalizeExt).includes(ext)) {
      config.excludeExtensions.push(ext);
    }
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
};
