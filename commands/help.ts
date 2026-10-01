import { RAG_COMMAND_META } from "./metadata.ts";
import type { RagCommandHandler } from "./types.ts";
import { setInfoWidget } from "./widget.ts";

const pad = (s: string, n: number) => s + " ".repeat(Math.max(0, n - s.length));

export const helpCommand: RagCommandHandler = ({ ctx }) => {
  const COL = 36;
  const th = ctx.ui.theme;
  const lines: string[] = [th.bold("pi-local-rag commands"), ""];
  for (const { usage, helpText } of RAG_COMMAND_META) {
    lines.push("  " + th.fg("success", pad(usage, COL)) + "  " + th.fg("dim", helpText));
  }
  setInfoWidget(ctx, lines);
};
