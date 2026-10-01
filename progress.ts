import type { ExtensionCommandContext } from "@mariozechner/pi-coding-agent";
import { RST, B, D, GREEN, CYAN } from "./constants.ts";
import type { ProgressCallbacks } from "./indexing.ts";

/** Progress callback factory used by the index/rebuild/refresh commands. */
export function makeProgressCallbacks(
  ctx: ExtensionCommandContext,
  label: string,
  doneLabel: string,
  includeEmbed = false
): ProgressCallbacks {
  const progressBar = (n: number, total: number, width = 24): string => {
    const filled = Math.round((n / total) * width);
    return CYAN + "█".repeat(filled) + D + "░".repeat(width - filled) + RST;
  };

  return {
    onFile(current, total, filename, skipped) {
      const pct = Math.round((current / total) * 100);
      const bar = progressBar(current, total);
      ctx.ui.setStatus("rag", `■ ${label} ${pct}% │ ${current}/${total} │ ${skipped} unchanged`);
      ctx.ui.setWidget("rag", [
        `${B}${CYAN}${label}${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
        `${D}file:    ${RST}${filename}`,
        `${D}done:    ${RST}${GREEN}${current - skipped} ${doneLabel}${RST}  ${D}${skipped} unchanged${RST}`,
      ]);
    },
    onChunk(ci, total, filename) {
      ctx.ui.setStatus("rag", `■ Embedding ${filename} — chunk ${ci}/${total}`);
    },
    onSave() {
      ctx.ui.setStatus("rag", `■ Saving index...`);
    },
    ...(includeEmbed && {
      onEmbed(done: number, total: number) {
        const pct = Math.round((done / total) * 100);
        const bar = progressBar(done, total);
        ctx.ui.setStatus("rag", `■ Embedding ${pct}% │ ${done}/${total} chunks`);
        ctx.ui.setWidget("rag", [
          `${B}${CYAN}Embedding${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
          `${D}chunks:  ${RST}${done}/${total}`,
        ]);
      },
    }),
  };
}
