import { loadConfig, saveConfig } from "../config.ts";
import type { RagCommandHandler } from "./types.ts";

function makeToggle(enabled: boolean): RagCommandHandler {
  return ({ ctx }) => {
    const config = loadConfig();
    config.ragEnabled = enabled;
    saveConfig(config);
    ctx.ui.notify(enabled ? "RAG auto-injection enabled" : "RAG auto-injection disabled", "info");
  };
}

export const onCommand: RagCommandHandler = makeToggle(true);
export const offCommand: RagCommandHandler = makeToggle(false);
