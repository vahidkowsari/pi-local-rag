import type { ExtensionCommandContext } from "@mariozechner/pi-coding-agent";

const WIDGET_KEY = "rag-info";
const WIDGET_TIMEOUT_MS = 45_000;

let activeTimer: ReturnType<typeof setTimeout> | undefined;

/**
 * Show a transient, read-only /rag result (search/status/find/ext/exclude/help).
 * All of these share one widget key so a new result replaces the previous one
 * instead of stacking, and auto-clears after WIDGET_TIMEOUT_MS so it doesn't
 * permanently eat screen space. Resets any pending timer from a prior call so
 * an old timeout can't clear a widget that replaced it.
 */
export function setInfoWidget(ctx: ExtensionCommandContext, lines: string[]) {
  if (activeTimer) {
    clearTimeout(activeTimer);
  }
  ctx.ui.setWidget(WIDGET_KEY, lines);
  activeTimer = setTimeout(() => {
    ctx.ui.setWidget(WIDGET_KEY, undefined);
    activeTimer = undefined;
  }, WIDGET_TIMEOUT_MS);
}
