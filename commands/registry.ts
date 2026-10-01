import { RAG_COMMAND_META } from "./metadata.ts";
import type { RagCommandDef, RagCommandHandler } from "./types.ts";
import { indexCommand } from "./indexCmd.ts";
import { searchCommand } from "./search.ts";
import { onCommand, offCommand } from "./toggle.ts";
import { rebuildCommand } from "./rebuild.ts";
import { refreshCommand } from "./refresh.ts";
import { extCommand } from "./ext.ts";
import { clearCommand } from "./clear.ts";
import { excludeCommand } from "./exclude.ts";
import { findCommand } from "./find.ts";
import { helpCommand } from "./help.ts";
import { statusCommand } from "./status.ts";

const HANDLERS: Record<string, RagCommandHandler> = {
  index: indexCommand,
  search: searchCommand,
  on: onCommand,
  off: offCommand,
  rebuild: rebuildCommand,
  refresh: refreshCommand,
  ext: extCommand,
  clear: clearCommand,
  exclude: excludeCommand,
  find: findCommand,
  help: helpCommand,
  status: statusCommand,
};

/** Top-level /rag command dispatch map: name → { label, description, usage, handler }. */
export const RAG_COMMANDS: Record<string, RagCommandDef> = 
  Object.fromEntries(RAG_COMMAND_META.map(({ name, label, description, usage }) => [
    name,
    { label, description, usage, handler: HANDLERS[name] },
  ])
);
