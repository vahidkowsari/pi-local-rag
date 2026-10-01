import type { ExtensionCommandContext } from "@mariozechner/pi-coding-agent";

/**
 * Everything a /rag subcommand handler needs. Handlers pull whatever else
 * they require (config, db connection, etc.) directly from the relevant
 * module — none of that is shared mutable state closed over by register(),
 * so there's nothing else to thread through here.
 */
export interface RagCommandArgs {
  ctx: ExtensionCommandContext;
  /** Everything after the subcommand name, e.g. "/rag ext add .foo" → ["add", ".foo"] */
  args: string[];
}

export type RagCommandHandler = (a: RagCommandArgs) => Promise<void> | void;

export interface RagCommandDef {
  label: string;
  description: string;
  /** Usage string shown in /rag help, e.g. "/rag ext list|add|remove|reset" */
  usage: string;
  handler: RagCommandHandler;
}
