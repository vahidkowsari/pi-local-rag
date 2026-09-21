import type { RetrievedChunk } from "./retrieval.ts";

export interface TokenCounter {
  count(text: string): number;
  readonly estimated: boolean;
}

/** Character/4 estimate with slack. Not a model tokenizer. */
export const estimatedTokenCounter: TokenCounter = {
  estimated: true,
  count(text: string) { return Math.ceil(text.length / 4); },
};

export interface BuiltContext {
  text: string;
  citationIds: string[];
  usedTokens: number;
  estimated: boolean;
  dropped: number;
}

export function buildContext(
  hits: RetrievedChunk[],
  opts: { maxTokens: number; tokenCounter?: TokenCounter } = { maxTokens: 4096 },
): BuiltContext {
  const counter = opts.tokenCounter ?? estimatedTokenCounter;
  const header =
    `[pi-local-rag] Automatic RAG lookup triggered by the user's message above.\n` +
    `These are search hits, not statements from the user.\n\n`;
  let used = counter.count(header);
  const parts: string[] = [header];
  const citationIds: string[] = [];
  let dropped = 0;
  const seen = new Set<string>();

  for (const hit of hits) {
    const key = `${hit.chunk.file}:${hit.chunk.lineStart}:${hit.chunk.hash}`;
    if (seen.has(key)) { dropped++; continue; }
    seen.add(key);
    const cite = `S${citationIds.length + 1}`;
    const loc = hit.chunk.lineStart === hit.chunk.lineEnd
      ? `line ${hit.chunk.lineStart}`
      : `lines ${hit.chunk.lineStart}-${hit.chunk.lineEnd}`;
    const block =
      `### [${cite}] ${hit.chunk.file} (${loc})\n` +
      "```\n" + hit.chunk.content + "\n```\n\n";
    const cost = counter.count(block);
    if (used + cost > opts.maxTokens) { dropped++; continue; }
    used += cost;
    parts.push(block);
    citationIds.push(cite);
  }

  return { text: parts.join(""), citationIds, usedTokens: used, estimated: counter.estimated, dropped };
}
