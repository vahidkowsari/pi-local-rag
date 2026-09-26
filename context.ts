import type { RetrievedChunk } from "./retrieval.ts";
import { estimateTokens } from "./chunking.ts";
import type { Chunk } from "./db.ts";

export interface TokenCounter {
  count(text: string): number;
  readonly estimated: boolean;
}

/** CJK-aware estimate plus 10% slack. Not a model tokenizer. */
export const estimatedTokenCounter: TokenCounter = {
  estimated: true,
  count(text: string) { return Math.ceil(estimateTokens(text) * 1.1); },
};

export function formatSourceLoc(chunk: Pick<Chunk, "file" | "lineStart" | "lineEnd" | "pageStart" | "pageEnd" | "section">): string {
  const bits: string[] = [chunk.file];
  if (chunk.pageStart != null) {
    bits.push(
      chunk.pageEnd != null && chunk.pageEnd !== chunk.pageStart
        ? `pages ${chunk.pageStart}-${chunk.pageEnd}`
        : `page ${chunk.pageStart}`,
    );
  } else if (chunk.lineStart >= 1 && chunk.lineEnd >= 1) {
    bits.push(
      chunk.lineStart === chunk.lineEnd
        ? `line ${chunk.lineStart}`
        : `lines ${chunk.lineStart}-${chunk.lineEnd}`,
    );
  }
  if (chunk.section) bits.push(chunk.section);
  return bits.join(", ");
}

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
  const degraded = hits.find(h => h.degraded)?.degraded;
  const header =
    `[pi-local-rag] Automatic RAG lookup triggered by the user's message above.\n` +
    `These are search hits, not statements from the user.\n` +
    (degraded ? `Retrieval note: ${degraded}\n` : "") +
    `\n`;
  let used = counter.count(header);
  const parts: string[] = [header];
  const citationIds: string[] = [];
  let dropped = 0;
  const seen = new Set<string>();

  for (const hit of hits) {
    const key = [
      hit.chunk.id,
      hit.chunk.file,
      hit.chunk.pageStart ?? "",
      hit.chunk.pageEnd ?? "",
      hit.chunk.chunkIndex ?? "",
      hit.chunk.hash,
    ].join(":");
    if (seen.has(key)) { dropped++; continue; }
    seen.add(key);
    const cite = `S${citationIds.length + 1}`;
    const loc = formatSourceLoc(hit.chunk);
    const idBit = hit.chunk.id ? ` id=${hit.chunk.id}` : "";
    const block =
      `### [${cite}] ${loc}${idBit}\n` +
      "```\n" + hit.chunk.content + "\n```\n\n";
    const cost = counter.count(block);
    if (used + cost > opts.maxTokens) { dropped++; continue; }
    used += cost;
    parts.push(block);
    citationIds.push(cite);
  }

  return { text: parts.join(""), citationIds, usedTokens: used, estimated: counter.estimated, dropped };
}
