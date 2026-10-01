import { Type, type Static } from "typebox";
import { loadConfig } from "../config.ts";
import { getIndexStats } from "../db.ts";
import { hybridSearch } from "../search.ts";

const parameters = Type.Object({
  query: Type.String({ description: "Search query" }),
  limit: Type.Optional(Type.Number({ description: "Max results (default 10)" })),
});

export const ragQueryTool = {
  name: "rag_query",
  label: "RAG query",
  description: "Search the local pi-local-rag index using hybrid BM25+vector search. Returns relevant chunks with file paths, line numbers, and relevance scores.",
  parameters,
  execute: async (_toolCallId: string, params: Static<typeof parameters>) => {
    if (getIndexStats().totalChunks === 0) return { content: [{ type: "text" as const, text: "pi-local-rag index is empty. Run rag_index first." }], details: undefined };
    const config = loadConfig();
    const results = await hybridSearch(params.query, params.limit ?? 10, config.ragAlpha);
    if (!results.length) return { content: [{ type: "text" as const, text: `No results for: ${params.query}` }], details: undefined };
    const text = JSON.stringify(results.map(r => ({
      file: r.chunk.file,
      lines: `${r.chunk.lineStart}-${r.chunk.lineEnd}`,
      tokens: r.chunk.tokens,
      scores: { bm25: r.bm25.toFixed(3), vector: r.vector.toFixed(3), hybrid: r.hybrid.toFixed(3) },
      preview: r.chunk.content.slice(0, 300),
    })), null, 2);
    return { content: [{ type: "text" as const, text }], details: undefined };
  },
};
