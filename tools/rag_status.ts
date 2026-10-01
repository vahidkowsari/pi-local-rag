import { Type } from "typebox";
import { loadConfig } from "../config.ts";
import { getIndexStats } from "../db.ts";
import { getRagDir, GLOBAL_RAG_DIR } from "../store.ts";

export const ragStatusTool = {
  name: "rag_status",
  label: "RAG status",
  description: "Show pi-local-rag index statistics: file count, chunk count, vector coverage, embedding model, RAG config.",
  parameters: Type.Object({}),
  execute: async (_toolCallId: string) => {
    const stats = getIndexStats();
    const config = loadConfig();
    const text = JSON.stringify({
      files: stats.totalFiles,
      chunks: stats.totalChunks,
      vectorsEmbedded: stats.embeddedCount,
      vectorCoverage: stats.totalChunks ? `${Math.round(stats.embeddedCount / stats.totalChunks * 100)}%` : "0%",
      embeddingModel: stats.embeddingModel || "none",
      totalTokens: stats.totalTokens,
      lastBuild: stats.lastBuild || "never",
      ragConfig: config,
      storagePath: getRagDir(),
      storageScope: getRagDir() === GLOBAL_RAG_DIR() ? "global" : "project",
    }, null, 2);
    return { content: [{ type: "text" as const, text }], details: undefined };
  },
};
