import { Type, type Static } from "typebox";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { getRagDir } from "../store.ts";
import { loadConfig, saveConfig } from "../config.ts";
import { collectFiles } from "../chunking.ts";
import { indexFiles } from "../indexing.ts";

const parameters = Type.Object({
  path: Type.String({ description: "File or directory path to index" }),
});

export const ragIndexTool = {
  name: "rag_index",
  label: "RAG index",
  description: "Index a file or directory into the local pi-local-rag pipeline. Chunks text files (including PDF and DOCX), generates embeddings, stores for hybrid BM25+vector search.",
  parameters,
  execute: async (_toolCallId: string, params: Static<typeof parameters>) => {
    if (!existsSync(params.path)) return { content: [{ type: "text" as const, text: `Path not found: ${params.path}` }], details: undefined };
    // Anchor a project-local store at cwd if there isn't one in scope yet.
    getRagDir({ createIfMissing: true });
    const config = loadConfig();
    const absPath = resolve(params.path);
    if (!config.trackedPaths.includes(absPath)) {
      config.trackedPaths.push(absPath);
      saveConfig(config);
    }
    const files = collectFiles(absPath, undefined, config.excludePatterns);
    if (!files.length) return { content: [{ type: "text" as const, text: `No indexable files found in: ${params.path}` }], details: undefined };
    const result = await indexFiles(files, {});
    process.stderr.write(`\n`);
    return { content: [{ type: "text" as const, text: `Indexed ${result.indexed} files (${result.chunks} chunks, embeddings generated). ${result.skipped} unchanged. ${(result.durationMs / 1000).toFixed(1)}s` }], details: undefined };
  },
};
