import { loadConfig, voyageApiKey, type RagConfig } from "../../config.ts";
import { NoneReranker } from "./none.ts";
import { VoyageReranker } from "./voyage.ts";
import type { Reranker } from "./types.ts";

export function createReranker(config: RagConfig = loadConfig()): Reranker {
  if (config.reranker.provider === "none") return new NoneReranker();
  if (config.reranker.provider === "voyage") {
    const apiKey = voyageApiKey();
    if (!apiKey) throw new Error("VOYAGE_API_KEY is required for the voyage reranker");
    return new VoyageReranker({
      model: config.reranker.model,
      apiKey,
      timeoutMs: config.http.timeoutMs,
      maxRetries: config.http.maxRetries,
    });
  }
  throw new Error(`Unsupported reranker provider "${config.reranker.provider}"`);
}
