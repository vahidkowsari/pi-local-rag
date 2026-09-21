import { assertValidConfig, loadConfig, voyageApiKey, type RagConfig } from "../../config.ts";
import { LocalEmbeddingProvider } from "./local.ts";
import { VoyageEmbeddingProvider } from "./voyage.ts";
import type { EmbeddingProvider } from "./types.ts";

/**
 * Build an embedding provider from config. Indexing/query production paths
 * stay on the local facade until Phase C wires this factory in.
 */
export function createEmbeddingProvider(config: RagConfig = loadConfig()): EmbeddingProvider {
  assertValidConfig(config);
  if (config.embedding.provider === "local") {
    return new LocalEmbeddingProvider({
      model: config.embedding.model,
      dimensions: config.embedding.dimensions,
    });
  }
  if (config.embedding.provider === "voyage") {
    return new VoyageEmbeddingProvider({
      model: config.embedding.model,
      dimensions: config.embedding.dimensions,
      apiKey: voyageApiKey() ?? "",
      timeoutMs: config.http.timeoutMs,
      maxRetries: config.http.maxRetries,
    });
  }
  throw new Error(`Unsupported embedding provider: ${config.embedding.provider}`);
}
