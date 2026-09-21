import { assertValidConfig, loadConfig, voyageApiKey, type RagConfig } from "../../config.ts";
import { LocalEmbeddingProvider, getLocalEmbeddingProvider } from "./local.ts";
import { VoyageEmbeddingProvider } from "./voyage.ts";
import type { EmbeddingProvider } from "./types.ts";
import type Database from "better-sqlite3";
import * as repo from "../../repository.ts";
import { parseEmbeddingFingerprint } from "../../fingerprint.ts";

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

/** Provider matching the ACTIVE index fingerprint (query and incremental writes). */
export function embeddingProviderForIndex(db: Database.Database, config: RagConfig = loadConfig()): EmbeddingProvider {
  const stored = parseEmbeddingFingerprint(repo.getMetadata(db, repo.MetadataKey.EmbeddingFingerprint));
  if (!stored) return getLocalEmbeddingProvider();
  if (stored.provider === "local") {
    return new LocalEmbeddingProvider({ model: stored.model, dimensions: stored.dimensions });
  }
  if (stored.provider === "voyage") {
    const apiKey = voyageApiKey();
    if (!apiKey) throw new Error("Active index uses Voyage embeddings but VOYAGE_API_KEY is not set");
    return new VoyageEmbeddingProvider({
      model: stored.model,
      dimensions: stored.dimensions,
      apiKey,
      timeoutMs: config.http.timeoutMs,
      maxRetries: config.http.maxRetries,
    });
  }
  throw new Error(`Active index has unsupported embedding provider "${stored.provider}"`);
}
