import { assertValidConfig, loadConfig, type RagConfig } from "../../config.ts";
import { LocalEmbeddingProvider, getLocalEmbeddingProvider } from "./local.ts";
import { VoyageEmbeddingProvider } from "./voyage.ts";
import type { EmbeddingProvider } from "./types.ts";
import type Database from "better-sqlite3";
import * as repo from "../../repository.ts";
import { parseEmbeddingFingerprint } from "../../fingerprint.ts";
import { IndexIncompatibleError } from "../../index-manager.ts";
import { EMBEDDING_MODEL, VECTOR_DIM } from "../../constants.ts";
import { getModelSpec, resolveModel } from "../../provider-config.ts";

const providerCache = new Map<string, EmbeddingProvider>();

/**
 * Cache only reusable contracts. The registry key, protocol type, endpoint,
 * model, dimensions, credential, and HTTP policy are all part of the key.
 * Including the endpoint prevents a changed proxy from reusing an object
 * created for the previous provider configuration.
 */
function cacheKey(opts: {
  id: string;
  type: string;
  baseUrl?: string;
  model: string;
  dimensions: number;
  apiKey?: string;
  http: { timeoutMs: number; maxRetries: number };
}): string {
  return [
    opts.id,
    opts.type,
    opts.baseUrl ?? "",
    opts.model,
    opts.dimensions,
    opts.apiKey ?? "",
    opts.http.timeoutMs,
    opts.http.maxRetries,
  ].join("|");
}

function buildProvider(opts: {
  id: string;
  type: string;
  baseUrl?: string;
  model: string;
  dimensions: number;
  timeoutMs: number;
  maxRetries: number;
  apiKey?: string;
}): EmbeddingProvider {
  if (opts.type === "transformers") {
    // Preserve the existing singleton for the canonical local model. Custom
    // registry aliases can still use the same adapter without changing code.
    if (opts.id === "local" && opts.model === EMBEDDING_MODEL && opts.dimensions === VECTOR_DIM) {
      return getLocalEmbeddingProvider();
    }
    return new LocalEmbeddingProvider({ model: opts.model, dimensions: opts.dimensions });
  }

  if (opts.type === "voyage") {
    if (!opts.apiKey) throw new Error("The configured Voyage embedding provider has no API key");
    if (!opts.baseUrl) throw new Error("The configured Voyage embedding provider has no baseUrl");
    return new VoyageEmbeddingProvider({
      id: opts.id,
      baseUrl: opts.baseUrl,
      model: opts.model,
      dimensions: opts.dimensions,
      apiKey: opts.apiKey,
      timeoutMs: opts.timeoutMs,
      maxRetries: opts.maxRetries,
    });
  }

  throw new Error(`Provider type "${opts.type}" cannot create embeddings`);
}

function cached(key: string, build: () => EmbeddingProvider): EmbeddingProvider {
  const hit = providerCache.get(key);
  if (hit) return hit;
  const created = build();
  providerCache.set(key, created);
  return created;
}

/** Build an embedding provider from config plus the active provider registry. */
export function createEmbeddingProvider(config: RagConfig = loadConfig()): EmbeddingProvider {
  assertValidConfig(config);
  const spec = resolveModel(config.embedding.provider, "embedding", config.embedding.model);
  const dimensions = spec.dimensions;
  if (dimensions === undefined) {
    throw new Error(`Embedding model ${spec.providerId}/${spec.model} has no dimensions in provider.json`);
  }
  const key = cacheKey({
    id: spec.providerId,
    type: spec.type,
    baseUrl: spec.baseUrl,
    model: spec.model,
    dimensions,
    apiKey: spec.apiKey,
    http: config.http,
  });
  return cached(key, () => buildProvider({
    id: spec.providerId,
    type: spec.type,
    baseUrl: spec.baseUrl,
    model: spec.model,
    dimensions,
    timeoutMs: config.http.timeoutMs,
    maxRetries: config.http.maxRetries,
    apiKey: spec.apiKey,
  }));
}

/** Provider matching the ACTIVE index fingerprint (query and incremental writes). */
export function embeddingProviderForIndex(db: Database.Database, config: RagConfig = loadConfig()): EmbeddingProvider {
  const stored = parseEmbeddingFingerprint(repo.getMetadata(db, repo.MetadataKey.EmbeddingFingerprint));
  if (!stored) {
    throw new IndexIncompatibleError(
      "Existing index has no fingerprint. Run /rag rebuild --force to rebuild under the current embedding contract.",
    );
  }

  // Resolve the historical provider/model through the same registry. If a
  // provider was removed or renamed, the error is explicit and the caller
  // can rebuild under the current contract instead of guessing credentials.
  let spec: ReturnType<typeof getModelSpec>;
  try {
    spec = getModelSpec(stored.provider, "embedding", stored.model);
  } catch (error) {
    throw new IndexIncompatibleError(
      `Active index provider ${stored.provider}/${stored.model} is no longer available in provider.json: ` +
      `${error instanceof Error ? error.message : String(error)} Rebuild required.`,
    );
  }
  if (spec.dimensions !== stored.dimensions) {
    throw new IndexIncompatibleError(
      `Active index uses ${stored.provider}/${stored.model}/${stored.dimensions}, but provider.json now declares ${spec.dimensions} dimensions. Rebuild required.`,
    );
  }
  const resolved = resolveModel(stored.provider, "embedding", stored.model);
  const key = cacheKey({
    id: resolved.providerId,
    type: resolved.type,
    baseUrl: resolved.baseUrl,
    model: resolved.model,
    dimensions: stored.dimensions,
    apiKey: resolved.apiKey,
    http: config.http,
  });
  return cached(key, () => buildProvider({
    id: resolved.providerId,
    type: resolved.type,
    baseUrl: resolved.baseUrl,
    model: resolved.model,
    dimensions: stored.dimensions,
    timeoutMs: config.http.timeoutMs,
    maxRetries: config.http.maxRetries,
    apiKey: resolved.apiKey,
  }));
}

/** Test-only: drop cached providers so a new contract can be constructed. */
export function resetEmbeddingProviderCache(): void {
  providerCache.clear();
}
