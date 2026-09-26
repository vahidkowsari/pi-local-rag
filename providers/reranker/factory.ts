import { loadConfig, type RagConfig } from "../../config.ts";
import { NoneReranker } from "./none.ts";
import { VoyageReranker } from "./voyage.ts";
import type { Reranker } from "./types.ts";
import { getModelSpec, resolveModel } from "../../provider-config.ts";

export function configuredRerankerIsEnabled(config: RagConfig): boolean {
  try {
    return getModelSpec(config.reranker.provider, "rerank", config.reranker.model).type !== "noop";
  } catch {
    // Validation reports the actionable error. Retrieval must still be able
    // to return a degraded result instead of crashing before its own error
    // handling is installed.
    return true;
  }
}

/**
 * Resolve a reranker from provider.json.
 *
 * The provider key may be an alias such as "company-voyage"; `type` decides
 * which wire protocol implementation is used. This is why the factory does
 * not compare the configured name with the literal string "voyage".
 */
export function createReranker(config: RagConfig = loadConfig()): Reranker {
  const spec = resolveModel(config.reranker.provider, "rerank", config.reranker.model);

  if (spec.type === "noop") return new NoneReranker();
  if (spec.type === "voyage") {
    if (!spec.apiKey) throw new Error("The configured Voyage reranker has no API key");
    if (!spec.baseUrl) throw new Error("The configured Voyage reranker has no baseUrl");
    return new VoyageReranker({
      id: spec.providerId,
      baseUrl: spec.baseUrl,
      model: spec.model,
      apiKey: spec.apiKey,
      timeoutMs: config.http.timeoutMs,
      maxRetries: config.http.maxRetries,
    });
  }
  throw new Error(`Provider type "${spec.type}" cannot create rerankers`);
}
