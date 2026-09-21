import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { configFile, getRagDir } from "./store.ts";
import { DEFAULT_TEXT_EXTS, EMBEDDING_MODEL, VECTOR_DIM } from "./constants.ts";

export type EmbeddingProviderId = "local" | "voyage";
export type RerankerProviderId = "none" | "voyage";

export interface EmbeddingConfig {
  provider: EmbeddingProviderId;
  model: string;
  dimensions: number;
}

export interface RerankerConfig {
  provider: RerankerProviderId;
  model: string;
}

export interface HttpConfig {
  timeoutMs: number;
  maxRetries: number;
}

export interface RagConfig {
  ragEnabled: boolean;
  ragTopK: number;
  ragScoreThreshold: number;
  ragAlpha: number; // 0 = pure vector, 1 = pure BM25
  extraExtensions: string[];
  excludeExtensions: string[];
  trackedPaths: string[];
  excludePatterns: string[];
  embedding: EmbeddingConfig;
  reranker: RerankerConfig;
  candidateTopK: number;
  maxContextTokens: number;
  http: HttpConfig;
  /** Cloud-mode auto-refresh. Default off; local mode keeps the 24h refresh. */
  cloudAutoRefresh: boolean;
}

export const DEFAULT_VOYAGE_EMBED_MODEL = "voyage-4-lite";
export const DEFAULT_VOYAGE_EMBED_DIM = 1024;
export const DEFAULT_VOYAGE_RERANK_MODEL = "rerank-2.5-lite";
export const CANDIDATE_TOP_K_MAX = 200;

export function defaultConfig(): RagConfig {
  return {
    ragEnabled: true, ragTopK: 5, ragScoreThreshold: 0.1, ragAlpha: 0.4,
    extraExtensions: [], excludeExtensions: [],
    trackedPaths: [], excludePatterns: [],
    embedding: { provider: "local", model: EMBEDDING_MODEL, dimensions: VECTOR_DIM },
    reranker: { provider: "none", model: DEFAULT_VOYAGE_RERANK_MODEL },
    candidateTopK: 30,
    maxContextTokens: 4096,
    http: { timeoutMs: 30_000, maxRetries: 3 },
    cloudAutoRefresh: false,
  };
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function mergeSaved(raw: unknown): RagConfig {
  const d = defaultConfig();
  if (!isPlainObject(raw)) return d;
  const embedding = isPlainObject(raw.embedding) ? { ...d.embedding, ...raw.embedding } : d.embedding;
  const reranker = isPlainObject(raw.reranker) ? { ...d.reranker, ...raw.reranker } : d.reranker;
  const http = isPlainObject(raw.http) ? { ...d.http, ...raw.http } : d.http;
  const { embedding: _e, reranker: _r, http: _h, ...rest } = raw;
  return { ...d, ...rest, embedding, reranker, http } as RagConfig;
}

function readSavedConfig(): RagConfig {
  const cfgFile = configFile(getRagDir());
  if (!existsSync(cfgFile)) return defaultConfig();
  try {
    return mergeSaved(JSON.parse(readFileSync(cfgFile, "utf-8")));
  } catch {
    return defaultConfig();
  }
}

function envString(name: string): string | undefined {
  const v = process.env[name];
  return v !== undefined && v !== "" ? v : undefined;
}

function envInt(name: string): number | undefined {
  const v = envString(name);
  if (v === undefined) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function envBool(name: string): boolean | undefined {
  const v = envString(name)?.toLowerCase();
  if (v === undefined) return undefined;
  if (v === "1" || v === "true" || v === "yes") return true;
  if (v === "0" || v === "false" || v === "no") return false;
  return undefined;
}

/** Apply PI_RAG_* (and only those) env overrides onto a config object. */
export function applyEnvOverrides(config: RagConfig): RagConfig {
  const c: RagConfig = {
    ...config,
    embedding: { ...config.embedding },
    reranker: { ...config.reranker },
    http: { ...config.http },
  };
  const provider = envString("PI_RAG_EMBEDDING_PROVIDER");
  if (provider) c.embedding.provider = provider as EmbeddingProviderId;
  const model = envString("PI_RAG_EMBEDDING_MODEL");
  if (model) c.embedding.model = model;
  const dim = envInt("PI_RAG_EMBEDDING_DIMENSIONS");
  if (dim !== undefined) c.embedding.dimensions = dim;
  const rp = envString("PI_RAG_RERANKER_PROVIDER");
  if (rp) c.reranker.provider = rp as RerankerProviderId;
  const rm = envString("PI_RAG_RERANKER_MODEL");
  if (rm) c.reranker.model = rm;
  const topk = envInt("PI_RAG_CANDIDATE_TOP_K");
  if (topk !== undefined) c.candidateTopK = topk;
  const ctx = envInt("PI_RAG_MAX_CONTEXT_TOKENS");
  if (ctx !== undefined) c.maxContextTokens = ctx;
  const timeout = envInt("PI_RAG_HTTP_TIMEOUT_MS");
  if (timeout !== undefined) c.http.timeoutMs = timeout;
  const retries = envInt("PI_RAG_HTTP_MAX_RETRIES");
  if (retries !== undefined) c.http.maxRetries = retries;
  const cloud = envBool("PI_RAG_CLOUD_AUTO_REFRESH");
  if (cloud !== undefined) c.cloudAutoRefresh = cloud;
  return c;
}

function stripSecrets(config: RagConfig): RagConfig {
  const copy = structuredClone(config) as RagConfig & { voyageApiKey?: unknown; apiKey?: unknown };
  delete copy.voyageApiKey;
  delete copy.apiKey;
  return copy;
}

/**
 * Persist config without writing env-only overlays or API keys.
 * Fields currently set via PI_RAG_* keep their previously saved values.
 */
export function saveConfig(config: RagConfig) {
  const saved = readSavedConfig();
  const next = stripSecrets({
    ...config,
    embedding: { ...config.embedding },
    reranker: { ...config.reranker },
    http: { ...config.http },
  });
  if (envString("PI_RAG_EMBEDDING_PROVIDER")) next.embedding.provider = saved.embedding.provider;
  if (envString("PI_RAG_EMBEDDING_MODEL")) next.embedding.model = saved.embedding.model;
  if (envString("PI_RAG_EMBEDDING_DIMENSIONS")) next.embedding.dimensions = saved.embedding.dimensions;
  if (envString("PI_RAG_RERANKER_PROVIDER")) next.reranker.provider = saved.reranker.provider;
  if (envString("PI_RAG_RERANKER_MODEL")) next.reranker.model = saved.reranker.model;
  if (envString("PI_RAG_CANDIDATE_TOP_K")) next.candidateTopK = saved.candidateTopK;
  if (envString("PI_RAG_MAX_CONTEXT_TOKENS")) next.maxContextTokens = saved.maxContextTokens;
  if (envString("PI_RAG_HTTP_TIMEOUT_MS")) next.http.timeoutMs = saved.http.timeoutMs;
  if (envString("PI_RAG_HTTP_MAX_RETRIES")) next.http.maxRetries = saved.http.maxRetries;
  if (envString("PI_RAG_CLOUD_AUTO_REFRESH")) next.cloudAutoRefresh = saved.cloudAutoRefresh;
  writeFileSync(configFile(getRagDir()), JSON.stringify(next, null, 2));
}

export function loadConfig(): RagConfig {
  return applyEnvOverrides(readSavedConfig());
}

export function voyageApiKey(): string | undefined {
  return envString("VOYAGE_API_KEY");
}

export function validateConfig(config: RagConfig): string[] {
  const issues: string[] = [];
  if (config.embedding.provider !== "local" && config.embedding.provider !== "voyage") {
    issues.push(`Unsupported embedding provider "${config.embedding.provider}". Use "local" or "voyage".`);
  }
  if (config.reranker.provider !== "none" && config.reranker.provider !== "voyage") {
    issues.push(`Unsupported reranker provider "${config.reranker.provider}". Use "none" or "voyage".`);
  }
  if (!Number.isInteger(config.embedding.dimensions) || config.embedding.dimensions <= 0) {
    issues.push(`embedding.dimensions must be a positive integer, got ${config.embedding.dimensions}.`);
  }
  if (!config.embedding.model || typeof config.embedding.model !== "string") {
    issues.push("embedding.model is required.");
  }
  if (config.embedding.provider === "voyage") {
    const allowed = [256, 512, 1024, 2048];
    if (!allowed.includes(config.embedding.dimensions)) {
      issues.push(`voyage embedding dimensions must be one of ${allowed.join(", ")} (got ${config.embedding.dimensions}).`);
    }
    if (!voyageApiKey()) {
      issues.push("VOYAGE_API_KEY is not set. Export it in the environment; it is never stored in config.json.");
    }
  }
  if (!Number.isInteger(config.ragTopK) || config.ragTopK < 1) {
    issues.push(`ragTopK must be a positive integer, got ${config.ragTopK}.`);
  }
  if (!Number.isInteger(config.candidateTopK) || config.candidateTopK < 1) {
    issues.push(`candidateTopK must be a positive integer, got ${config.candidateTopK}.`);
  } else if (config.candidateTopK > CANDIDATE_TOP_K_MAX) {
    issues.push(`candidateTopK ${config.candidateTopK} exceeds the limit of ${CANDIDATE_TOP_K_MAX}.`);
  } else if (config.candidateTopK < config.ragTopK) {
    issues.push(`candidateTopK (${config.candidateTopK}) must be >= ragTopK (${config.ragTopK}).`);
  }
  if (!Number.isInteger(config.maxContextTokens) || config.maxContextTokens < 64) {
    issues.push(`maxContextTokens must be an integer >= 64, got ${config.maxContextTokens}.`);
  }
  if (!Number.isInteger(config.http.timeoutMs) || config.http.timeoutMs < 1000) {
    issues.push(`http.timeoutMs must be an integer >= 1000, got ${config.http.timeoutMs}.`);
  }
  if (!Number.isInteger(config.http.maxRetries) || config.http.maxRetries < 0 || config.http.maxRetries > 8) {
    issues.push(`http.maxRetries must be an integer 0–8, got ${config.http.maxRetries}.`);
  }
  return issues;
}

/** Throw if config cannot be used for the requested provider construction. */
export function assertValidConfig(config: RagConfig): void {
  const issues = validateConfig(config);
  if (issues.length) throw new Error(issues.join("\n"));
}

/** Normalize a user-supplied extension to lowercase ".ext" form. */
export function normalizeExt(ext: string): string {
  const trimmed = ext.trim().toLowerCase();
  if (!trimmed) return "";
  return trimmed.startsWith(".") ? trimmed : `.${trimmed}`;
}

/** Build the effective extension allowlist from defaults + user config. */
export function resolveExtensions(config: Pick<RagConfig, "extraExtensions" | "excludeExtensions">): Set<string> {
  const set = new Set(DEFAULT_TEXT_EXTS);
  for (const e of config.extraExtensions) {
    const n = normalizeExt(e);
    if (n) set.add(n);
  }
  for (const e of config.excludeExtensions) {
    const n = normalizeExt(e);
    if (n) set.delete(n);
  }
  return set;
}
