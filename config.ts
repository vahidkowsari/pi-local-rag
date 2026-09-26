import { copyFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { configFile, getRagDir } from "./store.ts";
import {
  DEFAULT_TEXT_EXTS, EMBEDDING_MODEL, VECTOR_DIM,
} from "./constants.ts";
import {
  ProviderConfigError, getModelSpec, loadProviderFile,
  type ProviderFile,
} from "./provider-config.ts";

/**
 * Provider IDs are registry keys, not protocol names. For example, a custom
 * proxy may be called "my-voyage" while still using `type: "voyage"`.
 * The string type keeps provider.json open-ended without changing the shape
 * of the persisted config.
 */
export type EmbeddingProviderId = string;
export type RerankerProviderId = string;

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

/**
 * Legacy direct-construction fallbacks. Normal Extension resolution gets all
 * provider/model metadata from provider.json; these values only keep the
 * standalone Voyage classes backwards compatible for library consumers.
 */
export const DEFAULT_VOYAGE_EMBED_MODEL = "voyage-4-lite";
export const DEFAULT_VOYAGE_EMBED_DIM = 1024;
export const DEFAULT_VOYAGE_RERANK_MODEL = "rerank-3-lite";
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

function readSavedConfig(): { config: RagConfig; fileStatus: ConfigLoadResult["fileStatus"]; issue?: string } {
  const cfgFile = configFile(getRagDir());
  if (!existsSync(cfgFile)) return { config: defaultConfig(), fileStatus: "missing" };
  try {
    const raw = JSON.parse(readFileSync(cfgFile, "utf-8"));
    if (!isPlainObject(raw)) {
      return {
        config: defaultConfig(),
        fileStatus: "invalid",
        issue: `config.json is not a JSON object. Using defaults until it is repaired or replaced.`,
      };
    }
    return { config: mergeSaved(raw), fileStatus: "ok" };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      config: defaultConfig(),
      fileStatus: "invalid",
      issue: `config.json is invalid JSON (${msg}). Using defaults until it is repaired or replaced.`,
    };
  }
}

function envString(name: string): string | undefined {
  const v = process.env[name];
  return v !== undefined && v !== "" ? v : undefined;
}

const envParseIssues: string[] = [];

function noteEnvIssue(msg: string) {
  envParseIssues.push(msg);
}

function envInt(name: string): number | undefined {
  const v = envString(name);
  if (v === undefined) return undefined;
  const n = Number(v);
  if (!Number.isFinite(n)) {
    noteEnvIssue(`${name}=${JSON.stringify(v)} is not a number`);
    return undefined;
  }
  return n;
}

function envBool(name: string): boolean | undefined {
  const v = envString(name)?.toLowerCase();
  if (v === undefined) return undefined;
  if (v === "1" || v === "true" || v === "yes") return true;
  if (v === "0" || v === "false" || v === "no") return false;
  noteEnvIssue(`${name}=${JSON.stringify(process.env[name])} is not a boolean`);
  return undefined;
}

export interface ConfigLoadResult {
  config: RagConfig;
  issues: string[];
  /** "missing" = no file; "ok" = parsed; "invalid" = broken JSON or non-object. */
  fileStatus: "missing" | "ok" | "invalid";
}

let lastConfigLoad: ConfigLoadResult = {
  config: defaultConfig(),
  issues: [],
  fileStatus: "missing",
};

export function getConfigLoad(): ConfigLoadResult {
  return lastConfigLoad;
}

export class ConfigFileInvalidError extends Error {
  readonly issues: string[];
  constructor(issues: string[]) {
    super(issues.join("\n"));
    this.name = "ConfigFileInvalidError";
    this.issues = issues;
  }
}

export function requireWritableConfig(): RagConfig {
  const loaded = loadConfigDetailed();
  const blocking = blockingConfigIssues(loaded);
  if (blocking.length) {
    throw new ConfigFileInvalidError(blocking);
  }
  return loaded.config;
}

/** Copy a broken config.json aside and write defaults. Returns the backup path. */
export function resetBrokenConfig(): string {
  const cfgFile = configFile(getRagDir());
  let backup = "";
  if (existsSync(cfgFile)) {
    backup = `${cfgFile}.broken-${Date.now()}`;
    copyFileSync(cfgFile, backup);
  }
  const next = defaultConfig();
  writeFileSync(cfgFile, JSON.stringify(next, null, 2));
  lastConfigLoad = { config: applyEnvOverrides(next), issues: validateConfig(applyEnvOverrides(next)), fileStatus: "ok" };
  return backup;
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
export function saveConfig(config: RagConfig, opts?: { replaceInvalid?: boolean }) {
  const current = readSavedConfig();
  if (current.fileStatus === "invalid" && !opts?.replaceInvalid) {
    throw new ConfigFileInvalidError([current.issue ?? "config.json is invalid"]);
  }
  const saved = current.fileStatus === "ok" ? current.config : defaultConfig();
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
  return loadConfigDetailed().config;
}

export function loadConfigDetailed(): ConfigLoadResult {
  envParseIssues.length = 0;
  const saved = readSavedConfig();
  const config = applyEnvOverrides(saved.config);
  const issues: string[] = [];
  if (saved.issue) issues.push(saved.issue);
  issues.push(...envParseIssues);
  issues.push(...validateConfig(config));
  lastConfigLoad = { config, issues, fileStatus: saved.fileStatus };
  return lastConfigLoad;
}

/** @deprecated Provider resolution reads auth.env from provider.json. */
export function voyageApiKey(): string | undefined {
  return envString("VOYAGE_API_KEY");
}

export function validateConfig(config: RagConfig): string[] {
  const issues: string[] = [];
  if (!Number.isInteger(config.embedding.dimensions) || config.embedding.dimensions <= 0) {
    issues.push(`embedding.dimensions must be a positive integer, got ${config.embedding.dimensions}.`);
  }
  if (!config.embedding.model || typeof config.embedding.model !== "string") {
    issues.push("embedding.model is required.");
  }

  // provider.json is the source of truth for available providers, models,
  // dimensions, and credentials. This keeps model catalogs out of config.ts.
  let providers: ProviderFile | undefined;
  try {
    providers = loadProviderFile();
  } catch (error) {
    issues.push(error instanceof ProviderConfigError ? error.message : String(error));
  }

  if (providers) {
    try {
      const embedding = getModelSpec(config.embedding.provider, "embedding", config.embedding.model, providers);
      if (embedding.dimensions !== undefined && config.embedding.dimensions !== embedding.dimensions) {
        issues.push(
          `embedding ${config.embedding.provider}/${config.embedding.model} requires ${embedding.dimensions} dimensions ` +
          `(config.json requests ${config.embedding.dimensions}).`,
        );
      }
      const embeddingProvider = providers.providers[config.embedding.provider];
      if (embeddingProvider?.auth && !process.env[embeddingProvider.auth.env]) {
        issues.push(
          `${embeddingProvider.auth.env} is not set; it is required by provider ` +
          `"${config.embedding.provider}" and is never stored in provider.json.`,
        );
      }
    } catch (error) {
      issues.push(error instanceof ProviderConfigError ? error.message : String(error));
    }

    try {
      getModelSpec(config.reranker.provider, "rerank", config.reranker.model, providers);
      const rerankerProvider = providers.providers[config.reranker.provider];
      if (rerankerProvider?.auth && !process.env[rerankerProvider.auth.env]) {
        issues.push(
          `${rerankerProvider.auth.env} is not set; it is required by reranker ` +
          `provider "${config.reranker.provider}".`,
        );
      }
    } catch (error) {
      issues.push(error instanceof ProviderConfigError ? error.message : String(error));
    }
  }
  if (typeof config.ragAlpha !== "number" || !Number.isFinite(config.ragAlpha) || config.ragAlpha < 0 || config.ragAlpha > 1) {
    issues.push(`ragAlpha must be a number in [0, 1], got ${config.ragAlpha}.`);
  }
  if (typeof config.ragScoreThreshold !== "number" || !Number.isFinite(config.ragScoreThreshold) || config.ragScoreThreshold < 0 || config.ragScoreThreshold > 1) {
    issues.push(`ragScoreThreshold must be a number in [0, 1], got ${config.ragScoreThreshold}.`);
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
  if (typeof config.cloudAutoRefresh !== "boolean") {
    issues.push(`cloudAutoRefresh must be a boolean, got ${JSON.stringify(config.cloudAutoRefresh)}.`);
  }
  if (typeof config.ragEnabled !== "boolean") {
    issues.push(`ragEnabled must be a boolean, got ${JSON.stringify(config.ragEnabled)}.`);
  }
  return issues;
}

export function isTypeConfigIssue(msg: string): boolean {
  return /must be a boolean|must be a number|must be an integer|must be a JSON/.test(msg);
}

export function blockingConfigIssues(loaded: ConfigLoadResult): string[] {
  if (loaded.fileStatus === "invalid") return loaded.issues;
  return loaded.issues.filter(isTypeConfigIssue);
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
