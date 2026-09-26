/**
 * Provider registry for the RAG extension.
 *
 * This file deliberately contains no API keys. `provider.json` describes
 * providers, endpoints, model metadata, and the environment variable used for
 * authentication. Code in this file only resolves that metadata at runtime.
 *
 * The registry is intentionally separate from `config.json`:
 * - config.json selects a provider/model for the current RAG store;
 * - provider.json describes what providers and models are available;
 * - provider implementations (local, Voyage, noop) are selected by `type`.
 *
 * `provider.json` is read from the active RAG directory first. If it is not
 * present, the bundled file next to this module is used as a compatibility
 * default. This keeps existing installations working while allowing a user
 * to override the registry in a project-specific `.pi/rag/provider.json`.
 */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { getRagDir, providerFile } from "./store.ts";

export type ModelRole = "embedding" | "rerank";
export type ProviderType = "transformers" | "voyage" | "noop";

export interface ModelDefinition {
  /** Vector width for embedding models. Rerank/noop models may omit it. */
  dimensions?: number;
}

export interface ProviderDefinition {
  /** Built-in protocol adapter. This is not the same as the registry key. */
  type: ProviderType;
  baseUrl?: string;
  auth?: {
    type: "bearer";
    env: string;
  };
  models: {
    embedding?: Record<string, ModelDefinition>;
    rerank?: Record<string, ModelDefinition>;
  };
}

export interface ProviderFile {
  version: 1;
  providers: Record<string, ProviderDefinition>;
}

export interface ResolvedModel {
  providerId: string;
  type: ProviderType;
  baseUrl?: string;
  apiKey?: string;
  model: string;
  dimensions?: number;
}

export class ProviderConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ProviderConfigError";
  }
}

/** Path of the user/project override, or the bundled fallback when absent. */
export function providerFilePath(): string {
  const local = providerFile(getRagDir());
  if (existsSync(local)) return local;
  return fileURLToPath(new URL("./provider.json", import.meta.url));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Read and minimally validate provider.json.
 *
 * This is intentionally a registry parser, not a general schema validator.
 * Unknown fields are preserved for forward compatibility, while malformed
 * top-level shapes fail early with a useful error instead of silently falling
 * back to the previous Voyage-only implementation.
 */
export function loadProviderFile(): ProviderFile {
  const file = providerFilePath();
  let raw: unknown;

  try {
    raw = JSON.parse(readFileSync(file, "utf8"));
  } catch (error) {
    throw new ProviderConfigError(
      `provider.json is invalid JSON (${error instanceof Error ? error.message : String(error)}): ${file}`,
    );
  }

  if (!isRecord(raw) || raw.version !== 1 || !isRecord(raw.providers)) {
    throw new ProviderConfigError(
      `provider.json must contain version: 1 and a providers object: ${file}`,
    );
  }

  const providers: Record<string, ProviderDefinition> = {};
  for (const [id, value] of Object.entries(raw.providers)) {
    if (!isRecord(value) || typeof value.type !== "string" || !isRecord(value.models)) {
      throw new ProviderConfigError(`Provider "${id}" must define type and models in ${file}.`);
    }
    if (!["transformers", "voyage", "noop"].includes(value.type)) {
      throw new ProviderConfigError(
        `Provider "${id}" has unsupported type "${value.type}". Supported types: transformers, voyage, noop.`,
      );
    }
    providers[id] = value as unknown as ProviderDefinition;
  }

  return { version: 1, providers };
}

function requiredModel(
  provider: ProviderDefinition,
  providerId: string,
  role: ModelRole,
  model: string,
): ModelDefinition {
  const models = provider.models[role];
  if (!models || !Object.prototype.hasOwnProperty.call(models, model)) {
    throw new ProviderConfigError(
      `Provider "${providerId}" does not define ${role} model "${model}".`,
    );
  }
  return models[model];
}

/** Resolve metadata without requiring credentials; useful for validation/status. */
export function getModelSpec(
  providerId: string,
  role: ModelRole,
  model: string,
  file: ProviderFile = loadProviderFile(),
): ResolvedModel {
  const provider = file.providers[providerId];
  if (!provider) {
    throw new ProviderConfigError(
      `Unsupported ${role} provider "${providerId}". Configured providers: ${Object.keys(file.providers).join(", ")}`,
    );
  }
  const definition = requiredModel(provider, providerId, role, model);
  return {
    providerId,
    type: provider.type,
    baseUrl: provider.baseUrl,
    model,
    dimensions: definition.dimensions,
  };
}

/** Resolve metadata and the configured environment-variable credential. */
export function resolveModel(
  providerId: string,
  role: ModelRole,
  model: string,
  file: ProviderFile = loadProviderFile(),
): ResolvedModel {
  const spec = getModelSpec(providerId, role, model, file);
  const provider = file.providers[providerId];

  if (provider.auth) {
    const value = process.env[provider.auth.env];
    if (!value) {
      throw new ProviderConfigError(
        `${provider.auth.env} is required by provider "${providerId}" (${role}/${model}).`,
      );
    }
    spec.apiKey = value;
  } else if (provider.type === "voyage") {
    throw new ProviderConfigError(
      `Provider "${providerId}" uses the voyage adapter but has no auth.env configuration.`,
    );
  }

  return spec;
}
