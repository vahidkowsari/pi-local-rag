import { createHash } from "node:crypto";
import { getModelSpec } from "./provider-config.ts";
import {
  CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS,
  EMBEDDING_MODEL, VECTOR_DIM,
} from "./constants.ts";
import type { EmbeddingConfig, RagConfig } from "./config.ts";

/** Vector representation + normalization contract. Bump when distance assumptions change. */
export const VECTOR_CONTRACT_VERSION = "l2-unit-v1";
export const PARSER_VERSION = "blocks-v1";
/** token-v3: maxTokens is checked after overlap join; distinct sources are not suffix-dropped. */
export const CHUNKER_VERSION = "token-v3";
export const CHUNK_MAX_LINES = 50;

export interface EmbeddingFingerprint {
  provider: string;
  model: string;
  dimensions: number;
  /** Hash of non-secret provider/model metadata; absent in legacy indexes. */
  providerContract?: string;
  contract: string;
}

export interface ProcessingFingerprint {
  parser: string;
  chunker: string;
  maxLines: number;
  targetTokens: number;
  maxTokens: number;
  overlapTokens: number;
}

function providerContractHash(embedding: EmbeddingConfig): string | undefined {
  try {
    const spec = getModelSpec(embedding.provider, "embedding", embedding.model);
    // Never include the API key. A changed endpoint, adapter type, or model
    // metadata can change vector semantics and must trigger a rebuild.
    return createHash("sha256")
      .update(JSON.stringify({
        provider: embedding.provider,
        type: spec.type,
        baseUrl: spec.baseUrl ?? "",
        model: embedding.model,
        dimensions: spec.dimensions ?? embedding.dimensions,
      }))
      .digest("hex")
      .slice(0, 16);
  } catch {
    // Config validation will report the actionable provider error. Keeping
    // this optional makes fingerprint creation usable while reporting it.
    return undefined;
  }
}

export function embeddingFingerprintFromConfig(embedding: EmbeddingConfig): EmbeddingFingerprint {
  return {
    provider: embedding.provider,
    model: embedding.model,
    dimensions: embedding.dimensions,
    providerContract: providerContractHash(embedding),
    contract: VECTOR_CONTRACT_VERSION,
  };
}

export function defaultEmbeddingFingerprint(): EmbeddingFingerprint {
  return {
    provider: "local",
    model: EMBEDDING_MODEL,
    dimensions: VECTOR_DIM,
    providerContract: providerContractHash({ provider: "local", model: EMBEDDING_MODEL, dimensions: VECTOR_DIM }),
    contract: VECTOR_CONTRACT_VERSION,
  };
}

export function processingFingerprintFromConfig(_config?: Pick<RagConfig, "embedding">): ProcessingFingerprint {
  return {
    parser: PARSER_VERSION,
    chunker: CHUNKER_VERSION,
    maxLines: CHUNK_MAX_LINES,
    targetTokens: CHUNK_TARGET_TOKENS,
    maxTokens: CHUNK_MAX_TOKENS,
    overlapTokens: CHUNK_OVERLAP_TOKENS,
  };
}

export function serializeFingerprint(fp: EmbeddingFingerprint | ProcessingFingerprint): string {
  return JSON.stringify(fp);
}

export function parseEmbeddingFingerprint(raw: string | undefined): EmbeddingFingerprint | undefined {
  if (!raw) return undefined;
  try {
    const v = JSON.parse(raw) as EmbeddingFingerprint;
    if (!v.provider || !v.model || !v.dimensions || !v.contract) return undefined;
    return v;
  } catch {
    return undefined;
  }
}

export function parseProcessingFingerprint(raw: string | undefined): ProcessingFingerprint | undefined {
  if (!raw) return undefined;
  try {
    const v = JSON.parse(raw) as ProcessingFingerprint;
    if (!v.parser || !v.chunker || !v.maxLines) return undefined;
    return {
      parser: v.parser,
      chunker: v.chunker,
      maxLines: v.maxLines,
      targetTokens: v.targetTokens ?? 0,
      maxTokens: v.maxTokens ?? 0,
      overlapTokens: v.overlapTokens ?? 0,
    };
  } catch {
    return undefined;
  }
}

export function fingerprintsEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Compare embedding contracts while accepting indexes written before the
 * provider metadata hash was introduced. Once both sides have the hash, a
 * changed endpoint/model contract is intentionally incompatible.
 */
export function embeddingFingerprintsEqual(
  a: EmbeddingFingerprint,
  b: EmbeddingFingerprint,
): boolean {
  if (a.provider !== b.provider || a.model !== b.model || a.dimensions !== b.dimensions || a.contract !== b.contract) {
    return false;
  }
  return a.providerContract === undefined || b.providerContract === undefined || a.providerContract === b.providerContract;
}

export function indexIdFor(emb: EmbeddingFingerprint, proc: ProcessingFingerprint): string {
  const h = createHash("sha256").update(`${serializeFingerprint(emb)}|${serializeFingerprint(proc)}`).digest("hex");
  return `${emb.provider}-${emb.dimensions}-${h.slice(0, 10)}`;
}

export function safeVectorDimensions(n: number): number {
  if (!Number.isInteger(n) || n < 1 || n > 4096) {
    throw new Error(`Refusing to build a vector table with dimensions=${n}`);
  }
  return n;
}
