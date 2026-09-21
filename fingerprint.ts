import { createHash } from "node:crypto";
import { EMBEDDING_MODEL, VECTOR_DIM } from "./constants.ts";
import type { EmbeddingConfig, RagConfig } from "./config.ts";

/** Vector representation + normalization contract. Bump when distance assumptions change. */
export const VECTOR_CONTRACT_VERSION = "l2-unit-v1";
export const PARSER_VERSION = "extract-v1";
export const CHUNKER_VERSION = "lines-v1";
export const CHUNK_MAX_LINES = 50;

export interface EmbeddingFingerprint {
  provider: string;
  model: string;
  dimensions: number;
  contract: string;
}

export interface ProcessingFingerprint {
  parser: string;
  chunker: string;
  maxLines: number;
}

export function embeddingFingerprintFromConfig(embedding: EmbeddingConfig): EmbeddingFingerprint {
  return {
    provider: embedding.provider,
    model: embedding.model,
    dimensions: embedding.dimensions,
    contract: VECTOR_CONTRACT_VERSION,
  };
}

export function defaultEmbeddingFingerprint(): EmbeddingFingerprint {
  return {
    provider: "local",
    model: EMBEDDING_MODEL,
    dimensions: VECTOR_DIM,
    contract: VECTOR_CONTRACT_VERSION,
  };
}

export function processingFingerprintFromConfig(_config?: Pick<RagConfig, "embedding">): ProcessingFingerprint {
  return {
    parser: PARSER_VERSION,
    chunker: CHUNKER_VERSION,
    maxLines: CHUNK_MAX_LINES,
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
    return v;
  } catch {
    return undefined;
  }
}

export function fingerprintsEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
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
