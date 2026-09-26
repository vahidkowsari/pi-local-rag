import { describe, expect, it, afterEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { realpathSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { defaultConfig, validateConfig } from "../config.ts";
import { loadProviderFile, resolveModel } from "../provider-config.ts";
import { createEmbeddingProvider, resetEmbeddingProviderCache } from "../providers/embedding/factory.ts";
import { createReranker } from "../providers/reranker/factory.ts";

const savedRagDir = process.env.PI_RAG_DIR;
const savedKey = process.env.PI_RAG_TEST_PROVIDER_KEY;

afterEach(() => {
  if (savedRagDir === undefined) delete process.env.PI_RAG_DIR;
  else process.env.PI_RAG_DIR = savedRagDir;
  if (savedKey === undefined) delete process.env.PI_RAG_TEST_PROVIDER_KEY;
  else process.env.PI_RAG_TEST_PROVIDER_KEY = savedKey;
  resetEmbeddingProviderCache();
});

describe("provider.json registry", () => {
  it("resolves a provider alias, endpoint, dimensions, and env credential", () => {
    const dir = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-provider-")));
    process.env.PI_RAG_DIR = dir;
    process.env.PI_RAG_TEST_PROVIDER_KEY = "synthetic-provider-key";
    writeFileSync(join(dir, "provider.json"), JSON.stringify({
      version: 1,
      providers: {
        proxy: {
          type: "voyage",
          baseUrl: "https://proxy.example/v1",
          auth: { type: "bearer", env: "PI_RAG_TEST_PROVIDER_KEY" },
          models: {
            embedding: { "custom-embed": { dimensions: 7 } },
            rerank: { "custom-rerank": {} },
          },
        },
      },
    }));

    const embedding = resolveModel("proxy", "embedding", "custom-embed");
    expect(embedding).toMatchObject({
      providerId: "proxy",
      type: "voyage",
      baseUrl: "https://proxy.example/v1",
      model: "custom-embed",
      dimensions: 7,
      apiKey: "synthetic-provider-key",
    });

    const config = defaultConfig();
    config.embedding = { provider: "proxy", model: "custom-embed", dimensions: 7 };
    config.reranker = { provider: "proxy", model: "custom-rerank" };
    expect(validateConfig(config)).toEqual([]);

    const embeddingProvider = createEmbeddingProvider(config);
    expect(embeddingProvider.id).toBe("proxy");
    expect(embeddingProvider.dimensions).toBe(7);
    expect(createReranker(config).id).toBe("proxy");

    rmSync(dir, { recursive: true, force: true });
  });

  it("reports a missing provider model instead of silently using Voyage defaults", () => {
    const dir = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-provider-")));
    process.env.PI_RAG_DIR = dir;
    writeFileSync(join(dir, "provider.json"), JSON.stringify({
      version: 1,
      providers: {
        local: {
          type: "transformers",
          models: { embedding: { "known-model": { dimensions: 384 } } },
        },
      },
    }));

    expect(() => loadProviderFile()).not.toThrow();
    expect(() => resolveModel("local", "embedding", "unknown-model"))
      .toThrow(/does not define embedding model/);

    rmSync(dir, { recursive: true, force: true });
  });
});
