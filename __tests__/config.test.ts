import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, rmSync, realpathSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  defaultConfig, loadConfig, loadConfigDetailed, saveConfig, validateConfig, applyEnvOverrides, voyageApiKey,
  ConfigFileInvalidError, resetBrokenConfig, requireWritableConfig,
} from "../config.ts";
import { shouldAutoRefresh } from "../indexing.ts";
import { createEmbeddingProvider } from "../providers/embedding/factory.ts";

describe("config merge, env overlay, validation", () => {
  let ragDir: string;
  let savedRagDir: string | undefined;
  const envKeys = [
    "PI_RAG_EMBEDDING_PROVIDER", "PI_RAG_EMBEDDING_MODEL", "PI_RAG_EMBEDDING_DIMENSIONS",
    "PI_RAG_RERANKER_PROVIDER", "PI_RAG_CANDIDATE_TOP_K", "VOYAGE_API_KEY",
  ];
  const savedEnv: Record<string, string | undefined> = {};

  beforeEach(() => {
    ragDir = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-cfg-")));
    savedRagDir = process.env.PI_RAG_DIR;
    process.env.PI_RAG_DIR = ragDir;
    for (const k of envKeys) {
      savedEnv[k] = process.env[k];
      delete process.env[k];
    }
  });

  afterEach(() => {
    rmSync(ragDir, { recursive: true, force: true });
    if (savedRagDir !== undefined) process.env.PI_RAG_DIR = savedRagDir;
    else delete process.env.PI_RAG_DIR;
    for (const k of envKeys) {
      if (savedEnv[k] === undefined) delete process.env[k];
      else process.env[k] = savedEnv[k];
    }
  });

  it("old config files keep working and pick up nested defaults", () => {
    writeFileSync(join(ragDir, "config.json"), JSON.stringify({
      ragEnabled: true, ragTopK: 8, trackedPaths: ["/tmp/docs"], excludePatterns: ["*.log"],
    }));
    const cfg = loadConfig();
    expect(cfg.ragTopK).toBe(8);
    expect(cfg.trackedPaths).toEqual(["/tmp/docs"]);
    expect(cfg.excludePatterns).toEqual(["*.log"]);
    expect(cfg.embedding.provider).toBe("local");
    expect(cfg.embedding.dimensions).toBe(384);
    expect(cfg.reranker.provider).toBe("none");
    expect(cfg.candidateTopK).toBe(30);
    expect(cfg.cloudAutoRefresh).toBe(false);
  });

  it("partial nested embedding config merges over defaults", () => {
    writeFileSync(join(ragDir, "config.json"), JSON.stringify({
      embedding: { model: "custom-minilm" },
    }));
    const cfg = loadConfig();
    expect(cfg.embedding.provider).toBe("local");
    expect(cfg.embedding.model).toBe("custom-minilm");
    expect(cfg.embedding.dimensions).toBe(384);
    expect(cfg.http.timeoutMs).toBe(30_000);
  });

  it("PI_RAG_ env vars override saved config but saveConfig does not write them back", () => {
    saveConfig({ ...defaultConfig(), ragTopK: 7, embedding: { provider: "local", model: "Xenova/all-MiniLM-L6-v2", dimensions: 384 } });
    process.env.PI_RAG_EMBEDDING_PROVIDER = "voyage";
    process.env.PI_RAG_EMBEDDING_DIMENSIONS = "1024";
    const cfg = loadConfig();
    expect(cfg.embedding.provider).toBe("voyage");
    expect(cfg.embedding.dimensions).toBe(1024);
    expect(cfg.ragTopK).toBe(7);
    saveConfig(cfg);
    const raw = JSON.parse(readFileSync(join(ragDir, "config.json"), "utf-8"));
    expect(raw.embedding.provider).toBe("local");
    expect(raw.embedding.dimensions).toBe(384);
    expect(raw.ragTopK).toBe(7);
    expect(JSON.stringify(raw)).not.toMatch(/VOYAGE|apiKey|api_key/i);
  });

  it("rejects illegal provider, dimensions, and topK", () => {
    const badProvider = { ...defaultConfig(), embedding: { ...defaultConfig().embedding, provider: "openai" as "local" } };
    expect(validateConfig(badProvider).join("\n")).toMatch(/Unsupported embedding provider/);
    const badDim = { ...defaultConfig(), embedding: { ...defaultConfig().embedding, dimensions: 0 } };
    expect(validateConfig(badDim).join("\n")).toMatch(/positive integer/);
    const badTop = { ...defaultConfig(), candidateTopK: 2, ragTopK: 5 };
    expect(validateConfig(badTop).join("\n")).toMatch(/candidateTopK/);
    const huge = { ...defaultConfig(), candidateTopK: 9999 };
    expect(validateConfig(huge).join("\n")).toMatch(/exceeds the limit/);
  });

  it("voyage without VOYAGE_API_KEY is a reported config issue; key is never in the file", () => {
    const cfg = { ...defaultConfig(), embedding: { provider: "voyage" as const, model: "voyage-4-lite", dimensions: 1024 } };
    expect(voyageApiKey()).toBeUndefined();
    expect(validateConfig(cfg).join("\n")).toMatch(/VOYAGE_API_KEY/);
    saveConfig(cfg);
    const raw = readFileSync(join(ragDir, "config.json"), "utf-8");
    expect(raw).not.toMatch(/VOYAGE_API_KEY/);
    expect(existsSync(join(ragDir, "config.json"))).toBe(true);
  });

  it("factory builds the local provider by default and refuses voyage without a key", () => {
    const local = createEmbeddingProvider(defaultConfig());
    expect(local.id).toBe("local");
    expect(local.dimensions).toBe(384);
    expect(() => createEmbeddingProvider({
      ...defaultConfig(),
      embedding: { provider: "voyage", model: "voyage-4-lite", dimensions: 1024 },
    })).toThrow(/VOYAGE_API_KEY/);
  });

  it("broken JSON is reported as invalid instead of a silent default", () => {
    writeFileSync(join(ragDir, "config.json"), "{BROKEN");
    const loaded = loadConfigDetailed();
    expect(loaded.fileStatus).toBe("invalid");
    expect(loaded.issues.join("\n")).toMatch(/invalid JSON/i);
    expect(loaded.config.embedding.provider).toBe("local");
  });

  it("saveConfig refuses to overwrite an invalid config.json", () => {
    writeFileSync(join(ragDir, "config.json"), "{BROKEN");
    expect(() => saveConfig(defaultConfig())).toThrow(ConfigFileInvalidError);
    expect(readFileSync(join(ragDir, "config.json"), "utf-8")).toBe("{BROKEN");
  });

  it("resetBrokenConfig keeps the original file as a backup and writes defaults", () => {
    writeFileSync(join(ragDir, "config.json"), "{BROKEN");
    const backup = resetBrokenConfig();
    expect(existsSync(backup)).toBe(true);
    expect(readFileSync(backup, "utf-8")).toBe("{BROKEN");
    expect(loadConfigDetailed().fileStatus).toBe("ok");
    expect(loadConfig().embedding.provider).toBe("local");
  });

  it("string false is not a valid cloudAutoRefresh value", () => {
    const cfg = { ...defaultConfig(), cloudAutoRefresh: "false" as unknown as boolean };
    expect(validateConfig(cfg).join("\n")).toMatch(/cloudAutoRefresh must be a boolean/);
    const voyage = {
      ...cfg,
      embedding: { provider: "voyage" as const, model: "voyage-4-lite", dimensions: 1024 },
    };
    process.env.VOYAGE_API_KEY = "synthetic";
    expect(shouldAutoRefresh(voyage, {
      totalChunks: 1, totalFiles: 1, totalTokens: 1, embeddedCount: 1,
      lastBuild: "2020-01-01T00:00:00Z", embeddingModel: "voyage-4-lite",
    })).toBe(false);
    writeFileSync(join(ragDir, "config.json"), JSON.stringify({ cloudAutoRefresh: "false" }));
    const loaded = loadConfigDetailed();
    expect(loaded.issues.join("\n")).toMatch(/cloudAutoRefresh must be a boolean/);
    expect(() => requireWritableConfig()).toThrow(ConfigFileInvalidError);
    delete process.env.VOYAGE_API_KEY;
  });

  it("numeric cloudAutoRefresh is a type issue", () => {
    const cfg = { ...defaultConfig(), cloudAutoRefresh: 0 as unknown as boolean };
    expect(validateConfig(cfg).join("\n")).toMatch(/must be a boolean/);
  });

  it("local MiniLM with the wrong dimension is a config issue", () => {
    const cfg = defaultConfig();
    cfg.embedding.dimensions = 1024;
    expect(validateConfig(cfg).join("\n")).toMatch(/requires 384/);
  });

  it("applyEnvOverrides does not mutate the input object", () => {
    const cfg = defaultConfig();
    process.env.PI_RAG_CANDIDATE_TOP_K = "40";
    const next = applyEnvOverrides(cfg);
    expect(next.candidateTopK).toBe(40);
    expect(cfg.candidateTopK).toBe(30);
  });
});
