import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, rmSync, realpathSync, existsSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Database from "better-sqlite3";
import { load as loadVec } from "sqlite-vec";
import { initSchema } from "../repository.ts";
import * as repo from "../repository.ts";
import { defaultConfig } from "../config.ts";
import {
  checkIndexCompatibility, stampFingerprints, prepareStagingDir, publishActiveManifest, resolveActiveDbPath,
} from "../index-manager.ts";
import { embeddingFingerprintFromConfig, processingFingerprintFromConfig, serializeFingerprint } from "../fingerprint.ts";

describe("index compatibility and staging switch", () => {
  let ragDir: string;
  let saved: string | undefined;

  beforeEach(() => {
    ragDir = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-fp-")));
    saved = process.env.PI_RAG_DIR;
    process.env.PI_RAG_DIR = ragDir;
  });

  afterEach(() => {
    rmSync(ragDir, { recursive: true, force: true });
    if (saved !== undefined) process.env.PI_RAG_DIR = saved;
    else delete process.env.PI_RAG_DIR;
  });

  function mem(dim = 384) {
    const db = new Database(":memory:");
    loadVec(db);
    initSchema(db, dim);
    return db;
  }

  it("empty index is compatible with the current config", () => {
    const db = mem();
    const r = checkIndexCompatibility(db, defaultConfig());
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.empty).toBe(true);
    db.close();
  });

  it("non-empty index without fingerprint requires an explicit rebuild", () => {
    const db = mem();
    repo.insertChunk(db, {
      id: "c1", filePath: "/a.ts", content: "export const x = 1;",
      lineStart: 1, lineEnd: 1, hash: "h", indexedAt: new Date().toISOString(), tokens: 4,
    });
    const r = checkIndexCompatibility(db, defaultConfig());
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toMatch(/no fingerprint/);
    db.close();
  });

  it("same model and dimensions stay compatible; a different model does not", () => {
    const db = mem();
    repo.insertChunk(db, {
      id: "c1", filePath: "/a.ts", content: "export const x = 1;",
      lineStart: 1, lineEnd: 1, hash: "h", indexedAt: new Date().toISOString(), tokens: 4,
    });
    stampFingerprints(db, defaultConfig());
    expect(checkIndexCompatibility(db, defaultConfig()).ok).toBe(true);
    const other = defaultConfig();
    other.embedding = { ...other.embedding, model: "other-minilm" };
    const r = checkIndexCompatibility(db, other);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toMatch(/Rebuild required/);
    db.close();
  });

  it("384 → 1024 uses a separate staging path and does not replace legacy rag.db until publish", () => {
    mkdirSync(ragDir, { recursive: true });
    writeFileSync(join(ragDir, "rag.db"), "legacy");
    const cfg = defaultConfig();
    cfg.embedding = { provider: "voyage", model: "voyage-4-lite", dimensions: 1024 };
    const spec = prepareStagingDir(cfg, ragDir);
    expect(spec.dbPath).toContain("indexes");
    expect(spec.dbPath).not.toBe(join(ragDir, "rag.db"));
    expect(existsSync(join(ragDir, "rag.db"))).toBe(true);
    expect(resolveActiveDbPath(ragDir)).toBe(join(ragDir, "rag.db"));
    publishActiveManifest(ragDir, {
      version: 1,
      indexId: spec.indexId,
      relativeDbPath: spec.relativeDbPath,
      embeddingFingerprint: serializeFingerprint(embeddingFingerprintFromConfig(cfg.embedding)),
      processingFingerprint: serializeFingerprint(processingFingerprintFromConfig(cfg)),
      createdAt: new Date().toISOString(),
    });
    mkdirSync(join(spec.dbPath, ".."), { recursive: true });
    writeFileSync(spec.dbPath, "new");
    expect(resolveActiveDbPath(ragDir)).toBe(spec.dbPath);
    expect(existsSync(join(ragDir, "rag.db"))).toBe(true);
  });

  it("processing fingerprint change requires rebuild", () => {
    const db = mem();
    repo.insertChunk(db, {
      id: "c1", filePath: "/a.ts", content: "export const x = 1;",
      lineStart: 1, lineEnd: 1, hash: "h", indexedAt: new Date().toISOString(), tokens: 4,
    });
    stampFingerprints(db, defaultConfig());
    repo.setMetadata(db, repo.MetadataKey.ProcessingFingerprint, serializeFingerprint({
      parser: "extract-v1", chunker: "other-chunker", maxLines: 50,
    }));
    const r = checkIndexCompatibility(db, defaultConfig());
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toMatch(/processing fingerprint/);
    db.close();
  });
});
