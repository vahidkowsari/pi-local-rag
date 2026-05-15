import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, existsSync, writeFileSync, readFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Storage paths are read at module load. Set the env BEFORE the dynamic import
// so loadIndex / saveIndex / loadConfig / saveConfig point at a throwaway dir.
const ragDir = mkdtempSync(join(tmpdir(), "pi-rag-storage-"));
const legacyDir = mkdtempSync(join(tmpdir(), "pi-lens-legacy-"));
process.env.PI_RAG_DIR = ragDir;
process.env.PI_RAG_LEGACY_DIR = legacyDir;

// rmSync the placeholder dirs so we can exercise the "create on first use" + migration code paths.
rmSync(ragDir, { recursive: true, force: true });
rmSync(legacyDir, { recursive: true, force: true });

const mod = await import("../index.ts");
const { loadConfig, saveConfig, loadIndex, saveIndex } = mod;

after(() => {
  rmSync(ragDir, { recursive: true, force: true });
  rmSync(legacyDir, { recursive: true, force: true });
});

test("loadConfig: returns defaults when no config file exists", () => {
  const cfg = loadConfig();
  assert.equal(cfg.ragEnabled, true);
  assert.equal(cfg.ragTopK, 5);
  assert.equal(cfg.ragScoreThreshold, 0.1);
  assert.equal(cfg.ragAlpha, 0.4);
  assert.deepEqual(cfg.extraExtensions, []);
  assert.deepEqual(cfg.excludeExtensions, []);
});

test("saveConfig / loadConfig round-trip persists every field", () => {
  const written = {
    ragEnabled: false,
    ragTopK: 12,
    ragScoreThreshold: 0.25,
    ragAlpha: 0.7,
    extraExtensions: [".cs", ".tex"],
    excludeExtensions: [".md"],
  };
  saveConfig(written);
  const read = loadConfig();
  assert.deepEqual(read, written);

  // and verify it actually hit disk in the configured directory
  assert.ok(existsSync(join(ragDir, "config.json")));
  const raw = JSON.parse(readFileSync(join(ragDir, "config.json"), "utf-8"));
  assert.deepEqual(raw, written);
});

test("loadConfig: merges saved partial config over defaults", () => {
  // simulate a stored config that omits some fields
  mkdirSync(ragDir, { recursive: true });
  writeFileSync(join(ragDir, "config.json"), JSON.stringify({ ragTopK: 99 }));
  const cfg = loadConfig();
  assert.equal(cfg.ragTopK, 99);
  assert.equal(cfg.ragEnabled, true, "missing fields should fall back to defaults");
  assert.equal(cfg.ragAlpha, 0.4, "missing fields should fall back to defaults");
});

test("loadConfig: malformed JSON falls back to defaults instead of throwing", () => {
  writeFileSync(join(ragDir, "config.json"), "{not valid json");
  const cfg = loadConfig();
  assert.equal(cfg.ragEnabled, true);
  assert.equal(cfg.ragTopK, 5);
});

test("loadIndex: empty/missing index returns an empty IndexMeta shell", () => {
  // clean slate
  rmSync(join(ragDir, "index.json"), { force: true });
  const idx = loadIndex();
  assert.deepEqual(idx.chunks, []);
  assert.deepEqual(idx.files, {});
  assert.equal(idx.lastBuild, "");
});

test("saveIndex / loadIndex: round-trip preserves chunks, files map, lastBuild and model", () => {
  const written = {
    chunks: [{
      id: "abc-1",
      file: "/some/file.ts",
      content: "export const x = 1;",
      lineStart: 1,
      lineEnd: 1,
      hash: "deadbeef",
      indexed: "2026-05-15T00:00:00Z",
      tokens: 6,
      vector: [0.1, 0.2, 0.3],
    }],
    files: { "/some/file.ts": { hash: "deadbeef", chunks: 1, indexed: "2026-05-15T00:00:00Z", size: 19, embedded: true } },
    lastBuild: "2026-05-15T00:00:00Z",
    embeddingModel: "Xenova/all-MiniLM-L6-v2",
  };
  saveIndex(written);
  const read = loadIndex();
  assert.deepEqual(read, written);
});

test("loadIndex: corrupt index.json is treated as empty (no crash)", () => {
  writeFileSync(join(ragDir, "index.json"), "}}}not json{{{");
  const idx = loadIndex();
  assert.deepEqual(idx.chunks, []);
  assert.deepEqual(idx.files, {});
});

test("loadIndex: tolerates partial shapes (missing files or chunks key)", () => {
  writeFileSync(join(ragDir, "index.json"), JSON.stringify({ chunks: "not an array", files: null }));
  const idx = loadIndex();
  assert.deepEqual(idx.chunks, [], "non-array chunks should become []");
  assert.deepEqual(idx.files, {}, "null files should become {}");
});

test("ensureDir migration: legacy ~/.pi/lens → ~/.pi/rag is renamed on first use", () => {
  // Tear down the rag dir so the migration code path runs, and create a populated legacy dir
  rmSync(ragDir, { recursive: true, force: true });
  rmSync(legacyDir, { recursive: true, force: true });
  mkdirSync(legacyDir, { recursive: true });
  writeFileSync(join(legacyDir, "index.json"), JSON.stringify({
    chunks: [], files: {}, lastBuild: "from-legacy",
  }));

  // Any call that triggers ensureDir will migrate
  const idx = loadIndex();
  assert.equal(idx.lastBuild, "from-legacy", "data from legacy dir should be picked up after rename");
  assert.ok(existsSync(ragDir), "rag dir should now exist");
  assert.ok(!existsSync(legacyDir), "legacy dir should be gone (renamed)");
});
