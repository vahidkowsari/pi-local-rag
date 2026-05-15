import { test } from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_TEXT_EXTS, normalizeExt, resolveExtensions } from "../index.ts";

test("normalizeExt: adds leading dot and lowercases", () => {
  assert.equal(normalizeExt("cs"), ".cs");
  assert.equal(normalizeExt(".CS"), ".cs");
  assert.equal(normalizeExt("  .TeX  "), ".tex");
  assert.equal(normalizeExt(""), "");
  assert.equal(normalizeExt("   "), "");
});

test("resolveExtensions: returns the default set when no overrides", () => {
  const exts = resolveExtensions({ extraExtensions: [], excludeExtensions: [] });
  for (const e of DEFAULT_TEXT_EXTS) assert.ok(exts.has(e), `default ${e} missing`);
  assert.equal(exts.size, DEFAULT_TEXT_EXTS.length);
});

test("resolveExtensions: default set covers common languages including the ones from issue #9", () => {
  const exts = resolveExtensions({ extraExtensions: [], excludeExtensions: [] });
  for (const e of [".cs", ".tsx", ".jsx", ".kt", ".swift", ".rb", ".php", ".lua", ".vue", ".svelte"]) {
    assert.ok(exts.has(e), `expected default set to include ${e}`);
  }
});

test("resolveExtensions: extraExtensions are added and normalized", () => {
  const exts = resolveExtensions({ extraExtensions: ["tex", ".ZIG", " .nix "], excludeExtensions: [] });
  assert.ok(exts.has(".tex"));
  assert.ok(exts.has(".zig"));
  assert.ok(exts.has(".nix"));
});

test("resolveExtensions: excludeExtensions remove from the default set", () => {
  const exts = resolveExtensions({ extraExtensions: [], excludeExtensions: [".md", "JSON"] });
  assert.ok(!exts.has(".md"));
  assert.ok(!exts.has(".json"));
  assert.ok(exts.has(".ts"));
});

test("resolveExtensions: empty/whitespace entries are ignored", () => {
  const baseline = resolveExtensions({ extraExtensions: [], excludeExtensions: [] }).size;
  const exts = resolveExtensions({ extraExtensions: ["", "   "], excludeExtensions: ["", "  "] });
  assert.equal(exts.size, baseline);
});
