# Implementation progress

Plan: `pi-local-rag-grok-build-plan.md`. Branch: `feat/cloud-research-rag`. Start HEAD: `a48182c`.

Commands used unless noted:

```bash
SKIP_EMBEDDING_TESTS=1 npm test
npm run typecheck
npm pack --dry-run --ignore-scripts --json
```

Compiler: TypeScript **5.7.3** (locked in `devDependencies`). Runtime: Node v22.23.2.

## A1 — Fix module interfaces and package contents

- Status: **done**
- Commit: `a4d060d`
- Changes:
  - Unified DB exports: `getDbConn` / `closeDbConn` / `getFreshDbConn`; `openDb`/`getDb` aliases; `float32ToBuffer` re-exported.
  - Singleton is not closed by command/tool/hook handlers. Accidental `db.close()` on the singleton is detected and the next `getDbConn()` reopens.
  - `hybridSearch` call sites use `(query, limit, alpha, db?)`.
  - `isIndexStale` receives `IndexStats`.
  - `/rag clear` uses `clearIndex()` (SQLite wipe). Rebuild SQL goes through `repository.ts`.
  - `package.json` `files` includes `repository.ts`. Type-only imports fixed so Node strip-types can load the packed entry.
  - Integration tests: `__tests__/entrypoints.test.ts`.
- Tests:
  - `SKIP_EMBEDDING_TESTS=1 npm test` → 114 passed, 4 skipped (real ONNX, env skip).
  - `npm run typecheck` → pass (tsc 5.7.3).
  - `npm pack --dry-run --ignore-scripts --json` → 16 files, includes `repository.ts`.
  - Packed tarball extracted and `index.ts` imported with `--experimental-strip-types` → entry loaded (`default`, `hybridSearch`, `getDbConn`, `getFreshDbConn`, `indexFiles`).
- Unverified: loading the tarball from `node_modules` with stock Node 22 (`ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING`). Pi's own TS loader is the intended runtime. Real ONNX still skipped.
- Next: A2 BM25 direction + frozen ranking fixtures.

## A2 — BM25 direction

- Status: **done**
- Commit: `6beb81b`
- Changes:
  - `bm25ToRelevance()` inverts FTS5 `bm25()` (lower/more-negative = better) onto [0, 1].
  - Equal scores and a single candidate map to 1 so a lone hit is not filtered as `hybrid=0`.
  - Frozen fixtures: strong vs medium vs weak "quantum entanglement" ranking on pure BM25 (`alpha=1`) and hybrid (`alpha=0.5`).
- Tests: `SKIP_EMBEDDING_TESTS=1 npm test` → 119 passed, 4 skipped. `npm run typecheck` → pass.
- Unverified: real ONNX hybrid path (skipped).
- Next: A3 keep old index on failure.

## A3 — Keep old index on failure

- Status: **done**
- Commit: `693de2f`
- Changes:
  - `indexFiles` no longer deletes live chunks/vectors before embedding.
  - Embeddings are validated (count, dimension, finite, non-zero norm) then each file is replaced in its own transaction.
  - Embed/validation/write failures leave the previous file rows searchable and do not set a false `embedded=true`.
  - `/rag rebuild --force` no longer wipes the DB first.
- Tests: `__tests__/indexing-safety.test.ts`. `SKIP_EMBEDDING_TESTS=1 npm test` → 127 passed, 4 skipped. `npm run typecheck` → pass.
- Unverified: real ONNX model crash path (mocked embedBatch).
- Next: B1 LocalEmbeddingProvider.

## B1 — LocalEmbeddingProvider

- Status: **done** (this commit)
- Local MiniLM wrapped as `LocalEmbeddingProvider`; `embed.ts` is a facade (`embed` → `embedQuery`, `embedBatch` → `embedDocuments`). Default still Xenova/all-MiniLM-L6-v2 / 384-d.

## B2 — Config and factory

- Status: **done** (this commit)
- Nested `embedding` / `reranker` / `http` defaults merge over old config files. `PI_RAG_*` env overrides saved config; `saveConfig` does not write env overlays or keys. `VOYAGE_API_KEY` is env-only. Factory builds local by default; voyage requires a key.

## B3 — VoyageEmbeddingProvider

- Status: **done** (this commit)
- HTTP mock tests cover query/document `input_type`, index mapping, 401 no-retry, 429 Retry-After, empty input, abort, refuse-to-truncate.
- Smoke: `npm run smoke:voyage-embed` (exits 0 with UNVERIFIED when no key).
- Unverified: live Voyage API (no key in this environment).
- Next: C1 dynamic dimensions, fingerprint, rebuild/switch.

## C–E

- Status: **not started**
