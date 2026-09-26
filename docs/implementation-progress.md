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

## C1 — Dynamic dimensions, fingerprint, rebuild/switch

- Status: **done** (this commit)
- `initSchema(db, dimensions)` builds sqlite-vec with a validated integer dim. Active index selected via `active.json` (atomic rename). Legacy `rag.db` remains until a successful switch. Unfingerprinted non-empty indexes require `/rag rebuild --force`.

## C2 — Index/query provider wiring

- Status: **done** (this commit)
- Indexing uses `createEmbeddingProvider` / `embeddingProviderForIndex`. Query uses the active fingerprint's provider (`embedQuery`). `rag_status` reports wanted vs active model/dim and rebuild reason.
- Tests: `__tests__/fingerprint.test.ts`. `SKIP_EMBEDDING_TESTS=1 npm test` → 153 passed, 4 skipped.
- Unverified: live Voyage 384→1024 rebuild (no API key).
- Next: D1 unified retrieval.

## D1–D3 — retrieval, reranker, context

- Status: **done** (this commit)
- `retrieve()` is the shared entry for `/rag search`, `rag_query`, and auto-inject. NoneReranker keeps order. Voyage rerank maps by index; failures fall back to hybrid order. `buildContext` counts wrapper text with an estimated tokenizer.

## E1–E4 — parse, chunk, metadata, eval

- Status after first landing: **partial / E4 incomplete** (eval was a template). See Astra review 2026-09-21.
- Follow-up (this work):
  - E2: `chunkBlocks` splits oversized paragraphs; CJK-aware estimate; overlap-only tails dropped. Chunker fingerprint is `token-v2` and includes target/max/overlap.
  - E3: original markdown line ranges; PDF lines stay unknown and are not printed as `lines 1-2`; page/section/id/chunkIndex reach context, search, and `rag_query`.
  - E4: `npm run eval:retrieval` indexes a local corpus and writes real local-bm25 metrics. Cloud groups remain not-run without a key.
- Tests: `SKIP_EMBEDDING_TESTS=1 npm test` → 188 passed, 4 skipped. `npm run typecheck` → pass. Smoke scripts start under `--experimental-strip-types` without a key.
- Unverified: live Voyage embed/rerank, live Pi session, 3-page generated PDF, real paper spot-check, MiniLM-quality eval.

## Astra third-review follow-up (T1–T4)

- T1: `collectFromTrackedDetailed` returns files plus unavailable roots. `/rag rebuild --force` and refresh refuse to drop or publish when a tracked root cannot be stat/readdir.
- T2: `cloudAutoRefresh` and `ragEnabled` must be booleans; cloud auto-refresh requires `=== true`. String `"false"` is a reported type issue and does not refresh.
- T3: `/rag search` uses the same invalid-JSON guard as `rag_query`.
- T4: eval uses one `retrieveWithCandidates` call per question, stores `degraded`/`method`, and sets group status to `error` (all degraded) or `degraded` (partial).

## Astra recheck follow-up (F1–F7)

- F1: after overlap keep, a new piece is joined only if the estimate stays ≤ maxTokens; leftover overlap is shrunk or dropped. Probe `a×400 + b×960` is `[100, 240]`.
- F2: overlap-only tails use `addedSinceFlush`; identical text on different pages/sections is kept. Context dedup includes page and chunk id.
- F3: abort is re-checked after embed returns, before file writes, and before staging publish.
- F4: `saveConfig` / `rag_index` / mutating `/rag` commands refuse an invalid `config.json`. `/rag config reset` copies it to `.broken-*` then writes defaults.
- F5/F6: eval has four executable groups; recall uses a 30-candidate pool; sourceAccuracy is content-first then page.
- F7: Voyage provider cache keys include timeoutMs and maxRetries.

## Astra review follow-up (R1–R16)

- Index safety: parse errors count as failed and block staging publish; `--force` rebuilds on a unique generation; dropped files are not pruned until success; empty schema dimension is checked; old generations are not deleted when restaging.
- Query: compatibility is required; aborted signals throw; transient embed errors degrade to BM25; reranker config errors are visible; providers are cached per contract.
- Cloud: auto-inject refreshes only when `embedding.provider=local` or `cloudAutoRefresh=true`, with a 12s deadline.
- Config: invalid JSON is a visible issue; Voyage model lists and local dim pairing are validated.
