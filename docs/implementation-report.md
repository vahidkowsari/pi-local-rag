# Implementation report

Plan: cloud research RAG (A1–E4). Branch: `feat/cloud-research-rag`. Not pushed, not published, not merged.

## 1. Commits

- Start: `a48182c` (`main`)
- Branch HEAD before this report commit: `4ab0291`
- Commits on this branch:
  - `a4d060d` A1 — sync entrypoints with SQLite DB and search APIs
  - `6beb81b` A2 — invert FTS5 BM25
  - `693de2f` A3 — replace index rows only after embeddings validate
  - `b46aa8c` B1–B3 — local/Voyage embedding providers and config factory
  - `4ab0291` C1–C2 — fingerprints, dynamic dimensions, active manifest
  - `d546804` D1–E4 — retrieval, reranker, context, parsing, eval template
- (this work) Astra review R1–R16 follow-up on the same branch

## 2. A1–E4 status

Status after the 2026-09-21 Astra review and the follow-up fixes in this tree. “Done” means the code path exists and is covered by offline tests. “Unverified” means a live service or a human paper check was not run.

| Step | Status | Notes |
| --- | --- | --- |
| A1 | done | Call shapes, connection ownership, pack includes runtime modules including `abort.ts` |
| A2 | done | `bm25ToRelevance`; frozen ranking fixtures |
| A3 | done | Per-file transactional replace; command-level force rebuild now stages |
| B1 | done | `LocalEmbeddingProvider`; production queries reuse the cached pipeline |
| B2 | done | Nested config; broken JSON / illegal env / model-dim pairing are reported |
| B3 | done | Voyage embed HTTP mocks; smoke starts without a key (`UNVERIFIED`) |
| C1 | done (offline) | Unique staging generation; empty schema dim check; parse/embed failure does not publish |
| C2 | done (offline) | Query, index, and auto-inject share compatibility; cloud auto-refresh is off by default |
| D1 | done | `retrieve()`; NoneReranker identity slice |
| D2 | done (offline) | Voyage rerank `truncation:false` + token caps; config errors are visible; request failure degrades |
| D3 | done (offline) | Context budget has 10% slack; empty header-only inject is skipped; deadline on auto-inject |
| E1 | partial | Physical PDF pages at extract time; 3-page generated-PDF spot-check still unverified |
| E2 | partial | Estimate-based `maxTokens` is enforced after overlap join; CJK-aware estimate. Not a MiniLM tokenizer hard cap |
| E3 | done (offline) | page/section/id/chunkIndex flow to context, search, and `rag_query`; unknown lines omitted |
| E4 | partial | Four groups are selectable (`--groups=`). local-bm25 runs; local-hybrid skips when `SKIP_EMBEDDING_TESTS=1`; cloud groups execute when `VOYAGE_API_KEY` is set. Candidate recall uses k=30; sourceAccuracy scores page separately from content |

## 3. Module roles

- `providers/embedding/*` — local MiniLM and Voyage embed
- `providers/reranker/*` — none and Voyage rerank (`rerank-2.5-lite`, non-preview)
- `providers/http.ts` — retries, Retry-After, redaction
- `fingerprint.ts` / `index-manager.ts` — contracts and active index switch
- `retrieval.ts` / `context.ts` — unified retrieve + context builder
- `parsing.ts` — structured blocks with PDF pages
- `chunking.ts` — `chunkBlocks` plus legacy `chunkText`

Decisions: Voyage embed default `voyage-4-lite` / 1024-d / float; rerank default `rerank-2.5-lite` (GA, lower latency than `rerank-2.5`). Chunking uses a CJK-aware estimate (CJK ≈ 1 token/char, other ≈ 4 chars/token) plus MiniLM 180/240/30 bounds. Pi context uses the same estimate with 10% slack. These are still estimates, not a model tokenizer.

## 4. Config

Default remains `embedding.provider=local`, `reranker.provider=none`. Nested defaults merge over old `config.json`. Env `PI_RAG_*` overrides saved config; `saveConfig` does not write those overlays. `VOYAGE_API_KEY` is env-only. Example: `.env.example`.

## 5. Index switch

Unfingerprinted non-empty DBs require `/rag rebuild --force`. Queries on those indexes do not assume the local MiniLM contract. Full rebuilds (`--force` or incompatible) write a unique `staging/<generation>/` tree and publish to `indexes/<indexId>/<generation>/` only after parse+embed succeed and vector coverage matches. Previous generations are not deleted. Incremental non-force rebuilds stay in-place and prune dropped files only after a clean result.

## 6. Offline tests

```bash
SKIP_EMBEDDING_TESTS=1 npm test
npm run typecheck
npm pack --dry-run --ignore-scripts --json
```

- TypeScript **5.7.3** (locked in `devDependencies`)
- Node v22.23.2
- Last run: **194 passed, 4 skipped** (real ONNX, `SKIP_EMBEDDING_TESTS=1`)

## 7. Live services

| Path | Status |
| --- | --- |
| Real ONNX MiniLM | skipped (`SKIP_EMBEDDING_TESTS=1`) |
| Voyage embedding | **unverified** (no key); smoke starts and prints `UNVERIFIED` |
| Voyage rerank | **unverified** (no key); smoke starts and prints `UNVERIFIED` |
| Pi tool + auto-inject | handler tests with a fake Pi API; no live Pi session |

## 8. Evaluation

`eval/questions.json` has 23 items (including Chinese queries and a page-labeled sample PDF). `npm run eval:retrieval` indexes a local corpus and computes recall / hit@5 / MRR / unanswerable on **BM25-only**. Those numbers are pipeline metrics, not MiniLM or Voyage quality. Cloud embedding and rerank groups stay `not-run` without `VOYAGE_API_KEY`.

## 9. Known limits

- Token budgets are estimates, not a model tokenizer.
- Scanned/two-column/formula/table PDFs are not understood; pages are physical 1-based pages from extraction.
- One writer per process (lock); no multi-process writers.
- Cloud auto-refresh stays off unless `cloudAutoRefresh` is set.
- Pi `node_modules` TypeScript stripping cannot load this package; Pi's own loader is the intended runtime.

## 10. Reviewer reproduce

```bash
cd /Users/chouchao/Documents/RAG_Knowledge_Repository/pi-local-rag
git checkout feat/cloud-research-rag
SKIP_EMBEDDING_TESTS=1 npm test
npm run typecheck
npm pack --dry-run --ignore-scripts --json
npm run eval:retrieval
# optional, if you have a key:
# VOYAGE_API_KEY=... npm run smoke:voyage-embed
# VOYAGE_API_KEY=... npm run smoke:voyage-rerank
```

Do not point `PI_RAG_DIR` at a real `~/.pi/rag` while testing.
