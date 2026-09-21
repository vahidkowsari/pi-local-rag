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
  - (this commit) D1–E4 — retrieval, reranker, context, parsing, eval

## 2. A1–E4 status

| Step | Status | Notes |
| --- | --- | --- |
| A1 | done | Call shapes, connection ownership, `repository.ts` in pack, handler tests |
| A2 | done | `bm25ToRelevance`; frozen ranking fixtures |
| A3 | done | Per-file transactional replace; force-rebuild no longer wipes first |
| B1 | done | `LocalEmbeddingProvider`; `embed.ts` facade |
| B2 | done | Nested config, `PI_RAG_*` env, factory |
| B3 | done | Voyage embed HTTP mocks + `npm run smoke:voyage-embed` |
| C1 | done | Fingerprints, `indexes/<id>/rag.db`, `active.json` |
| C2 | done | Index/query use matching provider; `rag_status` shows rebuild reason |
| D1 | done | `retrieve()`; NoneReranker identity slice |
| D2 | done | Voyage rerank mock tests; degrade to hybrid on failure |
| D3 | done | `buildContext` with estimated token budget; auto-inject uses it |
| E1 | done | `extractBlocks` with 1-based PDF pages; markdown pages stay null |
| E2 | done | `chunkBlocks` token-aware; MiniLM budget 180/240/30 |
| E3 | done | `page_start`/`page_end`/`section`/`chunk_index` columns |
| E4 | done | `eval/questions.json` (20 items) + `npm run eval:retrieval` |

## 3. Module roles

- `providers/embedding/*` — local MiniLM and Voyage embed
- `providers/reranker/*` — none and Voyage rerank (`rerank-2.5-lite`, non-preview)
- `providers/http.ts` — retries, Retry-After, redaction
- `fingerprint.ts` / `index-manager.ts` — contracts and active index switch
- `retrieval.ts` / `context.ts` — unified retrieve + context builder
- `parsing.ts` — structured blocks with PDF pages
- `chunking.ts` — `chunkBlocks` plus legacy `chunkText`

Decisions: Voyage embed default `voyage-4-lite` / 1024-d / float; rerank default `rerank-2.5-lite` (GA, lower latency than `rerank-2.5`). Token counts for chunking and Pi context are character/4 **estimates**.

## 4. Config

Default remains `embedding.provider=local`, `reranker.provider=none`. Nested defaults merge over old `config.json`. Env `PI_RAG_*` overrides saved config; `saveConfig` does not write those overlays. `VOYAGE_API_KEY` is env-only. Example: `.env.example`.

## 5. Index switch

Unfingerprinted non-empty DBs require `/rag rebuild --force`. Model/dimension changes build `indexes/<id>/rag.db` and publish `active.json` only after a successful staging build. Legacy `rag.db` is not deleted. Failed force rebuild keeps the previous active index.

## 6. Offline tests

```bash
SKIP_EMBEDDING_TESTS=1 npm test
npm run typecheck
npm pack --dry-run --ignore-scripts --json
```

- TypeScript **5.7.3** (locked in `devDependencies`)
- Node v22.23.2
- Last run: **161 passed, 4 skipped** (real ONNX, `SKIP_EMBEDDING_TESTS=1`)

## 7. Live services

| Path | Status |
| --- | --- |
| Real ONNX MiniLM | skipped (`SKIP_EMBEDDING_TESTS=1`) |
| Voyage embedding | **unverified** (no key); smoke: `npm run smoke:voyage-embed` |
| Voyage rerank | **unverified** (no key); smoke: `npm run smoke:voyage-rerank` |
| Pi tool + auto-inject | handler tests with a fake Pi API; no live Pi session |

## 8. Evaluation

`eval/questions.json` has 20 fixed items (abbrev, synonym-ish, unanswerable). `npm run eval:retrieval` writes `eval/runs/` with metric **definitions** and `not-run` cloud columns. No simulated scores.

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
