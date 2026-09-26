# Changelog

## Unreleased

- **U1**: query embedding failure is returned on the search result object even when FTS has zero hits; eval and empty `rag_query`/`/rag search` keep the degraded diagnostic.
- **Astra third-review follow-up**: missing tracked roots no longer publish an empty index; `cloudAutoRefresh: "false"` is a type error and does not enable refresh; `/rag search` shares the broken-config guard; eval records per-question degradation and marks a group `error`/`degraded` instead of `ok` when rerank fails.
- **Astra recheck follow-up**: overlap join respects maxTokens; distinct-page identical text is kept; in-flight abort does not publish; broken config.json is not overwritten (`/rag config reset` backs it up); eval has four selectable groups with separate candidate/top-5 metrics; Voyage HTTP timeout/retry are part of the provider cache key.
- **Astra review follow-up**: parse/index failures no longer publish an empty staging index. Force rebuild uses a unique generation and leaves the live index untouched until success. Cloud auto-refresh stays off unless `cloudAutoRefresh` is set. Queries refuse unfingerprinted or mismatched indexes. Oversized paragraphs are split; short markdown sections are kept; page/section/id reach Pi context and `rag_query`. Cancel signals and HTTP backoff honor abort. Local embedding providers are cached per contract. Broken `config.json` is reported. Smoke scripts run under Node strip-types. `npm run eval:retrieval` computes local BM25 metrics.
- **A1: module interface sync**: `index.ts` now calls `hybridSearch(query, limit, alpha)` and `isIndexStale(IndexStats)`. Public DB access is `getDbConn()` (singleton, closed by `closeDbConn()`) and `getFreshDbConn()` (caller-owned). `openDb`/`getDb` remain aliases of `getDbConn`. Closing the singleton handle no longer permanently poisons later `getDbConn()` calls. `/rag clear` actually wipes SQLite instead of calling the no-op `saveIndex`. `package.json` `files` includes `repository.ts`. TypeScript 5.7.3 is a locked devDependency.
- **A2: BM25 direction**: FTS5 `bm25()` is lower-is-better; internal scores now invert that range so a strong keyword hit ranks first and is not mapped to 0 and filtered.
- **A3: safe reindex**: old chunks/vectors stay until embeddings validate and a per-file transaction replaces them. Failed embeds, bad dimensions, and force-rebuild errors keep the previous searchable index.
- **B1–B3: embedding providers**: `LocalEmbeddingProvider` wraps MiniLM; config grows nested `embedding`/`reranker`/`http` with `PI_RAG_*` env overlays; `VoyageEmbeddingProvider` talks to `POST /v1/embeddings` with `input_type` query/document. Default remains local. Cloud is not wired into the production index yet.
- **C1–C2: fingerprint + dynamic dimensions**: each index records embedding and processing fingerprints. Model/dimension changes rebuild into `indexes/<id>/rag.db` and publish `active.json` only after a successful build. Queries embed with the active index's provider.
- **D1–D3: retrieval + rerank + context**: `/rag search`, `rag_query`, and auto-inject share `retrieve()`. Default reranker is none. Voyage `rerank-2.5-lite` is optional; failures keep hybrid order. Context injection uses an estimated token budget that includes wrappers.
- **E1–E4: pages, token chunks, eval**: PDF extraction records 1-based physical pages; markdown does not invent pages. Chunker is token-aware with MiniLM-safe defaults. `eval/questions.json` is a 20-question set; `npm run eval:retrieval` does not fabricate scores.

## 0.4.1

- **Docs refresh**: README rewritten for 0.4.0 feature set — SQLite/FTS5/sqlite-vec storage, PDF/DOCX/HTML extraction, OCR fallback, per-project store, tracked paths + exclude patterns, 24 h auto-refresh, trailing-message auto-injection. Commands table expanded with `/rag find`, `/rag refresh`, `/rag rebuild --force`, `/rag exclude`, `/rag help`. Optional OCR install instructions (`brew install poppler tesseract tesseract-lang` / `apt install poppler-utils tesseract-ocr ...`). New "Testing" section noting `SKIP_EMBEDDING_TESTS` and the tesseract-absent OCR skip.
- **`package.json`**: description rewritten to mention SQLite + sqlite-vec + PDF/DOCX/HTML + OCR + per-project storage. Keywords += `sqlite`, `fts5`, `sqlite-vec`, `pdf`, `docx`, `ocr`.
- **`.gitignore`**: ignore `.pi/` so local RAG stores don't leak into commits.

## 0.4.0

- **SQLite storage** (replaces JSON): index now lives in a `rag.db` file using `better-sqlite3` + FTS5 virtual table for BM25 full-text search + `sqlite-vec` for vector similarity. Automatic one-shot migration from legacy `index.json` on first run, no data loss. WAL mode enabled for safe concurrent reads.
- **Per-project RAG store**: walks up from `process.cwd()` looking for `.pi/rag/`; falls back to `~/.pi/rag/` global store. First `/rag index` in a directory with no parent store creates one at cwd. Override with `$PI_RAG_DIR`.
- **Tracked paths + gitignore-style exclude patterns**: `trackedPaths` and `excludePatterns` in config; `/rag rebuild` re-walks tracked paths so new files are picked up automatically.
- **24h auto-refresh**: `before_agent_start` hook checks index age; re-indexes stale tracked paths in the background. `/rag refresh` command triggers manually. Configurable via `ragAutoRefresh`.
- **`/rag rebuild --force`**: wipes DB and re-embeds everything from scratch; fixes progress bar freezing during rebuild.
- **PDF + DOCX indexing**: `pdf-parse` for text PDFs, `mammoth` for DOCX files.
- **OCR fallback for image-only PDFs**: `pdftoppm` + `tesseract` pipeline for scanned documents (optional system deps).
- **HTML → Markdown via `turndown`** before chunking — cleaner chunks for web content.
- **`/rag find <glob>`**: list indexed files matching a glob pattern.
- **`/rag help`**: show all available subcommands.
- **`/rag` autocompletions**: working tab-completions for all subcommands.
- **Batched ONNX embeddings** (perf): `embedBatch()` now passes up to 64 texts per ONNX forward pass instead of 1-at-a-time (~64× fewer forward passes; ~219 passes for 13,955 chunks vs 13,955 previously).
- **Parallel file reads** (perf): Phase 1 of `indexFiles()` reads/chunks up to 32 files concurrently so I/O latency hides behind CPU work.
- **RAG context injected at end of prompt** (perf): avoids KV cache invalidation on models that support prefix caching.
- **Modular split**: `index.ts` refactored into 9 focused modules (`chunking.ts`, `embed.ts`, `indexing.ts`, `search.ts`, `store.ts`, `db.ts`, `config.ts`, `constants.ts`, `types/`).
- **104 tests** via vitest: covers chunking, math, BM25 search, SQLite storage round-trip, FTS5 triggers, vector normalization, PDF/DOCX/OCR extraction, per-project store resolution, 24h auto-refresh, and configurable extensions.
- **Fix**: tool definitions now include required `label` and `AgentToolResult.details` fields.
- **Fix**: silence `pdfjs` worker warnings in TUI.
- **Fix**: FTS5 query escaping for single quotes; split into individual terms.

## Unreleased

- **Configurable file extensions** (closes #9): expanded the default list to cover commonly-missing languages (`.cs`, `.tsx`, `.jsx`, `.kt`, `.swift`, `.rb`, `.php`, `.lua`, `.dart`, `.vue`, `.svelte`, `.scala`, `.scss`, `.tf`, `.hcl`, `.mdx`, …) and added `extraExtensions` / `excludeExtensions` to `RagConfig` plus a `/rag ext list|add|remove|reset` subcommand so users can extend the allowlist without forking. Includes 6 new tests for `normalizeExt` and `resolveExtensions`.
- **Test suite** (38 tests, no dev dependencies — uses `node --test` + `--experimental-strip-types`): covers cosine/normalize math, chunking, file collection against real tmp dirs, BM25 search ranking + phrase boost, storage I/O round-trip + legacy `~/.pi/lens` → `~/.pi/rag` migration, and live embedding/semantic-search against the real ONNX model. The model (`Xenova/all-MiniLM-L6-v2`, ~23 MB) is fetched from HuggingFace on the first run; set `SKIP_EMBEDDING_TESTS=1` to opt out in offline CI. Run with `npm test`.
- **Storage paths overridable via env**: `PI_RAG_DIR` and `PI_RAG_LEGACY_DIR` let the index live somewhere other than `~/.pi/rag` (useful for project-local indexes and isolated tests).

## 0.3.0

- **Renamed `/lens` → `/rag`**: all commands now use `/rag index|search|status|rebuild|clear|on|off`
- **Renamed tools**: `lens_index` → `rag_index`, `lens_query` → `rag_query`, `lens_status` → `rag_status`
- **Storage migrated**: index data moved from `~/.pi/lens/` to `~/.pi/rag/` — existing index is automatically migrated on first run, no data loss
- **`/rag on|off`**: simplified toggle (previously `/lens rag on|off`)
- **No file limit**: removed the 500-file cap — indexes all files in a directory
- **Live progress**: `setWidget` + `setStatus` updates during `/rag index` and `/rag rebuild`; stderr overwrite line for tool mode (`rag_index`)
- **Event loop yield**: `await setTimeout(0)` between files so the TUI can re-render and the agent doesn't appear hung

## 0.2.0

- **Hybrid RAG**: BM25 + local vector embeddings via `@xenova/transformers` (Transformers.js)
- **Auto-injection**: `before_agent_start` hook injects relevant chunks into every LLM prompt
- **Embedding model**: `Xenova/all-MiniLM-L6-v2` (384-dim, ~23MB, downloads once, runs fully offline)
- **Score transparency**: search results now show `bm25`, `vector`, and `hybrid` scores
- **`/lens rag on|off`**: toggle auto-injection at runtime *(renamed to `/rag on|off` in 0.3.0)*
- **`/lens status`**: now shows vector coverage % *(renamed to `/rag status` in 0.3.0)*
- **Config file**: `~/.pi/lens/config.json` for `ragEnabled`, `ragTopK`, `ragScoreThreshold`, `ragAlpha` *(moved to `~/.pi/rag/` in 0.3.0)*
- Bumped to `dependencies` for `@xenova/transformers`

## 0.1.0

- Initial release
- BM25 keyword search over local files
- Tools: `lens_index`, `lens_query`, `lens_status` *(renamed to `rag_*` in 0.3.0)*
- Commands: `/lens index|search|status|rebuild|clear|context` *(renamed to `/rag` in 0.3.0)*
