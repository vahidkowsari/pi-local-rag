# Changelog

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
