# pi-local-rag

Local BM25 RAG pipeline for the [Pi coding agent](https://github.com/badlogic/pi-mono). Index your local files and search them with keyword matching — **zero cloud dependency, works fully offline**.

## Features

- **Hybrid BM25 + vector search** — TF-IDF scoring with exact phrase and filename boosts, combined with local ONNX embeddings
- **Smart chunking** — splits files into ~50-line blocks at natural blank-line boundaries
- **Incremental indexing** — skips unchanged files (SHA-256 hash check)
- **Zero cloud dependency** — uses only Node.js built-ins + local Transformers.js model
- **3 AI tools** — `rag_index`, `rag_query`, `rag_status` for the agent to use directly

## Install

```bash
pi install npm:pi-local-rag
```

Or via git:

```bash
pi install git:github.com/vahidkowsari/pi-local-rag
```

## Commands

| Command | Description |
|---|---|
| `/rag index <path>` | Index a file or directory |
| `/rag search <query>` | Search indexed content |
| `/rag status` | Show index stats (files, chunks, tokens) |
| `/rag rebuild` | Re-index changed files, prune deleted |
| `/rag clear` | Wipe the entire index |
| `/rag on` | Enable auto-injection |
| `/rag off` | Disable auto-injection |

## AI Tools

The extension registers three tools the agent can call directly:

- **`rag_index`** — Index a path into the pipeline
- **`rag_query`** — Hybrid BM25+vector search, returns file paths + line numbers + previews
- **`rag_status`** — Show index stats and RAG config

## How It Works

1. **Index** — files are chunked (~50 lines each), embedded with `Xenova/all-MiniLM-L6-v2` (384-dim), and stored locally at `~/.pi/rag/`
2. **Search** — hybrid scoring: `alpha × BM25 + (1-alpha) × cosine_similarity` (default `alpha=0.4`)
3. **Auto-inject** — before every agent turn, the prompt is searched and relevant chunks are prepended to the system prompt

## Storage

Index data is stored at `~/.pi/rag/`. If you previously used an older version of this plugin (`~/.pi/lens/`), the directory is automatically migrated on first run.

## Configuration

Auto-injection is on by default. Tune via `/rag status`:

| Setting | Default | Description |
|---|---|---|
| `ragEnabled` | `true` | Auto-inject context before each turn |
| `ragTopK` | `5` | Max chunks to inject |
| `ragScoreThreshold` | `0.1` | Min hybrid score to include |
| `ragAlpha` | `0.4` | BM25/vector blend (0=pure vector, 1=pure BM25) |
