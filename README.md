# pi-local-rag

Local BM25 RAG pipeline for the [Pi coding agent](https://github.com/badlogic/pi-mono). Index your local files and search them with keyword matching — **zero cloud dependency, works fully offline**.

## Features

- **BM25 keyword search** — TF-IDF scoring with exact phrase and filename boosts
- **Smart chunking** — splits files into ~50-line blocks at natural blank-line boundaries
- **Incremental indexing** — skips unchanged files (SHA-256 hash check)
- **Zero dependencies** — uses only Node.js built-ins
- **3 AI tools** — `lens_index`, `lens_query`, `lens_status` for the agent to use directly

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
| `/lens index <path>` | Index a file or directory |
| `/lens search <query>` | Search indexed content |
| `/lens status` | Show index stats (files, chunks, tokens) |
| `/lens rebuild` | Re-index changed files, prune deleted |
| `/lens clear` | Wipe the entire index |
| `/lens context <query>` | Generate a context snippet for injection |

## AI Tools

The extension registers three tools the agent can call directly:

- **`lens_index`** — Index a path into the pipeline
- **`lens_query`** — BM25 search, returns file paths + line numbers + previews
- **`lens_status`** — Index stats (file count, chunk count, total tokens, last build)

## How It Works

1. Files are chunked into ~50-line blocks (splits at blank lines)
2. Chunks are stored in `~/.pi/lens/index.json`
3. Search scores each chunk with BM25 (TF × IDF), boosted for exact phrase matches and filename matches
4. Results include file path, line range, token count, and a content preview

## Supported File Types

`.md` `.txt` `.ts` `.js` `.py` `.rs` `.go` `.java` `.c` `.cpp` `.h` `.css` `.html` `.json` `.yaml` `.yml` `.toml` `.xml` `.csv` `.sh` `.sql` `.graphql` `.proto`

## Skipped Directories

`node_modules` `.git` `.next` `dist` `build` `__pycache__` `.venv` `venv` `.cache`

## Limits

- Max 500 files per index run
- Max 500KB per file

## Storage

Index is stored at `~/.pi/lens/index.json`.

## License

MIT
