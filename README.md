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
| `/rag ext list` | List active indexable file extensions |
| `/rag ext add <.ext>` | Add an extension (e.g. `.cs`, `.tex`, `.zig`) |
| `/rag ext remove <.ext>` | Stop indexing files with this extension |
| `/rag ext reset` | Restore the default extension list |

## Example session

```text
$ /rag index ~/code/my-app
Found 412 files to index
Indexing  ████████████████████████  100%
file:    src/server/handlers/payments.ts
done:    412 embedded  0 unchanged
✅ Indexed 412 files (1,847 chunks) · 0 unchanged · 38.4s

$ /rag status
🔍 pi-local-rag

  Files indexed:    412
  Chunks:           1847
  Vectors:          1847  (100% coverage)
  Total tokens:     438,219
  Embedding model:  Xenova/all-MiniLM-L6-v2
  Last build:       2026-05-15T20:14:03.221Z
  Storage:          /Users/you/.pi/rag

  RAG injection:    enabled  topK=5  threshold=0.1  alpha=0.4

  File types:
    .ts    231
    .tsx   118
    .md     34
    .json   18
    .yaml    7

$ /rag search "stripe webhook signature verification"
🔍 4 results for "stripe webhook signature verification"  hybrid BM25+vector

payments.ts:142-187  score=0.92
  export async function verifyStripeWebhook(req: Request) {
    const sig = req.headers.get("stripe-signature");
    if (!sig) throw new Error("missing signature header");

webhooks.md:1-23  score=0.71
  # Webhook signing
  All inbound webhooks are verified against the shared secret stored in
  STRIPE_WEBHOOK_SECRET. Stripe signs each request with a t= timestamp...

stripe-client.ts:54-71  score=0.58
  // Construct the event from the raw body and signature header
  stripe.webhooks.constructEvent(rawBody, sig, secret);

$ /rag ext add .zig
Added .zig to indexable extensions. Run /rag index <path> to pick up matching files.

$ /rag ext list
Active file extensions  (61)
  .astro .bash .c .cc .cjs .clj .cljs .cpp .cs .css .csv .cxx .dart .dockerfile
  .edn .env .erl .ex .exs .fish .fs .gitignore .gql .graphql .h .hcl .hpp .htm
  .html .hxx .ini .java .js .json .jsonc .jsx .kt .kts .less .lua .m .md .mdx
  .mjs .mm .pl .php .ps1 .proto .py .rb .rs .rst .sass .scala .scss .sh .sql
  .svelte .swift .tf .toml .ts .tsv .tsx .txt .vb .vue .xml .yaml .yml .zig .zsh
  extra:    .zig

  Edit via /rag ext add <.ext> / remove <.ext> / reset
```

> Output above is approximate — actual colors, spacing, and widget layout depend on your terminal theme and the Pi agent's UI.

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
| `extraExtensions` | `[]` | Extra file extensions to index beyond the defaults |
| `excludeExtensions` | `[]` | Default extensions to skip |

The default extension list covers common source/markup/config files (`.ts`, `.tsx`, `.js`, `.py`, `.rs`, `.go`, `.java`, `.kt`, `.swift`, `.cs`, `.cpp`, `.rb`, `.php`, `.lua`, `.vue`, `.svelte`, `.md`, `.json`, `.yaml`, `.toml`, `.sql`, …). Use `/rag ext add <.ext>` to extend it for project-specific formats (e.g. `.tex`, `.zig`, `.nix`).
