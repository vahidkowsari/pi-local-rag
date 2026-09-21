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
- Commit: recorded after `git commit` on this branch
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

- Status: **not started**

## A3 — Keep old index on failure

- Status: **not started**

## B–E

- Status: **not started**
