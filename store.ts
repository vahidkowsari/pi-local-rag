import { existsSync, mkdirSync, renameSync } from "node:fs";
import { join, dirname } from "node:path";
import { homedir } from "node:os";

// Legacy lens→rag rename target. Overridable via env for tests.
export const LEGACY_DIR = process.env.PI_RAG_LEGACY_DIR ?? join(homedir(), ".pi", "lens");

/** Global fallback store. Lazily evaluated so tests can override $HOME. */
export const GLOBAL_RAG_DIR = () => join(homedir(), ".pi", "rag");

// CLI flag override — set once at startup via setRagDir(), wins over everything.
let _ragDirGetter: (() => string | undefined) | undefined;

export function setRagDirGetter(getter: () => string | undefined): void {
  _ragDirGetter = getter;
}

/**
 * Resolve the active RAG store directory for the current cwd.
 *
 * 1. `--rag-dir` CLI flag — explicit override, wins over everything.
 * 2. `$PI_RAG_DIR` — environment variable override.
 * 3. Walk upward from `process.cwd()` looking for an existing `.pi/rag/`,
 *    stopping before `homedir()` so the global store at `~/.pi/rag/` is only
 *    reached as an explicit fallback (not via walk-up).
 * 4. With `createIfMissing`, create `${cwd}/.pi/rag/`.
 * 5. Otherwise, fall back to `${homedir()}/.pi/rag/`.
 */
export function getRagDir(opts: { createIfMissing?: boolean } = {}): string {
  // CLI flag takes highest priority
  // Check getter first — resolves lazily on first actual use
  const flagDir = _ragDirGetter?.();
  if (flagDir) {
    if (!existsSync(flagDir)) mkdirSync(flagDir, { recursive: true });
    return flagDir;
  }
  const override = process.env.PI_RAG_DIR;
  if (override) {
    if (!existsSync(override)) mkdirSync(override, { recursive: true });
    return override;
  }
  const home = homedir();
  let dir = process.cwd();
  // Walk-up search, stopping before $HOME so we don't accidentally pick up
  // ~/.pi/rag via the walk (that path is reached only as the explicit
  // fallback below).
  while (true) {
    if (dir === home) break;
    const candidate = join(dir, ".pi", "rag");
    if (existsSync(candidate)) return candidate;
    const parent = dirname(dir);
    if (parent === dir) break; // reached filesystem root
    dir = parent;
  }
  if (opts.createIfMissing) {
    const local = join(process.cwd(), ".pi", "rag");
    mkdirSync(local, { recursive: true });
    return local;
  }
  // Fallback: home-dir global. ensureDir handles creation + lens→rag migration.
  const global = GLOBAL_RAG_DIR();
  ensureDir(global);
  return global;
}

/** SQLite database file (legacy default when no active manifest exists). */
export function dbFile(ragDir: string): string { return join(ragDir, "rag.db"); }
export function indexesDir(ragDir: string): string { return join(ragDir, "indexes"); }
export function indexDbFile(ragDir: string, indexId: string): string {
  return join(indexesDir(ragDir), indexId, "rag.db");
}
export function activeManifestFile(ragDir: string): string { return join(ragDir, "active.json"); }
/** Legacy JSON index, kept for the one-shot auto-migration in getFreshDbConn. */
export function legacyIndexFile(ragDir: string): string { return join(ragDir, "index.json"); }
/** @deprecated use dbFile/legacyIndexFile. Kept temporarily for callers that still reach for the JSON path. */
export function indexFile(ragDir: string): string { return join(ragDir, "index.json"); }
export function configFile(ragDir: string): string { return join(ragDir, "config.json"); }

export function ensureDir(ragDir: string) {
  if (existsSync(ragDir)) return;
  // Lens→rag migration only applies at the home-dir global store.
  if (ragDir === GLOBAL_RAG_DIR() && existsSync(LEGACY_DIR)) {
    try {
      renameSync(LEGACY_DIR, ragDir);
      return;
    } catch { /* fall through to mkdir */ }
  }
  mkdirSync(ragDir, { recursive: true });
}
