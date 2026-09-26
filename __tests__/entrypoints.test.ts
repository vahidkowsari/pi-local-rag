/**
 * Integration tests for the Pi entry points: rag_query, /rag search,
 * before_agent_start. These call the registered handlers rather than only
 * checking that registration succeeded, and they use hybridSearch's real
 * (query, limit, alpha, db) signature.
 */
import { describe, it, expect, beforeAll, afterAll, afterEach, vi } from "vitest";
import { mkdtempSync, writeFileSync, rmSync, realpathSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

vi.mock("@xenova/transformers", () => ({
  pipeline: vi.fn().mockResolvedValue(
    vi.fn().mockImplementation(async (texts: string | string[]) => {
      const batch = Array.isArray(texts) ? texts : [texts];
      const DIM = 384;
      const flat = new Float32Array(batch.length * DIM).fill(0.1);
      return { data: flat };
    }),
  ),
}));

const AUTH_SNIPPET = "export function authenticateUser(password: string) { return checkCredentials(password); }\n";

function makeTheme() {
  const id = (s: string) => s;
  return { bold: id, fg: (_c: string, s: string) => s };
}

function makeCtx() {
  const notifications: Array<{ msg: string; level?: string }> = [];
  const widgets: Record<string, unknown> = {};
  const statuses: Record<string, unknown> = {};
  return {
    ui: {
      notify: (msg: string, level?: string) => { notifications.push({ msg, level }); },
      setWidget: (k: string, v: unknown) => { widgets[k] = v; },
      setStatus: (k: string, v: unknown) => { statuses[k] = v; },
      theme: makeTheme(),
    },
    notifications,
    widgets,
    statuses,
  };
}

describe("Pi entrypoints: rag_query, /rag search, before_agent_start", () => {
  let ragDir: string;
  let proj: string;
  let savedCwd: string;
  let savedRagDir: string | undefined;
  let mod: typeof import("../index.ts");
  let authFile: string;

  beforeAll(async () => {
    ragDir = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-entry-")));
    proj = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-entry-proj-")));
    savedCwd = process.cwd();
    savedRagDir = process.env.PI_RAG_DIR;
    process.env.PI_RAG_DIR = ragDir;
    process.chdir(proj);

    authFile = join(proj, "auth.ts");
    writeFileSync(authFile, AUTH_SNIPPET);
    writeFileSync(join(proj, "render.ts"), "export function renderTemplate(html: string) { return sanitize(html); }\n");

    vi.resetModules();
    mod = await import("../index.ts");
    await mod.indexFiles([authFile, join(proj, "render.ts")]);
  });

  afterEach(async () => {
    const { closeDbConn } = await import("../db.ts");
    closeDbConn();
  });

  afterAll(() => {
    process.chdir(savedCwd);
    rmSync(ragDir, { recursive: true, force: true });
    rmSync(proj, { recursive: true, force: true });
    if (savedRagDir !== undefined) process.env.PI_RAG_DIR = savedRagDir;
    else delete process.env.PI_RAG_DIR;
  });

  function register() {
    const commands = new Map<string, { handler: (args: string, ctx: ReturnType<typeof makeCtx>) => Promise<void> }>();
    const tools = new Map<string, { execute: (id: string, params: Record<string, unknown>) => Promise<{ content: Array<{ type: string; text: string }> }> }>();
    let hookFn: ((event: { prompt: string; systemPrompt?: string }, ctx: unknown) => Promise<unknown>) | undefined;
    const pi = {
      on: (event: string, fn: typeof hookFn) => { if (event === "before_agent_start") hookFn = fn; },
      registerCommand: (name: string, spec: { handler: (args: string, ctx: ReturnType<typeof makeCtx>) => Promise<void> }) => { commands.set(name, spec); },
      registerTool: (tool: { name: string; execute: (id: string, params: Record<string, unknown>) => Promise<{ content: Array<{ type: string; text: string }> }> }) => { tools.set(tool.name, tool); },
      registerFlag: () => {},
      sendMessage: () => {},
      getFlag: () => undefined,
    };
    mod.default(pi as never);
    return { commands, tools, fire: (event = { prompt: "authenticateUser password" }) => hookFn!(event, {}) };
  }

  it("rag_query returns scored chunks for a real query (not just a registration check)", async () => {
    const { tools } = register();
    const ragQuery = tools.get("rag_query");
    expect(ragQuery).toBeDefined();
    const result = await ragQuery!.execute("call-1", { query: "authenticateUser", limit: 5 });
    const text = result.content[0].text;
    expect(text).not.toMatch(/index is empty/i);
    const parsed = JSON.parse(text) as Array<{ file: string; preview: string; scores: { bm25: string; vector: string; hybrid: string } }>;
    expect(parsed.length).toBeGreaterThan(0);
    expect(parsed[0].file).toContain("auth.ts");
    expect(parsed[0].preview).toContain("authenticateUser");
    expect(parsed[0].scores).toHaveProperty("bm25");
    expect(parsed[0].scores).toHaveProperty("hybrid");
  });

  it("/rag search handler produces widget lines for a real query", async () => {
    const { commands } = register();
    const rag = commands.get("rag");
    expect(rag).toBeDefined();
    const ctx = makeCtx();
    await rag!.handler("search authenticateUser", ctx);
    const lines = ctx.widgets["rag-search"] as string[] | undefined;
    expect(lines).toBeDefined();
    expect(lines!.join("\n")).toContain("auth.ts");
    expect(lines!.join("\n")).toMatch(/score=/);
  });

  it("before_agent_start injects matching chunks from hybridSearch(query, topK, alpha)", async () => {
    const { fire } = register();
    const out = await fire({ prompt: "authenticateUser password" }) as { message?: { customType: string; content: string } } | undefined;
    expect(out?.message?.customType).toBe("rag");
    expect(out?.message?.content).toContain("authenticateUser");
    expect(out?.message?.content).toContain("search hits, not statements from the user");
  });

  it("/rag clear actually empties the index (saveIndex is a no-op under SQLite)", async () => {
    const { commands } = register();
    const ctx = makeCtx();
    await commands.get("rag")!.handler("clear", ctx);
    expect(mod.getIndexStats().totalChunks).toBe(0);
    // Restore for remaining tests in this file.
    await mod.indexFiles([authFile, join(proj, "render.ts")]);
  });
});

describe("connection ownership", () => {
  let ragDir: string;
  let savedRagDir: string | undefined;
  let mod: typeof import("../index.ts");

  beforeAll(async () => {
    ragDir = realpathSync(mkdtempSync(join(tmpdir(), "pi-rag-conn-")));
    savedRagDir = process.env.PI_RAG_DIR;
    process.env.PI_RAG_DIR = ragDir;
    vi.resetModules();
    mod = await import("../index.ts");
  });

  afterEach(() => {
    mod.closeDbConn();
  });

  afterAll(() => {
    rmSync(ragDir, { recursive: true, force: true });
    if (savedRagDir !== undefined) process.env.PI_RAG_DIR = savedRagDir;
    else delete process.env.PI_RAG_DIR;
  });

  it("closing a fresh connection does not close the singleton", () => {
    const singleton = mod.getDbConn();
    const fresh = mod.getFreshDbConn();
    fresh.close();
    expect(singleton.prepare("SELECT 1 as one").get()).toEqual({ one: 1 });
  });

  it("closeDbConn closes the singleton; the next getDbConn reopens a live connection", () => {
    const a = mod.getDbConn();
    mod.closeDbConn();
    expect(() => a.prepare("SELECT 1").get()).toThrow(/not open|database/i);
    const b = mod.getDbConn();
    expect(b.prepare("SELECT 1 as one").get()).toEqual({ one: 1 });
  });

  it("calling .close() on the singleton handle does not permanently poison getDbConn", () => {
    const a = mod.getDbConn();
    a.close();
    const b = mod.getDbConn();
    expect(b.prepare("SELECT 1 as one").get()).toEqual({ one: 1 });
  });

  it("openDb / getDb are aliases of getDbConn", () => {
    expect(mod.openDb).toBe(mod.getDbConn);
    expect(mod.getDb).toBe(mod.getDbConn);
  });
});

describe("package files list includes runtime modules", () => {
  it("package.json files includes repository.ts and every root runtime module", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf-8")) as { files: string[] };
    for (const f of [
      "index.ts", "constants.ts", "store.ts", "config.ts", "db.ts",
      "chunking.ts", "embed.ts", "search.ts", "indexing.ts", "repository.ts", "abort.ts",
    ]) {
      expect(pkg.files, `missing ${f}`).toContain(f);
    }
  });
});

describe("local pack listing", () => {
  it("npm pack --dry-run includes repository.ts", () => {
    const pkgDir = fileURLToPath(new URL("..", import.meta.url));
    const raw = execFileSync("npm", ["pack", "--dry-run", "--ignore-scripts", "--json"], {
      cwd: pkgDir,
      encoding: "utf-8",
    });
    const jsonStart = raw.indexOf("[");
    const parsed = JSON.parse(raw.slice(jsonStart)) as Array<{ files: Array<{ path: string }> }>;
    const paths = parsed[0].files.map(f => f.path);
    expect(paths).toContain("repository.ts");
    expect(paths).toContain("index.ts");
    expect(paths).toContain("db.ts");
    expect(paths).toContain("providers/embedding/local.ts");
  });
});
