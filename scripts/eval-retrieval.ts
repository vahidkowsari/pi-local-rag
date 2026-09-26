/**
 * Retrieval evaluation. Separate from `npm test`.
 *
 *   npm run eval:retrieval
 *   npm run eval:retrieval -- --groups=local-bm25,local-hybrid
 *   npm run eval:retrieval -- --corpus=/path/to/docs
 *
 * Groups:
 *   local-bm25                 FTS-only baseline (always runnable)
 *   local-hybrid               MiniLM hybrid (skipped when SKIP_EMBEDDING_TESTS=1)
 *   cloud-embedding            Voyage embeddings (skipped without VOYAGE_API_KEY)
 *   cloud-embedding+reranker   Voyage embeddings + rerank (skipped without key)
 *
 * Cloud execution paths exist; they are skipped only when the key is absent
 * or EVAL_SKIP_CLOUD=1. Never invents metric numbers.
 */
import { cpSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync, rmSync, existsSync } from "node:fs";
import { dirname, join, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { extractBlocks } from "../parsing.ts";
import { chunkBlocks, sha256, collectFiles } from "../chunking.ts";
import type Database from "better-sqlite3";
import { closeDbConn, openDbAt } from "../db.ts";
import * as repo from "../repository.ts";
import { defaultConfig, type RagConfig } from "../config.ts";
import { stampFingerprints } from "../index-manager.ts";
import { retrieveWithCandidates, type RetrieveMethod } from "../retrieval.ts";
import { indexFiles } from "../indexing.ts";
import { CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS } from "../constants.ts";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const questionsPath = join(root, "eval", "questions.json");
const questions = JSON.parse(readFileSync(questionsPath, "utf-8")) as Array<{
  id: string;
  query: string;
  acceptable: Array<{ file?: string; page?: number; snippet?: string }>;
  unanswerable?: boolean;
}>;

const ALL_GROUPS = ["local-bm25", "local-hybrid", "cloud-embedding", "cloud-embedding+reranker"] as const;
type GroupName = typeof ALL_GROUPS[number];

function parseArgs(argv: string[]): { groups: GroupName[]; corpus?: string } {
  let groups = [...ALL_GROUPS] as GroupName[];
  let corpus: string | undefined;
  for (const a of argv) {
    if (a.startsWith("--groups=")) {
      groups = a.slice(9).split(",").map(s => s.trim()).filter(Boolean) as GroupName[];
    } else if (a.startsWith("--corpus=")) {
      corpus = a.slice(9);
    }
  }
  return { groups, corpus };
}

const args = parseArgs(process.argv.slice(2));
const hasKey = !!process.env.VOYAGE_API_KEY;
const skipCloud = process.env.EVAL_SKIP_CLOUD === "1";
const skipOnnx = process.env.SKIP_EMBEDDING_TESTS === "1";
const outDir = join(root, "eval", "runs");
mkdirSync(outDir, { recursive: true });

function defaultFiles(): string[] {
  const files = [
    join(root, "README.md"),
    join(root, "search.ts"),
    join(root, "index.ts"),
    join(root, "constants.ts"),
    join(root, "config.ts"),
    join(root, "index-manager.ts"),
    join(root, "chunking.ts"),
    join(root, "parsing.ts"),
    join(root, "context.ts"),
    join(root, "providers/embedding/voyage.ts"),
    join(root, "providers/reranker/none.ts"),
    join(root, "indexing.ts"),
  ];
  const pdfSrc = join(root, "__tests__", "fixtures", "sample.pdf");
  const pdfDest = join(tmpdir(), "pi-rag-eval-sample.pdf");
  cpSync(pdfSrc, pdfDest);
  files.push(pdfDest);
  return files;
}

function corpusFiles(corpus?: string): string[] {
  if (!corpus) return defaultFiles();
  if (!existsSync(corpus)) throw new Error(`corpus path not found: ${corpus}`);
  const found = collectFiles(corpus);
  const pdfSrc = join(root, "__tests__", "fixtures", "sample.pdf");
  found.push(pdfSrc);
  return found;
}

function contentRelevant(
  hit: { chunk: { file: string; content: string } },
  acceptable: Array<{ file?: string; snippet?: string; page?: number }>,
): boolean {
  return acceptable.some(a => {
    if (a.file && !hit.chunk.file.endsWith(a.file) && !hit.chunk.file.includes(a.file)) return false;
    if (a.snippet && !hit.chunk.content.includes(a.snippet)) return false;
    return true;
  });
}

function firstContentHitPage(
  hits: Array<{ chunk: { file: string; content: string; pageStart?: number | null } }>,
  acceptable: Array<{ file?: string; snippet?: string; page?: number }>,
): number | null | undefined {
  const hit = hits.find(h => contentRelevant(h, acceptable));
  return hit ? hit.chunk.pageStart : undefined;
}

interface GroupMetrics {
  recallAtCandidates: number;
  hitAt5: number;
  mrrAt5: number;
  sourceAccuracy: number | null;
  unanswerable: number | null;
  questions: number;
  durationMs: number;
  degradedCount: number;
  methods: RetrieveMethod[];
}

function groupStatus(perQuestion: Array<Record<string, unknown>>): { status: string; reason?: string; degraded?: number } {
  const n = perQuestion.length;
  const deg = perQuestion.filter(q => q.degraded).length;
  if (deg === 0) return { status: "ok" };
  if (deg === n) return { status: "error", reason: `all ${n} questions degraded`, degraded: deg };
  return { status: "degraded", reason: `${deg}/${n} questions degraded`, degraded: deg };
}

async function scoreGroup(
  db: Database.Database,
  config: RagConfig,
  candidateK: number,
  topK: number,
): Promise<{ metrics: GroupMetrics; perQuestion: Array<Record<string, unknown>> }> {
  const perQuestion: Array<Record<string, unknown>> = [];
  let recallHits = 0;
  let hitAt5 = 0;
  let mrr = 0;
  let pageChecked = 0;
  let pageCorrect = 0;
  let unanswerableOk = 0;
  let unanswerableN = 0;
  const started = Date.now();
  const answerable = questions.filter(q => !q.unanswerable).length;
  const methods = new Set<RetrieveMethod>();

  for (const q of questions) {
    const t0 = Date.now();
    const bundle = await retrieveWithCandidates(q.query, {
      limit: topK,
      candidateTopK: candidateK,
      expandCandidates: true,
      alpha: config.ragAlpha,
      db,
      config,
    });
    const candidates = bundle.candidates;
    const final = bundle.hits;
    methods.add(bundle.method);
    const ms = Date.now() - t0;
    if (q.unanswerable) {
      unanswerableN++;
      const above = final.filter(h => h.hybrid >= config.ragScoreThreshold);
      if (!bundle.degraded && above.length === 0) unanswerableOk++;
      perQuestion.push({
        id: q.id, unanswerable: true, hits: above.length, candidateCount: candidates.length, ms,
        degraded: bundle.degraded ?? null, method: bundle.method,
      });
      continue;
    }
    const inCandidates = candidates.some(h => contentRelevant(h, q.acceptable));
    const rank = final.findIndex(h => contentRelevant(h, q.acceptable));
    if (inCandidates) recallHits++;
    if (rank >= 0) {
      hitAt5++;
      mrr += 1 / (rank + 1);
    }
    const pageLabeled = q.acceptable.filter(a => a.page != null);
    if (pageLabeled.length && rank >= 0) {
      pageChecked++;
      const page = firstContentHitPage(final, q.acceptable);
      if (pageLabeled.some(a => a.page === page)) pageCorrect++;
    }
    perQuestion.push({
      id: q.id,
      inCandidates,
      matchedTop5: rank >= 0,
      rank: rank >= 0 ? rank + 1 : null,
      candidateCount: candidates.length,
      ms,
      topFile: final[0]?.chunk.file ?? null,
      degraded: bundle.degraded ?? null,
      method: bundle.method,
    });
  }

  return {
    metrics: {
      recallAtCandidates: answerable ? recallHits / answerable : 0,
      hitAt5: answerable ? hitAt5 / answerable : 0,
      mrrAt5: answerable ? mrr / answerable : 0,
      sourceAccuracy: pageChecked ? pageCorrect / pageChecked : null,
      unanswerable: unanswerableN ? unanswerableOk / unanswerableN : null,
      questions: answerable,
      durationMs: Date.now() - started,
      degradedCount: perQuestion.filter(q => q.degraded).length,
      methods: [...methods],
    },
    perQuestion,
  };
}

async function indexBm25(files: string[], work: string) {
  const dbPath = join(work, "rag.db");
  const db = openDbAt(dbPath, 384);
  const indexedAt = new Date().toISOString();
  for (const fp of files) {
    const parsed = await extractBlocks(fp);
    const chunks = chunkBlocks(parsed.blocks, {
      targetTokens: CHUNK_TARGET_TOKENS,
      maxTokens: CHUNK_MAX_TOKENS,
      overlapTokens: CHUNK_OVERLAP_TOKENS,
    });
    const storedPath = basename(fp) === "sample.pdf" ? "sample.pdf" : fp;
    for (let i = 0; i < chunks.length; i++) {
      const c = chunks[i];
      repo.insertChunk(db, {
        id: `${sha256(fp)}-${i}`,
        filePath: storedPath,
        content: c.content,
        lineStart: c.lineStart,
        lineEnd: c.lineEnd,
        hash: sha256(c.content),
        indexedAt,
        tokens: Math.max(1, Math.ceil(c.content.length / 4)),
        pageStart: c.pageStart ?? null,
        pageEnd: c.pageEnd ?? null,
        section: c.section ?? null,
        chunkIndex: i,
      });
    }
    repo.upsertFile(db, storedPath, parsed.hash, chunks.length, indexedAt, parsed.size, false, {
      documentId: sha256(fp),
      title: basename(fp),
    });
  }
  const cfg = defaultConfig();
  cfg.ragAlpha = 1;
  stampFingerprints(db, cfg);
  repo.setMetadata(db, repo.MetadataKey.LastBuild, indexedAt);
  return { db, cfg };
}

async function runGroup(name: GroupName, files: string[]): Promise<Record<string, unknown>> {
  if (name === "local-hybrid" && skipOnnx) {
    return { name, status: "not-run", reason: "SKIP_EMBEDDING_TESTS=1; local MiniLM hybrid path is implemented but skipped." };
  }
  if ((name === "cloud-embedding" || name === "cloud-embedding+reranker") && skipCloud) {
    return { name, status: "not-run", reason: "EVAL_SKIP_CLOUD=1; cloud execution path is implemented but skipped." };
  }
  if ((name === "cloud-embedding" || name === "cloud-embedding+reranker") && !hasKey) {
    return { name, status: "not-run", reason: "VOYAGE_API_KEY is not set." };
  }

  const work = mkdtempSync(join(tmpdir(), `pi-rag-eval-${name}-`));
  const prevDir = process.env.PI_RAG_DIR;
  process.env.PI_RAG_DIR = work;
  closeDbConn();
  try {
    if (name === "local-bm25") {
      const { db, cfg } = await indexBm25(files, work);
      const scored = await scoreGroup(db, cfg, 30, 5);
      db.close();
      return { name, embedding: "none-fts", configuredReranker: cfg.reranker.provider, ...groupStatus(scored.perQuestion), ...scored.metrics, perQuestion: scored.perQuestion };
    }

    const cfg = defaultConfig();
    if (name.startsWith("cloud-")) {
      cfg.embedding = { provider: "voyage", model: "voyage-4-lite", dimensions: 1024 };
      cfg.http = { timeoutMs: 30_000, maxRetries: 2 };
    }
    if (name === "cloud-embedding+reranker") {
      cfg.reranker = { provider: "voyage", model: "rerank-2.5-lite" };
    }
    cfg.trackedPaths = files.map(f => dirname(f)).filter((v, i, a) => a.indexOf(v) === i);
    writeFileSync(join(work, "config.json"), JSON.stringify(cfg, null, 2));
    const result = await indexFiles(files, {}, undefined, true);
    if (result.failed > 0) {
      return { name, status: "error", reason: result.errors.join("; "), indexed: result.indexed, failed: result.failed };
    }
    const { getDbConn } = await import("../db.ts");
    const db = getDbConn();
    const scored = await scoreGroup(db, cfg, 30, 5);
    return {
      name,
      embedding: cfg.embedding.provider,
      configuredReranker: cfg.reranker.provider,
      ...groupStatus(scored.perQuestion),
      ...scored.metrics,
      perQuestion: scored.perQuestion,
    };
  } finally {
    closeDbConn();
    if (prevDir !== undefined) process.env.PI_RAG_DIR = prevDir;
    else delete process.env.PI_RAG_DIR;
    rmSync(work, { recursive: true, force: true });
  }
}

const files = corpusFiles(args.corpus);
const runs: Array<Record<string, unknown>> = [];
for (const g of args.groups) {
  if (!ALL_GROUPS.includes(g)) {
    runs.push({ name: g, status: "not-run", reason: `unknown group "${g}"` });
    continue;
  }
  process.stderr.write(`[eval] running ${g}…\n`);
  runs.push(await runGroup(g, files));
}

const report = {
  date: new Date().toISOString(),
  commit: process.env.GIT_COMMIT ?? "unknown",
  questions: questions.length,
  groupsRequested: args.groups,
  corpus: args.corpus ?? "builtin-repo-sources+sample.pdf",
  metricDefinitions: {
    recallAtCandidates: "fraction of answerable questions with a content/file match in the candidate pool (k=30)",
    hitAt5: "fraction of answerable questions with a content/file match in the final top 5",
    mrrAt5: "mean reciprocal rank of the first content/file match in the top 5",
    sourceAccuracy: "among page-labeled questions that have a content match in top 5, fraction whose first content match has the labeled page; wrong pages count as 0",
    unanswerable: "fraction of unanswerable questions with zero hits above threshold",
  },
  runs,
  questionsFile: questionsPath,
};

const out = join(outDir, `eval-${report.date.slice(0, 10)}.json`);
writeFileSync(out, JSON.stringify(report, null, 2));
const local = runs.find(r => r.name === "local-bm25");
process.stdout.write(`Wrote ${out}. groups=${runs.map(r => `${r.name}:${r.status}`).join(", ")}\n`);
if (local && local.status === "ok") {
  process.stdout.write(`local-bm25 hitAt5=${local.hitAt5} recallAtCandidates=${local.recallAtCandidates} mrrAt5=${local.mrrAt5}\n`);
}
