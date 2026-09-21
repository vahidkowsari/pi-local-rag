/**
 * Retrieval evaluation runner. Separate from `npm test`.
 *
 *   SKIP_EMBEDDING_TESTS=1 npx --yes tsx scripts/eval-retrieval.ts
 *   npm run eval:retrieval
 *
 * Without VOYAGE_API_KEY, cloud columns are recorded as "not-run".
 * Never invents metric numbers.
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const questionsPath = join(root, "eval", "questions.json");
const questions = JSON.parse(readFileSync(questionsPath, "utf-8")) as Array<{
  id: string;
  query: string;
  acceptable: Array<{ file?: string; page?: number; snippet?: string }>;
  unanswerable?: boolean;
}>;

const hasKey = !!process.env.VOYAGE_API_KEY;
const outDir = join(root, "eval", "runs");
mkdirSync(outDir, { recursive: true });

const report = {
  date: new Date().toISOString(),
  commit: process.env.GIT_COMMIT ?? "unknown",
  questions: questions.length,
  metrics: {
    recallAtCandidates: "defined as fraction of questions whose acceptable source appears in the candidate pool, after de-duping overlap chunks of the same document+page",
    hitAt5: "defined as fraction of questions with an acceptable source in the final top 5",
    mrrAt5: "mean reciprocal rank of the first acceptable source in the top 5",
    sourceAccuracy: "fraction of hits whose pageStart matches the labeled page, when a page is labeled",
    unanswerable: "fraction of unanswerable questions that return zero hits above threshold",
  },
  runs: [
    { name: "local-embedding", status: "not-run", reason: "Wire this runner to a prepared fixture corpus before recording numbers." },
    { name: "cloud-embedding", status: hasKey ? "not-run" : "not-run", reason: hasKey ? "Corpus not prepared in this invocation." : "VOYAGE_API_KEY is not set." },
    { name: "cloud-embedding+reranker", status: hasKey ? "not-run" : "not-run", reason: hasKey ? "Corpus not prepared in this invocation." : "VOYAGE_API_KEY is not set." },
  ],
  questionsFile: questionsPath,
};

const out = join(outDir, `eval-${report.date.slice(0, 10)}.json`);
writeFileSync(out, JSON.stringify(report, null, 2));
process.stdout.write(`Wrote ${out} with ${questions.length} questions. Cloud runs: ${hasKey ? "key present (not executed)" : "not-run (no key)"}.\n`);
