/**
 * Explicit Voyage rerank smoke test. Not part of `npm test`.
 *   VOYAGE_API_KEY=... npm run smoke:voyage-rerank
 */
import { VoyageReranker } from "../providers/reranker/voyage.ts";

if (!process.env.VOYAGE_API_KEY) {
  process.stderr.write("UNVERIFIED: VOYAGE_API_KEY is not set. Skipping live Voyage rerank smoke.\n");
  process.exit(0);
}

const reranker = new VoyageReranker({ apiKey: process.env.VOYAGE_API_KEY, model: "rerank-2.5-lite" });
const started = Date.now();
const hits = await reranker.rerank("hybrid search", [
  { id: "a", text: "hybrid BM25 and vector retrieval over local files" },
  { id: "b", text: "banana bread recipe with walnuts" },
]);
process.stdout.write(JSON.stringify({
  status: "ok",
  model: reranker.model,
  order: hits.map(h => h.id),
  requestCount: reranker.requestCount,
  latencyMs: Date.now() - started,
}, null, 2) + "\n");
