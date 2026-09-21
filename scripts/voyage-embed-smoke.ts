/**
 * Explicit Voyage embedding smoke test. Not part of `npm test`.
 *
 *   VOYAGE_API_KEY=... npm run smoke:voyage-embed
 *
 * Uses only short homemade strings. Does not read the user's knowledge base.
 */
import { createEmbeddingProvider } from "../providers/embedding/factory.ts";
import { defaultConfig } from "../config.ts";

const key = process.env.VOYAGE_API_KEY;
if (!key) {
  process.stderr.write("UNVERIFIED: VOYAGE_API_KEY is not set. Skipping live Voyage embedding smoke.\n");
  process.exit(0);
}

const config = defaultConfig();
config.embedding = { provider: "voyage", model: "voyage-4-lite", dimensions: 1024 };

const provider = createEmbeddingProvider(config);
const started = Date.now();
const query = await provider.embedQuery("short homemade query about hybrid search");
const docs = await provider.embedDocuments([
  "short homemade document one about retrieval",
  "short homemade document two about embeddings",
]);
const ms = Date.now() - started;

const report = {
  status: "ok",
  provider: provider.id,
  model: provider.model,
  dimensions: provider.dimensions,
  queryDim: query.length,
  documentCount: docs.length,
  documentDims: docs.map(d => d.length),
  requestCount: "requestCount" in provider ? (provider as { requestCount: number }).requestCount : undefined,
  latencyMs: ms,
};
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
