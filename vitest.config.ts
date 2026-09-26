import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["__tests__/**/*.test.ts"],
    // Several suites set process.env.PI_RAG_DIR / PI_RAG_LEGACY_DIR before
    // importing index.ts, so running them in parallel would race over the
    // shared module instance. Keep files sequential — runtime is < 1 s.
    fileParallelism: false,
    testTimeout: 10_000,
  },
});
