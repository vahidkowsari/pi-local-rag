import type { RerankHit, RerankInput, RerankOptions, Reranker } from "./types.ts";

/** Identity reranker: keep input order and slice to topK. */
export class NoneReranker implements Reranker {
  readonly id = "none";
  readonly model = "none";

  async rerank(query: string, candidates: RerankInput[], opts?: RerankOptions): Promise<RerankHit[]> {
    void query;
    const sliced = opts?.topK !== undefined ? candidates.slice(0, opts.topK) : candidates;
    return sliced.map(c => ({ id: c.id, rerankScore: 0 }));
  }
}
