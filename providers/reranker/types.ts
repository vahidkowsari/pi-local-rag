export interface RerankInput {
  id: string;
  text: string;
}

export interface RerankHit {
  id: string;
  rerankScore: number;
}

export interface RerankOptions {
  topK?: number;
  signal?: AbortSignal;
}

export interface Reranker {
  readonly id: string;
  readonly model: string;
  rerank(query: string, candidates: RerankInput[], opts?: RerankOptions): Promise<RerankHit[]>;
}
