export type EmbeddingRole = "query" | "document";

export interface EmbedBatchOptions {
  onProgress?: (done: number, total: number) => void;
  signal?: AbortSignal;
}

export interface EmbeddingProvider {
  readonly id: string;
  readonly model: string;
  readonly dimensions: number;
  embedQuery(text: string, opts?: { signal?: AbortSignal }): Promise<number[]>;
  embedDocuments(texts: string[], opts?: EmbedBatchOptions): Promise<number[][]>;
}
