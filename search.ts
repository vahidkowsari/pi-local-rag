import Database from "better-sqlite3";
import { embed } from "./embed.ts";
import { float32ToBuffer } from "./db.ts";
import { Chunk } from "./db.ts";

export interface ScoredChunk {
  chunk: Chunk;
  bm25: number;
  vector: number;
  hybrid: number;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function normalize(scores: number[]): number[] {
  const max = Math.max(...scores);
  const min = Math.min(...scores);
  const range = max - min;
  if (range === 0) return scores.map(() => 0);
  return scores.map(s => (s - min) / range);
}

function l2ToCosine(l2Dist: number): number {
  return 1 - (l2Dist * l2Dist) / 2;
}

/**
 * Hybrid search using SQLite FTS5 (BM25) + sqlite-vec (vector).
 */
export async function hybridSearch(
  query: string,
  limit = 10,
  alpha = 0.4,
  _db?: Database.Database
): Promise<ScoredChunk[]> {
  const database = _db ?? (await import("./db.ts")).getDb();
  const shouldClose = !_db;

  try {
    // Fast existence check — LIMIT 1 avoids full table scan
    const hasChunks = database.prepare("SELECT 1 FROM chunks LIMIT 1").get();
    if (!hasChunks) return [];

    // BM25 via FTS5 — cap candidates to avoid scanning entire index
    const ftsQuery = query.split(/\s+/).map(t => `"${t.replace(/"/g, '""')}"`).join(" ");
    const ftsLimit = Math.max(limit * 20, 200);
    const ftsResults = database.prepare(`
      SELECT chunks_fts.rowid, bm25(chunks_fts) as bm25_score
      FROM chunks_fts
      WHERE chunks_fts MATCH ?
      ORDER BY bm25(chunks_fts)
      LIMIT ?
    `).all(ftsQuery, ftsLimit);

    // Vector via sqlite-vec
    const queryVec = await embed(query);
    const queryBuf = float32ToBuffer(queryVec);
    const vecLimit = Math.max(limit * 10, 100);
    const vecResults = database.prepare(`
      SELECT rowid, distance
      FROM chunks_vec
      WHERE embedding MATCH ?
      LIMIT ?
    `).bind(queryBuf, vecLimit).all();

    const ftsRowIds = new Set((ftsResults as any[]).map((r: any) => r.rowid as number));
    const vecRowIds = new Set((vecResults as any[]).map((r: any) => r.rowid as number));
    const allRowIds: Set<number> = new Set([...ftsRowIds, ...vecRowIds]);

    if (allRowIds.size === 0) return [];

    const rowidPlaceholders = Array.from(allRowIds).map(() => "?").join(",");
    const rowidValues: number[] = Array.from(allRowIds);
    const chunks = database.prepare(`
      SELECT rowid, id, file_path, chunk_content, line_start, line_end,
             chunk_hash, indexed_at, tokens
      FROM chunks
      WHERE rowid IN (${rowidPlaceholders})
    `).all(...rowidValues) as Array<{
      rowid: number; id: string; file_path: string; chunk_content: string;
      line_start: number; line_end: number; chunk_hash: string;
      indexed_at: string; tokens: number;
    }>;

    const chunkMap = new Map<number, typeof chunks[0]>();
    for (const c of chunks) chunkMap.set(c.rowid, c);

    const bm25Map = new Map<number, number>();
    for (const r of ftsResults as any[]) bm25Map.set(r.rowid, r.bm25_score);

    const distMap = new Map<number, number>();
    for (const r of vecResults as any[]) distMap.set(r.rowid, r.distance);

    const bm25Scores = (ftsResults as any[]).map((r: any) => r.bm25_score);
    const hasBm25 = bm25Scores.length > 0;
    const distances = (vecResults as any[]).map((r: any) => r.distance);
    const hasVectors = distances.length > 0;

    // Normalize BM25
    const bm25NormMap = new Map<number, number>();
    if (hasBm25) {
      const bm25Max = Math.max(...bm25Scores);
      const bm25Min = Math.min(...bm25Scores);
      const bm25Range = bm25Max - bm25Min;
      if (bm25Range === 0) {
        for (const r of ftsResults as any[]) bm25NormMap.set(r.rowid, 1);
      } else {
        for (const r of ftsResults as any[]) {
          bm25NormMap.set(r.rowid, (r.bm25_score - bm25Min) / bm25Range);
        }
      }
    }

    // Normalize distances → cosine → min-max
    const vecNormMap = new Map<number, number>();
    if (hasVectors) {
      for (const r of vecResults as any[]) {
        vecNormMap.set(r.rowid, l2ToCosine(r.distance));
      }
      const cosines = Array.from(vecNormMap.values());
      const cosMax = Math.max(...cosines);
      const cosMin = Math.min(...cosines);
      const cosRange = cosMax - cosMin;
      if (cosRange > 0) {
        const normalized = new Map<number, number>();
        for (const [rowid, cos] of vecNormMap) {
          normalized.set(rowid, (cos - cosMin) / cosRange);
        }
        vecNormMap.clear();
        for (const [k, v] of normalized) vecNormMap.set(k, v);
      } else {
        for (const k of vecNormMap.keys()) vecNormMap.set(k, 1);
      }
    }

    // Build scored results
    const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);
    const scored: ScoredChunk[] = [];

    for (const rowid of allRowIds) {
      const c = chunkMap.get(rowid);
      if (!c) continue;

      const bm25Norm = bm25NormMap.get(rowid) ?? 0;
      const vecNorm = vecNormMap.get(rowid) ?? 0;

      let bm25Final = bm25Norm;
      if (c.file_path.toLowerCase().includes(terms[0] ?? "")) {
        bm25Final = Math.min(1, bm25Final * 1.5);
      }

      const hybrid = hasVectors
        ? alpha * bm25Final + (1 - alpha) * vecNorm
        : bm25Final;

      scored.push({
        chunk: {
          id: c.id, file: c.file_path, content: c.chunk_content,
          lineStart: c.line_start, lineEnd: c.line_end,
          hash: c.chunk_hash, indexed: c.indexed_at, tokens: c.tokens,
        },
        bm25: bm25Final, vector: vecNorm, hybrid,
      });
    }

    return scored
      .filter(s => s.hybrid > 0)
      .sort((a, b) => b.hybrid - a.hybrid)
      .slice(0, limit);
  } finally {
    if (shouldClose) database.close();
  }
}
