export interface Qrels {
  [queryId: string]: { [docId: string]: number };
}

export interface RankedResult {
  docId: string;
  score: number;
}

/**
 * Mean Reciprocal Rank @ K
 * MRR = 1/|Q| * Σ 1/rank_i (of first relevant doc)
 */
export function mrr(
  results: RankedResult[],
  relevant: Set<string>,
  k: number = 10,
): number {
  const topK = results.slice(0, k);
  for (let i = 0; i < topK.length; i++) {
    if (relevant.has(topK[i].docId)) {
      return 1 / (i + 1);
    }
  }
  return 0;
}

/**
 * Precision @ K = |relevant ∩ retrieved| / K
 */
export function precisionAtK(
  results: RankedResult[],
  relevant: Set<string>,
  k: number = 5,
): number {
  const topK = results.slice(0, k);
  let count = 0;
  for (const r of topK) {
    if (relevant.has(r.docId)) count++;
  }
  return count / k;
}

/**
 * Recall @ K = |relevant ∩ retrieved| / |relevant|
 */
export function recallAtK(
  results: RankedResult[],
  relevant: Set<string>,
  k: number = 5,
): number {
  if (relevant.size === 0) return 0;
  const topK = results.slice(0, k);
  let count = 0;
  for (const r of topK) {
    if (relevant.has(r.docId)) count++;
  }
  return count / relevant.size;
}

/**
 * NDCG @ K (Normalized Discounted Cumulative Gain)
 */
export function ndcgAtK(
  results: RankedResult[],
  qrels: { [docId: string]: number },
  k: number = 10,
): number {
  const topK = results.slice(0, k);

  // DCG
  let dcg = 0;
  for (let i = 0; i < topK.length; i++) {
    const rel = qrels[topK[i].docId] || 0;
    dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
  }

  // Ideal DCG
  const idealRels = Object.values(qrels)
    .sort((a, b) => b - a)
    .slice(0, k);
  let idcg = 0;
  for (let i = 0; i < idealRels.length; i++) {
    idcg += (Math.pow(2, idealRels[i]) - 1) / Math.log2(i + 2);
  }

  return idcg === 0 ? 0 : dcg / idcg;
}

export interface BenchmarkResult {
  method: string;
  mrrAt10: number;
  precisionAt5: number;
  recallAt5: number;
  ndcgAt10: number;
  avgLatencyMs: number;
  stdLatencyMs: number;
  perQueryResults?: {
    queryId: string;
    mrr: number;
    precision: number;
    recall: number;
    ndcg: number;
    latencyMs: number;
  }[];
}

export function aggregateMetrics(
  perQuery: { mrr: number; precision: number; recall: number; ndcg: number; latencyMs: number }[],
): { mean: number; std: number }[] {
  const keys = ['mrr', 'precision', 'recall', 'ndcg', 'latencyMs'] as const;
  return keys.map((key) => {
    const values = perQuery.map((q) => q[key]);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, v) => a + (v - mean) ** 2, 0) / values.length;
    return { mean, std: Math.sqrt(variance) };
  });
}
