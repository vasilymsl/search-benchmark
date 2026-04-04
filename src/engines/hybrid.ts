const RRF_K = 60;

export interface SearchResult {
  docIndex: number;
  score: number;
}

/**
 * Reciprocal Rank Fusion: combines ranked lists from multiple retrieval methods.
 * score(d) = Σ 1 / (k + rank_i(d))
 */
export function reciprocalRankFusion(
  ...rankedLists: SearchResult[][]
): SearchResult[] {
  const fusedScores = new Map<number, number>();

  for (const list of rankedLists) {
    for (let rank = 0; rank < list.length; rank++) {
      const docIndex = list[rank].docIndex;
      const rrfScore = 1 / (RRF_K + rank + 1); // rank is 0-based, formula uses 1-based
      fusedScores.set(docIndex, (fusedScores.get(docIndex) || 0) + rrfScore);
    }
  }

  const results: SearchResult[] = [];
  for (const [docIndex, score] of fusedScores) {
    results.push({ docIndex, score });
  }

  results.sort((a, b) => b.score - a.score);
  return results;
}

export function searchHybrid(
  bm25Results: SearchResult[],
  semanticResults: SearchResult[],
  topK: number = 10,
): SearchResult[] {
  return reciprocalRankFusion(bm25Results, semanticResults).slice(0, topK);
}
