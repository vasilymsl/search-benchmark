import { tokenize } from '../utils/tokenizer';

export interface BM25Index {
  /** Inverted index: term -> array of [docIndex, termFrequency] */
  invertedIndex: Map<string, [number, number][]>;
  /** Document lengths (in tokens) */
  docLengths: number[];
  /** Average document length */
  avgDocLength: number;
  /** Total number of documents */
  docCount: number;
}

const K1 = 1.2;
const B = 0.75;

export function buildBM25Index(documents: string[]): BM25Index {
  const invertedIndex = new Map<string, [number, number][]>();
  const docLengths: number[] = [];
  let totalLength = 0;

  for (let i = 0; i < documents.length; i++) {
    const tokens = tokenize(documents[i]);
    docLengths.push(tokens.length);
    totalLength += tokens.length;

    const termFreqs = new Map<string, number>();
    for (const token of tokens) {
      termFreqs.set(token, (termFreqs.get(token) || 0) + 1);
    }

    for (const [term, freq] of termFreqs) {
      if (!invertedIndex.has(term)) {
        invertedIndex.set(term, []);
      }
      invertedIndex.get(term)!.push([i, freq]);
    }
  }

  return {
    invertedIndex,
    docLengths,
    avgDocLength: totalLength / documents.length,
    docCount: documents.length,
  };
}

export function searchBM25(
  index: BM25Index,
  query: string,
  topK: number = 10,
): { docIndex: number; score: number }[] {
  const queryTokens = tokenize(query);
  const scores = new Float64Array(index.docCount);

  for (const token of queryTokens) {
    const postings = index.invertedIndex.get(token);
    if (!postings) continue;

    const df = postings.length;
    const idf = Math.log(
      (index.docCount - df + 0.5) / (df + 0.5) + 1,
    );

    for (const [docIdx, tf] of postings) {
      const dl = index.docLengths[docIdx];
      const tfNorm =
        (tf * (K1 + 1)) /
        (tf + K1 * (1 - B + B * (dl / index.avgDocLength)));
      scores[docIdx] += idf * tfNorm;
    }
  }

  const results: { docIndex: number; score: number }[] = [];
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] > 0) {
      results.push({ docIndex: i, score: scores[i] });
    }
  }

  results.sort((a, b) => b.score - a.score);
  return results.slice(0, topK);
}
