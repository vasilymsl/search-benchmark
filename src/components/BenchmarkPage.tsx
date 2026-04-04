import { useState, useCallback } from 'react';
import { useData } from './DataProvider';
import { searchBM25 } from '../engines/bm25';
import { searchSemantic } from '../engines/semantic';
import { searchHybrid } from '../engines/hybrid';
import {
  mrr,
  precisionAtK,
  recallAtK,
  ndcgAtK,
  type BenchmarkResult,
  type RankedResult,
} from '../evaluation/metrics';
import { BenchmarkCharts } from './BenchmarkCharts';

export function BenchmarkPage() {
  const { dataset, bm25Index, semanticIndex } = useData();
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');
  const [results, setResults] = useState<BenchmarkResult[] | null>(null);

  const runBenchmark = useCallback(async () => {
    if (!dataset || !bm25Index || !semanticIndex) return;
    setRunning(true);
    setResults(null);

    const methods = ['BM25', 'Semantic', 'Hybrid'] as const;
    const allResults: BenchmarkResult[] = [];

    for (const method of methods) {
      const perQuery: {
        queryId: string;
        mrr: number;
        precision: number;
        recall: number;
        ndcg: number;
        latencyMs: number;
      }[] = [];

      for (let qi = 0; qi < dataset.queries.length; qi++) {
        const q = dataset.queries[qi];
        setProgress(`${method}: query ${qi + 1}/${dataset.queries.length}`);

        const qrelEntry = dataset.qrels[q.id];
        if (!qrelEntry) continue;

        const relevantSet = new Set(
          Object.entries(qrelEntry)
            .filter(([, rel]) => rel > 0)
            .map(([docId]) => docId),
        );

        let rawResults: { docIndex: number; score: number }[];
        const t0 = performance.now();

        if (method === 'BM25') {
          rawResults = searchBM25(bm25Index, q.text, 10);
        } else if (method === 'Semantic') {
          rawResults = await searchSemantic(semanticIndex, q.text, 10);
        } else {
          const bm25 = searchBM25(bm25Index, q.text, 10);
          const sem = await searchSemantic(semanticIndex, q.text, 10);
          rawResults = searchHybrid(bm25, sem, 10);
        }

        const latencyMs = performance.now() - t0;

        const ranked: RankedResult[] = rawResults.map((r) => ({
          docId: dataset.corpus[r.docIndex].id,
          score: r.score,
        }));

        perQuery.push({
          queryId: q.id,
          mrr: mrr(ranked, relevantSet, 10),
          precision: precisionAtK(ranked, relevantSet, 5),
          recall: recallAtK(ranked, relevantSet, 5),
          ndcg: ndcgAtK(ranked, qrelEntry, 10),
          latencyMs,
        });

        // Yield to UI
        if (qi % 5 === 0) await new Promise((r) => setTimeout(r, 0));
      }

      const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
      const std = (arr: number[]) => {
        const m = avg(arr);
        return Math.sqrt(arr.reduce((a, v) => a + (v - m) ** 2, 0) / arr.length);
      };

      allResults.push({
        method,
        mrrAt10: avg(perQuery.map((q) => q.mrr)),
        precisionAt5: avg(perQuery.map((q) => q.precision)),
        recallAt5: avg(perQuery.map((q) => q.recall)),
        ndcgAt10: avg(perQuery.map((q) => q.ndcg)),
        avgLatencyMs: avg(perQuery.map((q) => q.latencyMs)),
        stdLatencyMs: std(perQuery.map((q) => q.latencyMs)),
        perQueryResults: perQuery,
      });
    }

    setResults(allResults);
    setRunning(false);
    setProgress('');
  }, [dataset, bm25Index, semanticIndex]);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold">Benchmark</h2>
          <p className="text-sm text-gray-500 mt-1">
            Run all queries and compare retrieval quality and latency
          </p>
        </div>
        <button
          onClick={runBenchmark}
          disabled={running}
          className="px-6 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {running ? progress : 'Run Benchmark'}
        </button>
      </div>

      {results && (
        <>
          {/* Results Table */}
          <div className="overflow-x-auto mb-8">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Method</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-medium">MRR@10</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-medium">P@5</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-medium">R@5</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-medium">NDCG@10</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-medium">Latency (ms)</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.method} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="py-3 px-4 font-medium text-white">{r.method}</td>
                    <td className="py-3 px-4 text-right font-mono">{r.mrrAt10.toFixed(4)}</td>
                    <td className="py-3 px-4 text-right font-mono">{r.precisionAt5.toFixed(4)}</td>
                    <td className="py-3 px-4 text-right font-mono">{r.recallAt5.toFixed(4)}</td>
                    <td className="py-3 px-4 text-right font-mono">{r.ndcgAt10.toFixed(4)}</td>
                    <td className="py-3 px-4 text-right font-mono">
                      {r.avgLatencyMs.toFixed(1)} <span className="text-gray-600">± {r.stdLatencyMs.toFixed(1)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Charts */}
          <BenchmarkCharts results={results} />
        </>
      )}

      {!results && !running && (
        <div className="text-center py-20 text-gray-600">
          <p className="text-lg">Press "Run Benchmark" to evaluate all methods</p>
          <p className="text-sm mt-2">
            {dataset ? `${dataset.queries.length} queries × 3 methods` : 'Loading...'}
          </p>
        </div>
      )}
    </div>
  );
}
