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

const METRIC_DESCRIPTIONS: Record<string, string> = {
  'MRR@10': 'Mean Reciprocal Rank at 10 — how quickly the first relevant result appears',
  'P@5': 'Precision at 5 — what fraction of top 5 results are relevant',
  'R@5': 'Recall at 5 — what fraction of all relevant docs appear in top 5',
  'NDCG@10': 'Normalized Discounted Cumulative Gain — overall ranking quality',
  'Latency': 'Average time per query in milliseconds',
};

function MetricHeader({ label }: { label: string }) {
  return (
    <th className="text-right py-3 px-4 font-medium">
      <span
        className="text-gray-400 cursor-help border-b border-dashed border-gray-600"
        title={METRIC_DESCRIPTIONS[label]}
      >
        {label}
      </span>
      <div className="text-xs text-gray-600 font-normal mt-0.5 max-w-[120px] ml-auto leading-tight">
        {METRIC_DESCRIPTIONS[label]}
      </div>
    </th>
  );
}

function KeyFindings({ results }: { results: BenchmarkResult[] }) {
  const bm25 = results.find((r) => r.method === 'BM25');
  const semantic = results.find((r) => r.method === 'Semantic');
  const hybrid = results.find((r) => r.method === 'Hybrid');

  if (!bm25 || !semantic || !hybrid) return null;

  const bestMrr = [...results].sort((a, b) => b.mrrAt10 - a.mrrAt10)[0];
  const bestP5 = [...results].sort((a, b) => b.precisionAt5 - a.precisionAt5)[0];
  const bestR5 = [...results].sort((a, b) => b.recallAt5 - a.recallAt5)[0];

  const latencyRatio = semantic.avgLatencyMs / bm25.avgLatencyMs;
  const latencyStr =
    latencyRatio >= 1
      ? `${latencyRatio.toFixed(1)}x`
      : `${(1 / latencyRatio).toFixed(1)}x`;
  const fasterMethod = bm25.avgLatencyMs <= semantic.avgLatencyMs ? 'BM25' : 'Semantic';
  const slowerMethod = fasterMethod === 'BM25' ? 'Semantic' : 'BM25';

  const others = (exclude: string) =>
    results
      .filter((r) => r.method !== exclude)
      .map((r) => `${r.method} (${r.mrrAt10.toFixed(3)})`)
      .join(' and ');

  const hybridBeatsBoth =
    hybrid.mrrAt10 >= bm25.mrrAt10 && hybrid.mrrAt10 >= semantic.mrrAt10;

  const findings: string[] = [];

  findings.push(
    `${bestMrr.method} achieves the highest MRR@10 (${bestMrr.mrrAt10.toFixed(3)}), outperforming ${others(bestMrr.method)}.`,
  );

  if (bestP5.method !== bestMrr.method) {
    const p5Others = results
      .filter((r) => r.method !== bestP5.method)
      .map((r) => `${r.method} (${r.precisionAt5.toFixed(3)})`)
      .join(' and ');
    findings.push(
      `${bestP5.method} leads in P@5 (${bestP5.precisionAt5.toFixed(3)}) vs ${p5Others}.`,
    );
  }

  if (bestR5.method !== bestMrr.method && bestR5.method !== bestP5.method) {
    const r5Others = results
      .filter((r) => r.method !== bestR5.method)
      .map((r) => `${r.method} (${r.recallAt5.toFixed(3)})`)
      .join(' and ');
    findings.push(
      `${bestR5.method} achieves the best R@5 (${bestR5.recallAt5.toFixed(3)}) vs ${r5Others}.`,
    );
  }

  findings.push(
    `${fasterMethod} is ${latencyStr} faster than ${slowerMethod} search (${bm25.avgLatencyMs.toFixed(1)} ms vs ${semantic.avgLatencyMs.toFixed(1)} ms per query), but trades off retrieval quality.`,
  );

  if (hybridBeatsBoth) {
    findings.push(
      `Hybrid search outperforms both individual methods on MRR@10, confirming that combining BM25 and Semantic signals is beneficial for this dataset.`,
    );
  } else {
    const hybridRank =
      [...results].sort((a, b) => b.mrrAt10 - a.mrrAt10).findIndex((r) => r.method === 'Hybrid') +
      1;
    findings.push(
      `Hybrid search ranks ${hybridRank === 2 ? 'second' : 'third'} in MRR@10 (${hybrid.mrrAt10.toFixed(3)}), suggesting that simple score fusion does not always outperform the best individual method.`,
    );
  }

  return (
    <div className="mt-8 p-5 bg-gray-800/40 border border-gray-700 rounded-lg">
      <h3 className="text-base font-semibold text-white mb-3">Key Findings</h3>
      <ul className="space-y-2">
        {findings.map((finding, i) => (
          <li key={i} className="flex gap-2 text-sm text-gray-300">
            <span className="text-blue-400 mt-0.5 shrink-0">•</span>
            <span>{finding}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

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
      {/* Intro section */}
      <div className="mb-6 p-4 bg-gray-800/30 border border-gray-700/50 rounded-lg">
        <p className="text-sm text-gray-400 leading-relaxed">
          This benchmark evaluates all three retrieval methods across 300 test queries from the
          SciFact dataset. Each query is a scientific claim, and the ground-truth relevant documents
          are determined by expert human annotators.
        </p>
      </div>

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
                  <MetricHeader label="MRR@10" />
                  <MetricHeader label="P@5" />
                  <MetricHeader label="R@5" />
                  <MetricHeader label="NDCG@10" />
                  <MetricHeader label="Latency" />
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
                      {r.avgLatencyMs.toFixed(1)}{' '}
                      <span className="text-gray-600">± {r.stdLatencyMs.toFixed(1)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Key Findings */}
          <KeyFindings results={results} />

          {/* Charts */}
          <div className="mt-8">
            <BenchmarkCharts results={results} />
          </div>
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
