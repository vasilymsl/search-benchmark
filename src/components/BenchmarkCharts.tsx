import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
} from 'recharts';
import type { BenchmarkResult } from '../evaluation/metrics';

interface Props {
  results: BenchmarkResult[];
}

const COLORS = {
  BM25: '#f59e0b',
  Semantic: '#a855f7',
  Hybrid: '#10b981',
};

export function BenchmarkCharts({ results }: Props) {
  // Quality metrics bar chart data
  const qualityData = [
    {
      metric: 'MRR@10',
      ...Object.fromEntries(results.map((r) => [r.method, r.mrrAt10])),
    },
    {
      metric: 'P@5',
      ...Object.fromEntries(results.map((r) => [r.method, r.precisionAt5])),
    },
    {
      metric: 'R@5',
      ...Object.fromEntries(results.map((r) => [r.method, r.recallAt5])),
    },
    {
      metric: 'NDCG@10',
      ...Object.fromEntries(results.map((r) => [r.method, r.ndcgAt10])),
    },
  ];

  // Latency bar chart data
  const latencyData = results.map((r) => ({
    method: r.method,
    latency: r.avgLatencyMs,
    std: r.stdLatencyMs,
  }));

  // Quality vs Latency scatter data
  const scatterData = results.map((r) => ({
    x: r.avgLatencyMs,
    y: r.mrrAt10,
    method: r.method,
  }));

  return (
    <div className="space-y-8">
      {/* Quality Metrics Comparison */}
      <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
        <h3 className="text-lg font-semibold mb-4">Retrieval Quality Comparison</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={qualityData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="metric" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" domain={[0, 1]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
              labelStyle={{ color: '#f3f4f6' }}
            />
            <Legend />
            {results.map((r) => (
              <Bar
                key={r.method}
                dataKey={r.method}
                fill={COLORS[r.method as keyof typeof COLORS]}
                radius={[4, 4, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Latency Comparison */}
      <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
        <h3 className="text-lg font-semibold mb-4">Average Latency per Query</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={latencyData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="method" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" label={{ value: 'ms', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
              formatter={(value) => [`${Number(value).toFixed(1)} ms`, 'Latency']}
            />
            <Bar
              dataKey="latency"
              radius={[4, 4, 0, 0]}
              fill="#3b82f6"
            >
              {latencyData.map((entry) => (
                <rect key={entry.method} fill={COLORS[entry.method as keyof typeof COLORS]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Quality vs Latency Scatter */}
      <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
        <h3 className="text-lg font-semibold mb-4">Quality vs Latency Trade-off</h3>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              type="number"
              dataKey="x"
              name="Latency"
              unit=" ms"
              stroke="#9ca3af"
              label={{ value: 'Latency (ms)', position: 'bottom', fill: '#9ca3af' }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="MRR@10"
              stroke="#9ca3af"
              domain={[0, 1]}
              label={{ value: 'MRR@10', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
            />
            <ZAxis range={[200, 200]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
              formatter={(value, name) => [
                name === 'Latency' ? `${Number(value).toFixed(1)} ms` : Number(value).toFixed(4),
                name,
              ]}
            />
            {results.map((r) => (
              <Scatter
                key={r.method}
                name={r.method}
                data={[scatterData.find((s) => s.method === r.method)!]}
                fill={COLORS[r.method as keyof typeof COLORS]}
              />
            ))}
            <Legend />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Export hint */}
      <p className="text-xs text-gray-600 text-center">
        Tip: Right-click on any chart and "Save Image As..." to export for your paper
      </p>
    </div>
  );
}
