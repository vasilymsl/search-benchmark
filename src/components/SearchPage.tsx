import { useState, useCallback } from 'react';
import { useData } from './DataProvider';
import { searchBM25 } from '../engines/bm25';
import { searchSemantic } from '../engines/semantic';
import { searchHybrid, type SearchResult } from '../engines/hybrid';
import { ResultColumn } from './ResultColumn';

interface MethodResult {
  results: SearchResult[];
  latencyMs: number;
}

export function SearchPage() {
  const { dataset, bm25Index, semanticIndex } = useData();
  const [query, setQuery] = useState('');
  const [bm25Results, setBm25Results] = useState<MethodResult | null>(null);
  const [semResults, setSemResults] = useState<MethodResult | null>(null);
  const [hybridResults, setHybridResults] = useState<MethodResult | null>(null);
  const [searching, setSearching] = useState(false);

  const handleSearch = useCallback(async () => {
    if (!query.trim() || !dataset || !bm25Index || !semanticIndex) return;

    setSearching(true);
    try {
      // BM25
      const t0 = performance.now();
      const bm25 = searchBM25(bm25Index, query, 10);
      const bm25Time = performance.now() - t0;
      setBm25Results({ results: bm25, latencyMs: bm25Time });

      // Semantic
      const t1 = performance.now();
      const sem = await searchSemantic(semanticIndex, query, 10);
      const semTime = performance.now() - t1;
      setSemResults({ results: sem, latencyMs: semTime });

      // Hybrid
      const t2 = performance.now();
      const hybrid = searchHybrid(bm25, sem, 10);
      const hybridTime = performance.now() - t2;
      setHybridResults({ results: hybrid, latencyMs: hybridTime });
    } finally {
      setSearching(false);
    }
  }, [query, dataset, bm25Index, semanticIndex]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch();
  };

  // Sample queries from dataset
  const sampleQueries = dataset?.queries.slice(0, 5) || [];

  return (
    <div>
      <div className="mb-6">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter a search query..."
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          />
          <button
            onClick={handleSearch}
            disabled={searching || !query.trim()}
            className="px-6 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {searching ? 'Searching...' : 'Search'}
          </button>
        </div>

        {sampleQueries.length > 0 && !bm25Results && (
          <div className="mt-3 flex flex-wrap gap-2">
            <span className="text-xs text-gray-500 py-1">Try:</span>
            {sampleQueries.map((q) => (
              <button
                key={q.id}
                onClick={() => setQuery(q.text)}
                className="text-xs px-3 py-1 bg-gray-800 border border-gray-700 rounded-full text-gray-400 hover:text-white hover:border-gray-600 transition-colors truncate max-w-xs"
              >
                {q.text}
              </button>
            ))}
          </div>
        )}
      </div>

      {bm25Results && dataset && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <ResultColumn
            title="BM25 (Lexical)"
            color="amber"
            results={bm25Results.results}
            latencyMs={bm25Results.latencyMs}
            corpus={dataset.corpus}
            query={query}
            qrels={dataset.qrels}
            queryId={findQueryId(query, dataset)}
          />
          <ResultColumn
            title="Semantic (Embeddings)"
            color="purple"
            results={semResults?.results || []}
            latencyMs={semResults?.latencyMs || 0}
            corpus={dataset.corpus}
            query={query}
            qrels={dataset.qrels}
            queryId={findQueryId(query, dataset)}
          />
          <ResultColumn
            title="Hybrid (RRF)"
            color="emerald"
            results={hybridResults?.results || []}
            latencyMs={hybridResults?.latencyMs || 0}
            corpus={dataset.corpus}
            query={query}
            qrels={dataset.qrels}
            queryId={findQueryId(query, dataset)}
          />
        </div>
      )}
    </div>
  );
}

function findQueryId(queryText: string, dataset: { queries: { id: string; text: string }[] }): string | null {
  const match = dataset.queries.find((q) => q.text === queryText);
  return match?.id || null;
}
