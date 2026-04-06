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

type QueryMode = 'dataset' | 'custom';

export function SearchPage() {
  const { dataset, bm25Index, semanticIndex } = useData();
  const [queryMode, setQueryMode] = useState<QueryMode>('dataset');
  const [selectedQueryId, setSelectedQueryId] = useState<string>('');
  const [customQuery, setCustomQuery] = useState('');
  const [queryFilter, setQueryFilter] = useState('');
  const [bm25Results, setBm25Results] = useState<MethodResult | null>(null);
  const [semResults, setSemResults] = useState<MethodResult | null>(null);
  const [hybridResults, setHybridResults] = useState<MethodResult | null>(null);
  const [searching, setSearching] = useState(false);

  const activeQuery =
    queryMode === 'dataset'
      ? dataset?.queries.find((q) => q.id === selectedQueryId)?.text ?? ''
      : customQuery;

  const activeQueryId =
    queryMode === 'dataset' ? selectedQueryId || null : null;

  const filteredQueries = dataset
    ? dataset.queries.filter((q) =>
        queryFilter.trim() === '' ||
        q.text.toLowerCase().includes(queryFilter.toLowerCase())
      )
    : [];

  const handleSearch = useCallback(async () => {
    if (!activeQuery.trim() || !dataset || !bm25Index || !semanticIndex) return;

    setSearching(true);
    try {
      const t0 = performance.now();
      const bm25 = searchBM25(bm25Index, activeQuery, 10);
      const bm25Time = performance.now() - t0;
      setBm25Results({ results: bm25, latencyMs: bm25Time });

      const t1 = performance.now();
      const sem = await searchSemantic(semanticIndex, activeQuery, 10);
      const semTime = performance.now() - t1;
      setSemResults({ results: sem, latencyMs: semTime });

      const t2 = performance.now();
      const hybrid = searchHybrid(bm25, sem, 10);
      const hybridTime = performance.now() - t2;
      setHybridResults({ results: hybrid, latencyMs: hybridTime });
    } finally {
      setSearching(false);
    }
  }, [activeQuery, dataset, bm25Index, semanticIndex]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch();
  };

  const canSearch =
    !searching &&
    (queryMode === 'dataset' ? !!selectedQueryId : !!customQuery.trim());

  return (
    <div>
      {/* Explanation panel */}
      <div className="mb-6 rounded-lg border border-blue-500/20 bg-blue-500/5 px-4 py-3 text-sm text-gray-300 leading-relaxed">
        Select a query from the SciFact test set below. These are scientific
        claims with human-annotated relevance judgments — results marked{' '}
        <span className="text-green-400 font-medium">relevant</span> are
        confirmed by expert annotators. You can also type a custom query, but
        relevance labels will only appear for dataset queries.
      </div>

      {/* Mode toggle */}
      <div className="mb-4 flex gap-2">
        <button
          onClick={() => setQueryMode('dataset')}
          className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
            queryMode === 'dataset'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:text-white border border-gray-700'
          }`}
        >
          Dataset queries
        </button>
        <button
          onClick={() => setQueryMode('custom')}
          className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
            queryMode === 'custom'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:text-white border border-gray-700'
          }`}
        >
          Custom query
        </button>
      </div>

      {/* Query input area */}
      <div className="mb-6">
        {queryMode === 'dataset' ? (
          <div className="space-y-2">
            <input
              type="text"
              value={queryFilter}
              onChange={(e) => setQueryFilter(e.target.value)}
              placeholder="Filter queries..."
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            />
            <select
              value={selectedQueryId}
              onChange={(e) => setSelectedQueryId(e.target.value)}
              size={6}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            >
              <option value="" disabled className="text-gray-500">
                -- select a query --
              </option>
              {filteredQueries.map((q) => (
                <option key={q.id} value={q.id} className="py-1">
                  {q.text.length > 100 ? q.text.slice(0, 97) + '...' : q.text}
                </option>
              ))}
            </select>
            {selectedQueryId && (
              <p className="text-xs text-gray-400 px-1">
                Selected:{' '}
                <span className="text-gray-200">
                  {dataset?.queries.find((q) => q.id === selectedQueryId)?.text}
                </span>
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-1">
            <input
              type="text"
              value={customQuery}
              onChange={(e) => setCustomQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter a custom search query..."
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 px-1">
              Relevance labels are not available for custom queries.
            </p>
          </div>
        )}

        <button
          onClick={handleSearch}
          disabled={!canSearch}
          className="mt-3 px-6 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {searching ? 'Searching...' : 'Search'}
        </button>
      </div>

      {bm25Results && dataset && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <ResultColumn
            title="BM25"
            color="amber"
            description="Keyword matching (term frequency × inverse document frequency)"
            results={bm25Results.results}
            latencyMs={bm25Results.latencyMs}
            corpus={dataset.corpus}
            query={activeQuery}
            qrels={dataset.qrels}
            queryId={activeQueryId}
          />
          <ResultColumn
            title="Semantic"
            color="purple"
            description="Meaning-based search using neural embeddings (cosine similarity)"
            results={semResults?.results || []}
            latencyMs={semResults?.latencyMs || 0}
            corpus={dataset.corpus}
            query={activeQuery}
            qrels={dataset.qrels}
            queryId={activeQueryId}
          />
          <ResultColumn
            title="Hybrid"
            color="emerald"
            description="Reciprocal Rank Fusion of BM25 + Semantic results"
            results={hybridResults?.results || []}
            latencyMs={hybridResults?.latencyMs || 0}
            corpus={dataset.corpus}
            query={activeQuery}
            qrels={dataset.qrels}
            queryId={activeQueryId}
          />
        </div>
      )}
    </div>
  );
}
