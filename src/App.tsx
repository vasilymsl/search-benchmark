import { useState } from 'react';
import './index.css';
import { SearchPage } from './components/SearchPage';
import { BenchmarkPage } from './components/BenchmarkPage';
import { DataProvider, useData } from './components/DataProvider';

type Tab = 'about' | 'search' | 'benchmark';

function AboutPage() {
  const { dataset } = useData();
  const corpus = dataset?.corpus ?? [];
  const queries = dataset?.queries ?? [];

  return (
    <div className="max-w-4xl mx-auto space-y-10 py-8">
      <div className="text-center space-y-3">
        <h1 className="text-3xl font-bold text-white">
          Browser-Side Information Retrieval Benchmark
        </h1>
        <p className="text-lg text-gray-400">
          Comparing BM25, Semantic Search, and Hybrid Retrieval in the Browser
        </p>
      </div>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold text-white border-b border-gray-800 pb-2">
          About This Project
        </h2>
        <p className="text-gray-300 leading-relaxed">
          This is a research project comparing lexical (BM25) and semantic (embedding-based) search
          methods, plus a hybrid approach that combines both — all running entirely client-side in
          the browser using WebAssembly. No server is required: the full retrieval pipeline,
          including neural text encoding, executes locally in your browser tab.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-white border-b border-gray-800 pb-2">
          Methods
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-yellow-400 inline-block"></span>
              <h3 className="font-semibold text-white">BM25 (Lexical)</h3>
            </div>
            <p className="text-sm text-gray-400 leading-relaxed">
              Keyword matching using term frequency and inverse document frequency. A classic
              probabilistic retrieval model that is fast and interpretable, but cannot understand
              synonyms or paraphrasing — only exact term overlap matters.
            </p>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-400 inline-block"></span>
              <h3 className="font-semibold text-white">Semantic Search</h3>
            </div>
            <p className="text-sm text-gray-400 leading-relaxed">
              Uses a neural network (all-MiniLM-L6-v2 via Transformers.js/ONNX) to encode text as
              384-dimensional dense vectors. Retrieval is performed by cosine similarity between
              query and document embeddings. Slower than BM25, but understands meaning and context.
            </p>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-400 inline-block"></span>
              <h3 className="font-semibold text-white">Hybrid (RRF)</h3>
            </div>
            <p className="text-sm text-gray-400 leading-relaxed">
              Reciprocal Rank Fusion combines the ranked result lists from both BM25 and semantic
              search without requiring score normalization. Documents appearing near the top of
              either list are promoted. Usually achieves the best overall retrieval quality.
            </p>
          </div>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold text-white border-b border-gray-800 pb-2">
          Dataset
        </h2>
        <p className="text-gray-300 leading-relaxed">
          The benchmark uses <span className="text-white font-medium">SciFact</span> from the BEIR
          benchmark suite — a collection of scientific paper abstracts paired with scientific claims
          as queries and human-annotated relevance judgments. It is a standard evaluation corpus for
          information retrieval research.
        </p>
        <div className="flex gap-6 mt-2">
          <div className="bg-gray-900 border border-gray-800 rounded-lg px-5 py-4 text-center">
            <div className="text-2xl font-bold text-white">{corpus.length.toLocaleString()}</div>
            <div className="text-sm text-gray-400 mt-1">Documents</div>
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-lg px-5 py-4 text-center">
            <div className="text-2xl font-bold text-white">{queries.length.toLocaleString()}</div>
            <div className="text-sm text-gray-400 mt-1">Test Queries</div>
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-lg px-5 py-4 text-center">
            <div className="text-2xl font-bold text-white">Human</div>
            <div className="text-sm text-gray-400 mt-1">Relevance Judgments</div>
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-white border-b border-gray-800 pb-2">
          Evaluation Metrics
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-1">
            <h3 className="font-semibold text-white text-sm">MRR@10</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
              Mean Reciprocal Rank — measures how quickly the first relevant result appears in the
              top 10. A score of 1.0 means the very first result is always relevant.
            </p>
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-1">
            <h3 className="font-semibold text-white text-sm">P@5</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
              Precision at 5 — the fraction of the top 5 returned results that are actually
              relevant. Higher means fewer irrelevant results in the first page.
            </p>
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-1">
            <h3 className="font-semibold text-white text-sm">R@5</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
              Recall at 5 — the fraction of all relevant documents that appear within the top 5
              results. Measures how well the system surfaces known relevant items.
            </p>
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-1">
            <h3 className="font-semibold text-white text-sm">NDCG@10</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
              Normalized Discounted Cumulative Gain — measures ranking quality by considering both
              the position and the degree of relevance of each result in the top 10.
            </p>
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-1">
            <h3 className="font-semibold text-white text-sm">Latency</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
              Time to process a single query in milliseconds, measured in the browser. Includes
              encoding for semantic search and ranking for all methods.
            </p>
          </div>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold text-white border-b border-gray-800 pb-2">
          Technology Stack
        </h2>
        <div className="flex flex-wrap gap-2">
          {[
            'Vite',
            'React',
            'TypeScript',
            'Tailwind CSS',
            'Transformers.js',
            'ONNX Runtime / WebAssembly',
            'Recharts',
          ].map((tech) => (
            <span
              key={tech}
              className="bg-gray-900 border border-gray-800 text-gray-300 text-sm px-3 py-1.5 rounded-full"
            >
              {tech}
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}

function AppContent() {
  const [tab, setTab] = useState<Tab>('about');
  const { loading, error, progress } = useData();

  if (error) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center p-8">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-400 mb-4">Error</h1>
          <p className="text-gray-400">{error}</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center p-8">
        <div className="text-center max-w-xs">
          <h1 className="text-xl font-bold mb-3">Search Benchmark</h1>
          <p className="text-sm text-blue-400 mb-4">{progress[0] || 'Initializing...'}</p>
          <div className="w-full bg-gray-800 rounded-full h-1.5">
            <div className="bg-blue-500 h-1.5 rounded-full animate-pulse" style={{ width: '60%' }} />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <h1 className="text-lg font-semibold">
            Search Benchmark
            <span className="text-xs text-gray-500 ml-2">BM25 vs Semantic vs Hybrid</span>
          </h1>
          <nav className="flex gap-1">
            <button
              onClick={() => setTab('about')}
              className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
                tab === 'about'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              About
            </button>
            <button
              onClick={() => setTab('search')}
              className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
                tab === 'search'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              Search
            </button>
            <button
              onClick={() => setTab('benchmark')}
              className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
                tab === 'benchmark'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              Benchmark
            </button>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {tab === 'about' && <AboutPage />}
        {tab === 'search' && <SearchPage />}
        {tab === 'benchmark' && <BenchmarkPage />}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <DataProvider>
      <AppContent />
    </DataProvider>
  );
}
