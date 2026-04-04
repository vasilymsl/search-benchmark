import { useState } from 'react';
import './index.css';
import { SearchPage } from './components/SearchPage';
import { BenchmarkPage } from './components/BenchmarkPage';
import { DataProvider, useData } from './components/DataProvider';

type Tab = 'search' | 'benchmark';

function AppContent() {
  const [tab, setTab] = useState<Tab>('search');
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
        <div className="text-center max-w-md">
          <h1 className="text-2xl font-bold mb-4">Loading Search Benchmark</h1>
          <div className="space-y-2 text-sm text-gray-400">
            {progress.map((msg, i) => (
              <p key={i} className={i === progress.length - 1 ? 'text-blue-400' : ''}>
                {msg}
              </p>
            ))}
          </div>
          <div className="mt-6 w-full bg-gray-800 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, progress.length * 20)}%` }}
            />
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
        {tab === 'search' ? <SearchPage /> : <BenchmarkPage />}
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
