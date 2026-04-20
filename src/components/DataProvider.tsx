import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { loadDataset, loadEmbeddings } from '../data/loader';
import { buildBM25Index, type BM25Index } from '../engines/bm25';
import {
  loadModel,
  buildSemanticIndex,
  type SemanticIndex,
} from '../engines/semantic';
import type { Dataset } from '../data/types';

interface DataContextValue {
  loading: boolean;
  error: string | null;
  progress: string[];
  dataset: Dataset | null;
  bm25Index: BM25Index | null;
  semanticIndex: SemanticIndex | null;
}

const DataContext = createContext<DataContextValue>({
  loading: true,
  error: null,
  progress: [],
  dataset: null,
  bm25Index: null,
  semanticIndex: null,
});

export function useData() {
  return useContext(DataContext);
}

export function DataProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<DataContextValue>({
    loading: true,
    error: null,
    progress: [],
    dataset: null,
    bm25Index: null,
    semanticIndex: null,
  });

  useEffect(() => {
    let cancelled = false;

    const setProgress = (msg: string) => {
      if (cancelled) return;
      setState((prev) => ({ ...prev, progress: [msg] }));
    };

    async function init() {
      try {
        setProgress('Loading dataset...');
        const dataset = await loadDataset();

        setProgress('Building search index...');
        const texts = dataset.corpus.map((d) => d.title + ' ' + d.text);
        const bm25Index = buildBM25Index(texts);

        setProgress('Loading embeddings...');
        const embeddings = await loadEmbeddings();
        const semanticIndex = buildSemanticIndex(embeddings);

        setProgress('Loading ML model...');
        await loadModel();


        if (!cancelled) {
          setState({
            loading: false,
            error: null,
            progress: [],
            dataset,
            bm25Index,
            semanticIndex,
          });
        }
      } catch (err) {
        if (!cancelled) {
          setState((prev) => ({
            ...prev,
            loading: false,
            error: err instanceof Error ? err.message : String(err),
          }));
        }
      }
    }

    init();
    return () => { cancelled = true; };
  }, []);

  return <DataContext.Provider value={state}>{children}</DataContext.Provider>;
}
