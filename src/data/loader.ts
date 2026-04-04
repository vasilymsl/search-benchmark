import type { Document, Query, Qrels, Dataset } from './types';

const DATA_BASE = import.meta.env.BASE_URL + 'data/';

export async function loadDataset(): Promise<Dataset> {
  const [corpusRes, queriesRes, qrelsRes] = await Promise.all([
    fetch(DATA_BASE + 'corpus.json'),
    fetch(DATA_BASE + 'queries.json'),
    fetch(DATA_BASE + 'qrels.json'),
  ]);

  const corpus: Document[] = await corpusRes.json();
  const queries: Query[] = await queriesRes.json();
  const qrels: Qrels = await qrelsRes.json();

  const docIdToIndex = new Map<string, number>();
  corpus.forEach((doc, i) => docIdToIndex.set(doc.id, i));

  return { corpus, queries, qrels, docIdToIndex };
}

export async function loadEmbeddings(): Promise<Float32Array[]> {
  const [metaRes, binRes] = await Promise.all([
    fetch(DATA_BASE + 'embeddings-meta.json'),
    fetch(DATA_BASE + 'embeddings.bin'),
  ]);

  const meta: { numDocs: number; dim: number } = await metaRes.json();
  const buffer = await binRes.arrayBuffer();
  const flat = new Float32Array(buffer);

  const embeddings: Float32Array[] = [];
  for (let i = 0; i < meta.numDocs; i++) {
    embeddings.push(flat.subarray(i * meta.dim, (i + 1) * meta.dim));
  }
  return embeddings;
}
