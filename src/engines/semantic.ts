import { pipeline, type FeatureExtractionPipeline } from '@huggingface/transformers';

const MODEL_NAME = 'Xenova/all-MiniLM-L6-v2';

let extractor: FeatureExtractionPipeline | null = null;

export async function loadModel(
  onProgress?: (progress: { status: string; progress?: number }) => void,
): Promise<void> {
  if (extractor) return;
  extractor = await pipeline('feature-extraction', MODEL_NAME, {
    progress_callback: onProgress,
  }) as FeatureExtractionPipeline;
}

export async function encode(text: string): Promise<Float32Array> {
  if (!extractor) throw new Error('Model not loaded. Call loadModel() first.');
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return new Float32Array(output.data as ArrayLike<number>);
}

export async function encodeBatch(texts: string[], batchSize = 16): Promise<Float32Array[]> {
  const results: Float32Array[] = [];
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    for (const text of batch) {
      results.push(await encode(text));
    }
  }
  return results;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot; // vectors are already normalized
}

export interface SemanticIndex {
  embeddings: Float32Array[];
  dimension: number;
}

export function buildSemanticIndex(embeddings: Float32Array[]): SemanticIndex {
  return {
    embeddings,
    dimension: embeddings.length > 0 ? embeddings[0].length : 0,
  };
}

export function loadPrecomputedEmbeddings(data: number[][]): Float32Array[] {
  return data.map((arr) => new Float32Array(arr));
}

export async function searchSemantic(
  index: SemanticIndex,
  query: string,
  topK: number = 10,
): Promise<{ docIndex: number; score: number }[]> {
  const queryEmbedding = await encode(query);

  const results: { docIndex: number; score: number }[] = [];
  for (let i = 0; i < index.embeddings.length; i++) {
    const score = cosineSimilarity(queryEmbedding, index.embeddings[i]);
    results.push({ docIndex: i, score });
  }

  results.sort((a, b) => b.score - a.score);
  return results.slice(0, topK);
}

export function isModelLoaded(): boolean {
  return extractor !== null;
}
