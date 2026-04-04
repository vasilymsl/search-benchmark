/**
 * Precompute document embeddings using Transformers.js (runs in Node).
 * Run: node scripts/precompute-embeddings.mjs
 */
import { pipeline } from '@huggingface/transformers';
import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

const DATA_DIR = join(import.meta.dirname, '..', 'public', 'data');

async function main() {
  console.log('Loading corpus...');
  const corpus = JSON.parse(readFileSync(join(DATA_DIR, 'corpus.json'), 'utf-8'));
  console.log(`  ${corpus.length} documents`);

  console.log('Loading model (Xenova/all-MiniLM-L6-v2)...');
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  console.log('Computing embeddings...');
  const embeddings = [];
  const startTime = Date.now();

  for (let i = 0; i < corpus.length; i++) {
    const text = (corpus[i].title + ' ' + corpus[i].text).slice(0, 512);
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    embeddings.push(Array.from(output.data));

    if ((i + 1) % 100 === 0 || i === corpus.length - 1) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const eta = (((Date.now() - startTime) / (i + 1)) * (corpus.length - i - 1) / 1000).toFixed(0);
      console.log(`  ${i + 1}/${corpus.length} (${elapsed}s elapsed, ~${eta}s remaining)`);
    }
  }

  const outPath = join(DATA_DIR, 'embeddings.json');
  writeFileSync(outPath, JSON.stringify(embeddings));

  const sizeMB = (Buffer.byteLength(JSON.stringify(embeddings)) / 1024 / 1024).toFixed(1);
  console.log(`\nEmbeddings saved to ${outPath} (${sizeMB} MB)`);
  console.log(`Dimension: ${embeddings[0].length}`);
  console.log(`Total time: ${((Date.now() - startTime) / 1000).toFixed(1)}s`);
}

main().catch(console.error);
