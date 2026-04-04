/**
 * Convert embeddings.json to a compact binary format (Float32 ArrayBuffer).
 * Reduces 40MB JSON to ~8MB binary.
 * Run: node scripts/compress-embeddings.mjs
 */
import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

const DATA_DIR = join(import.meta.dirname, '..', 'public', 'data');

const embeddings = JSON.parse(readFileSync(join(DATA_DIR, 'embeddings.json'), 'utf-8'));
const numDocs = embeddings.length;
const dim = embeddings[0].length;

console.log(`Converting ${numDocs} embeddings (dim=${dim})...`);

// Pack into a single Float32Array
const buffer = new Float32Array(numDocs * dim);
for (let i = 0; i < numDocs; i++) {
  for (let j = 0; j < dim; j++) {
    buffer[i * dim + j] = embeddings[i][j];
  }
}

const outPath = join(DATA_DIR, 'embeddings.bin');
writeFileSync(outPath, Buffer.from(buffer.buffer));

const sizeMB = (buffer.byteLength / 1024 / 1024).toFixed(1);
console.log(`Saved to ${outPath} (${sizeMB} MB)`);

// Also save metadata
writeFileSync(
  join(DATA_DIR, 'embeddings-meta.json'),
  JSON.stringify({ numDocs, dim }),
);
console.log('Metadata saved to embeddings-meta.json');
