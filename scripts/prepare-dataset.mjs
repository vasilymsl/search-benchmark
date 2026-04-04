/**
 * Download SciFact from BEIR and convert to JSON for browser use.
 * Run: node scripts/prepare-dataset.mjs
 */
import { writeFileSync, mkdirSync, existsSync, createWriteStream, readFileSync, createReadStream } from 'fs';
import { join } from 'path';
import { createGunzip } from 'zlib';
import { pipeline } from 'stream/promises';
import { createInterface } from 'readline';
import { tmpdir } from 'os';
import { execFileSync } from 'child_process';

const OUT_DIR = join(import.meta.dirname, '..', 'public', 'data');
const TMP_DIR = join(tmpdir(), 'scifact-download');

async function downloadFile(url, dest) {
  console.log(`  Fetching ${url}...`);
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  const fileStream = createWriteStream(dest);
  await pipeline(res.body, fileStream);
}

async function readGzJsonl(gzPath) {
  const results = [];
  const gunzip = createGunzip();
  const input = createReadStream(gzPath).pipe(gunzip);
  const rl = createInterface({ input });
  for await (const line of rl) {
    if (line.trim()) {
      results.push(JSON.parse(line));
    }
  }
  return results;
}

async function main() {
  for (const dir of [OUT_DIR, TMP_DIR]) {
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  }

  const HF_BASE = 'https://huggingface.co/datasets/BeIR/scifact/resolve/main';

  // Download corpus (.gz)
  console.log('Downloading SciFact corpus...');
  const corpusPath = join(TMP_DIR, 'corpus.jsonl.gz');
  if (!existsSync(corpusPath)) {
    await downloadFile(`${HF_BASE}/corpus.jsonl.gz`, corpusPath);
  }
  const corpusRaw = await readGzJsonl(corpusPath);
  const corpus = corpusRaw.map((doc) => ({
    id: String(doc._id),
    title: doc.title || '',
    text: doc.text || '',
  }));
  console.log(`  ${corpus.length} documents`);

  // Download queries (.gz)
  console.log('Downloading SciFact queries...');
  const queriesPath = join(TMP_DIR, 'queries.jsonl.gz');
  if (!existsSync(queriesPath)) {
    await downloadFile(`${HF_BASE}/queries.jsonl.gz`, queriesPath);
  }
  const queriesRaw = await readGzJsonl(queriesPath);
  const allQueries = queriesRaw.map((q) => ({
    id: String(q._id),
    text: q.text || '',
  }));
  console.log(`  ${allQueries.length} queries total`);

  // Download full SciFact zip (contains qrels)
  console.log('Downloading SciFact qrels...');
  const qrels = {};
  const zipPath = join(TMP_DIR, 'scifact.zip');
  if (!existsSync(zipPath)) {
    await downloadFile(
      'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip',
      zipPath,
    );
  }

  // Extract qrels from zip using execFileSync (safe, no shell injection)
  console.log('  Extracting qrels from zip...');
  execFileSync('unzip', ['-o', '-j', zipPath, 'scifact/qrels/test.tsv', '-d', TMP_DIR], {
    stdio: 'pipe',
  });

  const qrelsPath = join(TMP_DIR, 'test.tsv');
  const qrelsText = readFileSync(qrelsPath, 'utf-8');
  const lines = qrelsText.trim().split('\n');
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split('\t');
    if (parts.length < 3) continue;
    const [queryId, docId, relevance] = parts;
    if (!qrels[queryId]) qrels[queryId] = {};
    qrels[queryId][docId] = parseInt(relevance, 10);
  }

  // Filter queries to only those with qrels
  const queries = allQueries.filter((q) => qrels[q.id]);
  console.log(`  ${queries.length} queries with relevance judgments`);

  // Save
  writeFileSync(join(OUT_DIR, 'corpus.json'), JSON.stringify(corpus));
  writeFileSync(join(OUT_DIR, 'queries.json'), JSON.stringify(queries));
  writeFileSync(join(OUT_DIR, 'qrels.json'), JSON.stringify(qrels));

  const corpusSizeMB = (Buffer.byteLength(JSON.stringify(corpus)) / 1024 / 1024).toFixed(1);
  console.log(`\nDataset saved to ${OUT_DIR}`);
  console.log(`  corpus.json: ${corpusSizeMB} MB (${corpus.length} docs)`);
  console.log(`  queries.json: ${queries.length} queries`);
  console.log(`  qrels.json: ${Object.keys(qrels).length} query-doc pairs`);
  console.log('\nNext: run node scripts/precompute-embeddings.mjs');
}

main().catch(console.error);
