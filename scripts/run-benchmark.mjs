/**
 * Node.js benchmark runner.
 * Runs BM25, Semantic, and Hybrid (RRF) retrieval over all SciFact test queries
 * and computes MRR@10, P@5, R@5, NDCG@10, and latency.
 *
 * Usage: node scripts/run-benchmark.mjs
 *
 * Output:
 *   - benchmark-results.json  (full machine-readable results)
 *   - reports/results-table.md (markdown tables for the report)
 *
 * NOTE: Latency measured here is Node.js native ONNX runtime.
 * Browser latency (WebAssembly) is typically 2-5x slower — capture separately
 * via the "Export" button in the BenchmarkPage.
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import { pipeline } from '@huggingface/transformers';

const ROOT = join(import.meta.dirname, '..');
const DATA_DIR = join(ROOT, 'public', 'data');
const REPORTS_DIR = join(ROOT, 'reports');

// ---------- Tokenizer (ported from src/utils/tokenizer.ts) ----------

const STOP_WORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
  'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
  'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
  'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
  'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'you', 'your',
  'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them',
  'their', 'theirs', 'what', 'which', 'who', 'whom', 'whose',
  'where', 'when', 'how', 'not', 'no', 'nor', 'as', 'if', 'then',
  'than', 'too', 'very', 'just', 'about', 'above', 'after', 'again',
  'all', 'also', 'am', 'any', 'because', 'before', 'between', 'both',
  'each', 'few', 'get', 'got', 'here', 'into', 'more', 'most', 'new',
  'now', 'only', 'other', 'over', 'own', 'same', 'so', 'some', 'still',
  'such', 'take', 'through', 'under', 'up', 'while',
]);

function porterStem(word) {
  if (word.length < 3) return word;

  if (word.endsWith('sses')) word = word.slice(0, -2);
  else if (word.endsWith('ies')) word = word.slice(0, -2);
  else if (!word.endsWith('ss') && word.endsWith('s')) word = word.slice(0, -1);

  if (word.endsWith('eed')) {
    if (word.length > 4) word = word.slice(0, -1);
  } else if (word.endsWith('ed') && word.length > 4) {
    word = word.slice(0, -2);
    if (word.endsWith('at') || word.endsWith('bl') || word.endsWith('iz')) word += 'e';
  } else if (word.endsWith('ing') && word.length > 5) {
    word = word.slice(0, -3);
    if (word.endsWith('at') || word.endsWith('bl') || word.endsWith('iz')) word += 'e';
  }

  if (word.endsWith('y') && word.length > 2) word = word.slice(0, -1) + 'i';

  const step2 = {
    ational: 'ate', tional: 'tion', enci: 'ence', anci: 'ance',
    izer: 'ize', alli: 'al', entli: 'ent', eli: 'e',
    ousli: 'ous', ization: 'ize', ation: 'ate', ator: 'ate',
    alism: 'al', iveness: 'ive', fulness: 'ful', ousness: 'ous',
    aliti: 'al', iviti: 'ive', biliti: 'ble',
  };
  for (const [s, r] of Object.entries(step2)) {
    if (word.endsWith(s) && word.length - s.length > 1) {
      word = word.slice(0, -s.length) + r;
      break;
    }
  }

  const step3 = {
    icate: 'ic', ative: '', alize: 'al', iciti: 'ic',
    ical: 'ic', ful: '', ness: '',
  };
  for (const [s, r] of Object.entries(step3)) {
    if (word.endsWith(s) && word.length - s.length > 1) {
      word = word.slice(0, -s.length) + r;
      break;
    }
  }

  return word;
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP_WORDS.has(t))
    .map(porterStem);
}

// ---------- BM25 (ported from src/engines/bm25.ts) ----------

const K1 = 1.2;
const B = 0.75;

function buildBM25Index(documents) {
  const invertedIndex = new Map();
  const docLengths = [];
  let totalLength = 0;

  for (let i = 0; i < documents.length; i++) {
    const tokens = tokenize(documents[i]);
    docLengths.push(tokens.length);
    totalLength += tokens.length;
    const termFreqs = new Map();
    for (const t of tokens) termFreqs.set(t, (termFreqs.get(t) || 0) + 1);
    for (const [t, f] of termFreqs) {
      if (!invertedIndex.has(t)) invertedIndex.set(t, []);
      invertedIndex.get(t).push([i, f]);
    }
  }

  return {
    invertedIndex,
    docLengths,
    avgDocLength: totalLength / documents.length,
    docCount: documents.length,
  };
}

function searchBM25(index, query, topK = 10) {
  const queryTokens = tokenize(query);
  const scores = new Float64Array(index.docCount);

  for (const token of queryTokens) {
    const postings = index.invertedIndex.get(token);
    if (!postings) continue;
    const df = postings.length;
    const idf = Math.log((index.docCount - df + 0.5) / (df + 0.5) + 1);
    for (const [docIdx, tf] of postings) {
      const dl = index.docLengths[docIdx];
      const tfNorm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * (dl / index.avgDocLength)));
      scores[docIdx] += idf * tfNorm;
    }
  }

  const results = [];
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] > 0) results.push({ docIndex: i, score: scores[i] });
  }
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, topK);
}

// ---------- Semantic ----------

function cosineSimilarity(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

function searchSemantic(embeddings, queryEmb, topK = 10) {
  const results = [];
  for (let i = 0; i < embeddings.length; i++) {
    results.push({ docIndex: i, score: cosineSimilarity(queryEmb, embeddings[i]) });
  }
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, topK);
}

// ---------- Hybrid (RRF) ----------

const RRF_K = 60;

function searchHybrid(bm25Results, semanticResults, topK = 10) {
  const fused = new Map();
  for (const list of [bm25Results, semanticResults]) {
    for (let rank = 0; rank < list.length; rank++) {
      const idx = list[rank].docIndex;
      const rrf = 1 / (RRF_K + rank + 1);
      fused.set(idx, (fused.get(idx) || 0) + rrf);
    }
  }
  const results = [];
  for (const [docIndex, score] of fused) results.push({ docIndex, score });
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, topK);
}

// ---------- Metrics ----------

function mrr(results, relevant, k = 10) {
  for (let i = 0; i < Math.min(results.length, k); i++) {
    if (relevant.has(results[i].docId)) return 1 / (i + 1);
  }
  return 0;
}

function precisionAtK(results, relevant, k = 5) {
  const topK = results.slice(0, k);
  let count = 0;
  for (const r of topK) if (relevant.has(r.docId)) count++;
  return count / k;
}

function recallAtK(results, relevant, k = 5) {
  if (relevant.size === 0) return 0;
  const topK = results.slice(0, k);
  let count = 0;
  for (const r of topK) if (relevant.has(r.docId)) count++;
  return count / relevant.size;
}

function ndcgAtK(results, qrels, k = 10) {
  const topK = results.slice(0, k);
  let dcg = 0;
  for (let i = 0; i < topK.length; i++) {
    const rel = qrels[topK[i].docId] || 0;
    dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
  }
  const ideal = Object.values(qrels).sort((a, b) => b - a).slice(0, k);
  let idcg = 0;
  for (let i = 0; i < ideal.length; i++) {
    idcg += (Math.pow(2, ideal[i]) - 1) / Math.log2(i + 2);
  }
  return idcg === 0 ? 0 : dcg / idcg;
}

// ---------- Aggregation ----------

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function std(arr) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((a, v) => a + (v - m) ** 2, 0) / arr.length);
}

// ---------- Main ----------

async function main() {
  if (!existsSync(REPORTS_DIR)) mkdirSync(REPORTS_DIR, { recursive: true });

  console.log('Loading dataset...');
  const corpus = JSON.parse(readFileSync(join(DATA_DIR, 'corpus.json'), 'utf-8'));
  const queries = JSON.parse(readFileSync(join(DATA_DIR, 'queries.json'), 'utf-8'));
  const qrels = JSON.parse(readFileSync(join(DATA_DIR, 'qrels.json'), 'utf-8'));
  const meta = JSON.parse(readFileSync(join(DATA_DIR, 'embeddings-meta.json'), 'utf-8'));

  console.log(`  ${corpus.length} documents, ${queries.length} queries`);

  console.log('Loading embeddings...');
  const embBuf = readFileSync(join(DATA_DIR, 'embeddings.bin'));
  const flat = new Float32Array(embBuf.buffer, embBuf.byteOffset, embBuf.byteLength / 4);
  const embeddings = [];
  for (let i = 0; i < meta.numDocs; i++) {
    embeddings.push(flat.subarray(i * meta.dim, (i + 1) * meta.dim));
  }
  console.log(`  ${embeddings.length} × ${meta.dim}`);

  console.log('Building BM25 index...');
  const texts = corpus.map((d) => d.title + ' ' + d.text);
  const bm25Index = buildBM25Index(texts);
  console.log(`  avg doc length: ${bm25Index.avgDocLength.toFixed(1)}`);

  console.log('Loading ML model (this may take a minute)...');
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  const perMethod = { BM25: [], Semantic: [], Hybrid: [] };

  console.log(`\nRunning benchmark (${queries.length} queries × 3 methods)...`);
  const startTime = Date.now();

  for (let qi = 0; qi < queries.length; qi++) {
    const q = queries[qi];
    const qrelEntry = qrels[q.id];
    if (!qrelEntry) continue;

    const relevantSet = new Set(
      Object.entries(qrelEntry)
        .filter(([, rel]) => rel > 0)
        .map(([docId]) => docId),
    );

    // BM25
    const t0 = performance.now();
    const bm25Raw = searchBM25(bm25Index, q.text, 10);
    const bm25Latency = performance.now() - t0;

    // Semantic
    const t1 = performance.now();
    const output = await extractor(q.text, { pooling: 'mean', normalize: true });
    const queryEmb = new Float32Array(output.data);
    const semRaw = searchSemantic(embeddings, queryEmb, 10);
    const semLatency = performance.now() - t1;

    // Hybrid = BM25 + Semantic + RRF fusion (end-to-end, as the user sees it)
    const t2 = performance.now();
    const bm25ForHybrid = searchBM25(bm25Index, q.text, 10);
    const output2 = await extractor(q.text, { pooling: 'mean', normalize: true });
    const queryEmb2 = new Float32Array(output2.data);
    const semForHybrid = searchSemantic(embeddings, queryEmb2, 10);
    const hybridRaw = searchHybrid(bm25ForHybrid, semForHybrid, 10);
    const hybridLatency = performance.now() - t2;

    const toRanked = (raw) => raw.map((r) => ({ docId: corpus[r.docIndex].id, score: r.score }));
    const bm25Ranked = toRanked(bm25Raw);
    const semRanked = toRanked(semRaw);
    const hybridRanked = toRanked(hybridRaw);

    perMethod.BM25.push({
      queryId: q.id,
      mrr: mrr(bm25Ranked, relevantSet, 10),
      precision: precisionAtK(bm25Ranked, relevantSet, 5),
      recall: recallAtK(bm25Ranked, relevantSet, 5),
      ndcg: ndcgAtK(bm25Ranked, qrelEntry, 10),
      latencyMs: bm25Latency,
    });
    perMethod.Semantic.push({
      queryId: q.id,
      mrr: mrr(semRanked, relevantSet, 10),
      precision: precisionAtK(semRanked, relevantSet, 5),
      recall: recallAtK(semRanked, relevantSet, 5),
      ndcg: ndcgAtK(semRanked, qrelEntry, 10),
      latencyMs: semLatency,
    });
    perMethod.Hybrid.push({
      queryId: q.id,
      mrr: mrr(hybridRanked, relevantSet, 10),
      precision: precisionAtK(hybridRanked, relevantSet, 5),
      recall: recallAtK(hybridRanked, relevantSet, 5),
      ndcg: ndcgAtK(hybridRanked, qrelEntry, 10),
      latencyMs: hybridLatency,
    });

    if ((qi + 1) % 25 === 0 || qi === queries.length - 1) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`  ${qi + 1}/${queries.length} queries (${elapsed}s)`);
    }
  }

  // Aggregate
  console.log('\nAggregating...');
  const summary = [];
  for (const method of ['BM25', 'Semantic', 'Hybrid']) {
    const data = perMethod[method];
    summary.push({
      method,
      queries: data.length,
      mrrAt10: { mean: mean(data.map((d) => d.mrr)), std: std(data.map((d) => d.mrr)) },
      precisionAt5: { mean: mean(data.map((d) => d.precision)), std: std(data.map((d) => d.precision)) },
      recallAt5: { mean: mean(data.map((d) => d.recall)), std: std(data.map((d) => d.recall)) },
      ndcgAt10: { mean: mean(data.map((d) => d.ndcg)), std: std(data.map((d) => d.ndcg)) },
      latencyMs: { mean: mean(data.map((d) => d.latencyMs)), std: std(data.map((d) => d.latencyMs)) },
    });
  }

  // Environment metadata
  const env = {
    runtime: 'Node.js ' + process.version,
    platform: process.platform,
    arch: process.arch,
    timestamp: new Date().toISOString(),
    dataset: {
      name: 'SciFact (BEIR)',
      documents: corpus.length,
      queries: queries.length,
      avgDocLength: bm25Index.avgDocLength,
    },
    model: 'Xenova/all-MiniLM-L6-v2',
    embeddingDim: meta.dim,
    note: 'Latency is Node.js native ONNX runtime; in-browser latency differs (WebAssembly).',
  };

  // Save JSON
  const jsonOut = { env, summary, perMethod };
  writeFileSync(join(ROOT, 'benchmark-results.json'), JSON.stringify(jsonOut, null, 2));

  // Save markdown
  let md = '# Benchmark Results\n\n';
  md += `**Environment:** ${env.runtime} on ${env.platform} ${env.arch}, measured ${env.timestamp}.\n\n`;
  md += `**Dataset:** SciFact from BEIR — ${corpus.length} documents, ${queries.length} queries with expert-annotated relevance judgments.\n\n`;
  md += `**Model:** ${env.model} (${env.embeddingDim}-dim embeddings, precomputed offline).\n\n`;
  md += `> ⚠️ Latency values below are Node.js native ONNX runtime. In-browser latency (WebAssembly) is typically 2-5× slower.\n\n`;

  md += '## Table I — Retrieval Quality\n\n';
  md += '| Method | MRR@10 | P@5 | R@5 | NDCG@10 |\n';
  md += '|--------|--------|-----|-----|--------|\n';
  for (const s of summary) {
    md += `| **${s.method}** | ${s.mrrAt10.mean.toFixed(4)} | ${s.precisionAt5.mean.toFixed(4)} | ${s.recallAt5.mean.toFixed(4)} | ${s.ndcgAt10.mean.toFixed(4)} |\n`;
  }

  md += '\n## Table II — Query Latency (per-method)\n\n';
  md += '| Method | Mean (ms) | Std (ms) | Relative |\n';
  md += '|--------|-----------|----------|----------|\n';
  const bm25Mean = summary.find((s) => s.method === 'BM25').latencyMs.mean;
  for (const s of summary) {
    const rel = (s.latencyMs.mean / bm25Mean).toFixed(1) + '×';
    md += `| **${s.method}** | ${s.latencyMs.mean.toFixed(2)} | ${s.latencyMs.std.toFixed(2)} | ${rel} |\n`;
  }

  md += '\n## Key Findings\n\n';
  const best = (key) => summary.reduce((a, b) => (a[key].mean > b[key].mean ? a : b));
  const bestMrr = best('mrrAt10');
  const bestP = best('precisionAt5');
  const bestR = best('recallAt5');
  const bestN = best('ndcgAt10');
  md += `- **Best MRR@10**: ${bestMrr.method} (${bestMrr.mrrAt10.mean.toFixed(4)})\n`;
  md += `- **Best P@5**: ${bestP.method} (${bestP.precisionAt5.mean.toFixed(4)})\n`;
  md += `- **Best R@5**: ${bestR.method} (${bestR.recallAt5.mean.toFixed(4)})\n`;
  md += `- **Best NDCG@10**: ${bestN.method} (${bestN.ndcgAt10.mean.toFixed(4)})\n`;
  const semMean = summary.find((s) => s.method === 'Semantic').latencyMs.mean;
  md += `- **Latency**: BM25 is ${(semMean / bm25Mean).toFixed(0)}× faster than Semantic\n`;

  writeFileSync(join(REPORTS_DIR, 'results-table.md'), md);

  console.log('\n✓ Saved benchmark-results.json');
  console.log('✓ Saved reports/results-table.md');
  console.log('\nSummary:');
  console.table(
    summary.map((s) => ({
      Method: s.method,
      'MRR@10': s.mrrAt10.mean.toFixed(4),
      'P@5': s.precisionAt5.mean.toFixed(4),
      'R@5': s.recallAt5.mean.toFixed(4),
      'NDCG@10': s.ndcgAt10.mean.toFixed(4),
      'Latency (ms)': s.latencyMs.mean.toFixed(2),
    })),
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
