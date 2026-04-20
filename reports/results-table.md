# Benchmark Results

**Environment:** Node.js v25.2.1 on darwin arm64, measured 2026-04-20T18:51:12.788Z.

**Dataset:** SciFact from BEIR — 5183 documents, 300 queries with expert-annotated relevance judgments.

**Model:** Xenova/all-MiniLM-L6-v2 (384-dim embeddings, precomputed offline).

> ⚠️ Latency values below are Node.js native ONNX runtime. In-browser latency (WebAssembly) is typically 2-5× slower.

## Table I — Retrieval Quality

| Method | MRR@10 | P@5 | R@5 | NDCG@10 |
|--------|--------|-----|-----|--------|
| **BM25** | 0.6364 | 0.1587 | 0.7359 | 0.6740 |
| **Semantic** | 0.5622 | 0.1480 | 0.6762 | 0.6065 |
| **Hybrid** | 0.6534 | 0.1713 | 0.7869 | 0.6941 |

## Table II — Query Latency (per-method)

| Method | Mean (ms) | Std (ms) | Relative |
|--------|-----------|----------|----------|
| **BM25** | 0.48 | 0.23 | 1.0× |
| **Semantic** | 3.61 | 0.47 | 7.6× |
| **Hybrid** | 4.04 | 0.54 | 8.5× |

## Key Findings

- **Best MRR@10**: Hybrid (0.6534)
- **Best P@5**: Hybrid (0.1713)
- **Best R@5**: Hybrid (0.7869)
- **Best NDCG@10**: Hybrid (0.6941)
- **Latency**: BM25 is 8× faster than Semantic
