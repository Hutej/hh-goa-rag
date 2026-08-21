# Experiment 3 — Retrieval Latency Benchmark

Strategy: adaptive. Device: cpu(float32). 20 timed queries (warmup 5). Warmup + model/index load untimed. Real Qdrant dense + BM25 + RRF.

## Warm per-query percentiles (ms)

| Stage | P50 | P70 | P100 |
|---|---:|---:|---:|
| encode_ms | 219.81 | 265.07 | 543.88 |
| dense_ms | 278.8 | 300.4 | 553.75 |
| bm25_ms | 397.88 | 531.02 | 1240.9 |
| rrf_ms | 0.11 | 0.11 | 0.2 |
| total_ms | 928.61 | 1117.47 | 2059.6 |

Cold-start (first timed query, ms): `{'encode_ms': 543.88, 'dense_ms': 553.75, 'bm25_ms': 743.47, 'rrf_ms': 0.11, 'total_ms': 1841.23}`.

**Target (<200ms): NOT met** (retrieval-stage only — STT + LLM not included).

## Methodology

Follows the judge-provided benchmark methodology (`benchmarks/benchmark.py`):
warmup → repeated real queries → per-stage timing → percentile calculation.
The supplied script targets a FAISS `search()`/`warmup()` interface and is not
run directly against this architecture (different modules); our
`scripts/benchmark.py` implements the same methodology against Qdrant dense +
BM25 + RRF. No query cherry-picking (deterministic stratified sample, seed 12345).

## Earlier GPU measurement (30 warm queries, cuda/float16)

A prior run on the same adaptive index, GPU, 30 timed queries, measured:
embedding P50≈138ms, dense P50≈233ms, BM25 P50≈378ms, RRF P50≈0.11ms,
total P50≈747ms. The CPU run above (P50≈929ms total) is the saved summary; the
GPU run was not persisted to a result file. Both agree BM25 is the bottleneck.
The full-index dense latency should be re-measured on Kaggle (the laptop holds
only a partial 32,756-point Qdrant index; latency is real, dense recall is
incomplete there — recall is measured via bruteforce in the evaluation).

