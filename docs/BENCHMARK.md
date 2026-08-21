# Benchmark — Retrieval-Stage Latency

Measures per-query latency of the retrieval pipeline and reports **P50 / P70 /
P100** per stage. Uses the **real** Qdrant dense path, BM25, and RRF fusion.

## What is measured (per query, timed)

1. **encode** — BGE-M3 query encoding (`encode_batch`; the single normalization)
2. **dense** — Qdrant cosine `dense_search` (top-20 candidates)
3. **bm25** — BM25 `query` (top-20 candidates)
4. **rrf** — `rrf_fuse` fusion to top-k
5. **total** — the above end-to-end

## Cold-start vs warm

- Model + BM25 index load is **untimed** (one-time setup).
- A configurable number of **warmup queries** (default 5) run before timing.
- The **first timed query** is reported separately as `cold_start_ms` (still
  warm after warmup, but the first measurement).
- Warm percentiles (P50/P70/P100) are computed over the remaining timed queries.

## Percentiles

- P50, P70 via `numpy.percentile` (linear interpolation).
- **P100 = observed maximum** (not interpolated) — the worst observed case,
  which is what the official `<200 ms` target cares about.

## Scope caveat (important)

This is a **retrieval-stage benchmark only**. It does **NOT** include:

- STT (voice input)
- LLM answer generation

The official Task 2 requirement is a full-pipeline target of **<200 ms**
(voice → STT → retrieval → answer) with **P50 / P70 / P100** reporting. Until STT
and answer-generation are measured, **no claim is made that the <200 ms target is
met**. This benchmark establishes the retrieval-stage baseline only.

## Files

- `scripts/benchmark.py` — CLI + `percentiles()` helper
- `docs/BENCHMARK.md` — this doc

## Run

```bash
venv/bin/python scripts/benchmark.py --n-queries 50
```

Flags: `--strategy` (adaptive), `--n-queries` (50), `--device`, `--seed`,
`--warmup` (5).

## Kaggle (full artifacts)

The benchmark needs the **full Qdrant index** (Kaggle has it; the laptop has a
partial 32k index). On Kaggle:

```bash
venv/bin/python scripts/benchmark.py --n-queries 100
```
