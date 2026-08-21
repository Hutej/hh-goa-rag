# Evaluation — Retrieval Recall@k

Measures Dense, BM25, and Hybrid (RRF) retrieval quality on the Adaptive corpus.

## Relevance model

Ground truth = the dataset's `is_selected == 1` passages per query. These are the
human-annotated relevant passages. **`answer` / `answer_en` are NOT relevance
labels and are never used as such.**

- 20,000 queries, each with 1–82 chunks (median 10).
- 12,354 queries (61.8%) have ≥1 selected passage → **evaluatable**. The other
  7,646 have no relevant passage and are skipped (a query with no relevant chunk
  cannot score a hit).

## Metrics

For each query and each `k ∈ {1, 5, 10}`: a binary hit = did any relevant chunk_id
appear in the top-k retrieved? `Recall@k` = mean of these binary hits across
evaluatable queries (fraction of queries whose relevant passage was retrieved
within top-k).

## Dense ranking (bruteforce == Qdrant)

Dense Recall@k is measured by **exact cosine over the full local
`embeddings.npy`** (bruteforce). Because the embeddings are unit-norm and Qdrant
uses Cosine distance, this produces the **identical ranking Qdrant returns** — so
dense Recall@k here equals dense Recall@k on Kaggle's full Qdrant index, **without
rebuilding Qdrant locally**. BM25 and Hybrid reuse the Phase 4 / Phase 5 code.

## Deterministic sampling

`sample_queries(n)` is **deterministic** (seeded) and **stratified across
`query_type`** (DESCRIPTION / NUMERIC / LOCATION / ENTITY / PERSON), allocating
roughly proportionally. Same `(n, seed)` always yields the same query_ids — the
eval is reproducible.

## Bilingual query text (important)

The dataset is bilingual. Of the 12,354 evaluatable queries:
- **6,331** have a non-empty Hindi `query` text.
- **6,023** have `query = None` (Hindi missing) but **all have a non-empty
  `query_en`** (English fallback, e.g. "what is a gaucho?").

`load_eval_corpus` prefers the Hindi `query` and falls back to `query_en` when
Hindi is missing. Queries with **no usable text in either language are dropped**
(BGE-M3 cannot encode `None`). So the evaluatable set used here is the ~6,331
queries with Hindi text (the Hindi-voice-RAG condition), plus English-fallback
queries if they fall in the sample. Reported Recall@k reflects whichever query
text is available per sampled query.

## Files

- `backend/rag/evaluation.py` — `load_eval_corpus`, `sample_queries`, `hit_at_k`,
  `recall_at_k`, `dense_bruteforce`, `evaluate`
- `scripts/evaluate_retrieval.py` — CLI

## Run

```bash
# 100 queries (stratified)
venv/bin/python scripts/evaluate_retrieval.py --n-queries 100

# all evaluatable queries
venv/bin/python scripts/evaluate_retrieval.py --n-queries 0

# subset of rankers
venv/bin/python scripts/evaluate_retrieval.py --n-queries 100 --rankers dense hybrid
```

Output: a Recall@1/@5/@10 table for each ranker + query_type coverage + JSON.

## Kaggle (full artifacts)

The eval runs locally (full `.npy` is local), but on Kaggle with GPU it's faster:

```bash
venv/bin/python scripts/evaluate_retrieval.py --n-queries 500
```
