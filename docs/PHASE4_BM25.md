# Phase 4 — BM25 Lexical Retrieval

Lexical (keyword) retrieval over the Adaptive chunk corpus, complementary to the
Phase 3B dense retriever. BM25 matches exact surface forms; dense matches
semantic meaning — together they feed the later hybrid/RRF phase.

## Approach

- **Library:** `rank_bm25.BM25Okapi` (pure Python, already in the venv).
- **Tokenization:** whitespace + punctuation-strip + lowercase. Keeps Devanagari
  words whole (`मैनहट्टन` stays one token). A `\w+` regex would fragment Hindi
  combining matras — so it is not used. BM25 needs only surface forms; the dense
  retriever covers semantic fuzz.
- **Index:** built ONCE over `data/processed/chunks/adaptive.parquet`, persisted to
  `data/processed/bm25/{strategy}/` as `bm25.pkl` (the picklable `BM25Okapi`) +
  `metadata.parquet` (row-index → chunk_id/document_id/query_id/chunk_index/
  is_selected/text). `get_scores` is row-index-aligned, so BM25 row i ↔ metadata row i.

## Files

- `backend/rag/bm25.py` — `tokenize`, `build_index`, `BM25Index.load/.query`
- `scripts/retrieve_bm25.py` — CLI
- `tests/test_bm25.py` — synthetic tests

## CLI

```bash
# query (builds+persists the index on first run; loads the pickle thereafter)
venv/bin/python scripts/retrieve_bm25.py --query "मैनहट्टन परियोजना" --top-k 5

# force rebuild
venv/bin/python scripts/retrieve_bm25.py --query "..." --rebuild

# filter to selected passages only
venv/bin/python scripts/retrieve_bm25.py --query "..." --only-selected
```

Flags: `--strategy` (default adaptive), `--query`, `--top-k` (default 10),
`--only-selected`, `--rebuild`.

## Result fields

Per hit: `rank, chunk_id, document_id, query_id, chunk_index, text, is_selected,
score, strategy` — same shape as `retrieve_dense.py` so the hybrid phase can fuse
them directly. `score` is a BM25 score (unbounded positive, **not** a cosine
similarity). Output also reports `index_load_ms`, `search_ms`, `latency_ms`
(measured; no latency target claimed).

## Notes

- `score <= 0` hits are dropped (no term overlap = not a real lexical hit).
- Index is per-strategy; only adaptive is built here. fixed/semantic supported
  via `--strategy` but indexed later if time permits.
- BM25 index storage lives under `data/processed/bm25/` (git-ignored).
