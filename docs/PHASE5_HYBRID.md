# Phase 5 — Hybrid Retrieval (Reciprocal Rank Fusion)

Fuses dense (Phase 3B Qdrant) and BM25 (Phase 4) rankings with standard RRF.
No learned weights, no score normalization — just rank-based fusion.

## RRF formula

```
RRF_score(d) = sum over retrievers of   weight / (k + rank)
```

- `rank` is 1-indexed within each retriever (rank 1 = most relevant).
- A document present in only one retriever contributes only that retriever's term.
- Documents are matched across retrievers by `chunk_id` (both return it).
- Defaults: `k = 60`, `dense_weight = 1.0`, `bm25_weight = 1.0`.

## Files

- `backend/rag/hybrid.py` — `rrf_fuse` (pure), `hybrid_search` (orchestration)
- `scripts/retrieve_hybrid.py` — CLI
- `tests/test_hybrid.py` — pure-function tests (no Qdrant/model)
- `backend/rag/qdrant_index.py` — added reusable `dense_search` (shared with `retrieve_dense.py`)

## Retrieval flow

```
query ──► encode_batch (BGE-M3) ──► dense_search (Qdrant cosine) ──┐
                                                                   ├──► rrf_fuse ──► top-k
query ───────────────────────────► BM25Index.query (BM25)     ─────┘
```

Dense and BM25 implementations are **reused, not duplicated**.

## CLI

```bash
venv/bin/python scripts/retrieve_hybrid.py --query "मैनहट्टन परियोजना" --top-k 5
```

Flags and defaults:

| flag | default | meaning |
|---|---|---|
| `--query` | required | query text |
| `--strategy` | adaptive | chunking strategy |
| `--top-k` | 5 | final fused results returned |
| `--dense-k` | 20 | candidates pulled from dense |
| `--bm25-k` | 20 | candidates pulled from BM25 |
| `--rrf-k` | 60 | RRF k constant |
| `--dense-weight` | 1.0 | dense retriever weight |
| `--bm25-weight` | 1.0 | BM25 retriever weight |
| `--only-selected` | off | filter both retrievers to `is_selected==1` |
| `--device` / `--dtype` | auto | BGE-M3 device override |

## Result fields

Each hybrid hit: `rank, chunk_id, document_id, text, query_id, chunk_index,
is_selected, dense_rank (or None), bm25_rank (or None), rrf_score, strategy`.

The JSON also prints the raw `dense_results` and `bm25_results` so the fusion is
auditable.

## Latency

Output reports per-stage timing: `encode_ms`, `dense_ms`, `bm25_ms`, `rrf_ms`,
`total_ms`. This is **retrieval-stage latency only** — it does NOT include STT,
LLM generation, or guardrails, and it is **not** a claim against the official
`<200 ms` end-to-end target. Final P50/P70/P100 benchmarking is a later phase.

## Notes

- Requires both artifacts present in the environment: the Qdrant collection
  `hhgoa_<strategy>` and the persisted BM25 index `data/processed/bm25/<strategy>/`.
- The full 204k-point Qdrant index may exist only on Kaggle. On the laptop, run
  the synthetic unit tests (`pytest tests/test_hybrid.py`) — they need no Qdrant.
- Full hybrid query on Kaggle:
  ```bash
  venv/bin/python scripts/retrieve_hybrid.py --query "मैनहट्टन परियोजना" --top-k 5
  ```
