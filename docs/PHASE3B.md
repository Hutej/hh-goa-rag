# Phase 3B — Qdrant Indexing + Dense Retrieval

Loads Phase 3A BGE-M3 embeddings into a **local Qdrant** store (no Docker) and
serves dense retrieval over them.

## Collection

- **Name:** `hhgoa_<strategy>` — one per chunking strategy (`adaptive`, `fixed`, `semantic`).
- **Vector dim:** `1024` (BGE-M3; derived from the embedding artifact, asserted at create time).
- **Distance:** `Cosine`. Embeddings are already L2-normalized by Phase 3A — used as-is.
- **Point id:** `embedding_index` (int) — upserts are idempotent; re-running is safe.
- **Store:** `data/processed/qdrant/` (git-ignored). Local embedded mode, single-process.

## Payload fields

Per point: `chunk_id`, `document_id`, `query_id`, `chunk_index`, `chunk_strategy`,
`language`, `is_selected`, `text`. Text is stored so retrieval returns it in one call.

> Note: text-in-payload is for retrieval output convenience only. It does **not** enable
> BM25 — a later phase needs sparse vectors for that.

## Indexing

Streams the memmap `.npy` (sliced) + chunk parquet `text`/metadata (positional —
`embeddings[i] ↔ mapping[i] ↔ chunks[i]`) into batched `upsert`s. Resume is just
`count()`: a re-run picks up from the existing point count (no progress file).

```bash
# test mode (100 vectors)
venv/bin/python scripts/index_qdrant.py --strategy adaptive --limit 100 --recreate

# full adaptive index
venv/bin/python scripts/index_qdrant.py --strategy adaptive

# other strategies (code supports them; index later if time permits)
venv/bin/python scripts/index_qdrant.py --strategy fixed
```

Flags: `--batch-size` (default 256), `--limit N` (0 = all), `--recreate` (drop + rebuild).

## Retrieval

Encodes the query with the Phase 3A BGE-M3 utilities (`encode_batch` is the single
normalization) and runs `query_points`. Prints JSON with measured latency.

```bash
venv/bin/python scripts/retrieve_dense.py --strategy adaptive \
    --query "मैनहट्टन परियोजना" --top-k 5

venv/bin/python scripts/retrieve_dense.py --strategy adaptive \
    --query "..." --only-selected
```

Output fields per result: `rank, chunk_id, document_id, score, text, query_id,
chunk_index, is_selected, strategy`. `score` is cosine similarity (range `[-1, 1]`).
`latency_ms` is the measured end-to-end time (encode + search); no latency target is claimed.

## Resource notes

- Vectors ~800 MiB/strategy on disk (float32); text payload ~80 MiB.
- RAM bounded by batch size; vectors are memmap-sliced, never fully loaded.
- Qdrant local mode warns above 20k points (expected, non-fatal — no Docker per constraints).
