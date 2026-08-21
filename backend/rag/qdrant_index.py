"""Minimal Qdrant local-mode indexing helpers (Phase 3B).

Loads Phase 3A BGE-M3 embeddings (``data/processed/embeddings/{strategy}``) into a
local Qdrant store (no Docker) and provides the shared pieces used by
``scripts/index_qdrant.py`` and ``scripts/retrieve_dense.py``.

Kept intentionally small: only what both scripts need. No elaborate resumability
or validation framework — correctness first, infrastructure later.

Embeddings are float32, shape ``(n, 1024)``, already L2-normalized by Phase 3A, so
they are used as-is for cosine similarity (do NOT re-normalize). The positional
invariant ``embeddings[i] <-> mapping[i] <-> chunks[i]`` holds, so chunk ``text``
and the 7 metadata fields are streamed positionally from the chunk parquet in the
same pass as the memmap-sliced vectors — no chunk_id join, no full-RAM load.

Qdrant point id = ``embedding_index`` (int), making upserts idempotent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from backend.rag.embeddings import EMBED_DIM  # noqa: E402  (1024, verified)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
QDRANT_DIR = PROJECT_ROOT / "data" / "processed" / "qdrant"
EMB_ROOT = PROJECT_ROOT / "data" / "processed" / "embeddings"
CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"

COLLECTION_PREFIX = "hhgoa"

# 7 required metadata fields + text (text is needed for retrieval output, req #12).
PAYLOAD_FIELDS = [
    "chunk_id", "document_id", "query_id", "chunk_index",
    "chunk_strategy", "language", "is_selected", "text",
]

STRATEGIES = ["adaptive", "fixed", "semantic"]


def collection_name(strategy: str) -> str:
    return f"{COLLECTION_PREFIX}_{strategy}"


def get_client(path: Path | None = None) -> QdrantClient:
    """Open the local Qdrant store. Callers MUST ``client.close()`` in ``finally``
    (QdrantClient has no context manager; a live process holds the file lock)."""
    return QdrantClient(path=str(path or QDRANT_DIR))


def ensure_collection(client: QdrantClient, strategy: str, dim: int = EMBED_DIM):
    """Create the strategy collection (cosine) if it does not exist. Returns the name."""
    if dim != EMBED_DIM:
        raise ValueError(f"vector dim {dim} != expected {EMBED_DIM}")
    name = collection_name(strategy)
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    return name


def validate_artifacts(strategy: str) -> int:
    """Basic validation of the Phase 3A artifacts for ``strategy``. Returns n_rows.

    Checks: embeddings.npy rows == mapping.parquet rows; dim == 1024;
    embedding_index is contiguous 0..n-1; no duplicate chunk_id in the mapping.
    Raises ValueError on any failure.
    """
    emb_path = EMB_ROOT / strategy / "embeddings.npy"
    map_path = EMB_ROOT / strategy / "mapping.parquet"
    if not emb_path.exists():
        raise ValueError(f"missing embeddings: {emb_path}")
    if not map_path.exists():
        raise ValueError(f"missing mapping: {map_path}")

    emb = np.load(emb_path, mmap_mode="r")
    if emb.ndim != 2 or emb.shape[1] != EMBED_DIM:
        raise ValueError(
            f"embeddings shape {emb.shape}: expected (n, {EMBED_DIM})")
    n = int(emb.shape[0])

    pf = pq.ParquetFile(map_path)
    m = pf.metadata.num_rows
    if m != n:
        raise ValueError(f"embeddings rows {n} != mapping rows {m}")

    # read the mapping columns we need for validation + streaming (single pass)
    cols = ["embedding_index", "chunk_id"]
    table = pq.read_table(map_path, columns=cols)
    eidx = np.asarray(table.column("embedding_index").to_pylist(), dtype=np.int64)
    if not np.array_equal(eidx, np.arange(n)):
        raise ValueError("mapping.embedding_index is not contiguous 0..n-1")

    chunk_ids = table.column("chunk_id").to_pylist()
    if len(set(chunk_ids)) != n:
        raise ValueError(f"{n - len(set(chunk_ids))} duplicate chunk_id(s) in mapping")
    return n


def dense_search(client: QdrantClient, strategy: str, query_vector,
                 top_k: int = 10, only_selected: bool = False) -> list[dict]:
    """Run a dense (cosine) search against the strategy's Qdrant collection.

    Reused by retrieve_dense.py and retrieve_hybrid.py so the dense path is not
    duplicated. ``query_vector`` is a (1024,) unit-norm vector (encode_batch is the
    single normalization — do NOT re-normalize). Returns hit dicts keyed by
    chunk_id with rank/score + the payload fields. The client is owned by the
    caller (which must ``close()`` it in a ``finally``).
    """
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue
    name = collection_name(strategy)
    if not client.collection_exists(name):
        raise RuntimeError(f"collection {name} does not exist — run index_qdrant.py")
    qfilter = None
    if only_selected:
        qfilter = Filter(must=[FieldCondition(key="is_selected",
                                             match=MatchValue(value=1))])
    # accept (1024,) or (1, 1024) numpy / list; Qdrant wants a 1-D list
    import numpy as _np
    vec = _np.asarray(query_vector, dtype=_np.float32).reshape(-1).tolist()
    res = client.query_points(
        name, query=vec, limit=top_k, query_filter=qfilter,
        with_payload=True, with_vectors=False)
    out = []
    for r, p in enumerate(res.points, start=1):
        pl = p.payload or {}
        out.append({
            "rank": r,
            "chunk_id": pl.get("chunk_id"),
            "document_id": pl.get("document_id"),
            "score": round(float(p.score), 6),
            "text": pl.get("text"),
            "query_id": pl.get("query_id"),
            "chunk_index": pl.get("chunk_index"),
            "is_selected": pl.get("is_selected"),
            "strategy": strategy,
        })
    return out


def stream_points(strategy: str, start: int, end: int):
    """Yield ``(point_id, vector_list, payload_dict)`` for rows [start, end).

    Streams the chunk parquet row-group by row-group; for each row group reads
    ``text`` + the 7 metadata fields, and slices the memmap for the matching
    vector rows. Positional (chunks[i] == mapping[i] == embeddings[i]), so no join.
    Bounded RAM (one row group at a time). ``end`` is exclusive.
    """
    emb_path = EMB_ROOT / strategy / "embeddings.npy"
    chunk_path = CHUNK_DIR / f"{strategy}.parquet"
    emb = np.load(emb_path, mmap_mode="r")
    pf = pq.ParquetFile(chunk_path)

    read_cols = [
        "chunk_id", "document_id", "query_id", "chunk_index",
        "chunk_strategy", "language", "is_selected", "text",
    ]
    cur = 0  # global row index
    for rg in range(pf.metadata.num_row_groups):
        if cur >= end:
            break
        rg_size = pf.metadata.row_group(rg).num_rows
        rg_end = cur + rg_size
        if rg_end <= start:
            cur = rg_end
            continue
        lo = max(0, start - cur)
        hi = min(rg_size, end - cur)
        tbl = pf.read_row_group(rg, columns=read_cols)
        cols = {c: tbl.column(c).to_pylist() for c in read_cols}
        for i in range(lo, hi):
            idx = cur + i
            payload = {
                "chunk_id": cols["chunk_id"][i],
                "document_id": cols["document_id"][i],
                "query_id": int(cols["query_id"][i]),
                "chunk_index": int(cols["chunk_index"][i]),
                "chunk_strategy": cols["chunk_strategy"][i],
                "language": cols["language"][i],
                "is_selected": int(cols["is_selected"][i]),
                "text": cols["text"][i],
            }
            yield idx, emb[idx].tolist(), payload
        cur = rg_end


__all__ = [
    "EMBED_DIM", "QDRANT_DIR", "PAYLOAD_FIELDS", "STRATEGIES",
    "collection_name", "get_client", "ensure_collection",
    "validate_artifacts", "dense_search", "stream_points",
]
