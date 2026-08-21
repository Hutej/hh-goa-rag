"""Hybrid retrieval via Reciprocal Rank Fusion (Phase 5).

Fuses dense (Phase 3B Qdrant) + BM25 (Phase 4) rankings using standard RRF —
no learned weights, no score normalization. Reuses the existing dense and BM25
implementations (no duplication):

    RRF_score(d) = sum over retrievers of  weight / (k + rank)

A document present in only one retriever contributes only that retriever's
term. ``rank`` is 1-indexed (rank 1 = most relevant). Defaults: k=60, weights 1.0.

The dense and BM25 retrievers are keyed by ``chunk_id`` (both return it in their
payloads), so fusion is by chunk identity across the two ranked lists.
"""

from __future__ import annotations

from backend.rag.bm25 import BM25Index
from backend.rag.qdrant_index import dense_search, get_client

RRF_K_DEFAULT = 60


def rrf_fuse(dense_hits: list[dict], bm25_hits: list[dict],
             rrf_k: int = RRF_K_DEFAULT,
             dense_weight: float = 1.0,
             bm25_weight: float = 1.0) -> list[dict]:
    """Fuse two ranked lists (each hit has ``chunk_id`` + ``rank``) by RRF.

    Returns hits sorted by descending RRF score, each carrying:
    rank, chunk_id, document_id, text, query_id, chunk_index, is_selected,
    dense_rank (or None), bm25_rank (or None), rrf_score, strategy.

    Pure function — no I/O, no model. Rank is 1-indexed within each input list.
    """
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    ranks: dict[str, dict] = {}

    def add(hits, weight, src):
        for h in hits:
            cid = h.get("chunk_id")
            if cid is None:
                continue
            scores[cid] = scores.get(cid, 0.0) + weight / (rrf_k + h["rank"])
            ranks.setdefault(cid, {})[src] = h["rank"]
            # keep the richest metadata (text etc.) — prefer the first seen
            if cid not in meta:
                meta[cid] = h

    add(dense_hits, dense_weight, "dense")
    add(bm25_hits, bm25_weight, "bm25")

    fused = []
    for cid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        m = meta[cid]
        rk = ranks[cid]
        fused.append({
            "rank": len(fused) + 1,
            "chunk_id": cid,
            "document_id": m.get("document_id"),
            "text": m.get("text"),
            "query_id": m.get("query_id"),
            "chunk_index": m.get("chunk_index"),
            "is_selected": m.get("is_selected"),
            "dense_rank": rk.get("dense"),
            "bm25_rank": rk.get("bm25"),
            "rrf_score": round(score, 6),
            "strategy": m.get("strategy"),
        })
    return fused


def hybrid_search(model, query: str, strategy: str = "adaptive",
                 top_k: int = 5, dense_k: int = 20, bm25_k: int = 20,
                 rrf_k: int = RRF_K_DEFAULT, dense_weight: float = 1.0,
                 bm25_weight: float = 1.0, only_selected: bool = False,
                 bm25_index: BM25Index | None = None):
    """Run dense + BM25 for ``query`` and fuse with RRF. Returns a dict with
    timing (ms) and the three result lists.

    ``model`` is a loaded BGE-M3 (from ``load_embedder``). The BM25 index is
    built/loaded once if not supplied. The Qdrant client is opened+closed here.
    """
    import time
    from backend.rag.embeddings import encode_batch

    t0 = time.time()
    qvec = encode_batch(model, [query], batch_size=1)  # (1, 1024), unit-norm
    encode_ms = (time.time() - t0) * 1000

    if bm25_index is None:
        bm25_index = BM25Index.load(strategy)

    client = get_client()
    try:
        t_d = time.time()
        dense_hits = dense_search(client, strategy, qvec, top_k=dense_k,
                                  only_selected=only_selected)
        dense_ms = (time.time() - t_d) * 1000
    finally:
        client.close()

    t_b = time.time()
    bm25_hits = bm25_index.query(query, top_k=bm25_k, only_selected=only_selected)
    bm25_ms = (time.time() - t_b) * 1000

    t_r = time.time()
    fused = rrf_fuse(dense_hits, bm25_hits, rrf_k=rrf_k,
                     dense_weight=dense_weight, bm25_weight=bm25_weight)[:top_k]
    rrf_ms = (time.time() - t_r) * 1000

    return {
        "dense_results": dense_hits,
        "bm25_results": bm25_hits,
        "hybrid_results": fused,
        "timing": {
            "encode_ms": round(encode_ms, 1),
            "dense_ms": round(dense_ms, 1),
            "bm25_ms": round(bm25_ms, 1),
            "rrf_ms": round(rrf_ms, 1),
            "total_ms": round((time.time() - t0) * 1000, 1),
        },
        "config": {
            "strategy": strategy, "top_k": top_k, "dense_k": dense_k,
            "bm25_k": bm25_k, "rrf_k": rrf_k,
            "dense_weight": dense_weight, "bm25_weight": bm25_weight,
            "only_selected": only_selected,
        },
    }


__all__ = ["rrf_fuse", "hybrid_search", "RRF_K_DEFAULT"]
