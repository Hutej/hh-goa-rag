"""Retrieval evaluation for the Adaptive corpus (Phase 6).

Relevance = the dataset's ``is_selected == 1`` passages per query. This is the
ground truth used here. ``answer`` / ``answer_en`` are NOT relevance labels and
are never treated as such.

Each query has 1..N chunks (median 10); 61.8% of queries have >=1 selected
passage. Only queries with >=1 selected are evaluatable (a query with no
relevant passage cannot score a hit) — those are skipped.

The dense ranking here uses **exact cosine over the full local embeddings.npy**
(bruteforce). Because the embeddings are unit-norm and Qdrant uses Cosine
distance, this produces the **identical ranking Qdrant would return** — so
dense Recall@k measured by bruteforce equals dense Recall@k on Kaggle's full
Qdrant index, without rebuilding it. BM25 and Hybrid reuse the Phase 4 / Phase 5
code.

Sampling is deterministic (seeded by query_id) and stratified across
``query_type`` so the eval set covers all query types, not arbitrary rows.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from backend.rag.bm25 import BM25Index, tokenize
from backend.rag.hybrid import rrf_fuse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
EMB_ROOT = PROJECT_ROOT / "data" / "processed" / "embeddings"

K_VALUES = [1, 5, 10]
RANKER_DENSE = "dense"
RANKER_BM25 = "bm25"
RANKER_HYBRID = "hybrid"
RANKERS = [RANKER_DENSE, RANKER_BM25, RANKER_HYBRID]


def load_eval_corpus(strategy: str = "adaptive") -> dict:
    """Load the per-query relevance model from the chunk parquet.

    Returns dict with:
      qid -> {"query": text, "query_type": str, "relevant": set(chunk_id),
              "rows": [npy row indices for this query (== chunk row order)]}
    Only queries with >=1 selected passage are included (evaluatable).
    """
    path = CHUNK_DIR / f"{strategy}.parquet"
    t = pq.read_table(path, columns=[
        "query_id", "is_selected", "query", "query_en", "query_type", "chunk_id"])
    qids = t.column("query_id").to_pylist()
    sel = np.array(t.column("is_selected").to_pylist())
    qtexts = t.column("query").to_pylist()
    qtexts_en = t.column("query_en").to_pylist()
    qtypes = t.column("query_type").to_pylist()
    cids = t.column("chunk_id").to_pylist()

    queries: dict[int, dict] = {}
    for i, q in enumerate(qids):
        if q not in queries:
            # Prefer the Hindi query text; fall back to English if Hindi is
            # missing/empty (the dataset is bilingual — ~49% of evaluatable
            # queries have query=None but a non-empty query_en).
            hi = qtexts[i]
            en = qtexts_en[i]
            text = hi if (hi is not None and str(hi).strip()) else en
            queries[q] = {"query": text, "query_hi": hi, "query_en": en,
                          "query_type": qtypes[i], "relevant": set(), "rows": []}
        queries[q]["rows"].append(i)
        if sel[i] == 1:
            queries[q]["relevant"].add(cids[i])
    # keep only evaluatable queries WITH usable query text (dense needs a string)
    usable = {q: v for q, v in queries.items()
              if v["relevant"] and v["query"] is not None and str(v["query"]).strip()}
    return usable


def sample_queries(corpus: dict, n: int = 100, seed: int = 12345) -> list[int]:
    """Deterministic stratified sample across query_type.

    Allocates ``n`` roughly proportionally to each query_type's share, rounding
    down and filling the remainder with the largest type. Deterministic: same
    (n, seed) always yields the same query_ids.
    """
    by_type: dict[str, list[int]] = defaultdict(list)
    for qid, v in corpus.items():
        by_type[v["query_type"]].append(qid)
    # sort each bucket for determinism, then deterministic shuffle via seeded rng
    rng = np.random.default_rng(seed)
    type_items = []
    for qt, qs in by_type.items():
        qs = sorted(qs)
        perm = rng.permutation(len(qs))
        type_items.append((qt, [qs[i] for i in perm]))

    total = sum(len(qs) for _, qs in type_items)
    if n >= total:
        return sorted(q for _, qs in type_items for q in qs)

    # proportional allocation, floor, fill remainder into largest type
    alloc = {qt: (len(qs) * n) // total for qt, qs in type_items}
    allocated = sum(alloc.values())
    remainder = n - allocated
    largest = max(type_items, key=lambda kv: len(kv[1]))[0]
    alloc[largest] += remainder

    out = []
    for qt, qs in type_items:
        k = min(alloc[qt], len(qs))
        out.extend(qs[:k])
    return sorted(out)


def hit_at_k(retrieved_chunk_ids: list, relevant: set, k: int) -> int:
    """1 if any relevant chunk_id appears in the top-k retrieved, else 0."""
    topk = retrieved_chunk_ids[:k]
    return 1 if any(c in relevant for c in topk) else 0


def recall_at_k(hits: list[int]) -> float:
    """Mean hit@k over queries (binary recall: was the relevant passage found)."""
    if not hits:
        return 0.0
    return sum(hits) / len(hits)


def _rows_to_chunk_ids(rows: list[int], strategy: str) -> list[str]:
    """Map chunk-parquet row indices -> chunk_ids via the mapping parquet
    (embedding_index == row order)."""
    mp = pq.read_table(EMB_ROOT / strategy / "mapping.parquet",
                       columns=["chunk_id"])
    all_ids = mp.column("chunk_id").to_pylist()
    return [all_ids[r] for r in rows]


def dense_bruteforce(strategy: str, query_vector: np.ndarray, top_k: int = 10) -> list[str]:
    """Exact cosine top-k via the full local .npy. Returns chunk_ids in rank order.

    Identical ranking to Qdrant (unit-norm + cosine == dot product). For eval only.
    """
    emb = np.load(EMB_ROOT / strategy / "embeddings.npy", mmap_mode="r")
    qv = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    qv = qv / np.linalg.norm(qv)  # defensive: ensure unit norm for cosine
    scores = emb @ qv
    if top_k >= len(scores):
        order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, top_k)[:top_k]
        order = part[np.argsort(-scores[part])]
    rows = order.tolist()
    return _rows_to_chunk_ids(rows, strategy)


def evaluate(rankers: list[str], corpus: dict, qids: list[int],
             bm25: BM25Index, encode_fn, top_k: int = 10,
             dense_k: int = 20, bm25_k: int = 20, rrf_k: int = 60,
             strategy: str = "adaptive", dense_weight: float = 1.0,
             bm25_weight: float = 1.0) -> dict:
    """Run ``rankers`` over the sampled query_ids and return Recall@{1,5,10}.

    ``encode_fn(query_text) -> (1, 1024) unit-norm np array`` (BGE-M3, the single
    normalization). ``bm25`` is a loaded BM25Index for ``strategy``. Dense uses
    exact cosine bruteforce over the full local embeddings.npy for ``strategy``
    (identical to Qdrant cosine ranking). Reused, not duplicated.
    """
    results = {r: {k: [] for k in K_VALUES} for r in rankers}
    for qid in qids:
        q = corpus[qid]
        rel = q["relevant"]
        qvec = encode_fn(q["query"])

        per_ranker = {}
        if RANKER_DENSE in rankers:
            per_ranker[RANKER_DENSE] = dense_bruteforce(strategy, qvec, top_k)
        if RANKER_BM25 in rankers:
            per_ranker[RANKER_BM25] = [h["chunk_id"] for h in
                                       bm25.query(q["query"], top_k=top_k)]
        if RANKER_HYBRID in rankers:
            dh = [{"rank": i + 1, "chunk_id": c} for i, c in
                  enumerate(dense_bruteforce(strategy, qvec, dense_k))]
            bh = [{"rank": i + 1, "chunk_id": h["chunk_id"]} for i, h in
                  enumerate(bm25.query(q["query"], top_k=bm25_k))]
            per_ranker[RANKER_HYBRID] = [h["chunk_id"] for h in
                                        rrf_fuse(dh, bh, rrf_k=rrf_k,
                                                 dense_weight=dense_weight,
                                                 bm25_weight=bm25_weight)[:top_k]
                                        if h["chunk_id"] is not None]

        for r in rankers:
            cids = per_ranker[r]
            for k in K_VALUES:
                results[r][k].append(hit_at_k(cids, rel, k))

    return {r: {k: round(recall_at_k(results[r][k]), 4) for k in K_VALUES}
            for r in rankers}


__all__ = ["load_eval_corpus", "sample_queries", "hit_at_k", "recall_at_k",
           "dense_bruteforce", "evaluate", "K_VALUES", "RANKERS"]
