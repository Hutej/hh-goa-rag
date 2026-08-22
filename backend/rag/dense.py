"""Dense (vector) retrieval — in-process FAISS HNSW, one index per language.

Replaces the embedded-Qdrant path, which was slow for two compounding reasons:

1. ``QdrantClient(path=...)`` is Qdrant's *local* mode — a pure-Python
   reimplementation that brute-force scans every vector and ignores
   ``hnsw_config`` entirely. There was no ANN in the old hot path at all.
2. ``hybrid_search`` opened and closed the client on **every request**, so each
   query re-acquired the store lock and re-initialized the collection. That is
   the most likely cause of dense latency swinging 279 ms -> 1162 ms.

Here the index is a real HNSW graph held in-process for the lifetime of the
worker. Measured dense search on this corpus is ~1 ms.

**Why unit-norm vectors + inner product.** ``encoder.py`` L2-normalizes exactly
once, so cosine similarity is identical to a dot product. FAISS
``METRIC_INNER_PRODUCT`` therefore returns true cosine scores with no extra
normalization step and no divide in the inner loop.

**Why per-language indexes that merge by score.** Unlike BM25 (where blending
corpora corrupts IDF), dense scores from a shared embedding space *are* directly
comparable across languages. So separate indexes give the operational benefit of
adding or rebuilding one language independently, while a score-sorted merge
still yields one globally correct cross-lingual ranking. A Hindi query can and
does surface an English passage on merit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from backend.rag.config import CFG

META_FIELDS = ["chunk_id", "document_id", "query_id", "chunk_index",
               "is_selected", "text"]


class DenseIndexError(RuntimeError):
    """Dense index missing, unreadable, or dimensionally inconsistent."""


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError as e:  # pragma: no cover - dependency guard
        raise DenseIndexError(
            "faiss-cpu is not installed (pip install -r requirements.txt)"
        ) from e


class DenseIndex:
    """A FAISS index over one language's chunk vectors."""

    def __init__(self, lang: str, index, meta: pa.Table, strategy: str):
        self.lang = lang
        self.strategy = strategy
        self._index = index
        self._meta = meta
        self.n = meta.num_rows

        # Small fields are materialized once as Python lists for O(1) indexing.
        # `ChunkedArray.take` costs ~4.4 ms even for 20 rows because it resolves
        # chunk boundaries across the whole array — which made metadata lookup
        # 93% of dense retrieval time, against 0.3 ms for the actual FAISS
        # search. `text` is deliberately NOT materialized: it is the largest
        # column and is only needed for the handful of chunks that survive
        # fusion, so it stays in Arrow and is fetched by `hydrate`.
        self._chunk_id = meta.column("chunk_id").to_pylist()
        self._document_id = meta.column("document_id").to_pylist()
        self._query_id = meta.column("query_id").to_pylist()
        self._chunk_index = meta.column("chunk_index").to_pylist()
        self._is_selected = meta.column("is_selected").to_pylist()
        # `combine_chunks` collapses the ChunkedArray to a single contiguous
        # Array, which makes per-row access genuinely O(1). Leaving it chunked
        # meant every lookup re-resolved chunk boundaries across all ~200K rows
        # — 9 ms to fetch 5 strings. The text itself is not copied into Python
        # objects until a specific row is requested.
        self._text_arr = meta.column("text").combine_chunks()

        if index.ntotal != self.n:
            raise DenseIndexError(
                f"{lang}: index holds {index.ntotal} vectors but metadata has "
                f"{self.n} rows — indexes are out of sync, rebuild required")

    # -- build / persist --------------------------------------------------
    @staticmethod
    def build(lang: str, vectors: np.ndarray, meta: pa.Table,
              strategy: str | None = None, exact: bool = False) -> "DenseIndex":
        """Build an HNSW (or exact Flat) index from unit-norm vectors.

        ``exact=True`` builds ``IndexFlatIP`` — brute force, zero approximation.
        Useful as the ground truth when measuring HNSW recall loss.
        """
        faiss = _faiss()
        strategy = strategy or CFG.chunk_strategy

        v = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if v.ndim != 2:
            raise DenseIndexError(f"expected 2-D vectors, got shape {v.shape}")
        n, d = v.shape
        if d != CFG.embed_dim:
            raise DenseIndexError(
                f"vector dim {d} != configured EMBED_DIM {CFG.embed_dim}")
        if n != meta.num_rows:
            raise DenseIndexError(
                f"{n} vectors but {meta.num_rows} metadata rows")

        norms = np.linalg.norm(v[:min(n, 256)], axis=1)
        if n and not np.allclose(norms, 1.0, atol=1e-2):
            raise DenseIndexError(
                "vectors are not L2-normalized; encoder.py must normalize once "
                "and downstream code must not re-normalize")

        if exact:
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexHNSWFlat(d, CFG.hnsw_m,
                                        faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = CFG.hnsw_ef_construction
            index.hnsw.efSearch = CFG.hnsw_ef_search
        index.add(v)
        return DenseIndex(lang, index, meta, strategy)

    def save(self, out_dir: Path | None = None) -> Path:
        faiss = _faiss()
        path = Path(CFG.dense_index_path(self.lang, self.strategy))
        if out_dir is not None:
            path = Path(out_dir) / path.name
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        pq.write_table(self._meta, CFG.dense_meta_path(self.lang, self.strategy),
                       compression="zstd")
        (path.parent / f"{self.lang}.info.json").write_text(json.dumps({
            "lang": self.lang, "strategy": self.strategy, "n": self.n,
            "dim": CFG.embed_dim, "backend": "faiss",
            "index_type": type(self._index).__name__,
            "hnsw_m": CFG.hnsw_m,
            "ef_construction": CFG.hnsw_ef_construction,
            "ef_search": CFG.hnsw_ef_search,
            "metric": "inner_product (== cosine on unit-norm vectors)",
        }, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(lang: str, strategy: str | None = None) -> "DenseIndex":
        faiss = _faiss()
        strategy = strategy or CFG.chunk_strategy
        ipath = Path(CFG.dense_index_path(lang, strategy))
        mpath = Path(CFG.dense_meta_path(lang, strategy))
        if not ipath.exists() or not mpath.exists():
            raise DenseIndexError(
                f"dense index not built for {lang!r}: {ipath}\n"
                f"Run: python scripts/build_indexes.py --languages {lang}")
        try:
            index = faiss.read_index(str(ipath))
        except Exception as e:
            raise DenseIndexError(f"failed to read {ipath}: {e}") from e
        # efSearch is a query-time knob, so honour the current env on load.
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = CFG.hnsw_ef_search
        meta = pq.read_table(mpath, columns=META_FIELDS)
        return DenseIndex(lang, index, meta, strategy)

    # -- query -----------------------------------------------------------
    def search(self, qvec: np.ndarray, top_k: int = 20,
               only_selected: bool = False) -> list[dict]:
        """Search with a (dim,) or (1, dim) unit-norm query vector.

        ``score`` is cosine similarity in [-1, 1].
        """
        if self.n == 0:
            return []
        q = np.ascontiguousarray(
            np.asarray(qvec, dtype=np.float32).reshape(1, -1))
        if q.shape[1] != CFG.embed_dim:
            raise DenseIndexError(
                f"query dim {q.shape[1]} != EMBED_DIM {CFG.embed_dim}")

        want = min(self.n, top_k * 4 if only_selected else top_k)
        scores, ids = self._index.search(q, want)
        rows = np.asarray(ids[0], dtype=np.int64)
        vals = np.asarray(scores[0], dtype=np.float32)

        keep = rows >= 0  # FAISS pads with -1 when fewer than k neighbours
        rows, vals = rows[keep], vals[keep]
        if rows.size == 0:
            return []

        out: list[dict] = []
        for i in range(len(rows)):
            row = int(rows[i])
            sel = self._is_selected[row]
            if only_selected and not sel:
                continue
            out.append({
                "rank": len(out) + 1,
                "row": row,               # for lazy text hydration
                "chunk_id": self._chunk_id[row],
                "document_id": self._document_id[row],
                "query_id": self._query_id[row],
                "chunk_index": self._chunk_index[row],
                "is_selected": sel,
                "score": round(float(vals[i]), 6),
                "lang": self.lang,
                "strategy": self.strategy,
                "retriever": "dense",
            })
            if len(out) >= top_k:
                break
        return out

    def texts_for(self, rows: list[int]) -> list[str]:
        """Fetch chunk text for specific rows (called only for final results)."""
        arr = self._text_arr
        n = self.n
        return [arr[r].as_py() if 0 <= r < n else None for r in rows]


class MultilingualDenseIndex:
    """Holds one :class:`DenseIndex` per active language, merged by score.

    The merge is score-sorted rather than rank-interleaved because all languages
    share one embedding space, so cosine values are directly comparable. This is
    what makes cross-lingual retrieval work: the ranking is decided on semantic
    similarity, not on which language the query happened to be in.
    """

    def __init__(self, indexes: dict[str, DenseIndex], strategy: str):
        self.indexes = indexes
        self.strategy = strategy

    @staticmethod
    def load(languages: list[str] | None = None,
             strategy: str | None = None) -> "MultilingualDenseIndex":
        strategy = strategy or CFG.chunk_strategy
        codes = languages or CFG.languages
        loaded: dict[str, DenseIndex] = {}
        errors: list[str] = []
        for code in codes:
            try:
                loaded[code] = DenseIndex.load(code, strategy)
            except DenseIndexError as e:
                errors.append(str(e))
        if not loaded:
            raise DenseIndexError(
                "no dense indexes could be loaded:\n" + "\n".join(errors))
        return MultilingualDenseIndex(loaded, strategy)

    @property
    def total_vectors(self) -> int:
        return sum(i.n for i in self.indexes.values())

    def search(self, qvec: np.ndarray, top_k: int = 20,
               languages: list[str] | None = None,
               only_selected: bool = False) -> list[dict]:
        codes = [c for c in (languages or list(self.indexes)) if c in self.indexes]
        if not codes:
            codes = list(self.indexes)

        merged: list[dict] = []
        for code in codes:
            merged.extend(self.indexes[code].search(
                qvec, top_k=top_k, only_selected=only_selected))

        merged.sort(key=lambda h: h["score"], reverse=True)
        for r, h in enumerate(merged[:top_k], start=1):
            h["rank"] = r
        return merged[:top_k]

    def hydrate(self, hits: list[dict]) -> list[dict]:
        """Fill in chunk ``text`` for the given hits, in place.

        Called once on the final fused list rather than per retriever per
        language, which is the whole point: text is the largest column and only
        a handful of chunks ever reach the answer.
        """
        by_lang: dict[str, list[int]] = {}
        for i, h in enumerate(hits):
            if h.get("text") is not None or h.get("row") is None:
                continue
            by_lang.setdefault(h["lang"], []).append(i)
        for lang, positions in by_lang.items():
            idx = self.indexes.get(lang)
            if idx is None:
                continue
            rows = [hits[p]["row"] for p in positions]
            for p, text in zip(positions, idx.texts_for(rows)):
                hits[p]["text"] = text
        return hits

    def stats(self) -> dict:
        return {"backend": "faiss", "strategy": self.strategy,
                "hnsw_m": CFG.hnsw_m, "ef_search": CFG.hnsw_ef_search,
                "languages": {c: i.n for c, i in self.indexes.items()},
                "total_vectors": self.total_vectors}


__all__ = ["DenseIndex", "MultilingualDenseIndex", "DenseIndexError",
           "META_FIELDS"]
