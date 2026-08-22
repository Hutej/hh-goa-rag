"""Sparse (BM25) retrieval — one index per language, backed by ``bm25s``.

Replaces ``rank_bm25.BM25Okapi``, which was the measured #1 latency bottleneck
(P50 398 ms, P100 1241 ms over ~204K docs). ``BM25Okapi.get_scores`` walks the
whole corpus in Python per query and the caller then did a full ``np.argsort``
over every score. ``bm25s`` instead scores through a scipy sparse term-document
matrix and selects top-k directly, which is the same arithmetic done ~100x
faster.

Two further design points:

**Per-language indexes, not one shared index.** IDF is a corpus statistic; if
Hindi, Marathi and English share one index the document frequencies blend and
term weighting degrades for all three. Separate indexes keep IDF meaningful and
let the router search only plausible languages.

**Metadata stays columnar.** The previous implementation materialized every row
— including the full chunk text — as a Python list of dicts at load time, which
cost 1-2 GB of RSS. Here metadata is an Arrow table and only the top-k rows are
ever converted to Python objects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from backend.rag.config import CFG

# Columns carried per chunk. `text` is kept so retrieval can return the passage
# without a second lookup, but it is only decoded for returned hits.
META_FIELDS = ["chunk_id", "document_id", "query_id", "chunk_index",
               "is_selected", "text"]

# Punctuation stripped from token edges. Includes the Devanagari danda (U+0964)
# and double danda (U+0965) alongside ASCII punctuation.
_STRIP = "\u0964\u0965.,;:!?\"'()[]{}|/\\-–—…*#@&%+=<>~`"


def tokenize(text: str) -> list[str]:
    """Whitespace split, strip edge punctuation, lowercase.

    Deliberately not ``\\w+``: Python's ``\\w`` does not match Devanagari
    combining vowel signs, so a regex tokenizer shreds Hindi and Marathi words
    (``मैनहट्टन`` comes apart at the matras). Splitting already-chunked text on
    whitespace yields whole words for Devanagari and Latin alike, which is what
    BM25 needs — the dense retriever covers semantic fuzz.

    Single characters are dropped: they carry almost no lexical signal and
    inflate the vocabulary.
    """
    if not text:
        return []
    out: list[str] = []
    for w in text.split():
        w = w.strip(_STRIP).lower()
        if len(w) >= 2:
            out.append(w)
    return out


class SparseIndexError(RuntimeError):
    """Sparse index missing or unreadable."""


class SparseIndex:
    """A BM25 index over one language's chunks."""

    def __init__(self, lang: str, retriever, meta: pa.Table, strategy: str):
        self.lang = lang
        self.strategy = strategy
        self._retriever = retriever
        self._meta = meta
        self.n = meta.num_rows

        # Small fields as Python lists for O(1) lookup; `text` stays in Arrow.
        # `ChunkedArray.take` resolves chunk boundaries over the whole array, so
        # it cost ~5 ms per query against 2.8 ms for the actual BM25 scoring.
        # Text is fetched by `hydrate` for final results only.
        self._chunk_id = meta.column("chunk_id").to_pylist()
        self._document_id = meta.column("document_id").to_pylist()
        self._query_id = meta.column("query_id").to_pylist()
        self._chunk_index = meta.column("chunk_index").to_pylist()
        self._is_selected = meta.column("is_selected").to_pylist()
        # Single contiguous array so per-row text access is O(1); see dense.py.
        self._text_arr = meta.column("text").combine_chunks()

    # -- build / persist --------------------------------------------------
    @staticmethod
    def build(lang: str, strategy: str | None = None,
              chunks_path: Path | None = None) -> "SparseIndex":
        """Tokenize a language's chunk parquet and build the BM25 index.

        Streams row group by row group so peak RAM stays bounded regardless of
        corpus size.
        """
        import bm25s

        strategy = strategy or CFG.chunk_strategy
        path = Path(chunks_path or CFG.chunks_path(lang, strategy))
        if not path.exists():
            raise SparseIndexError(f"missing chunk parquet: {path}")

        pf = pq.ParquetFile(path)
        corpus_tokens: list[list[str]] = []
        batches: list[pa.RecordBatch] = []

        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=META_FIELDS)
            for t in tbl.column("text").to_pylist():
                corpus_tokens.append(tokenize(t or ""))
            batches.extend(tbl.to_batches())

        if not corpus_tokens:
            raise SparseIndexError(f"no rows in {path}")

        meta = pa.Table.from_batches(batches).combine_chunks()

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        return SparseIndex(lang, retriever, meta, strategy)

    def save(self, out_dir: Path | None = None) -> Path:
        d = Path(out_dir or CFG.sparse_dir(self.lang, self.strategy))
        d.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(d / "bm25"), corpus=None)
        pq.write_table(self._meta, d / "metadata.parquet", compression="zstd")
        (d / "info.json").write_text(json.dumps({
            "lang": self.lang, "strategy": self.strategy, "n": self.n,
            "backend": "bm25s",
        }, indent=2), encoding="utf-8")
        return d

    @staticmethod
    def load(lang: str, strategy: str | None = None,
             index_dir: Path | None = None) -> "SparseIndex":
        import bm25s

        strategy = strategy or CFG.chunk_strategy
        d = Path(index_dir or CFG.sparse_dir(lang, strategy))
        meta_path = d / "metadata.parquet"
        if not meta_path.exists():
            raise SparseIndexError(
                f"sparse index not built for {lang!r}: {d}\n"
                f"Run: python scripts/build_indexes.py --languages {lang}")
        try:
            retriever = bm25s.BM25.load(str(d / "bm25"), mmap=True,
                                        load_corpus=False)
        except Exception as e:
            raise SparseIndexError(
                f"failed to load bm25s index at {d}: {e}") from e
        meta = pq.read_table(meta_path, columns=META_FIELDS)
        return SparseIndex(lang, retriever, meta, strategy)

    # -- query -----------------------------------------------------------
    def query(self, text: str, top_k: int = 20,
              only_selected: bool = False) -> list[dict]:
        """Return up to ``top_k`` hits, highest BM25 score first.

        ``score`` is a raw BM25 score (unbounded, not a similarity). Only rank
        is used by RRF fusion, so the scale does not need normalizing.
        """
        tokens = tokenize(text)
        if not tokens or self.n == 0:
            return []

        # Over-fetch when filtering, so post-filtering can still fill top_k.
        want = min(self.n, top_k * 4 if only_selected else top_k)
        try:
            # n_threads=0 lets bm25s pick; measured ~30% faster than pinning 1.
            idx, scores = self._retriever.retrieve(
                [tokens], k=want, show_progress=False, n_threads=0)
        except Exception as e:
            raise SparseIndexError(f"bm25s retrieve failed: {e}") from e

        rows = np.asarray(idx[0], dtype=np.int64)
        vals = np.asarray(scores[0], dtype=np.float64)

        # bm25s pads with score 0 when fewer than k documents match any term.
        keep = vals > 0
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
                "retriever": "sparse",
            })
            if len(out) >= top_k:
                break
        return out

    def texts_for(self, rows: list[int]) -> list[str]:
        """Fetch chunk text for specific rows (called only for final results)."""
        arr = self._text_arr
        n = self.n
        return [arr[r].as_py() if 0 <= r < n else None for r in rows]


class MultilingualSparseIndex:
    """Holds one :class:`SparseIndex` per active language and unions results.

    Routing is by script, which is the only reliable signal available at zero
    latency: Latin text can only be English here, Devanagari is Hindi *or*
    Marathi. Because those two share a script, a Devanagari query searches both
    and lets fusion decide — no language classifier on the hot path, and no
    silent misroute. A caller that knows the language (e.g. from Sarvam's
    returned ``language_code``) can pin it via ``languages=``.
    """

    def __init__(self, indexes: dict[str, SparseIndex], strategy: str):
        self.indexes = indexes
        self.strategy = strategy

    @staticmethod
    def load(languages: list[str] | None = None,
             strategy: str | None = None) -> "MultilingualSparseIndex":
        strategy = strategy or CFG.chunk_strategy
        codes = languages or CFG.languages
        loaded: dict[str, SparseIndex] = {}
        errors: list[str] = []
        for code in codes:
            try:
                loaded[code] = SparseIndex.load(code, strategy)
            except SparseIndexError as e:
                errors.append(str(e))
        if not loaded:
            raise SparseIndexError(
                "no sparse indexes could be loaded:\n" + "\n".join(errors))
        return MultilingualSparseIndex(loaded, strategy)

    @property
    def total_docs(self) -> int:
        return sum(i.n for i in self.indexes.values())

    def query(self, text: str, top_k: int = 20,
              languages: list[str] | None = None,
              only_selected: bool = False) -> list[dict]:
        """Search the selected languages and merge into one ranked list.

        Scores from different indexes are not directly comparable (different
        IDF), so the merge sorts by score only to establish a *within-sparse*
        ordering for RRF. Cross-retriever comparison is rank-based by design.
        """
        codes = [c for c in (languages or list(self.indexes)) if c in self.indexes]
        if not codes:
            codes = list(self.indexes)

        merged: list[dict] = []
        for code in codes:
            merged.extend(self.indexes[code].query(
                text, top_k=top_k, only_selected=only_selected))

        merged.sort(key=lambda h: h["score"], reverse=True)
        for r, h in enumerate(merged[:top_k], start=1):
            h["rank"] = r
        return merged[:top_k]

    def stats(self) -> dict:
        return {"backend": "bm25s", "strategy": self.strategy,
                "languages": {c: i.n for c, i in self.indexes.items()},
                "total_docs": self.total_docs}


__all__ = ["tokenize", "SparseIndex", "MultilingualSparseIndex",
           "SparseIndexError", "META_FIELDS"]
