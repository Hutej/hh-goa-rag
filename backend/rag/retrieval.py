"""Hybrid retrieval: dense + sparse, fused by Reciprocal Rank Fusion.

Replaces ``hybrid.py``. Same fusion mathematics, but the two things that made
the old path slow are gone: the index handles are process-lifetime singletons
(no per-request client open/close) and both retrievers now use real top-k
structures instead of scoring the whole corpus in Python.

Why RRF and not score blending
------------------------------
Dense scores are cosine similarities in [-1, 1]; BM25 scores are unbounded and
corpus-dependent. There is no principled scale on which to add them, and
min-max normalizing per query makes the combination depend on the *worst*
candidate retrieved. RRF sidesteps this entirely by combining **ranks**:

    score(d) = sum over retrievers of  weight / (rrf_k + rank(d))

Rank is scale-free, so the fusion is stable across queries and languages.
``rrf_k=60`` is the standard damping constant; ``sparse_weight=0.25`` was the
verified best on this corpus (word-level BM25 is weak on morphologically rich
Devanagari, so it earns a supporting vote rather than an equal one).

Language handling
-----------------
Dense search runs over every active language and merges by score, which is valid
because all languages share one embedding space — this is what lets a Hindi
question retrieve an English passage on merit.

Sparse search is routed by **script**, the only language signal available at
zero cost that is never wrong: Latin implies English, Devanagari implies Hindi
*or* Marathi (they share the script, so both are searched and fusion decides).
A caller who knows the language — e.g. from Sarvam's returned ``language_code``
— can pin it explicitly instead.
"""

from __future__ import annotations

import threading
import time
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from backend.rag.config import CFG
from backend.rag.dense import DenseIndexError, MultilingualDenseIndex
from backend.rag.encoder import get_encoder
from backend.rag.sparse import MultilingualSparseIndex, SparseIndexError


class RetrievalError(RuntimeError):
    """Retrieval could not run (indexes missing or inconsistent)."""


# --------------------------------------------------------------------------
# script detection (zero-cost language routing)
# --------------------------------------------------------------------------
def detect_script(text: str, sample: int = 120) -> str:
    """Dominant Unicode script of ``text``: ``Devanagari``, ``Latin`` or ``Unknown``.

    Counts letters only, over a short prefix — a few microseconds. This is
    deliberately *not* a language classifier: it cannot separate Hindi from
    Marathi and does not pretend to. It narrows the sparse search to the
    plausible set, and fusion resolves the rest.
    """
    dev = lat = 0
    for ch in text[:sample]:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if name.startswith("DEVANAGARI"):
            dev += 1
        elif name.startswith("LATIN"):
            lat += 1
    if dev == 0 and lat == 0:
        return "Unknown"
    return "Devanagari" if dev >= lat else "Latin"


def route_languages(text: str, pinned: str | None = None) -> list[str]:
    """Active language codes worth searching lexically for ``text``."""
    if pinned:
        code = pinned.split("-")[0].lower()
        if code in CFG.languages:
            return [code]
    script = detect_script(text)
    if script == "Unknown":
        return list(CFG.languages)
    codes = CFG.languages_for_script(script)
    return codes or list(CFG.languages)


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------
def rrf_fuse(dense_hits: list[dict], sparse_hits: list[dict],
             rrf_k: int | None = None,
             dense_weight: float | None = None,
             sparse_weight: float | None = None,
             max_per_document: int | None = None) -> list[dict]:
    """Fuse two ranked lists by weighted RRF. Pure function, no I/O.

    ``max_per_document`` caps how many chunks a single source document may
    contribute. Adjacent chunks of one passage are near-duplicates: without a
    cap they crowd out genuinely different evidence, which both wastes the
    generation context window and makes the answer look narrower than the corpus
    actually supports.
    """
    rrf_k = CFG.rrf_k if rrf_k is None else int(rrf_k)
    dense_weight = CFG.dense_weight if dense_weight is None else float(dense_weight)
    sparse_weight = CFG.sparse_weight if sparse_weight is None else float(sparse_weight)
    cap = CFG.max_per_document if max_per_document is None else int(max_per_document)

    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    ranks: dict[str, dict] = {}
    raw: dict[str, dict] = {}

    def add(hits: list[dict], weight: float, src: str) -> None:
        for h in hits:
            cid = h.get("chunk_id")
            if cid is None:
                continue
            scores[cid] = scores.get(cid, 0.0) + weight / (rrf_k + h["rank"])
            ranks.setdefault(cid, {})[src] = h["rank"]
            raw.setdefault(cid, {})[src] = h.get("score")
            if cid not in meta:
                meta[cid] = h

    add(dense_hits, dense_weight, "dense")
    add(sparse_hits, sparse_weight, "sparse")

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    fused: list[dict] = []
    per_doc: dict[str, int] = {}
    for cid, score in ordered:
        m = meta[cid]
        doc = m.get("document_id")
        if cap > 0 and doc is not None:
            if per_doc.get(doc, 0) >= cap:
                continue
            per_doc[doc] = per_doc.get(doc, 0) + 1
        r = ranks[cid]
        fused.append({
            "rank": len(fused) + 1,
            "chunk_id": cid,
            "document_id": doc,
            "row": m.get("row"),
            "text": m.get("text"),
            "query_id": m.get("query_id"),
            "chunk_index": m.get("chunk_index"),
            "is_selected": m.get("is_selected"),
            "lang": m.get("lang"),
            "dense_rank": r.get("dense"),
            "sparse_rank": r.get("sparse"),
            "dense_score": raw[cid].get("dense"),
            "sparse_score": raw[cid].get("sparse"),
            "rrf_score": round(score, 6),
        })
    return fused


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    query: str
    hits: list[dict]
    dense_hits: list[dict] = field(default_factory=list)
    sparse_hits: list[dict] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    routed_languages: list[str] = field(default_factory=list)
    script: str = "Unknown"

    @property
    def best_rrf(self) -> float:
        return max((h.get("rrf_score") or 0.0) for h in self.hits) \
            if self.hits else 0.0

    @property
    def best_cosine(self) -> float:
        """Highest raw dense cosine among fused hits.

        More interpretable than the RRF score for a confidence gate: it answers
        "does anything in the corpus actually resemble this query?" on a fixed
        [-1, 1] scale, independent of how many retrievers voted.
        """
        vals = [h.get("dense_score") for h in self.hits
                if h.get("dense_score") is not None]
        return max(vals) if vals else 0.0

    @property
    def score_margin(self) -> float:
        """Gap between the top two fused scores.

        A flat distribution means the retriever could not discriminate, which is
        a different failure from "nothing matched" and worth treating separately.
        """
        if len(self.hits) < 2:
            return 1.0
        a = self.hits[0].get("rrf_score") or 0.0
        b = self.hits[1].get("rrf_score") or 0.0
        return round(a - b, 6)

    def to_dict(self) -> dict:
        return {"query": self.query, "hits": self.hits, "timing": self.timing,
                "routed_languages": self.routed_languages, "script": self.script,
                "best_rrf": self.best_rrf, "best_cosine": self.best_cosine,
                "score_margin": self.score_margin}


# --------------------------------------------------------------------------
# retriever
# --------------------------------------------------------------------------
class HybridRetriever:
    """Owns the encoder + both index sets for the lifetime of the process."""

    def __init__(self, languages: list[str] | None = None,
                 strategy: str | None = None):
        self.strategy = strategy or CFG.chunk_strategy
        self.languages = languages or CFG.languages
        self.encoder = get_encoder()
        try:
            self.dense = MultilingualDenseIndex.load(self.languages, self.strategy)
        except DenseIndexError as e:
            raise RetrievalError(str(e)) from e
        try:
            self.sparse = MultilingualSparseIndex.load(self.languages, self.strategy)
        except SparseIndexError as e:
            raise RetrievalError(str(e)) from e

    def search(self, query: str, top_k: int | None = None,
               dense_k: int | None = None, sparse_k: int | None = None,
               languages: list[str] | None = None,
               pinned_language: str | None = None,
               only_selected: bool = False,
               use_cache: bool = True) -> RetrievalResult:
        top_k = CFG.top_k if top_k is None else int(top_k)
        dense_k = CFG.dense_k if dense_k is None else int(dense_k)
        sparse_k = CFG.sparse_k if sparse_k is None else int(sparse_k)

        timing: dict[str, float] = {}
        t_all = time.perf_counter()

        t = time.perf_counter()
        qvec = self.encoder.encode_query(query, use_cache=use_cache)
        timing["encode_ms"] = round((time.perf_counter() - t) * 1000, 2)

        script = detect_script(query)
        sparse_langs = languages or route_languages(query, pinned_language)

        t = time.perf_counter()
        dense_hits = self.dense.search(qvec, top_k=dense_k,
                                       languages=languages,
                                       only_selected=only_selected)
        timing["dense_ms"] = round((time.perf_counter() - t) * 1000, 2)

        t = time.perf_counter()
        sparse_hits = self.sparse.query(query, top_k=sparse_k,
                                       languages=sparse_langs,
                                       only_selected=only_selected)
        timing["sparse_ms"] = round((time.perf_counter() - t) * 1000, 2)

        t = time.perf_counter()
        fused = rrf_fuse(dense_hits, sparse_hits)[:top_k]
        timing["fuse_ms"] = round((time.perf_counter() - t) * 1000, 2)

        # Chunk text is fetched only now, for the few chunks that survived
        # fusion. Doing it inside each retriever cost ~9 ms per query.
        t = time.perf_counter()
        self.dense.hydrate(fused)
        timing["hydrate_ms"] = round((time.perf_counter() - t) * 1000, 2)

        timing["retrieval_ms"] = round((time.perf_counter() - t_all) * 1000, 2)

        return RetrievalResult(query=query, hits=fused, dense_hits=dense_hits,
                               sparse_hits=sparse_hits, timing=timing,
                               routed_languages=sparse_langs, script=script)

    def stats(self) -> dict:
        return {"strategy": self.strategy, "languages": self.languages,
                "encoder": self.encoder.stats(),
                "dense": self.dense.stats(), "sparse": self.sparse.stats()}


_retriever: HybridRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = HybridRetriever()
    return _retriever


def reset_retriever() -> None:
    global _retriever
    with _retriever_lock:
        _retriever = None


__all__ = ["HybridRetriever", "RetrievalResult", "RetrievalError", "rrf_fuse",
           "detect_script", "route_languages", "get_retriever", "reset_retriever"]
