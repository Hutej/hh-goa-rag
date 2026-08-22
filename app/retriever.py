"""Compatibility shim: ``search()`` / ``warmup()`` for the supplied benchmark.

``benchmarks/benchmark.py`` expects a ``search(query, top_k)`` returning an
object with ``total_ms``, ``embed_ms`` and ``search_ms``, plus a ``warmup()``.
The real pipeline reports a finer breakdown (encode / dense / sparse / fuse /
hydrate), so the mapping onto the coarser interface is made explicit here:

    embed_ms  = encode_ms                        (query encoding)
    search_ms = dense + sparse + fuse + hydrate  (all retrieval work)
    total_ms  = end-to-end retrieval core

The caching note matters for honesty: ``use_cache=False`` is passed, so repeated
queries in the benchmark loop each pay full cost. The supplied script cycles a
short query list, and with the LRU query cache enabled every iteration after the
first would be a cache hit and the reported percentiles would be meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.rag.config import CFG
from backend.rag.retrieval import get_retriever


@dataclass
class SearchResponse:
    """Shape expected by the supplied benchmark script."""
    query: str
    results: list[dict] = field(default_factory=list)
    total_ms: float = 0.0
    embed_ms: float = 0.0
    search_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.results)


def warmup() -> None:
    """Load the encoder and all indexes (idempotent)."""
    get_retriever()


def search(query: str, top_k: int = None) -> SearchResponse:
    """Hybrid retrieval for ``query``, timed for the supplied benchmark."""
    result = get_retriever().search(
        query, top_k=CFG.top_k if top_k is None else int(top_k),
        use_cache=False)
    t = result.timing
    embed_ms = t.get("encode_ms", 0.0)
    search_ms = (t.get("dense_ms", 0.0) + t.get("sparse_ms", 0.0)
                 + t.get("fuse_ms", 0.0) + t.get("hydrate_ms", 0.0))
    return SearchResponse(
        query=query,
        results=[{"chunk_id": h["chunk_id"], "score": h["rrf_score"],
                  "lang": h.get("lang"), "text": h.get("text")}
                 for h in result.hits],
        total_ms=t.get("retrieval_ms", embed_ms + search_ms),
        embed_ms=embed_ms,
        search_ms=round(search_ms, 2),
    )


__all__ = ["search", "warmup", "SearchResponse"]
