"""Strategy 3 — Adaptive (length-aware) chunking.

Routes a passage to a sub-strategy based on its content-token length, so the
chunking behaviour adapts to document characteristics instead of using one
rule for everything.

Thresholds (content tokens, BGE-M3, add_special_tokens=False):

    SHORT    <= 128 tokens  -> ONE chunk (the whole passage).
    MEDIUM   129–512 tokens  -> sentence-aware grouping (semantic).
    LONG     >  512 tokens  -> overlapping fixed-token windows (fixed 256/32).

These thresholds are NOT invented: they are justified by the measured
token-length distribution of the real subset
(``docs/chunking_length_stats.json``):

    <= 128  : 82.6% of passages (164,834)  -> SHORT  (one chunk each)
    129–512 : 17.2% (34,342)               -> MEDIUM (sentence-aware)
    >  512  :  0.2% (414)                   -> LONG   (overlapping fixed)

So the vast majority of passages (the SHORT band, median 88 tokens) are kept
whole — no artificial fragmentation of already-small passages — while only
the genuinely long ones pay the cost of overlap, and the medium band gets
sentence-boundary-aware grouping. The thresholds match the task spec's
suggested policy, and the data confirms they sit at natural distribution
boundaries (P90=153, P99=228; only 0.5% exceed chunk_size 256).

Determinism: routing key is the exact token count (deterministic), and the
sub-strategies (fixed, semantic) are deterministic.
"""

from __future__ import annotations

from typing import Any

from backend.rag.chunkers.base import make_chunks
from backend.rag.chunkers.fixed import split_fixed
from backend.rag.chunkers.semantic import split_semantic
from backend.rag.chunkers.tokenizer import count_tokens

# Thresholds (content tokens).
SHORT_MAX = 128      # <= SHORT_MAX  -> short path
MEDIUM_MAX = 512     # SHORT_MAX+1 .. MEDIUM_MAX -> medium path; > MEDIUM_MAX -> long

STRATEGY = "adaptive"

# Sub-strategy names used for routing (exposed for diagnostics/validation).
PATH_SHORT = "short"
PATH_MEDIUM = "medium"
PATH_LONG = "long"


def adaptive_path(text: str) -> str:
    """Return the routing path name for ``text`` without chunking it."""
    n = count_tokens(text)
    if n <= SHORT_MAX:
        return PATH_SHORT
    if n <= MEDIUM_MAX:
        return PATH_MEDIUM
    return PATH_LONG


def split_adaptive(text: str) -> list[str]:
    """Route ``text`` by length and split with the chosen sub-strategy."""
    if not text or not text.strip():
        return []
    n = count_tokens(text)
    if n <= SHORT_MAX:
        return [text]
    if n <= MEDIUM_MAX:
        return split_semantic(text)
    return split_fixed(text)


def chunk(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Adaptive length-aware chunker."""
    pieces = split_adaptive(doc["text"])
    return make_chunks(doc, pieces, STRATEGY)


class AdaptiveChunker:
    name = STRATEGY

    def chunk(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        return chunk(doc)


__all__ = [
    "SHORT_MAX", "MEDIUM_MAX", "STRATEGY",
    "PATH_SHORT", "PATH_MEDIUM", "PATH_LONG",
    "adaptive_path", "split_adaptive", "chunk", "AdaptiveChunker",
]
