"""Strategy 2 — Semantic / sentence-aware chunking.

Goal: avoid cutting passages arbitrarily mid-sentence. Group whole sentences
into chunks while respecting a maximum token size.

* Target chunk size ~ 256 content tokens.
* Maximum chunk size 384 content tokens (never exceeded by *grouping*; a
  single sentence longer than 384 tokens is split by the fixed-token fallback).
* If a single sentence exceeds ``max_tokens``, it is split with the fixed
  256/32 strategy (``backend.rag.chunkers.fixed.split_fixed``) so we never
  produce an oversized chunk and never lose text.
* No embeddings, no neural segmentation — just regex sentence boundaries
  (``backend.rag.chunkers.sentences``) + token counting. Fast enough for
  preprocessing the whole subset on CPU.

Algorithm
---------
1. Split the passage into sentences (Devanagari danda, ``?``, ``!``, and a
   conservative ASCII-period rule).
2. Greedily accumulate sentences into the current chunk until adding the next
   sentence would exceed ``max_tokens``. Then close the chunk and start a new
   one. The first sentence that would overflow is NOT merged (so a chunk
   never exceeds ``max_tokens``), EXCEPT when the current chunk is still empty
   (a single sentence already > ``max_tokens``) — that sentence goes through
   the fixed-token fallback.
3. A chunk may legitimately contain a single short sentence.

Determinism: identical inputs yield identical outputs (deterministic
sentence split + deterministic greedy grouping + deterministic fixed fallback).
"""

from __future__ import annotations

from typing import Any

from backend.rag.chunkers.base import make_chunks
from backend.rag.chunkers.fixed import split_fixed
from backend.rag.chunkers.sentences import split_sentences
from backend.rag.chunkers.tokenizer import count_tokens

# Strategy parameters (content tokens).
TARGET_TOKENS = 256
MAX_TOKENS = 384

STRATEGY = "semantic"


def split_semantic(text: str) -> list[str]:
    """Sentence-aware split of ``text`` into chunks <= ``MAX_TOKENS`` tokens.

    Returns the ordered list of exact-substring chunks.
    """
    if not text or not text.strip():
        return []
    sentences = split_sentences(text)
    if not sentences:
        return []

    # Fast path: whole passage fits in one chunk.
    if count_tokens(text) <= MAX_TOKENS:
        return [text]

    pieces: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    def flush():
        nonlocal cur, cur_tokens
        if cur:
            pieces.append("".join(cur))
            cur = []
            cur_tokens = 0

    for sent in sentences:
        stoks = count_tokens(sent)
        if stoks > MAX_TOKENS:
            # Close the current chunk first (flush), then split the oversized
            # sentence with the fixed-token fallback.
            flush()
            for piece in split_fixed(sent):
                pieces.append(piece)
            continue
        if cur_tokens + stoks > MAX_TOKENS:
            # adding this sentence would overflow -> close current chunk
            flush()
        cur.append(sent)
        cur_tokens += stoks
    flush()
    return pieces


def chunk(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Semantic sentence-aware chunker."""
    pieces = split_semantic(doc["text"])
    return make_chunks(doc, pieces, STRATEGY)


class SemanticChunker:
    name = STRATEGY

    def chunk(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        return chunk(doc)


__all__ = ["TARGET_TOKENS", "MAX_TOKENS", "STRATEGY",
           "split_semantic", "chunk", "SemanticChunker"]
