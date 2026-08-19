"""Strategy 1 — Fixed-size overlapping token windows.

``chunk_size = 256`` content tokens, ``overlap = 32`` content tokens, so the
stride between windows is ``chunk_size - overlap = 224``.

Windows (token indices, content tokens via BGE-M3, add_special_tokens=False):

    0 ----------- 256
            224 -------- 480
                    448 -------- 704
                ...

Behaviour
---------
* A passage with <= ``chunk_size`` tokens -> exactly one chunk (the whole text).
* A longer passage -> overlapping windows of ``chunk_size`` tokens each,
  stepping by ``stride = chunk_size - overlap``.
* Last-window merge: if the final window would start so late that it only
  covers <= ``overlap`` tokens beyond the previous window's end, that final
  window is *not* emitted separately — instead the PREVIOUS window is extended
  to the end of the passage. This avoids a needless tiny trailing chunk while
  never exceeding ``chunk_size`` by more than the overlap (see ``max_size``
  note below). Concretely: we stop spawning new windows once fewer than
  ``stride`` fresh tokens remain; the last emitted window runs to the end.
* Empty/whitespace-only passage -> no chunks (handled by ``make_chunks``).
* Deterministic: a given text always yields the same token windows because
  the tokenizer is deterministic.

Token windows -> text
---------------------
Token counts are computed from the ORIGINAL encoding (window token count =
``j - i``, exact). The chunk text is taken from the tokenizer's offset map as
``text[offs[i][0] : offs[j-1][1]]`` — an EXACT substring of the original
passage, so Devanagari text, whitespace, and punctuation are never truncated
or corrupted. (Re-tokenizing the substring would give a different token count
due to BPE context effects — that's why counts come from the original
encoding; see ``backend/rag/chunkers/tokenizer``.)

``max_size`` honesty
--------------------
A normal window has exactly ``chunk_size`` tokens. The merged final window
can have up to ``chunk_size + overlap - 1`` tokens (287) — i.e. it may exceed
``chunk_size`` but never by more than the overlap. This is the documented
trade for avoiding a tiny dangling chunk and is reported as "chunks exceeding
chunk_size" in the stats.
"""

from __future__ import annotations

from typing import Any

from backend.rag.chunkers.base import make_chunks
from backend.rag.chunkers.tokenizer import encode_with_offsets

# Strategy parameters (content tokens).
CHUNK_SIZE = 256
OVERLAP = 32
STRIDE = CHUNK_SIZE - OVERLAP  # 224

STRATEGY = "fixed"


def split_fixed(text: str) -> list[str]:
    """Split ``text`` into overlapping fixed-size windows of original text.

    Returns the ordered list of exact-substring chunks. Token counts are
    computed from the original encoding; chunk text is taken from the
    tokenizer offset map (no re-tokenization, no character truncation).
    """
    if not text or not text.strip():
        return []
    ids, offs = encode_with_offsets(text)
    n = len(ids)
    if n == 0:
        return []
    if n <= CHUNK_SIZE:
        # short passage -> one chunk = whole text
        return [text[offs[0][0]:offs[n - 1][1]]]

    pieces: list[str] = []
    start = 0
    while start < n:
        end = start + CHUNK_SIZE
        if end >= n:
            # final window runs to the end of the passage
            pieces.append(text[offs[start][0]:offs[n - 1][1]])
            break
        pieces.append(text[offs[start][0]:offs[end - 1][1]])
        # advance by stride; if fewer than `stride` tokens remain after that,
        # the next window would only cover <= overlap fresh tokens -> stop and
        # let the loop's `end >= n` branch merge it into the last window.
        next_start = start + STRIDE
        if next_start + STRIDE > n and next_start < n:
            # remaining fresh tokens (n - next_start) <= STRIDE; spawn one final
            # window starting at next_start that runs to the end (covers the
            # tail), then stop.
            pieces.append(text[offs[next_start][0]:offs[n - 1][1]])
            break
        start = next_start
    return pieces


def chunk(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Fixed 256/32 overlapping-window chunker."""
    pieces = split_fixed(doc["text"])
    return make_chunks(doc, pieces, STRATEGY)


# Expose a simple object with the common interface.
class FixedChunker:
    name = STRATEGY

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        # parameterised path (used by tests); default path uses module consts
        if self.chunk_size == CHUNK_SIZE and self.overlap == OVERLAP:
            return chunk(doc)
        return _chunk_param(doc, self.chunk_size, self.overlap)


def _chunk_param(doc: dict[str, Any], chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    """Parameterised fixed chunker (for tests with non-default sizes)."""
    text = doc["text"]
    if not text or not text.strip():
        return []
    ids, offs = encode_with_offsets(text)
    n = len(ids)
    if n == 0:
        return []
    if n <= chunk_size:
        return make_chunks(doc, [text[offs[0][0]:offs[n - 1][1]]], STRATEGY)
    stride = chunk_size - overlap
    pieces: list[str] = []
    start = 0
    while start < n:
        end = start + chunk_size
        if end >= n:
            pieces.append(text[offs[start][0]:offs[n - 1][1]])
            break
        pieces.append(text[offs[start][0]:offs[end - 1][1]])
        next_start = start + stride
        if next_start + stride > n and next_start < n:
            pieces.append(text[offs[next_start][0]:offs[n - 1][1]])
            break
        start = next_start
    return make_chunks(doc, pieces, STRATEGY)


__all__ = ["CHUNK_SIZE", "OVERLAP", "STRIDE", "STRATEGY",
           "split_fixed", "chunk", "FixedChunker"]
