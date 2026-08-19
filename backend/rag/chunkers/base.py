"""Common chunker interface.

A chunker takes ONE normalized per-passage document (a dict matching
``backend.rag.normalize.CANONICAL_FIELDS``) and returns a list of chunk
dicts. Each chunk is a piece of the passage's ``text`` (Hindi) carrying the
full canonical metadata, plus chunk-specific identity fields.

Chunk fields (superset of the passage metadata):

    chunk_id        str   f"{document_id}_c{chunk_index}"
    document_id     str   parent passage document id
    query_id        int   parent query id
    language        str   parent language ("hi")
    text            str   chunk text (a substring of the parent passage text)
    text_en         str   parallel English — full parent passage kept aligned
    query           str   parent query (Hindi)
    query_en        str   parent query (English)
    query_type      str   parent query type
    is_selected     int   parent is_selected ({0,1})
    source          str   "MSMARCO-XI"
    source_file     str   parent source file
    chunk_strategy  str   "fixed" | "semantic" | "adaptive"
    chunk_index     int   0-based position of this chunk within the passage

English handling (see task §7):
    ``text_en`` (the parallel English passage) is NOT independently chunked —
    chunking it separately would destroy alignment with the Hindi chunk.
    Instead the FULL parent English passage is carried on every Hindi chunk
    as metadata, so the English text stays available for cross-lingual
    fallback and citations. A separate English corpus is NOT built here.

Why a tiny shared module instead of an ABC:
    The three strategies differ in HOW they cut text but share identical
    metadata plumbing. Keeping that plumbing in one place (``make_chunks``)
    avoids duplication (Ponytail rule 5) without an over-engineered class
    hierarchy (rule 4). Each strategy implements a function that returns
    *text pieces + chunk strategy name*; this module wraps them with the
    canonical metadata.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from backend.rag.normalize import CANONICAL_FIELDS

# Fields a chunk carries that are NOT in the canonical passage schema.
CHUNK_ONLY_FIELDS = ["chunk_id", "chunk_index", "chunk_strategy", "text"]

# Full chunk schema = canonical metadata + chunk identity fields.
# `text` is shared (it exists in the canonical schema but is overwritten with
# the chunk's substring); every other canonical field is copied verbatim.
CHUNK_FIELDS = ["chunk_id", "chunk_index", "chunk_strategy"] + CANONICAL_FIELDS

# Metadata copied verbatim from the passage to every chunk.
_META_FIELDS = [f for f in CANONICAL_FIELDS if f != "text"]


class TextSplitter(Protocol):
    """A function that splits a passage's Hindi text into ordered pieces."""

    def __call__(self, text: str) -> list[str]: ...


def make_chunks(
    doc: dict[str, Any],
    pieces: list[str],
    chunk_strategy: str,
) -> list[dict[str, Any]]:
    """Wrap text pieces in chunk dicts carrying the parent metadata.

    ``pieces`` is the ordered list of text substrings produced by a strategy.
    Empty pieces are dropped (a chunk must have non-empty text). If ALL pieces
    are empty (e.g. an empty passage), returns ``[]`` — the caller is
    responsible for reporting that a passage produced zero chunks.
    """
    chunks: list[dict[str, Any]] = []
    idx = 0
    for piece in pieces:
        if not piece or not piece.strip():
            continue
        chunk: dict[str, Any] = {}
        for f in _META_FIELDS:
            chunk[f] = doc.get(f)
        chunk["text"] = piece
        chunk["chunk_strategy"] = chunk_strategy
        chunk["chunk_index"] = idx
        chunk["chunk_id"] = f"{doc.get('document_id')}_c{idx}"
        chunks.append(chunk)
        idx += 1
    return chunks


class Chunker(Protocol):
    """Common interface: ``chunker.chunk(doc) -> list[chunk]``."""

    name: str

    def chunk(self, doc: dict[str, Any]) -> list[dict[str, Any]]: ...


def run_chunker(
    chunker: Chunker,
    doc: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply a chunker to one document (kept for symmetry with the task's
    ``chunks = chunker.chunk(document)`` sketch)."""
    return chunker.chunk(doc)


__all__ = [
    "CHUNK_FIELDS", "CHUNK_ONLY_FIELDS", "make_chunks",
    "Chunker", "run_chunker",
]
