"""Multi-strategy chunking of normalized MSMARCO-XI passages.

Phase 2. Each chunker takes one normalized per-passage document (as produced
by ``backend.rag.normalize``) and returns a list of *chunks* — sub-pieces of
the passage's ``text`` (Hindi) that carry the full canonical metadata needed
for retrieval, evaluation, citations, debugging, and benchmark comparison.

Three strategies share a common interface (``backend.rag.chunkers.base``):

* ``fixed``     — 256-token sliding window, 32-token overlap.
* ``semantic``  — Hindi sentence-boundary-aware grouping (no embeddings).
* ``adaptive``  — length-aware routing (short / medium / long).

Token counting uses the tokenizer of the planned embedding model
(``backend.rag.chunkers.tokenizer``) — NOT whitespace splitting.
"""
