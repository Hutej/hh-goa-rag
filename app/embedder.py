"""Embedder interface for `rag-local-eval-loop` (default module `app.embedder`).

Contract required by that suite (see its TARGET_INTERFACE.md):

    embed(texts: list[str]) -> array-like, shape (len(texts), dim)
    embed_one(text: str)    -> array-like, shape (dim,)
    get_model()             -> anything; only the side effect matters

This is a thin adapter over `backend/rag/encoder.py` — the same int8 ONNX
encoder that serves production queries. Nothing is reimplemented here, so the
suite grades the real embedding model.

The prefix split is the important part
--------------------------------------
E5-family models are **prefix-conditioned**: queries must be prefixed
``"query: "`` and documents ``"passage: "``. Applying the wrong one, or the same
one to both sides, measurably degrades retrieval — and silently, since the
vectors still look perfectly valid.

The eval suite happens to make the distinction unambiguous, verified by reading
its source rather than assumed:

* ``eval/index_build.py`` calls ``embed(texts)`` on **candidate passage chunks**
  only, when building its throwaway index.
* ``eval/pipeline.py::_search`` calls ``embed_one(query)`` on **queries** only.

So this module maps ``embed`` -> passage encoding and ``embed_one`` -> query
encoding. That is the correct asymmetric pairing for E5, and it is why the
suite's retrieval numbers will reflect this model's real quality.

``embed_one`` is also called once with the literal string ``"dimension probe"``
to infer the embedding width. Encoding that as a query is harmless — only the
shape is read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# The suite puts the project root on sys.path before importing this module, but
# make it explicit so `python -m app.embedder` and direct imports both work.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.rag.encoder import get_encoder  # noqa: E402


def get_model():
    """Load the ONNX encoder once. Returns it, though the suite ignores the value.

    Called before the suite starts timing, so the ~1.7 s session construction is
    not charged to the first embed call.
    """
    return get_encoder()


def embed(texts: list[str]) -> np.ndarray:
    """Encode **passages** to a ``(len(texts), dim)`` float32 array.

    Vectors are L2-normalized exactly once, inside the encoder, so the suite's
    ``faiss.METRIC_INNER_PRODUCT`` index yields true cosine similarity.
    """
    if texts is None:
        return np.zeros((0, get_encoder().dim), dtype=np.float32)
    if isinstance(texts, str):        # tolerate a bare string
        texts = [texts]
    return get_encoder().encode_passages(list(texts), batch_size=64)


def embed_one(text: str) -> np.ndarray:
    """Encode a single **query** to a ``(dim,)`` float32 vector.

    Caching is disabled: the suite measures embed latency per call, and serving a
    cached vector would report a cache hit as embedding speed.
    """
    return get_encoder().encode_query(text or "", use_cache=False)


__all__ = ["embed", "embed_one", "get_model"]
