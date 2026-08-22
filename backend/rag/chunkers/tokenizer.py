"""Shared tokenizer for the chunkers (token-count convention).

Why a real tokenizer
--------------------
Chunk sizes are specified in *tokens*, not whitespace-split words, so the token
count must match the model that will actually embed the text. Otherwise a
"256-token" chunk is 256 of nothing in particular.

This matters much more across languages than it looks. A sentencepiece
tokenizer trained mostly on Latin script fragments Devanagari far more
aggressively, so the *same* 256-token budget holds substantially less content
in Hindi or Marathi than in English. That is precisely why the adaptive
thresholds are calibrated per language (see ``scripts/measure_lengths.py``)
rather than shared.

Only the tokenizer is loaded — no model weights, no forward pass. It is read
from the local ONNX encoder directory via the ``tokenizers`` library, so
neither ``transformers`` nor torch is needed anywhere in this project.

Tokenizer
---------
* Model:    whatever ``EMBED_MODEL`` points at (default
            ``intfloat/multilingual-e5-small``, XLM-RoBERTa vocab, 250k).
* Source:   ``data/models/<model>/tokenizer.json``
            (fetched by ``scripts/download_encoder.py``).

Token-count convention
----------------------
All chunk sizes and thresholds count **content tokens only**
(``add_special_tokens=False``):

* The passage text is what is being split; ``<s>``/``</s>`` are not part of the
  passage and would inflate every count by a constant 2.
* The embedding model adds its own special tokens at encode time anyway, so the
  chunker's job is to count content length.
* One convention across all strategies keeps chunk statistics comparable.

Instance isolation
------------------
A dedicated ``Tokenizer`` instance is used here, deliberately *not* the one
inside ``encoder.py``. The encoder enables truncation and padding on its
instance for batched inference; chunking needs the complete, untruncated token
sequence. Sharing one object would let those settings silently clip long
passages mid-chunk.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from backend.rag.config import CFG

# Content-token convention: do not count <s>/</s>.
ADD_SPECIAL_TOKENS = False

MODEL_NAME = CFG.embed_model


class TokenizerUnavailable(RuntimeError):
    """tokenizer.json is missing — the encoder has not been downloaded."""


@lru_cache(maxsize=1)
def get_tokenizer() -> Any:
    """Return the process-wide chunking tokenizer (loaded once).

    Truncation and padding are explicitly disabled so callers always receive
    the full token sequence for a passage.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError as e:  # pragma: no cover - dependency guard
        raise TokenizerUnavailable(
            "the `tokenizers` package is not installed "
            "(pip install -r requirements.txt)") from e

    path = CFG.tokenizer_path
    if not path.exists():
        raise TokenizerUnavailable(
            f"tokenizer.json not found at {path}\n"
            f"Run: python scripts/download_encoder.py")

    tok = Tokenizer.from_file(str(path))
    tok.no_truncation()
    tok.no_padding()
    return tok


def count_tokens(text: str) -> int:
    """Number of content tokens in ``text`` (``add_special_tokens=False``)."""
    if not text:
        return 0
    enc = get_tokenizer().encode(text, add_special_tokens=ADD_SPECIAL_TOKENS)
    return len(enc.ids)


def encode_with_offsets(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Return ``(input_ids, offsets)`` where ``offsets[i]`` is a char span.

    ``input_ids`` are content tokens, so they align 1:1 with ``offsets``. Each
    offset is a half-open character span into ``text``, which lets a chunker
    slice the EXACT original substring for a token window — preserving
    whitespace, punctuation and Devanagari combining marks. Re-tokenizing a
    substring would yield a different count because of BPE context effects,
    which is why counts always come from the original encoding.
    """
    if not text:
        return [], []
    enc = get_tokenizer().encode(text, add_special_tokens=ADD_SPECIAL_TOKENS)
    return list(enc.ids), [tuple(o) for o in enc.offsets]


def count_tokens_batch(texts: list[str]) -> list[int]:
    """Token counts for many texts in one call (much faster than a loop).

    ``encode_batch`` releases the GIL and parallelizes internally, which makes
    a real difference when measuring length distributions over ~200K passages.
    """
    if not texts:
        return []
    encs = get_tokenizer().encode_batch(
        texts, add_special_tokens=ADD_SPECIAL_TOKENS)
    return [len(e.ids) for e in encs]


__all__ = [
    "MODEL_NAME", "ADD_SPECIAL_TOKENS", "TokenizerUnavailable",
    "get_tokenizer", "count_tokens", "encode_with_offsets",
    "count_tokens_batch",
]
