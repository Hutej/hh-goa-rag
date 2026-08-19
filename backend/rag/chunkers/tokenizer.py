"""Shared tokenizer for the chunkers (token-count convention).

Why a real tokenizer
--------------------
Phase 2 needs a reproducible token-count definition for the ``fixed`` and
``adaptive`` strategies (chunk sizes are specified in *tokens*, not
whitespace-split words). The token count must also be consistent with the
later retrieval/embedding stage, so the obvious choice is the tokenizer of
the planned multilingual embedding model.

Only the tokenizer is loaded — NO model weights, NO embeddings, NO neural
forward pass. A tokenizer is a few MB and runs on CPU; this keeps the chunking
phase fast and memory-light (the host box has ~1.5 GiB free RAM).

Tokenizer
---------
* Model:   ``BAAI/bge-m3`` (XLM-RoBERTa-large tokenizer, vocab 250 002).
* Version: provided by ``transformers`` 5.x / ``tokenizers`` 0.22 in the venv.
* Specials: ``<s> </s> <unk> <pad> <mask>``.

Token-count convention
----------------------
All chunk sizes and length thresholds count **content tokens only**, i.e.
``add_special_tokens=False``. Rationale:

* The passage text is what is being split; ``<s>``/``</s>`` are not part of
  the passage and would inflate counts by a constant 2.
* The downstream embedding model adds its own special tokens at embedding
  time anyway, so the chunker's job is to count *content* length.
* Using ``add_special_tokens=False`` everywhere keeps the convention
  consistent and reproducible across the three strategies and the later
  P50/P70/P100 latency benchmarks.

Fast-path vs slow-path
----------------------
For *counting* tokens we use the fast tokenizer directly (``encode``),
which is fast and allocation-light. For *span reconstruction* (mapping a
token window back to an exact character slice of the original text) we use
``return_offsets_mapping=True`` so the chunker never has to re-tokenize or
guess where a token window starts/ends — the tokenizer tells it.

The tokenizer is a process-wide singleton (loaded once, ~32 s on first
load). All three chunkers share it via :func:`get_tokenizer`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from transformers import AutoTokenizer

# The embedding model the project plans to use (PROJECT_CONTEXT.md §11).
# We load its TOKENIZER ONLY — never its weights.
MODEL_NAME = "BAAI/bge-m3"
# Content-token convention: do not count <s>/</s> special tokens.
ADD_SPECIAL_TOKENS = False


@lru_cache(maxsize=1)
def get_tokenizer() -> Any:
    """Return the process-wide BGE-M3 tokenizer (loaded once)."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def count_tokens(text: str) -> int:
    """Number of content tokens in ``text`` (add_special_tokens=False)."""
    if not text:
        return 0
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=ADD_SPECIAL_TOKENS))


def encode_with_offsets(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Return (input_ids, offsets) where offsets[i] = (start, end) char span.

    ``input_ids`` are content token ids (add_special_tokens=False) so they
    align 1:1 with ``offsets``. Each offset is a half-open character span into
    ``text``; the chunker uses these to slice the EXACT original substring for
    a token window, preserving original whitespace and Devanagari text.
    """
    tok = get_tokenizer()
    enc = tok(text, add_special_tokens=ADD_SPECIAL_TOKENS,
              return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


__all__ = [
    "MODEL_NAME", "ADD_SPECIAL_TOKENS",
    "get_tokenizer", "count_tokens", "encode_with_offsets",
]
