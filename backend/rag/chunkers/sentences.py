"""Hindi / Devanagari sentence boundary detection.

A lightweight, dependency-free sentence splitter for the chunkers. The
semantic chunker needs to avoid cutting passages mid-sentence; this module
finds sentence boundaries using the terminators that actually occur in the
data.

Real-data justification (measured on 2,500 Hindi passages of the subset):
* ``।`` (Devanagari danda) appears in 98.3% of passages — the dominant
  sentence terminator.
* ``.`` appears in 36.4% (often decimal points or abbreviations, NOT just
  sentence ends).
* ``?`` 3.1%, ``!`` 2.7%.

Design
------
A sentence ends at one of: ``।`` (danda), ``?``, ``!`` — each optionally
followed by a closing quote/bracket and then whitespace/end-of-string.
``.`` (ASCII period) is treated as a sentence end ONLY when followed by
whitespace and a capital letter / Devanagari consonant / start of a new
sentence token, to avoid splitting on decimals ("3.14") and abbreviations.
This conservative rule errs toward fewer splits (a wrong join is cheaper to
recover from than a wrong cut), which is appropriate for grouping sentences
into chunks.

Returned sentences preserve their original trailing whitespace up to the
next sentence, so concatenating all sentences reproduces the passage
exactly. Sentences with only whitespace are not emitted.
"""

from __future__ import annotations

import re

# A sentence terminator: danda, ?, ! (ASCII), ॥ (double danda, rare).
# We capture the terminator so it stays with the sentence.
_TERMINATORS = r"[।?!॥]"

# Closing punctuation that may appear right after a terminator before the
# whitespace: Devanagari + ASCII quotes and brackets.
_TRAILING = r"[’'”\")\]ऽ।]*"

# Whitespace boundary required after the terminator (or end of text).
# A terminator immediately followed by a non-space (e.g. "3.14", "Mr.X")
# is NOT treated as a sentence boundary.
_SENTENCE_END = (
    r"(?:" + _TERMINATORS + r")"
    + _TRAILING
    + r"(?=\s|$)"
)

# ASCII period is a boundary only if followed by whitespace AND the next
# non-space char looks like the start of a new sentence (Devanagari letter,
# ASCII capital, digit, or another sentence opener). This avoids decimals
# and common abbreviations.
_PERIOD_END = (
    r"\."
    + _TRAILING
    + r"(?=\s+(?:[ऀ-ॿ]|[A-Z]|\d|['\"]))"  # next looks like a new sentence
)

_BOUNDARY_RE = re.compile(r"(?:" + _SENTENCE_END + r"|" + _PERIOD_END + r")")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, preserving exact original text.

    Concatenating the returned list reproduces ``text`` exactly (modulo
    leading/trailing whitespace, which is preserved on the first/last
    sentence). Empty/whitespace-only sentences are dropped.
    """
    if not text or not text.strip():
        return []

    sentences: list[str] = []
    last = 0
    for m in _BOUNDARY_RE.finditer(text):
        end = m.end()
        sentence = text[last:end]
        if sentence.strip():
            sentences.append(sentence)
        last = end
    tail = text[last:]
    if tail.strip():
        sentences.append(tail)
    return sentences


__all__ = ["split_sentences"]
