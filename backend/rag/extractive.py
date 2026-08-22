"""Extractive answering — a grounded answer available in ~1 ms.

Why this exists
---------------
No hosted LLM's time-to-first-token can be relied on to fit a 200 ms budget.
Independent measurement puts even the fastest providers in the hundreds of
milliseconds, and it varies with load, region and prompt size. Betting the
latency target on somebody else's queue is not engineering.

So the moment retrieval finishes, the grounded text is already in hand. This
module selects the best-supported sentence(s) from the retrieved chunks and
returns them verbatim. That answer is:

* **fast** — sentence splitting plus lexical scoring, no model call;
* **more grounded than a generated answer, not less** — it is copied from the
  retrieved context, so hallucination is impossible by construction;
* **useful three ways** — it is the low-latency answer, the fallback when the
  LLM errors or times out, and a reference signal for checking whether the
  generated answer drifted away from the evidence.

Scoring
-------
Sentences are ranked by lexical overlap with the query, deliberately not by
embedding similarity: an extra encode call would cost more than everything else
here combined, and for "which sentence of this passage answers the question"
term overlap is a strong signal. The score blends

* **coverage** — fraction of the query's content terms present in the sentence,
  which is what actually indicates answerhood; and
* **density** — matched terms per sentence length, which penalises a long
  sentence that happens to contain every term incidentally.

Retrieval rank contributes a mild prior, so a sentence from the top-ranked chunk
wins ties against an equally-matching sentence from a weaker chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.rag.chunkers.sentences import split_sentences
from backend.rag.sparse import tokenize

# Very common words carry no discriminative signal, so they are excluded from
# coverage. Kept deliberately small and multilingual rather than a full stopword
# list per language: over-filtering would strip content words in Marathi, and
# the density term already handles verbosity.
STOPWORDS = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "at", "to", "for", "and", "or", "as", "by", "it", "its", "this",
    "that", "these", "those", "with", "from", "what", "which", "who", "whom",
    "how", "why", "when", "where", "do", "does", "did", "can", "could",
    # Hindi / Marathi (shared Devanagari function words)
    "है", "हैं", "था", "थी", "थे", "का", "की", "के", "को", "में", "पर", "से",
    "और", "या", "यह", "वह", "क्या", "कौन", "कैसे", "कब", "कहाँ", "क्यों",
    "आहे", "आहेत", "होता", "होती", "होते", "चा", "ची", "चे", "ला", "मध्ये",
    "आणि", "किंवा", "हे", "ते", "काय", "कोण", "कसे", "केव्हा", "कुठे", "का",
}

MIN_SENTENCE_CHARS = 20
MAX_ANSWER_CHARS = 400


@dataclass
class ExtractiveAnswer:
    """A verbatim answer assembled from retrieved text."""
    answer: str
    chunk_ids: list[str] = field(default_factory=list)
    score: float = 0.0
    coverage: float = 0.0
    lang: str | None = None
    sentences: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.answer.strip()

    def to_dict(self) -> dict:
        return {"answer": self.answer, "chunk_ids": self.chunk_ids,
                "score": round(self.score, 4),
                "coverage": round(self.coverage, 4),
                "lang": self.lang, "sentences": self.sentences}


def content_terms(text: str) -> set[str]:
    """Discriminative terms of ``text`` (tokenized, stopwords removed)."""
    return {t for t in tokenize(text) if t not in STOPWORDS}


def score_sentence(sentence: str, query_terms: set[str]) -> tuple[float, float]:
    """Return ``(score, coverage)`` for a candidate sentence.

    ``coverage`` is the share of query terms found. ``score`` combines coverage
    with match density so a focused sentence beats a rambling one that contains
    the same terms.
    """
    if not query_terms:
        return 0.0, 0.0
    s_terms = tokenize(sentence)
    if not s_terms:
        return 0.0, 0.0
    s_set = set(s_terms)
    matched = query_terms & s_set
    if not matched:
        return 0.0, 0.0
    coverage = len(matched) / len(query_terms)
    density = len(matched) / (len(s_terms) ** 0.5)
    return coverage * 0.75 + min(density, 1.0) * 0.25, coverage


def extract_answer(query: str, hits: list[dict], max_sentences: int = 2,
                   max_chunks: int = 3) -> ExtractiveAnswer:
    """Select the best-supported sentence(s) from the top retrieved chunks.

    Only the first ``max_chunks`` hits are considered: past that, retrieval
    confidence is low enough that a verbatim span is more likely to mislead than
    to help.
    """
    q_terms = content_terms(query)
    if not q_terms or not hits:
        return ExtractiveAnswer(answer="")

    candidates: list[tuple[float, float, str, str, str | None]] = []
    for rank, hit in enumerate(hits[:max_chunks], start=1):
        text = (hit.get("text") or "").strip()
        if not text:
            continue
        chunk_id = hit.get("chunk_id")
        lang = hit.get("lang")
        # A mild rank prior breaks ties toward better-retrieved chunks without
        # letting rank override a clearly better sentence further down.
        rank_prior = 1.0 / (1.0 + 0.15 * (rank - 1))

        sentences = split_sentences(text) or [text]
        for sent in sentences:
            s = sent.strip()
            if len(s) < MIN_SENTENCE_CHARS:
                continue
            base, coverage = score_sentence(s, q_terms)
            if base <= 0:
                continue
            candidates.append((base * rank_prior, coverage, s, chunk_id, lang))

    if not candidates:
        # Nothing matched lexically. Fall back to the opening of the top chunk:
        # it is still grounded, and retrieval already judged it most relevant.
        top = hits[0]
        text = (top.get("text") or "").strip()
        if not text:
            return ExtractiveAnswer(answer="")
        snippet = text[:MAX_ANSWER_CHARS].rstrip()
        return ExtractiveAnswer(answer=snippet,
                                chunk_ids=[top.get("chunk_id")],
                                score=0.0, coverage=0.0,
                                lang=top.get("lang"), sentences=1)

    candidates.sort(key=lambda c: c[0], reverse=True)
    chosen = candidates[:max_sentences]

    # Re-order the chosen sentences by their original appearance so the answer
    # reads naturally rather than in score order.
    ordering = {c[2]: i for i, c in enumerate(candidates)}
    chosen.sort(key=lambda c: ordering[c[2]])

    parts: list[str] = []
    ids: list[str] = []
    total = 0
    for score, coverage, sent, chunk_id, lang in chosen:
        if total + len(sent) > MAX_ANSWER_CHARS and parts:
            break
        parts.append(sent)
        total += len(sent)
        if chunk_id and chunk_id not in ids:
            ids.append(chunk_id)

    best = candidates[0]
    answer = " ".join(p.strip() for p in parts).strip()
    answer = re.sub(r"\s+", " ", answer)

    return ExtractiveAnswer(answer=answer, chunk_ids=ids, score=best[0],
                            coverage=best[1], lang=best[4],
                            sentences=len(parts))


__all__ = ["ExtractiveAnswer", "extract_answer", "score_sentence",
           "content_terms", "STOPWORDS"]
