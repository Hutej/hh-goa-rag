"""Layered guardrails — knowing when *not* to answer.

The previous implementation was a single float comparison on the fused RRF
score. That catches an empty result set and nothing else. This module adds four
checkpoints around the model, each answering a different question:

1. :func:`check_input` — *should this query be processed at all?*
   Length bounds, script/language recognition, unsafe content, prompt injection.

2. :func:`check_retrieval` — *did retrieval degenerate?* Plus advisory
   confidence signals. See the measured findings below: this layer is
   deliberately lenient, because retrieval scores turned out to be poor
   off-topic detectors on this corpus.

3. :func:`check_generation` — *did the model itself claim it could answer?*
   Uses the structured ``sufficient`` flag and ``confidence`` it returned,
   turning refusal into a checkable field instead of string-matching prose.

4. :func:`check_answer` — *is the answer actually supported by the context?*
   Post-hoc groundedness by term overlap against the retrieved text, citation
   validation, and an answer-language check.

Measured: why there is no cosine threshold here
-----------------------------------------------
The obvious design is a similarity floor — refuse when the best retrieved chunk
is not similar enough. It was implemented, measured against 80 in-corpus queries
(Hindi and English) and 13 deliberately off-topic ones, and **it does not work**:

    in-corpus best cosine   min 0.848   P50 0.915
    off-topic best cosine   min 0.838   P50 0.866   max 0.896

The distributions almost entirely overlap. Two reasons, both real:

* **Embedding anisotropy.** E5-family vectors occupy a narrow cone, so cosine
  similarities compress into a high, tight band (~0.83-0.95) and absolute values
  carry little information.
* **The corpus is web-scale.** MS MARCO covers a vast topic range, so a query
  like "capital of France" genuinely retrieves French geography passages. Whether
  that counts as "off-topic" is ambiguous in the data itself, not just to the
  measurement.

Rank-shape signals were then tried, scored by Youden's J:

    dense score std (top 20)   J = 0.648  (keeps 72% in-corpus, admits 8% off-topic)
    dense gap (top1 - top10)   J = 0.598  (keeps 68% in-corpus, admits 8% off-topic)
    max BM25 score             J = 0.236
    dense/sparse rank overlap  J = 0.188

The best of those still costs ~28% false refusals, which is worse for a user
than occasionally attempting a weak question. So the design changed to match the
evidence: the retrieval layer blocks only genuinely degenerate results, the
shape signals are reported as **advisory confidence** rather than enforced, and
the real weight moves to layers 3 and 4 — the model's grounded self-assessment
(it sees the actual passages and the question, and judges "these do not answer
it" far better than any threshold) and post-hoc groundedness verification of
what it wrote.

Measured: the model's self-reported confidence does not discriminate either
---------------------------------------------------------------------------
:func:`check_generation` also has a ``min_confidence`` floor, and the obvious
next move is to tune it. That was measured too, over 50 MS MARCO rows split
evenly between answerable and unanswerable, and **it carries no usable signal**.
On every case the model chose to answer, it reported confidence in a very narrow
high band regardless of whether the query was answerable:

    answerable,   answered (n=16):  0.9-1.0
    unanswerable, answered (n=10):  0.9-1.0

Sweeping the threshold is therefore flat — false-refusal plus false-confidence
sums to 0.760 at every value from 0.25 to 0.90, and *worsens* to 0.840 at 0.95
because the only cases it starts rejecting are correct answers. The floor is kept
at a low value purely to catch a degenerate ``confidence: 0`` response; it is not
a tuning knob, and treating it as one would be false precision.

Both of this module's calibration attempts landed the same way: the plausible
scalar signal (retrieval cosine, then model confidence) turned out to be
uninformative, and the check that actually works is the model's binary
``sufficient`` judgement plus lexical verification of what it wrote against the
evidence.

On the unsafe-content layer, honestly
-------------------------------------
The pattern lists below are a demo-grade filter, not a safety classifier. They
catch obvious cases in three languages and are easy to defeat. A production
system would put a trained multilingual moderation model here. What this layer
*does* provide is a real, inspectable decision point with structured output, so
the surrounding harness behaviour can be demonstrated and tested.

Every check returns a :class:`GuardrailVerdict` carrying the measured signals,
so a refusal is explainable rather than a black box.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from backend.rag.config import CFG

# --------------------------------------------------------------------------
# reason codes
# --------------------------------------------------------------------------
OK = "ok"
EMPTY_QUERY = "empty_query"
QUERY_TOO_SHORT = "query_too_short"
QUERY_TOO_LONG = "query_too_long"
UNSUPPORTED_SCRIPT = "unsupported_script"
UNSAFE_CONTENT = "unsafe_content"
PROMPT_INJECTION = "prompt_injection"
NO_RELEVANT_CONTEXT = "no_relevant_context"
LOW_SIMILARITY = "low_similarity"
AMBIGUOUS_RETRIEVAL = "ambiguous_retrieval"
MODEL_INSUFFICIENT = "model_reported_insufficient"
LOW_CONFIDENCE = "low_model_confidence"
UNGROUNDED_ANSWER = "ungrounded_answer"
INVALID_CITATIONS = "invalid_citations"
LANGUAGE_MISMATCH = "language_mismatch"

REFUSAL_MESSAGES = {
    EMPTY_QUERY: "Please provide a question.",
    QUERY_TOO_SHORT: "That question is too short to search on. Could you give a "
                     "little more detail?",
    QUERY_TOO_LONG: "That question is too long. Please shorten it.",
    UNSUPPORTED_SCRIPT: "This system answers questions in Hindi, Marathi or "
                        "English. Please rephrase in one of those languages.",
    UNSAFE_CONTENT: "I can't help with that request.",
    PROMPT_INJECTION: "I can only answer questions about the content in my "
                      "knowledge base.",
    NO_RELEVANT_CONTEXT: "I couldn't find relevant information in the knowledge "
                         "base to answer this question.",
    LOW_SIMILARITY: "I couldn't find anything in the knowledge base that closely "
                    "matches this question, so I'd rather not guess.",
    AMBIGUOUS_RETRIEVAL: "I couldn't confidently identify relevant information "
                         "for this question.",
    MODEL_INSUFFICIENT: "The retrieved context doesn't contain enough "
                        "information to answer this question.",
    LOW_CONFIDENCE: "I'm not confident enough in an answer from the available "
                    "context.",
    UNGROUNDED_ANSWER: "I couldn't produce an answer reliably supported by the "
                       "knowledge base, so I'd rather not answer.",
    INVALID_CITATIONS: "I couldn't verify the sources for that answer.",
    LANGUAGE_MISMATCH: "I couldn't answer in the language of your question "
                       "reliably.",
}


@dataclass
class GuardrailVerdict:
    """Outcome of one guardrail layer."""
    allowed: bool
    stage: str
    reason: str = OK
    message: str = ""
    signals: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return not self.allowed

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "stage": self.stage,
                "reason": self.reason, "message": self.message,
                "signals": self.signals}

    @staticmethod
    def allow(stage: str, **signals) -> "GuardrailVerdict":
        return GuardrailVerdict(True, stage, OK, "", signals)

    @staticmethod
    def block(stage: str, reason: str, **signals) -> "GuardrailVerdict":
        return GuardrailVerdict(
            False, stage, reason,
            REFUSAL_MESSAGES.get(reason, "I can't answer that."), signals)


# --------------------------------------------------------------------------
# layer 1: input
# --------------------------------------------------------------------------
# Obvious harmful-intent markers in the three served languages. Word-boundary
# anchored where the script supports it, to limit false positives.
UNSAFE_PATTERNS = [
    r"\bhow (?:to|do i) (?:make|build|synthesi[sz]e) (?:a )?(?:bomb|explosive|nerve agent|napalm)\b",
    r"\b(?:make|build|construct) (?:a )?(?:pipe bomb|dirty bomb|suicide vest)\b",
    r"\bhow (?:to|do i) (?:kill|murder|poison) (?:someone|somebody|a person|my)\b",
    r"\bchild (?:porn|sexual abuse material)\b|\bcsam\b",
    r"\bhow (?:to|do i) (?:hack|breach) (?:into )?(?:someone|somebody|his|her|their)\b",
    r"बम\s*(?:कैसे|कसे)\s*बना",
    r"(?:किसी को|कोणाला)\s*(?:कैसे|कसे)\s*(?:मार|जान से मार)",
    r"ज़हर\s*(?:कैसे|कसे)\s*(?:दे|बना)",
]

# Attempts to override the system prompt or exfiltrate it. This matters more
# than usual here: the whole value proposition is that answers stay grounded in
# the corpus, so an instruction-override is an attack on groundedness itself.
INJECTION_PATTERNS = [
    r"\bignore (?:all )?(?:previous|prior|above) (?:instructions|prompts?|rules)\b",
    r"\bdisregard (?:all )?(?:previous|prior|your) (?:instructions|rules)\b",
    r"\b(?:you are|act as|pretend to be) (?:now )?(?:a |an )?(?:different|new)\b",
    r"\b(?:reveal|show|print|repeat|output) (?:me )?(?:your |the )?system prompt\b",
    r"\bwhat (?:are|were) your (?:original )?instructions\b",
    r"\bdeveloper mode\b|\bjailbreak\b|\bDAN mode\b",
    r"पिछले\s*(?:सभी\s*)?निर्देश.*(?:भूल|अनदेखा)",
    r"मागील\s*(?:सर्व\s*)?सूचना.*(?:विसर|दुर्लक्ष)",
]

_UNSAFE_RE = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# Scripts the corpus actually covers. A query in Tamil or Cyrillic is not
# unsafe, it is simply unanswerable here, and saying so is better than
# retrieving noise.
SUPPORTED_SCRIPTS = {"DEVANAGARI", "LATIN"}


def dominant_script(text: str, sample: int = 160) -> tuple[str, float]:
    """Return ``(script_name, share)`` for the letters in ``text``."""
    counts: dict[str, int] = {}
    letters = 0
    for ch in text[:sample]:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch).split()[0]
        except ValueError:
            continue
        counts[name] = counts.get(name, 0) + 1
        letters += 1
    if not letters:
        return "NONE", 0.0
    top = max(counts, key=counts.get)
    return top, counts[top] / letters


def check_input(query: str) -> GuardrailVerdict:
    """Validate the query before spending any compute on it."""
    q = (query or "").strip()
    signals: dict = {"chars": len(q)}

    if not q:
        return GuardrailVerdict.block("input", EMPTY_QUERY, **signals)
    if len(q) < CFG.min_query_chars:
        return GuardrailVerdict.block("input", QUERY_TOO_SHORT, **signals)
    if len(q) > CFG.max_query_chars:
        return GuardrailVerdict.block("input", QUERY_TOO_LONG, **signals)

    for rx in _INJECTION_RE:
        if rx.search(q):
            signals["pattern"] = rx.pattern[:60]
            return GuardrailVerdict.block("input", PROMPT_INJECTION, **signals)
    for rx in _UNSAFE_RE:
        if rx.search(q):
            signals["pattern"] = rx.pattern[:60]
            return GuardrailVerdict.block("input", UNSAFE_CONTENT, **signals)

    script, share = dominant_script(q)
    signals["script"] = script
    signals["script_share"] = round(share, 3)
    # Digits-and-punctuation-only queries have no script; let them through and
    # let the retrieval confidence layer judge, rather than guessing here.
    if script not in SUPPORTED_SCRIPTS and script != "NONE":
        return GuardrailVerdict.block("input", UNSUPPORTED_SCRIPT, **signals)

    return GuardrailVerdict.allow("input", **signals)


# --------------------------------------------------------------------------
# layer 2: retrieval confidence
# --------------------------------------------------------------------------
def retrieval_confidence(result) -> dict:
    """Advisory confidence signals from the shape of the score distribution.

    These are *reported*, not enforced. A confident retrieval has one clearly
    best match, so the top dense scores are spread out; an unanswerable query
    produces a flat profile where everything is equally mediocre. Measured
    Youden's J for the best of these signals is 0.648 — informative enough to
    surface to a caller or a UI, not reliable enough to refuse on.
    """
    dense = [h.get("score") for h in result.dense_hits
             if h.get("score") is not None]
    if not dense:
        return {"dense_std": 0.0, "dense_gap": 0.0, "level": "none"}
    top = dense[:20]
    mean = sum(top) / len(top)
    var = sum((x - mean) ** 2 for x in top) / len(top)
    std = var ** 0.5
    gap = top[0] - top[min(9, len(top) - 1)]

    # Thresholds from the measured in-corpus P5 / off-topic P50 crossover.
    if std >= 0.010 and gap >= 0.030:
        level = "high"
    elif std >= 0.006 or gap >= 0.020:
        level = "medium"
    else:
        level = "low"
    return {"dense_std": round(std, 5), "dense_gap": round(gap, 5),
            "level": level}


def check_retrieval(result) -> GuardrailVerdict:
    """Block only genuinely degenerate retrieval; attach advisory confidence.

    ``result`` is a :class:`~backend.rag.retrieval.RetrievalResult`.

    Note what is deliberately *not* here: a similarity threshold tuned to reject
    off-topic queries. It was measured and does not separate the classes (see
    the module docstring). Refusal for unanswerable-but-plausible queries is the
    job of the generation and answer layers.
    """
    conf = retrieval_confidence(result)
    signals = {
        "hits": len(result.hits),
        "best_rrf": round(result.best_rrf, 6),
        "best_cosine": round(result.best_cosine, 4),
        "score_margin": result.score_margin,
        "confidence": conf,
        "thresholds": {"min_relevance": CFG.min_relevance,
                       "min_cosine_floor": CFG.min_cosine},
    }

    if not result.hits:
        return GuardrailVerdict.block("retrieval", NO_RELEVANT_CONTEXT, **signals)
    if result.best_rrf < CFG.min_relevance:
        return GuardrailVerdict.block("retrieval", NO_RELEVANT_CONTEXT, **signals)
    # A degenerate floor only. In-corpus queries measured a minimum of 0.848, so
    # anything below ~0.80 means the encoder or index is malfunctioning rather
    # than the question being hard. This is a sanity trip, not topic detection.
    if result.best_cosine < CFG.min_cosine:
        return GuardrailVerdict.block("retrieval", LOW_SIMILARITY, **signals)

    v = GuardrailVerdict.allow("retrieval", **signals)
    return v


# --------------------------------------------------------------------------
# layer 3: the model's own report
# --------------------------------------------------------------------------
def check_generation(answer, min_confidence: float = 0.25) -> GuardrailVerdict:
    """Honour the model's structured self-assessment.

    ``answer`` is a :class:`~backend.rag.llm.GroundedAnswer`.
    """
    signals = {"sufficient": answer.sufficient,
               "confidence": answer.confidence,
               "valid_json": answer.valid_json,
               "repaired": answer.repaired}
    if not answer.sufficient:
        return GuardrailVerdict.block("generation", MODEL_INSUFFICIENT, **signals)
    if answer.confidence < min_confidence:
        return GuardrailVerdict.block("generation", LOW_CONFIDENCE, **signals)
    if not (answer.answer or "").strip():
        return GuardrailVerdict.block("generation", MODEL_INSUFFICIENT, **signals)
    return GuardrailVerdict.allow("generation", **signals)


# --------------------------------------------------------------------------
# layer 4: post-hoc grounding + language
# --------------------------------------------------------------------------
def groundedness(answer_text: str, sources: list[dict] | list) -> float:
    """Share of the answer's content terms that appear in the retrieved context.

    Lexical rather than embedding-based, deliberately: it costs microseconds, it
    needs no model, and it behaves identically across the three languages. It
    detects the failure that matters — the model asserting specifics (numbers,
    names, entities) that are not in the evidence.

    It is a necessary-not-sufficient check. Paraphrase lowers the score without
    meaning the answer is wrong, which is why the threshold is a floor for
    refusal rather than a correctness score.
    """
    from backend.rag.extractive import content_terms

    a_terms = content_terms(answer_text or "")
    if not a_terms:
        return 0.0
    ctx: list[str] = []
    for s in sources:
        text = s.get("text") if isinstance(s, dict) else getattr(s, "text", "")
        if text:
            ctx.append(text)
    if not ctx:
        return 0.0
    c_terms = content_terms(" ".join(ctx))
    if not c_terms:
        return 0.0
    return len(a_terms & c_terms) / len(a_terms)


def check_answer(answer_text: str, sources: list, query: str,
                 cited_ids: list[str] | None = None) -> GuardrailVerdict:
    """Final gate: grounding, citation validity, answer language."""
    grounded = groundedness(answer_text, sources)
    q_script, _ = dominant_script(query)
    a_script, _ = dominant_script(answer_text)

    signals = {
        "groundedness": round(grounded, 4),
        "min_groundedness": CFG.min_groundedness,
        "query_script": q_script,
        "answer_script": a_script,
    }

    if cited_ids is not None:
        valid = {s.get("chunk_id") if isinstance(s, dict)
                 else getattr(s, "chunk_id", None) for s in sources}
        bad = [c for c in cited_ids if c not in valid]
        signals["cited"] = len(cited_ids)
        signals["invalid_citations"] = len(bad)
        if bad:
            return GuardrailVerdict.block("answer", INVALID_CITATIONS, **signals)

    if grounded < CFG.min_groundedness:
        return GuardrailVerdict.block("answer", UNGROUNDED_ANSWER, **signals)

    # Answering a Hindi question in English is a real and common failure. It is
    # reported rather than blocked: the content may still be correct, and
    # refusing a right answer over its script would be worse than flagging it.
    if q_script in SUPPORTED_SCRIPTS and a_script in SUPPORTED_SCRIPTS \
            and q_script != a_script:
        v = GuardrailVerdict.allow("answer", **signals)
        v.reason = LANGUAGE_MISMATCH
        v.message = ""
        return v

    return GuardrailVerdict.allow("answer", **signals)


def describe() -> dict:
    """Guardrail configuration summary for /api/health."""
    return {
        "layers": ["input", "retrieval", "generation", "answer"],
        "input": {"min_chars": CFG.min_query_chars,
                  "max_chars": CFG.max_query_chars,
                  "supported_scripts": sorted(SUPPORTED_SCRIPTS),
                  "unsafe_patterns": len(UNSAFE_PATTERNS),
                  "injection_patterns": len(INJECTION_PATTERNS)},
        "retrieval": {"min_relevance": CFG.min_relevance,
                      "min_cosine": CFG.min_cosine},
        "answer": {"min_groundedness": CFG.min_groundedness},
    }


__all__ = ["GuardrailVerdict", "check_input", "check_retrieval",
           "check_generation", "check_answer", "groundedness",
           "dominant_script", "describe", "REFUSAL_MESSAGES",
           "OK", "EMPTY_QUERY", "QUERY_TOO_SHORT", "QUERY_TOO_LONG",
           "UNSUPPORTED_SCRIPT", "UNSAFE_CONTENT", "PROMPT_INJECTION",
           "NO_RELEVANT_CONTEXT", "LOW_SIMILARITY", "AMBIGUOUS_RETRIEVAL",
           "MODEL_INSUFFICIENT", "LOW_CONFIDENCE", "UNGROUNDED_ANSWER",
           "INVALID_CITATIONS", "LANGUAGE_MISMATCH"]
