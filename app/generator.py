"""Generator interface for `rag-local-eval-loop` (default module `app.generator`).

Contract required by that suite (see its TARGET_INTERFACE.md):

    generate_answer(query: str, results: list) -> answer object

Each item in ``results`` exposes ``.text`` and ``.source``. The returned object
must expose ``.text: str``, ``.grounded: bool``, ``.generation_ms: float`` and
``.model: str``.

This runs the **real** generation path and the **real** guardrail layers from
``backend/rag/``. The one deliberate difference from the HTTP pipeline: retrieval
does not run here. The suite supplies context from its own throwaway index, and
grading generation against context the target re-fetched for itself would
measure something other than what was asked.

What ``grounded`` means here, and why it is set carefully
--------------------------------------------------------
``grounded`` drives the suite's "lying factor" reliability check, a 2x2 of
dataset-answerable against system-answered:

* **False confidence** — the dataset says no candidate answers this query, but
  the system answered anyway.
* **False refusal** — the dataset says it is answerable, but the system declined.

The suite notes that a generator always reporting ``grounded=True`` can never be
caught fabricating. So it is reported truthfully: ``True`` only when this system
actually believes it produced a supported answer, which means both the model's
own structured ``sufficient`` flag and the post-hoc groundedness check passed.

Two consequences worth stating plainly, because they cut in opposite directions
and are not being quietly optimised for:

* When the grounding check fails, this returns a refusal with
  ``grounded=False`` rather than the extractive-span fallback the web pipeline
  serves. Serving a verbatim snippet that does not answer the question, while
  reporting ``grounded=True``, would be indistinguishable from fabrication by the
  reliability check — and the honest reading is that the system did *not* believe
  it had a supported answer.
* No threshold here was tuned against the reliability metric. Suppressing false
  confidence by refusing more would simply move the error into false refusal,
  and both are reported side by side.

Robustness: this function never raises. ``eval/pipeline.py`` records an exception
as a lost example, so a transient provider error would silently shrink the sample
instead of being measured. Failures degrade to a refusal with ``grounded=False``
and an explanatory ``model`` label.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.rag import guardrails as G  # noqa: E402
from backend.rag.config import CFG  # noqa: E402
from backend.rag.llm import (  # noqa: E402
    GenerationError, INSUFFICIENT_ANSWER, Source, get_client,
)


# --------------------------------------------------------------------------
# decision logging
# --------------------------------------------------------------------------
# The eval harness only sees `grounded: bool`, which collapses three very
# different outcomes into one: the model judging the context insufficient, the
# answer failing the grounding check, and the provider erroring out. Without
# separating them, a rate-limited run is indistinguishable from genuine
# over-refusal — which is exactly the confusion that produced a false-refusal
# regression here.
#
# Set HHGOA_DECISION_LOG to a path to append one JSON line per decision.
_DECISIONS: Counter = Counter()
_DECISION_LOCK = threading.Lock()


def _record(reason: str, grounded: bool, query: str, ms: float,
            confidence: float = 0.0, groundedness: float = 0.0) -> None:
    with _DECISION_LOCK:
        _DECISIONS[reason] += 1
    path = os.environ.get("HHGOA_DECISION_LOG")
    if not path:
        return
    try:
        line = json.dumps({"reason": reason, "grounded": grounded,
                           "ms": round(ms, 1),
                           "confidence": confidence,
                           "groundedness": round(groundedness, 4),
                           "query": (query or "")[:200]},
                          ensure_ascii=False)
        with _DECISION_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:
        pass  # diagnostics must never break a run


def decision_stats() -> dict:
    """Reason-code tally for this process (diagnostics, not part of the contract)."""
    with _DECISION_LOCK:
        return dict(_DECISIONS)


def reset_decision_stats() -> None:
    with _DECISION_LOCK:
        _DECISIONS.clear()


@dataclass
class Answer:
    """The answer object shape the eval suite reads."""
    text: str
    grounded: bool
    generation_ms: float
    model: str

    # Extra diagnostics, ignored by the suite but useful when inspecting a run.
    reason: str = "ok"
    confidence: float = 0.0
    groundedness: float = 0.0

    def to_dict(self) -> dict:
        return {"text": self.text, "grounded": self.grounded,
                "generation_ms": round(self.generation_ms, 2),
                "model": self.model, "reason": self.reason,
                "confidence": self.confidence,
                "groundedness": round(self.groundedness, 4)}


def _sources_from(results: list) -> list[Source]:
    """Adapt the suite's duck-typed context objects to our Source dataclass.

    Only ``.text`` and ``.source`` are guaranteed, so nothing else is assumed.
    ``source`` becomes the chunk id, which keeps citation validation meaningful.
    """
    out: list[Source] = []
    for r in results or []:
        text = (getattr(r, "text", "") or "").strip()
        if not text:
            continue
        out.append(Source(chunk_id=str(getattr(r, "source", "") or f"ctx_{len(out)}"),
                          document_id=None, score=getattr(r, "score", None),
                          text=text, lang=None))
    return out


def generate_answer(query: str, results: list) -> Answer:
    """Answer ``query`` using only the supplied context, with guardrails applied."""
    t0 = time.perf_counter()
    model_label = f"{CFG.llm_provider}/{CFG.llm_model}"

    def elapsed() -> float:
        return (time.perf_counter() - t0) * 1000

    def done(a: Answer) -> Answer:
        _record(a.reason, a.grounded, query, a.generation_ms,
                a.confidence, a.groundedness)
        return a

    # --- input layer -----------------------------------------------------
    v_input = G.check_input(query or "")
    if v_input.blocked:
        return done(Answer(text=v_input.message, grounded=False,
                           generation_ms=elapsed(), model=model_label,
                           reason=v_input.reason))

    sources = _sources_from(results)
    if not sources:
        # No context at all: declining is the only defensible response.
        return done(Answer(text=INSUFFICIENT_ANSWER, grounded=False,
                           generation_ms=elapsed(), model=model_label,
                           reason="no_context"))

    # Only `generation_k` chunks are sent, matching production: fewer prefill
    # tokens means lower time-to-first-token, and the tail of the list rarely
    # adds evidence the top few do not already carry.
    gen_sources = sources[:CFG.generation_k]

    # --- generation ------------------------------------------------------
    try:
        client = get_client()
    except GenerationError as e:
        return done(Answer(text=INSUFFICIENT_ANSWER, grounded=False,
                           generation_ms=elapsed(),
                           model=f"{model_label} (unconfigured)",
                           reason=f"llm_unavailable: {str(e)[:80]}"))

    try:
        answer_obj, meta = client.complete(query, gen_sources)
    except GenerationError as e:
        # Never raise: a lost example would shrink the sample rather than be
        # measured. Report the failure as a non-grounded response instead.
        return done(Answer(text=INSUFFICIENT_ANSWER, grounded=False,
                           generation_ms=elapsed(),
                           model=f"{model_label} (error)",
                           reason=f"generation_failed: {str(e)[:80]}"))

    gen_ms = meta.get("ms", elapsed())

    # --- the model's own report ------------------------------------------
    v_gen = G.check_generation(answer_obj)
    if v_gen.blocked:
        # The model has seen the actual passages and reports they do not answer
        # the question. That judgement is more reliable than any retrieval-score
        # threshold (see guardrails.py for the measurement behind that claim).
        return done(Answer(text=answer_obj.answer or INSUFFICIENT_ANSWER,
                           grounded=False, generation_ms=gen_ms,
                           model=model_label, reason=v_gen.reason,
                           confidence=answer_obj.confidence))

    # --- post-hoc grounding + citation validity --------------------------
    v_ans = G.check_answer(answer_obj.answer, gen_sources, query,
                           cited_ids=answer_obj.used_source_ids or None)
    grounded_score = float(v_ans.signals.get("groundedness", 0.0))

    if v_ans.blocked:
        return done(Answer(text=INSUFFICIENT_ANSWER, grounded=False,
                           generation_ms=gen_ms, model=model_label,
                           reason=v_ans.reason,
                           confidence=answer_obj.confidence,
                           groundedness=grounded_score))

    return done(Answer(text=answer_obj.answer, grounded=True,
                       generation_ms=gen_ms, model=model_label, reason="ok",
                       confidence=answer_obj.confidence,
                       groundedness=grounded_score))


__all__ = ["generate_answer", "Answer", "decision_stats", "reset_decision_stats"]
