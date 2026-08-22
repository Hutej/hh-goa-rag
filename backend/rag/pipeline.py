"""The harness: staged orchestration of the voice RAG pipeline.

This is deliberately not a prompt-in / text-out function. Each stage is
explicit, individually timed, individually failable, and wrapped in a guardrail
decision, so the system's behaviour is inspectable rather than emergent:

    audio ─▶ STT ─▶ [input guard] ─▶ retrieval ─▶ [retrieval guard]
                                          │
                                          ├─▶ extractive answer  (~0.3 ms, always)
                                          │
                                          └─▶ LLM ─▶ [generation guard]
                                                        ─▶ [answer guard]

Design decisions worth stating
------------------------------
**The extractive answer is computed unconditionally**, immediately after
retrieval, before the LLM is contacted. It costs ~0.3 ms and it serves three
distinct purposes: it is the sub-millisecond grounded answer, it is the fallback
when the LLM errors or times out, and it is a reference point for judging
whether the generated answer stayed near the evidence. One component, three
jobs.

**Degradation is layered, not binary.** A failing LLM does not fail the request:
the harness serves the extractive answer and marks the response degraded. An
*ungrounded* LLM answer likewise falls back to extractive rather than refusing,
because the extractive text is grounded by construction — strictly better than
both a hallucination and a refusal. Only a genuine "the corpus cannot answer
this" refuses outright.

**Every response carries its own audit trail**: a trace_id, per-stage timings,
and the full list of guardrail verdicts with the signals each one measured. A
refusal can always be explained.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from backend.rag import guardrails as G
from backend.rag.config import CFG, load_env
from backend.rag.extractive import ExtractiveAnswer, extract_answer
from backend.rag.llm import (
    GenerationError, GroundedAnswer, LLMClient, Source, get_client,
)
from backend.rag.retrieval import (
    HybridRetriever, RetrievalError, get_retriever,
)

log = logging.getLogger("hhgoa.pipeline")

# Answer provenance, surfaced to the caller so a UI can label it honestly.
MODE_GENERATED = "generated"      # LLM answer, passed all guardrails
MODE_EXTRACTIVE = "extractive"    # verbatim span from retrieved context
MODE_REFUSED = "refused"          # system declined to answer


class PipelineError(Exception):
    """User-facing failure (readable message, never leaks keys or stack traces)."""


@dataclass
class PipelineResult:
    trace_id: str
    query: str
    answer: str
    answer_mode: str
    sources: list[dict] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    guardrails: list[dict] = field(default_factory=list)
    language: str | None = None
    script: str = "Unknown"
    routed_languages: list[str] = field(default_factory=list)
    extractive_answer: str = ""
    refused: bool = False
    refusal_reason: str | None = None
    degraded: bool = False
    degraded_reason: str | None = None
    confidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "answer": self.answer,
            "answer_mode": self.answer_mode,
            "sources": self.sources,
            "timing": self.timing,
            "guardrails": self.guardrails,
            "language": self.language,
            "script": self.script,
            "routed_languages": self.routed_languages,
            "extractive_answer": self.extractive_answer,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "confidence": self.confidence,
            # Kept for backwards compatibility with the existing frontend.
            "guardrail_triggered": self.refused,
        }


# --------------------------------------------------------------------------
# component registry (process-wide, loaded once)
# --------------------------------------------------------------------------
class Components:
    """Lazily-constructed singletons. Nothing heavy is built per request."""

    _retriever: HybridRetriever | None = None
    _llm: LLMClient | None = None
    _stt = None
    _stt_tried = False

    @classmethod
    def retriever(cls) -> HybridRetriever:
        if cls._retriever is None:
            cls._retriever = get_retriever()
        return cls._retriever

    @classmethod
    def llm(cls) -> LLMClient | None:
        """The LLM client, or None if it cannot be configured.

        Returning None rather than raising is intentional: without an LLM the
        system still answers extractively, which is a degraded service rather
        than an outage.
        """
        if cls._llm is None:
            try:
                cls._llm = get_client()
            except GenerationError as e:
                log.warning("llm unavailable: %s", e)
                return None
        return cls._llm

    @classmethod
    def stt(cls):
        """The STT provider, or None if no key is configured.

        Cached including the failure, but the failure is recorded once so
        repeated health polls do not retry provider construction on every call.
        """
        if cls._stt is None and not cls._stt_tried:
            cls._stt_tried = True
            try:
                from backend.rag.stt import STTError, get_provider
                cls._stt = get_provider(language=CFG.stt_language)
            except Exception as e:
                log.warning("stt unavailable: %s", e)
                cls._stt = None
        return cls._stt

    @classmethod
    def ready(cls) -> bool:
        return cls._retriever is not None

    @classmethod
    def reset(cls) -> None:
        cls._retriever = None
        cls._llm = None
        cls._stt = None
        cls._stt_tried = False


def warmup() -> dict:
    """Load everything expensive before the first user request.

    Also pings the LLM so DNS, TLS and the HTTP/2 session are established at
    boot instead of inside somebody's query — the single largest source of
    latency variance measured in the original implementation (9,995 ms cold
    against 112 ms warm on identical work).
    """
    load_env()
    report: dict = {}
    t0 = time.perf_counter()

    try:
        from backend.rag.bootstrap import bootstrap_serve_data, serve_data_ready
        if not serve_data_ready():
            bootstrap_serve_data()
    except Exception as e:
        report["bootstrap_error"] = str(e)[:200]

    t = time.perf_counter()
    try:
        Components.retriever()
        report["retriever_ms"] = round((time.perf_counter() - t) * 1000, 1)
    except RetrievalError as e:
        report["retriever_error"] = str(e)[:300]

    client = Components.llm()
    if client is not None:
        report["llm_warmup"] = client.warmup()
    else:
        report["llm_warmup"] = {"warmed": False, "error": "not configured"}

    report["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info("warmup complete: %s", report)
    return report


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _sources_from(hits: list[dict]) -> list[Source]:
    return [Source(chunk_id=h["chunk_id"], document_id=h.get("document_id"),
                   score=h.get("rrf_score"), text=h.get("text") or "",
                   lang=h.get("lang"))
            for h in hits]


def _public_sources(hits: list[dict]) -> list[dict]:
    """Retrieved chunks as returned to the caller.

    ``row`` is stripped: it is an internal index offset and meaningless outside
    the process.
    """
    out = []
    for h in hits:
        d = {k: v for k, v in h.items() if k != "row"}
        out.append(d)
    return out


def _transcribe(audio_path: str, timing: dict) -> tuple[str, str | None]:
    stt = Components.stt()
    if stt is None:
        raise PipelineError(
            "Voice input is unavailable because no Sarvam API key is "
            "configured. Please type your question instead.")
    t = time.perf_counter()
    try:
        tr = stt.transcribe(audio_path)
    except Exception as e:
        raise PipelineError(f"Transcription failed: {e}") from e
    timing["stt_ms"] = round((time.perf_counter() - t) * 1000, 1)
    text = (tr.text or "").strip()
    if not text:
        raise PipelineError(
            "I couldn't hear anything in that recording. "
            "Please try again or type your question.")
    return text, getattr(tr, "language", None)


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------
def run_pipeline(audio: str | None = None, query: str | None = None,
                 top_k: int | None = None, use_llm: bool = True,
                 use_cache: bool = True,
                 trace_id: str | None = None) -> PipelineResult:
    """Run the full pipeline for one request.

    Exactly one input is needed: a non-empty ``query`` (text path, no STT) or an
    ``audio`` path (voice path). Raises :class:`PipelineError` only for failures
    the user must act on; everything else degrades to a served answer.
    """
    trace_id = trace_id or uuid.uuid4().hex[:12]
    top_k = CFG.top_k if top_k is None else int(top_k)
    timing: dict = {}
    verdicts: list[dict] = []
    t_total = time.perf_counter()
    language: str | None = None

    def finish(result: PipelineResult) -> PipelineResult:
        result.timing["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)
        log.info("trace=%s mode=%s refused=%s degraded=%s total_ms=%s",
                 result.trace_id, result.answer_mode, result.refused,
                 result.degraded, result.timing.get("total_ms"))
        return result

    # -- stage 1: obtain query text -------------------------------------
    q_text = (query or "").strip()
    if not q_text:
        if not audio:
            raise PipelineError(
                "No input provided. Speak a question or type one.")
        q_text, language = _transcribe(audio, timing)

    # -- stage 2: input guardrail ---------------------------------------
    t = time.perf_counter()
    v_input = G.check_input(q_text)
    timing["guard_input_ms"] = round((time.perf_counter() - t) * 1000, 2)
    verdicts.append(v_input.to_dict())
    if v_input.blocked:
        return finish(PipelineResult(
            trace_id=trace_id, query=q_text, answer=v_input.message,
            answer_mode=MODE_REFUSED, timing=timing, guardrails=verdicts,
            language=language, refused=True, refusal_reason=v_input.reason))

    # -- stage 3: retrieval ---------------------------------------------
    try:
        result = Components.retriever().search(
            q_text, top_k=top_k, pinned_language=language, use_cache=use_cache)
    except RetrievalError as e:
        raise PipelineError(f"Retrieval is unavailable: {e}") from e
    timing.update(result.timing)

    # -- stage 4: retrieval guardrail -----------------------------------
    t = time.perf_counter()
    v_retr = G.check_retrieval(result)
    timing["guard_retrieval_ms"] = round((time.perf_counter() - t) * 1000, 2)
    verdicts.append(v_retr.to_dict())
    confidence = v_retr.signals.get("confidence", {})
    sources_public = _public_sources(result.hits)

    if v_retr.blocked:
        return finish(PipelineResult(
            trace_id=trace_id, query=q_text, answer=v_retr.message,
            answer_mode=MODE_REFUSED, sources=sources_public, timing=timing,
            guardrails=verdicts, language=language, script=result.script,
            routed_languages=result.routed_languages, refused=True,
            refusal_reason=v_retr.reason, confidence=confidence))

    # -- stage 5: extractive answer (always; fast path + fallback) ------
    t = time.perf_counter()
    extractive: ExtractiveAnswer = extract_answer(q_text, result.hits)
    timing["extractive_ms"] = round((time.perf_counter() - t) * 1000, 2)
    timing["first_answer_ms"] = round((time.perf_counter() - t_total) * 1000, 2)

    def extractive_result(reason: str | None, degraded: bool) -> PipelineResult:
        return PipelineResult(
            trace_id=trace_id, query=q_text, answer=extractive.answer,
            answer_mode=MODE_EXTRACTIVE, sources=sources_public, timing=timing,
            guardrails=verdicts, language=language, script=result.script,
            routed_languages=result.routed_languages,
            extractive_answer=extractive.answer, degraded=degraded,
            degraded_reason=reason, confidence=confidence)

    if not use_llm:
        return finish(extractive_result(None, False))

    client = Components.llm()
    if client is None:
        return finish(extractive_result("llm_not_configured", True))

    # -- stage 6: grounded generation -----------------------------------
    # Only `generation_k` chunks go to the model: fewer prefill tokens means
    # lower time-to-first-token, and the tail of the fused list rarely adds
    # evidence the top few do not already carry.
    gen_sources = _sources_from(result.hits[:CFG.generation_k])
    try:
        answer_obj, gen_meta = client.complete(q_text, gen_sources)
    except GenerationError as e:
        log.warning("trace=%s generation failed, serving extractive: %s",
                    trace_id, e)
        timing["generation_ms"] = 0.0
        return finish(extractive_result("generation_failed", True))

    timing["generation_ms"] = gen_meta.get("ms", 0.0)
    timing["generation_attempts"] = gen_meta.get("attempts", 1)

    # -- stage 7: the model's own report --------------------------------
    t = time.perf_counter()
    v_gen = G.check_generation(answer_obj)
    timing["guard_generation_ms"] = round((time.perf_counter() - t) * 1000, 2)
    verdicts.append(v_gen.to_dict())
    if v_gen.blocked:
        # The model, having seen the passages, says they do not answer the
        # question. That judgement is better than any retrieval threshold, so
        # it is honoured as a refusal.
        return finish(PipelineResult(
            trace_id=trace_id, query=q_text, answer=v_gen.message,
            answer_mode=MODE_REFUSED, sources=sources_public, timing=timing,
            guardrails=verdicts, language=language, script=result.script,
            routed_languages=result.routed_languages,
            extractive_answer=extractive.answer, refused=True,
            refusal_reason=v_gen.reason, confidence=confidence))

    # -- stage 8: post-hoc grounding + citations + language -------------
    t = time.perf_counter()
    v_ans = G.check_answer(answer_obj.answer, gen_sources, q_text,
                           cited_ids=answer_obj.used_source_ids or None)
    timing["guard_answer_ms"] = round((time.perf_counter() - t) * 1000, 2)
    verdicts.append(v_ans.to_dict())

    if v_ans.blocked:
        # The generated answer is not supported by the evidence. Serving the
        # extractive span instead is strictly better than either shipping a
        # hallucination or refusing outright: it is grounded by construction.
        log.warning("trace=%s answer failed grounding (%s), serving extractive",
                    trace_id, v_ans.reason)
        if extractive.is_empty:
            return finish(PipelineResult(
                trace_id=trace_id, query=q_text, answer=v_ans.message,
                answer_mode=MODE_REFUSED, sources=sources_public,
                timing=timing, guardrails=verdicts, language=language,
                script=result.script,
                routed_languages=result.routed_languages, refused=True,
                refusal_reason=v_ans.reason, confidence=confidence))
        return finish(extractive_result(v_ans.reason, True))

    return finish(PipelineResult(
        trace_id=trace_id, query=q_text, answer=answer_obj.answer,
        answer_mode=MODE_GENERATED, sources=sources_public, timing=timing,
        guardrails=verdicts, language=language, script=result.script,
        routed_languages=result.routed_languages,
        extractive_answer=extractive.answer,
        confidence={**confidence,
                    "model_confidence": answer_obj.confidence,
                    "used_source_ids": answer_obj.used_source_ids}))


def transcribe_only(audio_path: str) -> dict:
    """Transcribe audio without retrieval (backs ``/api/transcribe``)."""
    timing: dict = {}
    text, language = _transcribe(audio_path, timing)
    return {"text": text, "language": language,
            "latency_ms": timing.get("stt_ms", 0.0)}


def describe() -> dict:
    """Serving configuration + guardrail summary, free of secrets."""
    return {"config": CFG.describe(), "guardrails": G.describe()}


__all__ = ["run_pipeline", "transcribe_only", "warmup", "describe",
           "PipelineResult", "PipelineError", "Components",
           "MODE_GENERATED", "MODE_EXTRACTIVE", "MODE_REFUSED"]
