"""End-to-end RAG pipeline orchestrator (final demo phase).

Single function ``run_pipeline(audio=None, query=None, top_k=5)`` wires the
existing phases WITHOUT duplicating their logic:

    audio? --> stt.transcribe (Phase 8)        [skipped if query given]
    query  --> hybrid.hybrid_search (Phase 5) --> generation.generate_answer (Phase 7)
            --> answer + sources + timing

Heavy components (BGE-M3, BM25 index, LLM provider) are loaded ONCE and cached
in-process (lazily) so warm requests reuse them — the backend does NOT reload
them per request.

Serving configuration (verified final config):
    strategy=adaptive, rrf_k=60, dense_weight=1.0, bm25_weight=0.25,
    dense_k=20, bm25_k=20, top_k=5.

Guardrail (simple, deterministic):
    If no retrieved hybrid candidate has ``rrf_score >= MIN_RELEVANCE_SCORE``
    (configurable via env ``RAG_MIN_RELEVANCE``, default 0.005), the system does
    NOT generate an answer and instead returns a fixed insufficient-context
    message. The threshold is a retrieval signal (RRF score), not a per-query
    heuristic. Generation always uses retrieved context only (Phase 7 grounding).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.rag.bm25 import BM25Index
from backend.rag.bootstrap import bootstrap_serve_data, serve_data_ready
from backend.rag.embeddings import load_embedder
from backend.rag.generation import (
    GenerationError, Source, generate_answer, get_provider as gen_get_provider,
)
from backend.rag.hybrid import hybrid_search
from backend.rag.stt import STTError, get_provider as stt_get_provider, load_env

# ---- serving config (verified final) ----
STRATEGY = "adaptive"
TOP_K = 5
DENSE_K = 20
BM25_K = 20
RRF_K = 60
DENSE_WEIGHT = 1.0
BM25_WEIGHT = 0.25  # verified best (Exp 2): R@1 0.27 / R@10 0.74 > dense-only
STT_LANGUAGE = "hi-IN"

# Guardrail threshold on RRF score. Default 0.005: with k=60 and a candidate in
# both retrievers at rank 1, score = 2/(60+1) ≈ 0.0328; a candidate in only one
# retriever at rank 20 scores 1/(60+20) = 0.0125; rank 60 in one retriever scores
# 1/120 = 0.0083. So 0.005 admits weak-but-present matches while rejecting an
# empty/noise result (top score < 0.005 means no candidate ranked meaningfully).
MIN_RELEVANCE_SCORE = float(os.environ.get("RAG_MIN_RELEVANCE", "0.005"))

INSUFFICIENT_ANSWER = (
    "I couldn't find enough relevant information in the knowledge base "
    "to answer this question."
)


class PipelineError(Exception):
    """User-facing pipeline error (readable message, no keys/stacktraces)."""


@dataclass
class PipelineResult:
    answer: str
    query: str
    language: str | None
    sources: list[dict]
    timing: dict
    guardrail_triggered: bool = False

    def to_dict(self) -> dict:
        return {"answer": self.answer, "query": self.query,
                "language": self.language, "sources": self.sources,
                "timing": self.timing, "guardrail_triggered": self.guardrail_triggered}


# ---- lazy singleton cache (loaded once per process) ----
class _Components:
    _model = None
    _bm25 = None
    _llm = None
    _stt = None
    _device = None
    _dtype = None

    @classmethod
    def model(cls):
        if cls._model is None:
            load_env()
            cls._model, cls._device, cls._dtype = load_embedder(None, None)
        return cls._model

    @classmethod
    def bm25(cls):
        if cls._bm25 is None:
            cls._bm25 = BM25Index.load(STRATEGY)
        return cls._bm25

    @classmethod
    def llm(cls):
        if cls._llm is None:
            load_env()
            cls._llm = gen_get_provider()
        return cls._llm

    @classmethod
    def stt(cls):
        if cls._stt is None:
            try:
                cls._stt = stt_get_provider(language=STT_LANGUAGE)
            except STTError:
                cls._stt = None  # STT unavailable (no key) -> voice path errors
        return cls._stt

    @classmethod
    def ready(cls) -> bool:
        return cls._model is not None and cls._bm25 is not None


def warmup() -> None:
    """Eagerly load heavy components (call at app startup so the first user
    request is warm). Safe to call multiple times.

    On a fresh deployment (e.g. a HF Space) the serve data is not on disk yet;
    bootstrap it from the configured HF dataset repo first (no-op locally,
    where data already exists). Then load BGE-M3 + BM25.
    """
    if not serve_data_ready():
        bootstrap_serve_data()  # no-op if data present or no HHGOA_DATA_REPO set
    _Components.model()
    _Components.bm25()


def _sources_from(hybrid_results: list[dict]) -> list[Source]:
    return [Source(chunk_id=h["chunk_id"], document_id=h.get("document_id"),
                   score=h.get("rrf_score"), text=h.get("text"))
            for h in hybrid_results]


def _sources_to_dict(sources: list[Source]) -> list[dict]:
    return [s.to_dict() for s in sources]


def run_pipeline(audio: str | None = None, query: str | None = None,
                 top_k: int = TOP_K, strategy: str = STRATEGY) -> PipelineResult:
    """Run the full pipeline. Exactly one of (audio, query) should be meaningful:
    if a non-empty ``query`` is given it is used directly (text path, no STT);
    otherwise ``audio`` is transcribed (voice path).

    Returns a PipelineResult. Raises PipelineError on user-facing failures.
    """
    t_total = time.time()
    timing = {"stt_ms": 0.0, "encode_ms": 0.0, "dense_ms": 0.0,
              "bm25_ms": 0.0, "rrf_ms": 0.0, "generation_ms": 0.0,
              "total_ms": 0.0}
    language = None

    # --- 1. obtain query text (STT if no text given) ---
    q_text = (query or "").strip()
    if not q_text:
        if not audio:
            raise PipelineError("No input provided: provide a text query or audio.")
        stt = _Components.stt()
        if stt is None:
            raise PipelineError(
                "Voice input is unavailable: SARAVAM_API_KEY is not configured. "
                "Please type your question instead.")
        try:
            t = time.time()
            tr = stt.transcribe(audio)
            timing["stt_ms"] = round((time.time() - t) * 1000, 1)
        except STTError as e:
            raise PipelineError(f"Transcription failed: {e}") from e
        q_text = tr.text.strip()
        language = tr.language
        if not q_text:
            raise PipelineError(
                "Transcription returned empty text. Please try again or type your question.")

    # --- 2. hybrid retrieval (reuses Phase 5; best RRF weights baked in) ---
    try:
        r = hybrid_search(
            _Components.model(), q_text, strategy=strategy, top_k=top_k,
            dense_k=DENSE_K, bm25_k=BM25_K, rrf_k=RRF_K,
            dense_weight=DENSE_WEIGHT, bm25_weight=BM25_WEIGHT,
            only_selected=False, bm25_index=_Components.bm25(),
        )
    except Exception as e:
        raise PipelineError(f"Retrieval failed: {e}") from e
    ht = r["timing"]
    timing["encode_ms"] = ht["encode_ms"]
    timing["dense_ms"] = ht["dense_ms"]
    timing["bm25_ms"] = ht["bm25_ms"]
    timing["rrf_ms"] = ht["rrf_ms"]

    hybrid_results = r["hybrid_results"]
    sources = _sources_from(hybrid_results)

    # --- 3. guardrail: refuse if no candidate clears the relevance threshold ---
    best_score = max((h.get("rrf_score") or 0.0) for h in hybrid_results) \
        if hybrid_results else 0.0
    if best_score < MIN_RELEVANCE_SCORE:
        timing["generation_ms"] = 0.0
        timing["total_ms"] = round((time.time() - t_total) * 1000, 1)
        return PipelineResult(
            answer=INSUFFICIENT_ANSWER, query=q_text, language=language,
            sources=_sources_to_dict(sources), timing=timing,
            guardrail_triggered=True)

    # --- 4. grounded generation (reuses Phase 7) ---
    t = time.time()
    try:
        answer = generate_answer(q_text, sources, provider=_Components.llm())
    except GenerationError as e:
        raise PipelineError(f"Answer generation failed: {e}") from e
    timing["generation_ms"] = round((time.time() - t) * 1000, 1)
    timing["total_ms"] = round((time.time() - t_total) * 1000, 1)

    return PipelineResult(answer=answer, query=q_text, language=language,
                          sources=_sources_to_dict(sources), timing=timing,
                          guardrail_triggered=False)


def transcribe_only(audio_path: str) -> dict:
    """Transcribe audio only (for /api/transcribe). Raises PipelineError."""
    stt = _Components.stt()
    if stt is None:
        raise PipelineError(
            "Voice input is unavailable: SARAVAM_API_KEY is not configured.")
    try:
        tr = stt.transcribe(audio_path)
    except STTError as e:
        raise PipelineError(f"Transcription failed: {e}") from e
    return {"text": tr.text, "language": tr.language,
            "provider": tr.provider, "latency_ms": round(tr.latency_ms, 1)}


__all__ = ["run_pipeline", "transcribe_only", "warmup", "PipelineResult",
           "PipelineError", "MIN_RELEVANCE_SCORE", "INSUFFICIENT_ANSWER"]
