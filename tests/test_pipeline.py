"""Tests for the RAG pipeline orchestrator + FastAPI app.

Mocks STT/hybrid/generation (no live APIs, no model load). Verifies the wiring,
guardrail, timing, and the API surface.

Run:
    venv/bin/python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.rag import pipeline as P


# --- fixtures / stubs -------------------------------------------------------

class _StubTranscription:
    def __init__(self, text="मैनहट्टन परियोजना क्या थी?", language="hi-IN"):
        self.text = text; self.language = language; self.provider = "stub"
        self.latency_ms = 123.0


def _stub_source(cid="hi_1_p0_c0", score=0.03, text="manhattan project text"):
    return {"chunk_id": cid, "document_id": "hi_1_p0", "rrf_score": score,
            "text": text, "is_selected": 1}


@pytest.fixture
def patched(monkeypatch):
    """Stub heavy components + the reused phase functions."""
    # bypass lazy model/bm25/llm load
    monkeypatch.setattr(P._Components, "model", classmethod(lambda cls: "fake-model"))
    monkeypatch.setattr(P._Components, "bm25", classmethod(lambda cls: "fake-bm25"))

    # stub STT
    class _Stt:
        def transcribe(self, audio): return _StubTranscription()
    monkeypatch.setattr(P._Components, "stt", classmethod(lambda cls: _Stt()))

    # stub LLM provider
    monkeypatch.setattr(P._Components, "llm", classmethod(lambda cls: "fake-llm"))

    return monkeypatch


def _hybrid_return(scores):
    return {"hybrid_results": [_stub_source(score=s) for s in scores],
            "dense_results": [], "bm25_results": [],
            "timing": {"encode_ms": 138.0, "dense_ms": 233.0,
                       "bm25_ms": 378.0, "rrf_ms": 0.1, "total_ms": 749.1},
            "config": {}}


# --- pipeline tests --------------------------------------------------------

class TestPipeline:

    def test_text_query_reaches_retrieval_and_generation(self, patched):
        patched.setattr(P, "hybrid_search",
                        lambda *a, **k: _hybrid_return([0.03, 0.025]))
        calls = {}
        def gen(q, sources, provider=None):
            calls["q"] = q; calls["n"] = len(sources); return "an answer"
        patched.setattr(P, "generate_answer", gen)
        r = P.run_pipeline(query="मैनहट्टन परियोजना क्या थी?", top_k=5)
        assert r.answer == "an answer"
        assert calls["n"] == 2
        assert r.guardrail_triggered is False
        assert r.query == "मैनहट्टन परियोजना क्या थी?"

    def test_audio_query_invokes_stt(self, patched):
        patched.setattr(P, "hybrid_search",
                        lambda *a, **k: _hybrid_return([0.03]))
        patched.setattr(P, "generate_answer", lambda *a, **k: "ans")
        r = P.run_pipeline(audio="/tmp/fake.wav", top_k=5)
        assert r.query == "मैनहट्टन परियोजना क्या थी?"  # from stub STT
        assert r.language == "hi-IN"
        assert "stt_ms" in r.timing  # wall-clock; >=0 (instant stub may be 0)

    def test_answer_contains_sources(self, patched):
        patched.setattr(P, "hybrid_search",
                        lambda *a, **k: _hybrid_return([0.03, 0.02]))
        patched.setattr(P, "generate_answer", lambda *a, **k: "ans")
        r = P.run_pipeline(query="q", top_k=5)
        assert len(r.sources) == 2
        assert all("chunk_id" in s for s in r.sources)
        assert r.sources[0]["chunk_id"] == "hi_1_p0_c0"
        assert "rrf_score" in str(r.sources[0]["score"]) or r.sources[0]["score"] == 0.03

    def test_guardrail_refuses_low_relevance(self, patched):
        # all scores below MIN_RELEVANCE_SCORE (0.005) -> refuse, no generation
        patched.setattr(P, "hybrid_search",
                        lambda *a, **k: _hybrid_return([0.002, 0.001]))
        called = {"g": False}
        def gen(*a, **k):
            called["g"] = True; return "should not be called"
        patched.setattr(P, "generate_answer", gen)
        r = P.run_pipeline(query="nonsense xyz", top_k=5)
        assert r.guardrail_triggered is True
        assert r.answer == P.INSUFFICIENT_ANSWER
        assert called["g"] is False  # generation skipped

    def test_guardrail_passes_when_one_score_clears(self, patched):
        patched.setattr(P, "hybrid_search",
                        lambda *a, **k: _hybrid_return([0.002, 0.03]))
        patched.setattr(P, "generate_answer", lambda *a, **k: "ans")
        r = P.run_pipeline(query="q", top_k=5)
        assert r.guardrail_triggered is False
        assert r.answer == "ans"

    def test_timing_fields_returned(self, patched):
        patched.setattr(P, "hybrid_search",
                        lambda *a, **k: _hybrid_return([0.03]))
        patched.setattr(P, "generate_answer", lambda *a, **k: "ans")
        r = P.run_pipeline(query="q", top_k=5)
        for k in ["encode_ms", "dense_ms", "bm25_ms", "rrf_ms",
                  "generation_ms", "total_ms"]:
            assert k in r.timing
        assert r.timing["encode_ms"] == 138.0
        assert r.timing["generation_ms"] >= 0

    def test_no_input_raises(self, patched):
        with pytest.raises(P.PipelineError, match="No input"):
            P.run_pipeline(audio=None, query=None)

    def test_empty_query_with_no_audio_raises(self, patched):
        with pytest.raises(P.PipelineError, match="No input"):
            P.run_pipeline(query="   ", audio=None)

    def test_empty_transcription_raises(self, patched):
        class _EmptyStt:
            def transcribe(self, audio): return _StubTranscription(text="  ")
        patched.setattr(P._Components, "stt", classmethod(lambda cls: _EmptyStt()))
        with pytest.raises(P.PipelineError, match="empty"):
            P.run_pipeline(audio="/tmp/x.wav")

    def test_stt_unavailable_raises_readable_error(self, patched):
        patched.setattr(P._Components, "stt", classmethod(lambda cls: None))
        with pytest.raises(P.PipelineError, match="SARAVAM_API_KEY"):
            P.run_pipeline(audio="/tmp/x.wav")

    def test_retrieval_failure_propagates_as_pipeline_error(self, patched):
        def boom(*a, **k): raise RuntimeError("qdrant down")
        patched.setattr(P, "hybrid_search", boom)
        with pytest.raises(P.PipelineError, match="Retrieval failed"):
            P.run_pipeline(query="q")

    def test_generation_failure_propagates(self, patched):
        patched.setattr(P, "hybrid_search",
                        lambda *a, **k: _hybrid_return([0.03]))
        from backend.rag.generation import GenerationError
        patched.setattr(P, "generate_answer",
                        lambda *a, **k: (_ for _ in ()).throw(GenerationError("llm down")))
        with pytest.raises(P.PipelineError, match="Answer generation failed"):
            P.run_pipeline(query="q")


# --- FastAPI app tests -----------------------------------------------------

class TestApp:
    def _client(self, patched):
        # avoid real warmup/model load at import
        patched.setattr(P, "warmup", lambda: None)
        from fastapi.testclient import TestClient
        from backend.app import app
        return TestClient(app)

    def test_health(self, patched):
        c = self._client(patched)
        r = c.get("/api/health")
        assert r.status_code == 200
        j = r.json()
        assert j["status"] == "ok"
        assert "ready" in j and "stt_available" in j
        assert j["serving_config"]["bm25_weight"] == 0.25

    def test_query_text_endpoint(self, patched):
        patched.setattr(P, "warmup", lambda: None)
        patched.setattr(P._Components, "model", classmethod(lambda cls: "m"))
        patched.setattr(P._Components, "bm25", classmethod(lambda cls: "b"))
        patched.setattr(P, "hybrid_search",
                        lambda *a, **k: _hybrid_return([0.03]))
        patched.setattr(P, "generate_answer", lambda *a, **k: "ans")
        from fastapi.testclient import TestClient
        from backend.app import app
        c = TestClient(app)
        r = c.post("/api/query", data={"text": "मैनहट्टन परियोजना क्या थी?", "top_k": "5"})
        assert r.status_code == 200
        j = r.json()
        assert j["answer"] == "ans"
        assert len(j["sources"]) == 1
        assert "timing" in j
