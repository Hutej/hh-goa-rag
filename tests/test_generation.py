"""Tests for Phase 7 grounded answer generation.

Uses an offline EchoProvider mock — no external API calls.

Run:
    venv/bin/python -m pytest tests/test_generation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.rag.generation import (
    EchoProvider, GenerationError, INSUFFICIENT_ANSWER, Source,
    build_prompt, generate_answer,
)


def _src(i: int, text: str | None = None) -> Source:
    return Source(chunk_id=f"hi_{i}_p0_c0", document_id=f"hi_{i}_p0",
                  score=0.5, text=text if text is not None else f"text-{i}")


class TestPrompt:

    def test_context_included_in_prompt(self):
        sources = [_src(1, "मैनहट्टन परियोजना थी"), _src(2, "atom bomb")]
        sys_p, user = build_prompt("What is it?", sources)
        assert "मैनहट्टन परियोजना थी" in user
        assert "atom bomb" in user
        assert "[Source 1]" in user and "[Source 2]" in user
        assert "chunk_id: hi_1_p0_c0" in user
        assert "What is it?" in user
        assert "ONLY" in sys_p

    def test_chunk_ids_preserved_in_prompt(self):
        sources = [_src(7, "x")]
        _, user = build_prompt("q", sources)
        assert "chunk_id: hi_7_p0_c0" in user


class TestGeneration:

    def test_generated_with_context(self):
        sources = [_src(1, "the answer is 42")]
        ans = generate_answer("q?", sources, provider=EchoProvider())
        assert ans.startswith("[echo]")
        assert "q?" in ans

    def test_empty_context_insufficient(self):
        ans = generate_answer("q?", [], provider=EchoProvider())
        assert ans == INSUFFICIENT_ANSWER
        assert "insufficient" in ans.lower()

    def test_sources_preserved_as_objects(self):
        sources = [_src(3, "txt")]
        ans = generate_answer("q", sources, provider=EchoProvider())
        assert ans  # produced something
        # the sources list itself is untouched + still traceable
        assert [s.chunk_id for s in sources] == ["hi_3_p0_c0"]
        assert sources[0].to_dict()["document_id"] == "hi_3_p0"

    def test_provider_failure_controlled_error(self):
        class BoomProvider:
            name = "boom"
            def generate(self, system, user):
                raise RuntimeError("network down")
        with pytest.raises(GenerationError) as ei:
            generate_answer("q", [_src(1, "x")], provider=BoomProvider())
        assert "boom" in str(ei.value) or "network" in str(ei.value)

    def test_response_structure_via_cli(self, monkeypatch, capsys, tmp_path):
        # drive the CLI main() with the echo provider (no network, no Qdrant)
        # by monkeypatching hybrid_search to return canned hybrid hits.
        import scripts.answer_query as aq
        canned = [{"chunk_id": "hi_1_p0_c0", "document_id": "hi_1_p0",
                   "rrf_score": 0.03, "text": "मैनहट्टन परियोजना द्वितीय विश्व युद्ध..."}]
        monkeypatch.setattr(aq, "hybrid_search", lambda *a, **k: {"hybrid_results": canned})
        monkeypatch.setattr(aq, "load_embedder", lambda *a, **k: (None, "cpu", "float32"))
        monkeypatch.setattr(aq, "get_provider", lambda: EchoProvider())
        monkeypatch.setattr(aq, "generate_answer", lambda q, s, provider=None: "मैनहट्टन परियोजना एक R&D उपक्रम था।")
        import json
        old = sys.argv
        sys.argv = ["answer_query.py", "--query", "मैनहट्टन परियोजना क्या थी?"]
        try:
            aq.main()
        finally:
            sys.argv = old
        out = capsys.readouterr().out
        data = json.loads(out)
        assert set(data.keys()) >= {"query", "answer", "sources",
                                    "retrieval_latency_ms", "generation_latency_ms",
                                    "total_latency_ms", "provider"}
        assert data["answer"]
        assert data["sources"][0]["chunk_id"] == "hi_1_p0_c0"
        assert isinstance(data["retrieval_latency_ms"], (int, float))
        assert isinstance(data["generation_latency_ms"], (int, float))
