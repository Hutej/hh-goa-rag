"""Tests for Phase 8 STT (Sarvam).

Uses an offline EchoSTTProvider mock — no live Sarvam API calls. The Sarvam
provider's config/error paths are tested by stubbing the sarvamai client so no
network is used.

Run:
    venv/bin/python -m pytest tests/test_stt.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.rag.stt import (
    EchoSTTProvider, SarvamSTTProvider, STTError, Transcription, transcribe,
)


class TestEchoProvider:

    def test_successful_transcription(self, tmp_path):
        audio = tmp_path / "मैनहट्टन परियोजना क्या थी.wav"
        audio.write_bytes(b"fake")
        r = transcribe(str(audio), provider=EchoSTTProvider())
        assert isinstance(r, Transcription)
        assert r.text
        assert r.provider == "echo"

    def test_language_and_provider_fields(self, tmp_path):
        audio = tmp_path / "hello.wav"; audio.write_bytes(b"x")
        r = transcribe(str(audio), provider=EchoSTTProvider())
        assert r.language == "hi-IN"
        assert r.provider == "echo"
        d = r.to_dict()
        assert set(d.keys()) == {"text", "language", "provider", "latency_ms"}

    def test_transcription_passed_through(self, tmp_path):
        audio = tmp_path / "my_query.wav"; audio.write_bytes(b"x")
        r = transcribe(str(audio), provider=EchoSTTProvider())
        assert r.text == "my query"  # underscores -> spaces

    def test_missing_audio_file_raises(self):
        with pytest.raises(STTError, match="not found"):
            transcribe("/nonexistent/audio.wav", provider=EchoSTTProvider())


class TestSarvamProviderConfig:
    """Sarvam provider config + error paths without a live API call."""

    def test_missing_api_key_raises(self, monkeypatch):
        # Both spellings must be absent: the provider accepts SARVAM_API_KEY
        # (correct) and SARAVAM_API_KEY (this project's historic misspelling).
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        monkeypatch.delenv("SARAVAM_API_KEY", raising=False)
        monkeypatch.setattr("backend.rag.stt.load_env", lambda *a, **k: None)
        with pytest.raises(STTError, match="SARVAM_API_KEY"):
            SarvamSTTProvider()

    def test_accepts_correctly_spelled_key(self, monkeypatch):
        monkeypatch.delenv("SARAVAM_API_KEY", raising=False)
        monkeypatch.setenv("SARVAM_API_KEY", "sk-fake-key")
        monkeypatch.setattr("backend.rag.stt.load_env", lambda *a, **k: None)
        assert SarvamSTTProvider().api_key == "sk-fake-key"

    def test_codec_inferred_from_container_extension(self, monkeypatch):
        """The browser records WebM; labelling it mp3 (the previous behaviour)
        sent every microphone recording with the wrong codec hint."""
        monkeypatch.setenv("SARVAM_API_KEY", "sk-fake-key")
        monkeypatch.setattr("backend.rag.stt.load_env", lambda *a, **k: None)
        codecs = SarvamSTTProvider.CODEC_BY_SUFFIX
        assert codecs[".webm"] == "webm"
        assert codecs[".wav"] == "wav"
        assert codecs[".mp3"] == "mp3"
        assert codecs[".ogg"] == "ogg"

    def test_missing_audio_file_raises(self, monkeypatch):
        monkeypatch.setenv("SARAVAM_API_KEY", "sk-fake-key")
        monkeypatch.setattr("backend.rag.stt.load_env", lambda *a, **k: None)
        prov = SarvamSTTProvider()
        with pytest.raises(STTError, match="not found"):
            prov.transcribe("/nonexistent/audio.wav")

    def test_api_failure_raises_controlled_error(self, monkeypatch):
        monkeypatch.setenv("SARAVAM_API_KEY", "sk-fake-key")
        monkeypatch.setattr("backend.rag.stt.load_env", lambda *a, **k: None)
        prov = SarvamSTTProvider()
        # stub the sarvamai client to raise on transcribe
        class BoomSTT:
            class speech_to_text:
                @staticmethod
                def transcribe(**kw):
                    raise RuntimeError("network down / unauthorized")
        prov.client = BoomSTT()
        audio = Path("/tmp/fake.wav")
        # write a real temp file so the file-exists check passes
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake")
            audio = Path(f.name)
        try:
            with pytest.raises(STTError, match="sarvam transcription failed"):
                prov.transcribe(str(audio))
        finally:
            audio.unlink(missing_ok=True)

    def test_successful_transcription_via_stub(self, monkeypatch):
        monkeypatch.setenv("SARAVAM_API_KEY", "sk-fake-key")
        monkeypatch.setattr("backend.rag.stt.load_env", lambda *a, **k: None)
        prov = SarvamSTTProvider()

        class FakeResp:
            transcript = "मैनहट्टन परियोजना क्या थी?"
            language_code = "hi-IN"

        class FakeSTT:
            class speech_to_text:
                @staticmethod
                def transcribe(**kw):
                    assert kw["model"] == "saaras:v4"
                    assert kw["language_code"] == "hi-IN"
                    return FakeResp()
        prov.client = FakeSTT()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake")
            audio = Path(f.name)
        try:
            r = prov.transcribe(str(audio))
            assert r.text == "मैनहट्टन परियोजना क्या थी?"
            assert r.language == "hi-IN"
            assert r.provider == "sarvam"
            assert r.latency_ms >= 0
        finally:
            audio.unlink(missing_ok=True)
