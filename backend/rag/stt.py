"""Speech-to-text (STT) via Sarvam (Phase 8).

Minimal replaceable STT layer. audio -> Sarvam STT -> transcribed text, which
then feeds the existing hybrid retrieval + grounded answer generation.

Provider interface (replaceable with ElevenLabs/etc. later):
    transcribe(audio_path) -> {"text", "language", "provider", "latency_ms"}

Sarvam SDK used (inspected from the installed sarvamai 0.1.30):
    client = SarvamAI(api_subscription_key=KEY)
    resp = client.speech_to_text.transcribe(
        file=<path|bytes>, model="saaras:v4", mode="transcribe",
        language_code="hi-IN")
    # resp.transcript (str), resp.language_code (str|None)

Configuration from environment (NEVER hardcode keys):
    SARAVAM_API_KEY   (note the spelling, matching the project .env)
    STT_LANGUAGE     default language_code, default "hi-IN"

A ``.env`` at the project root is auto-loaded (bare KEY=VALUE format, no quotes)
so the CLI works without manual ``export`` — no python-dotenv dependency added.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_LANGUAGE = "hi-IN"


def load_env(env_path: Path | None = None) -> None:
    """Load bare KEY=VALUE lines from .env into os.environ (idempotent).

    Only sets vars not already present in the real environment (real env wins).
    Skips blank/comment lines and strips surrounding quotes.
    """
    p = env_path or (PROJECT_ROOT / ".env")
    if not p.exists():
        return
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            if k and k not in os.environ:
                os.environ[k] = v


class STTError(Exception):
    """Controlled error for missing config or provider/API failure."""


@dataclass
class Transcription:
    text: str
    language: str | None
    provider: str
    latency_ms: float

    def to_dict(self) -> dict:
        return {"text": self.text, "language": self.language,
                "provider": self.provider, "latency_ms": round(self.latency_ms, 1)}


class STTProvider:
    name = "base"

    def transcribe(self, audio_path: str) -> Transcription:
        raise NotImplementedError


class EchoSTTProvider(STTProvider):
    """Offline mock for tests. Derives text from the filename (no network)."""

    name = "echo"

    def transcribe(self, audio_path: str) -> Transcription:
        t0 = time.time()
        if not Path(audio_path).exists():
            raise STTError(f"audio file not found: {audio_path}")
        stem = Path(audio_path).stem
        text = stem.replace("_", " ")
        return Transcription(text=text, language="hi-IN", provider=self.name,
                             latency_ms=(time.time() - t0) * 1000)


class SarvamSTTProvider(STTProvider):
    """Sarvam speech-to-text using the installed sarvamai SDK (real interface)."""

    name = "sarvam"

    def __init__(self, api_key: str | None = None, language: str = DEFAULT_LANGUAGE):
        load_env()
        self.api_key = api_key or os.environ.get("SARAVAM_API_KEY")
        if not self.api_key:
            raise STTError("SARAVAM_API_KEY not set (configure .env or export it)")
        self.language = language
        from sarvamai import SarvamAI
        self.client = SarvamAI(api_subscription_key=self.api_key)

    def transcribe(self, audio_path: str) -> Transcription:
        p = Path(audio_path)
        if not p.exists():
            raise STTError(f"audio file not found: {audio_path}")
        # Send raw bytes (the path-string upload form is rejected server-side
        # with HTTP 400 "Failed to read the file"; bytes with an explicit
        # input_audio_codec works). Infer codec from extension.
        codec = "wav" if p.suffix.lower() in (".wav", ".wave") else "mp3"
        data = p.read_bytes()
        t0 = time.time()
        try:
            resp = self.client.speech_to_text.transcribe(
                file=data,
                model="saaras:v4",
                mode="transcribe",
                language_code=self.language,
                input_audio_codec=codec,
            )
        except STTError:
            raise
        except Exception as e:
            raise STTError(f"sarvam transcription failed: {e}") from e
        latency_ms = (time.time() - t0) * 1000
        text = (resp.transcript or "").strip()
        lang = getattr(resp, "language_code", None) or self.language
        return Transcription(text=text, language=lang, provider=self.name,
                             latency_ms=latency_ms)


def get_provider(provider: str | None = None, language: str = DEFAULT_LANGUAGE) -> STTProvider:
    """Build a provider. ``provider`` from env STT_PROVIDER (default 'sarvam')
    or 'echo' for tests."""
    load_env()
    name = (provider or os.environ.get("STT_PROVIDER") or "sarvam").lower()
    if name == "echo":
        return EchoSTTProvider()
    return SarvamSTTProvider(language=language)


def transcribe(audio_path: str, provider: STTProvider | None = None,
               language: str = DEFAULT_LANGUAGE) -> Transcription:
    prov = provider or get_provider(language=language)
    return prov.transcribe(audio_path)


__all__ = ["load_env", "Transcription", "STTError", "STTProvider",
           "EchoSTTProvider", "SarvamSTTProvider", "get_provider", "transcribe",
           "DEFAULT_LANGUAGE"]
