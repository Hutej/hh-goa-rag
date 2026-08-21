# Phase 8 — Speech-to-Text (Sarvam)

audio → Sarvam STT → transcribed text → (existing) hybrid retrieval → (existing)
grounded answer generation. The STT layer is a small replaceable provider; it
does not duplicate retrieval or generation.

## Sarvam SDK / API used

Inspected from the installed **`sarvamai` 0.1.30** and used as-is (not guessed):

```python
from sarvamai import SarvamAI
client = SarvamAI(api_subscription_key=KEY)
resp = client.speech_to_text.transcribe(
    file=<path|bytes>,
    model="saaras:v4",
    mode="transcribe",
    language_code="hi-IN")
# resp.transcript (str), resp.language_code (str|None)
```

Response type `SpeechToTextResponse` fields: `transcript`, `language_code`,
`language_probability`, `timestamps`, `request_id`.

## Provider interface

`STTProvider.transcribe(audio_path) -> Transcription(text, language, provider, latency_ms)`.
Implementations: `SarvamSTTProvider` (live), `EchoSTTProvider` (offline mock).
Replaceable with ElevenLabs/etc. later by adding another subclass.

## Configuration (env, never hardcoded)

| env var | meaning |
|---|---|
| `SARAVAM_API_KEY` | Sarvam API subscription key (note the spelling, matches project `.env`) |
| `STT_PROVIDER` | `sarvam` (default) or `echo` (mock) |
| `STT_LANGUAGE` | language_code, default `hi-IN` |

A `.env` at the project root is **auto-loaded** (bare `KEY=VALUE` format) so the
CLI works without manual `export` — no `python-dotenv` dependency added. Real
environment variables take precedence over `.env`.

## CLI

```bash
venv/bin/python scripts/transcribe.py --audio path/to/audio.wav
venv/bin/python scripts/transcribe.py --audio audio.wav --language hi-IN
```

Output JSON:
```json
{"text": "...", "language": "hi-IN", "provider": "sarvam", "latency_ms": ...}
```

## Integration with the existing pipeline

The transcribed text feeds directly into the existing scripts (no duplication):

```bash
# 1. transcribe
venv/bin/python scripts/transcribe.py --audio q.wav > /tmp/t.json
# 2. retrieve (use the transcribed text as --query)
venv/bin/python scripts/retrieve_hybrid.py --query "$(jq -r .text /tmp/t.json)" --top-k 5
# 3. grounded answer
venv/bin/python scripts/answer_query.py --query "$(jq -r .text /tmp/t.json)"
```

## Latency

`transcribe()` reports `latency_ms` (the Sarvam round-trip). This is **not** a
claim against the official `<200 ms` end-to-end target. The final benchmark
(STT + retrieval + generation) comes after the demo pipeline exists.

## Files

- `backend/rag/stt.py` — provider interface, Sarvam + Echo providers, `load_env`
- `scripts/transcribe.py` — CLI
- `tests/test_stt.py` — mock-based tests (no live API)
