"""FastAPI backend for the voice-enabled RAG demo.

Endpoints
---------
    GET  /api/health          service state, serving config, guardrail config,
                              live index statistics
    POST /api/transcribe      audio -> transcript (Sarvam)
    POST /api/query           text and/or audio -> grounded answer + sources
                              + per-stage timings + guardrail audit trail
    POST /api/query/stream    same, as Server-Sent Events: the extractive answer
                              is emitted immediately, then generated tokens

Concurrency note
----------------
The handlers are declared ``def``, not ``async def``, on purpose. The pipeline is
synchronous CPU work (ONNX inference, FAISS search, BM25 scoring) plus blocking
network calls. Declaring them ``async`` — as the previous version did — ran that
work directly on the event loop, so a single in-flight query blocked every other
request including health checks. FastAPI dispatches sync handlers to a thread
pool, which is the correct home for this workload.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.rag import guardrails as G
from backend.rag.config import CFG
from backend.rag.pipeline import (
    Components, MODE_EXTRACTIVE, PipelineError, describe, run_pipeline,
    transcribe_only, warmup,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("hhgoa.api")

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "frontend" / "static"

# Audio container extensions the browser or a client might send. The suffix is
# preserved because Sarvam needs an accurate codec hint (see stt.py).
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".webm", ".ogg", ".oga", ".aac",
                    ".flac", ".opus")

_WARMUP: dict = {"state": "pending"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load indexes and pre-establish the LLM connection before serving.

    Failure is non-fatal: the app still starts so ``/api/health`` can report
    what went wrong, rather than the container dying silently on boot.
    """
    global _WARMUP
    try:
        t0 = time.perf_counter()
        _WARMUP = {"state": "ok", **warmup()}
        log.info("warmup finished in %.1fs", time.perf_counter() - t0)
    except Exception as e:
        _WARMUP = {"state": "failed", "error": str(e)[:300]}
        log.exception("warmup failed; will retry lazily on first request")
    yield


app = FastAPI(title="Voice-Enabled Multilingual RAG", version="2.0.0",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # public read-only demo; no auth, no user data stored
    allow_methods=["*"],
    allow_headers=["*"],
)


def _suffix(name: str | None) -> str:
    """Container extension for an upload, defaulting to ``.webm``.

    The browser records WebM/Opus, so that is the honest default. The previous
    version defaulted to ``.wav``, which made ``stt.py`` mislabel every
    microphone recording's codec.
    """
    if not name:
        return ".webm"
    lowered = name.lower()
    for ext in AUDIO_EXTENSIONS:
        if lowered.endswith(ext):
            return ext
    return ".webm"


def _save_upload(upload: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(suffix=_suffix(upload.filename),
                                     delete=False) as f:
        f.write(upload.file.read())
        return f.name


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """Service state plus the full serving configuration.

    Deliberately verbose: it doubles as the reproducibility record for a live
    deployment, so anyone can see exactly which encoder, index parameters,
    fusion weights and guardrail thresholds produced a given answer. Contains no
    secrets — key presence is reported as a boolean.
    """
    ready = Components.ready()
    stt_ready = Components.stt() is not None

    stats = None
    if ready:
        try:
            stats = Components.retriever().stats()
        except Exception as e:  # pragma: no cover - defensive
            stats = {"error": str(e)[:200]}

    llm = Components.llm()
    return {
        "status": "ok" if ready else "starting",
        "ready": ready,
        "warmup": _WARMUP,
        "stt_available": stt_ready,
        "llm_available": llm is not None,
        "llm": llm.describe() if llm is not None else None,
        "indexes": stats,
        **describe(),
    }


# --------------------------------------------------------------------------
# transcription only
# --------------------------------------------------------------------------
@app.post("/api/transcribe")
def transcribe(audio: UploadFile = File(...)) -> JSONResponse:
    path = None
    try:
        path = _save_upload(audio)
        return JSONResponse(transcribe_only(path))
    except PipelineError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        log.exception("transcribe failed")
        return JSONResponse({"error": f"Transcription failed: {e}"},
                            status_code=500)
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


# --------------------------------------------------------------------------
# query
# --------------------------------------------------------------------------
@app.post("/api/query")
def query(
    text: str | None = Form(None),
    audio: UploadFile | None = File(None),
    top_k: int = Form(CFG.top_k),
    use_llm: bool = Form(True),
) -> JSONResponse:
    """Full pipeline. Accepts a typed question, a recording, or both.

    ``use_llm=false`` returns the extractive answer only — useful for
    demonstrating the retrieval-core latency without a provider round trip.
    """
    audio_path = None
    try:
        if audio is not None and audio.filename:
            audio_path = _save_upload(audio)
        try:
            result = run_pipeline(audio=audio_path, query=text,
                                  top_k=top_k, use_llm=use_llm)
        except PipelineError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(result.to_dict())
    except Exception as e:
        log.exception("query failed")
        return JSONResponse({"error": f"Request failed: {e}"}, status_code=500)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/query/stream")
def query_stream(
    text: str | None = Form(None),
    audio: UploadFile | None = File(None),
    top_k: int = Form(CFG.top_k),
) -> StreamingResponse:
    """Server-Sent Events version, ordered by how fast each part is available.

    Event sequence:
        ``retrieval``  sources + per-stage timings         (~20 ms)
        ``extractive`` the grounded verbatim answer        (~21 ms)
        ``token``      generated text deltas               (first at TTFT)
        ``done``       final answer, guardrail audit, timings
        ``error``      user-facing failure

    The extractive answer lands roughly 30x sooner than the first generated
    token, which is the entire reason it exists: the user has a grounded answer
    on screen while the model is still being contacted.
    """
    audio_path = None
    if audio is not None and audio.filename:
        audio_path = _save_upload(audio)

    def generate():
        nonlocal audio_path
        t0 = time.perf_counter()
        try:
            from backend.rag.extractive import extract_answer
            from backend.rag.llm import Source
            from backend.rag.pipeline import _public_sources, _transcribe

            timing: dict = {}
            q_text = (text or "").strip()
            language = None
            if not q_text:
                if not audio_path:
                    yield _sse("error", {"error": "No input provided. "
                                                  "Speak a question or type one."})
                    return
                q_text, language = _transcribe(audio_path, timing)
                yield _sse("transcript", {"text": q_text, "language": language,
                                          "stt_ms": timing.get("stt_ms")})

            v_input = G.check_input(q_text)
            if v_input.blocked:
                yield _sse("done", {"answer": v_input.message, "refused": True,
                                    "refusal_reason": v_input.reason,
                                    "answer_mode": "refused",
                                    "guardrails": [v_input.to_dict()],
                                    "sources": [], "timing": timing})
                return

            result = Components.retriever().search(q_text, top_k=top_k,
                                                   pinned_language=language)
            timing.update(result.timing)
            v_retr = G.check_retrieval(result)
            sources = _public_sources(result.hits)
            yield _sse("retrieval", {"sources": sources, "timing": timing,
                                     "script": result.script,
                                     "routed_languages": result.routed_languages,
                                     "confidence": v_retr.signals.get("confidence")})

            if v_retr.blocked:
                yield _sse("done", {"answer": v_retr.message, "refused": True,
                                    "refusal_reason": v_retr.reason,
                                    "answer_mode": "refused",
                                    "guardrails": [v_input.to_dict(),
                                                   v_retr.to_dict()],
                                    "sources": sources, "timing": timing})
                return

            extractive = extract_answer(q_text, result.hits)
            timing["first_answer_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            yield _sse("extractive", {"answer": extractive.answer,
                                      "chunk_ids": extractive.chunk_ids,
                                      "coverage": round(extractive.coverage, 3),
                                      "first_answer_ms": timing["first_answer_ms"]})

            client = Components.llm()
            if client is None:
                yield _sse("done", {"answer": extractive.answer,
                                    "answer_mode": MODE_EXTRACTIVE,
                                    "degraded": True,
                                    "degraded_reason": "llm_not_configured",
                                    "guardrails": [v_input.to_dict(),
                                                   v_retr.to_dict()],
                                    "sources": sources, "timing": timing})
                return

            gen_sources = [Source(chunk_id=h["chunk_id"],
                                  document_id=h.get("document_id"),
                                  score=h.get("rrf_score"),
                                  text=h.get("text") or "", lang=h.get("lang"))
                           for h in result.hits[:CFG.generation_k]]

            ttft: list[float] = []
            pieces: list[str] = []
            try:
                for piece in client.stream(
                        q_text, gen_sources,
                        on_first_token=lambda ms: ttft.append(ms)):
                    pieces.append(piece)
                    yield _sse("token", {"delta": piece})
            except Exception as e:
                log.warning("stream generation failed: %s", e)
                yield _sse("done", {"answer": extractive.answer,
                                    "answer_mode": MODE_EXTRACTIVE,
                                    "degraded": True,
                                    "degraded_reason": "generation_failed",
                                    "guardrails": [v_input.to_dict(),
                                                   v_retr.to_dict()],
                                    "sources": sources, "timing": timing})
                return

            generated = "".join(pieces).strip()
            if ttft:
                timing["ttft_ms"] = round(ttft[0], 1)
            timing["generation_ms"] = round((time.perf_counter() - t0) * 1000
                                            - timing["first_answer_ms"], 1)

            # Streaming cannot use the structured `sufficient` flag, so the
            # groundedness check on the completed text carries that weight here.
            v_ans = G.check_answer(generated, gen_sources, q_text)
            guards = [v_input.to_dict(), v_retr.to_dict(), v_ans.to_dict()]

            if v_ans.blocked and not extractive.is_empty:
                yield _sse("done", {"answer": extractive.answer,
                                    "answer_mode": MODE_EXTRACTIVE,
                                    "degraded": True,
                                    "degraded_reason": v_ans.reason,
                                    "guardrails": guards, "sources": sources,
                                    "timing": timing})
                return

            timing["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            yield _sse("done", {"answer": generated, "answer_mode": "generated",
                                "refused": False, "degraded": False,
                                "guardrails": guards, "sources": sources,
                                "extractive_answer": extractive.answer,
                                "timing": timing})
        except PipelineError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:
            log.exception("stream failed")
            yield _sse("error", {"error": f"Request failed: {e}"})
        finally:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------
# static frontend
# --------------------------------------------------------------------------
if STATIC_DIR.exists():
    if (STATIC_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"),
                  name="assets")

    @app.get("/")
    def _index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/{full_path:path}")
    def _spa(full_path: str) -> FileResponse:
        candidate = STATIC_DIR / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=7860,
                workers=1, log_level="info")
