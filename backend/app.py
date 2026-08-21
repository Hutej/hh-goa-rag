"""FastAPI backend for the Voice-Enabled RAG demo.

Endpoints:
    GET  /api/health       -> {"status", "ready", "stt_available"}
    POST /api/transcribe   -> {text, language, provider, latency_ms}   (audio file)
    POST /api/query        -> {answer, query, language, sources, timing}  (text and/or audio)

Components load lazily once (via backend.rag.pipeline.warmup / _Components) so
warm requests reuse BGE-M3, BM25 and the LLM provider. Errors return a readable
JSON message (no keys/stacktraces). Serves the static frontend from
frontend/static/ when present.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.rag.pipeline import (
    MIN_RELEVANCE_SCORE, PipelineError, run_pipeline, transcribe_only, warmup,
)

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "frontend" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly warm heavy components so the first user request is not cold on
    # retrieval (BGE-M3 + BM25). LLM/STT load lazily on first use.
    try:
        warmup()
    except Exception as e:
        print(f"[startup] warmup failed (will lazy-load on first request): {e}",
              flush=True)
    yield


app = FastAPI(title="Voice-Enabled RAG", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # demo; permissive for local dev + static hosting
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    from backend.rag.pipeline import _Components
    from backend.rag.bootstrap import serve_data_ready
    ready = _Components.ready()
    stt = _Components.stt()  # None if no SARAVAM_API_KEY
    return {"status": "ok", "ready": ready, "stt_available": stt is not None,
            "serve_data_present": serve_data_ready(),
            "guardrail_min_relevance": MIN_RELEVANCE_SCORE,
            "serving_config": {"strategy": "adaptive", "rrf_k": 60,
                               "dense_weight": 1.0, "bm25_weight": 0.25,
                               "dense_k": 20, "bm25_k": 20, "top_k": 5}}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> JSONResponse:
    try:
        with tempfile.NamedTemporaryFile(suffix=_suffix(audio.filename),
                                          delete=False) as f:
            f.write(await audio.read())
            path = f.name
        try:
            return JSONResponse(transcribe_only(path))
        finally:
            os.unlink(path)
    except PipelineError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Transcription failed: {e}"},
                            status_code=500)


@app.post("/api/query")
async def query(
    text: str | None = Form(None),
    audio: UploadFile | None = File(None),
    top_k: int = Form(5),
) -> JSONResponse:
    # save audio to a temp file if provided
    audio_path = None
    if audio is not None and audio.filename:
        with tempfile.NamedTemporaryFile(suffix=_suffix(audio.filename),
                                          delete=False) as f:
            f.write(await audio.read())
            audio_path = f.name
    try:
        try:
            res = run_pipeline(audio=audio_path, query=text, top_k=top_k)
        except PipelineError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(res.to_dict())
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


def _suffix(name: str | None) -> str:
    if not name:
        return ".wav"
    n = name.lower()
    for ext in (".wav", ".mp3", ".m4a", ".webm", ".ogg", ".aac", ".flac"):
        if n.endswith(ext):
            return ext
    return ".wav"


# serve the static frontend (SPA) if built/present
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"),
              name="assets") if (STATIC_DIR / "assets").exists() else None

    @app.get("/")
    def _index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/{full_path:path}")
    def _spa(full_path: str) -> FileResponse:
        # client-side routing fallback
        f = STATIC_DIR / full_path
        if f.is_file():
            return FileResponse(f)
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=7860,
                workers=1, log_level="info")
