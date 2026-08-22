# HH Goa — Voice-Enabled Multilingual RAG — Hugging Face Spaces (Docker)
#
# One image serves the same FastAPI app locally and on a Space, on port 7860.
#
# The image contains NO torch, no transformers and no sentence-transformers.
# The query and corpus encoder is an int8 ONNX graph run by onnxruntime, and the
# tokenizer is loaded by `tokenizers` from tokenizer.json. That takes the image
# from roughly 4 GB to well under 1 GB and removes a multi-GB cold start.
# Build-time-only dependencies (indexing, evaluation) live in
# requirements-build.txt and are deliberately not installed here.
#
# Serving artifacts expected under data/processed/ (PROJECT_ROOT-relative, so
# local and Space paths are identical):
#   data/models/multilingual-e5-small/onnx/model_int8.onnx   (~113 MB)
#   data/models/multilingual-e5-small/tokenizer.json         (~17 MB)
#   data/processed/dense/adaptive/{hi,en,mr}.hnsw            (~1.1 GB total)
#   data/processed/dense/adaptive/{hi,en,mr}.meta.parquet
#   data/processed/sparse/adaptive/{hi,en,mr}/               (~304 MB total)
# On a fresh Space these are fetched by backend/rag/bootstrap.py from the
# dataset repo named in HHGOA_DATA_REPO.

FROM python:3.11-slim

# HF Spaces run as uid 1000; create the user and the persistent /data dir.
RUN useradd -m -u 1000 user && mkdir -p /data && chown -R user:user /data

# libgomp1 is required by faiss-cpu and onnxruntime (OpenMP runtime).
# No build-essential: every serve-time wheel is prebuilt for manylinux.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer caches independently of source changes.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the codebase.
COPY --chown=user:user . .

# Persistent /data volume. Symlinked so data/processed/ paths in the repo point
# at the volume with no code changes.
RUN mkdir -p /data/processed /data/models && chown -R user:user /data \
    && rm -rf /app/data && ln -s /data /app/data

# HF cache on the persistent volume so bootstrap downloads survive a restart.
ENV HF_HOME=/data/.cache \
    HF_HUB_CACHE=/data/.cache/hub

# Thread pinning. A Space CPU is shared and oversubscribing the OpenMP pools in
# onnxruntime and faiss makes tail latency noticeably worse than leaving each
# library to guess the core count independently.
ENV OMP_NUM_THREADS=4 \
    EMBED_THREADS=4

# Secrets are injected by the platform as environment variables; .env is not
# used on a Space. Required: SARVAM_API_KEY (STT), LLM_API_KEY (+ LLM_BASE_URL
# and LLM_MODEL for a non-OpenAI provider). Optional: HHGOA_DATA_REPO, HF_TOKEN.

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

USER user
EXPOSE 7860

# Single worker on purpose: the indexes are held in-process and are ~1.4 GB
# resident, so additional workers would multiply memory rather than throughput.
# Request handlers are sync `def`, so FastAPI dispatches them to a thread pool
# and concurrent requests do not block the event loop.
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--workers", "1", "--timeout-keep-alive", "65"]
