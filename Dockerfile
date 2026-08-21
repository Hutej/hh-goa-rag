# HH Goa — Voice-Enabled RAG — Hugging Face Spaces (Docker)
#
# One image runs the same FastAPI app locally and on a Space. The app listens
# on 7860 (HF Spaces' required port). Retrieval data (Qdrant index + BM25 +
# chunk text) lives under data/processed/ so PROJECT_ROOT-relative paths in
# backend/rag/*.py resolve identically in both environments.
#
# Layout on the Space (persistent data goes in /data, mounted as a Space
# persistent volume; we symlink data/ -> /data so code paths are unchanged):
#   /data/processed/qdrant/collection/hhgoa_adaptive/   (Qdrant index)
#   /data/processed/bm25/adaptive/{bm25.pkl,metadata.parquet}
#   /data/processed/chunks/adaptive.parquet
# BGE-M3 is downloaded from the HF Hub at startup (cached under /data/.cache).

FROM python:3.11-slim

# HF Spaces run as uid 1000; create the user and the persistent /data dir.
RUN useradd -m -u 1000 user && mkdir -p /data && chown -R user:user /data

# System deps: build tools for a couple of Python wheels + libgomp for torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer).
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the codebase.
COPY --chown=user:user . .

# Persistent /data volume (HF Spaces persistent storage). Symlink data/ so the
# repo's data/processed/ paths point at the volume without code changes.
RUN mkdir -p /data/processed && chown -R user:user /data \
    && rm -rf /app/data && ln -s /data /app/data

# Model + HF cache on the persistent volume so BGE-M3 is not re-downloaded
# on every restart (when a persistent volume is attached).
ENV HF_HOME=/data/.cache \
    TRANSFORMERS_CACHE=/data/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/data/.cache/sentence-transformers

# Secrets: HF Spaces exposes them as env vars at runtime.
#   SARAVAM_API_KEY  (STT)   OPENAI_API_KEY  (LLM)   LLM_BASE_URL / LLM_MODEL (optional)
# .env is not used on the Space; secrets are injected as env vars by the platform.

ENV PYTHONUNBUFFERED=1

USER user
EXPOSE 7860

# One command for both local and Space: serve the FastAPI app on 7860.
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
