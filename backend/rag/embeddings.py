"""BGE-M3 embedder — load once per process, encode in batches.

Phase 3A. Generates dense embeddings for chunk text using BAAI/bge-m3 via
``sentence-transformers`` (already in the venv; production-quality, handles
pooling + normalization). Designed to run on **CPU** or **NVIDIA CUDA**
with automatic device selection, so the same code runs on a GTX 1650, an
RTX 4050, or a CPU-only box with no source edits.

Key facts (verified from the actual loaded model — not assumed):
* model: BAAI/bge-m3 (XLM-RoBERTa-based, multilingual — fits the Hindi corpus)
* embedding dimension: 1024 (verified)
* max_seq_length: 8192
* dtype: float16 on CUDA (fits in 4 GB VRAM, ~1.1 GB resident), float32 on CPU
* normalization: ``normalize_embeddings=True`` is the SINGLE place vectors are
  L2-normalized — output vectors have unit L2 norm, ready for cosine similarity.
  Do NOT normalize again downstream.

The model is loaded ONCE per process and reused for every chunk (the tokenizer
is part of the model; the Phase 2 standalone tokenizer module is not used here).
"""

from __future__ import annotations

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024  # verified from actual encode; asserted at load time

# default dtype per device
DTYPE_DEFAULT = {"cuda": "float16", "cpu": "float32"}


def select_device(override: str | None = None) -> str:
    """Return the device to use. ``override`` wins; else CUDA if available."""
    if override:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_dtype(dtype: str):
    return {"float16": torch.float16, "float32": torch.float32,
            "bfloat16": torch.bfloat16}[dtype]


def load_embedder(device: str | None = None, dtype: str | None = None):
    """Load BGE-M3 once. Returns (model, device, dtype_str).

    * device: 'cuda' if available (or override), else 'cpu'.
    * dtype: defaults to float16 on CUDA, float32 on CPU (or override).
    * Verifies the actual embedding dimension == 1024 from a probe encode.
    """
    device = select_device(device)
    if dtype is None:
        dtype = DTYPE_DEFAULT.get(device, "float32")
    model_kwargs = {}
    if device == "cuda" and dtype in ("float16", "bfloat16"):
        model_kwargs["torch_dtype"] = _torch_dtype(dtype)
    model = SentenceTransformer(MODEL_NAME, device=device, model_kwargs=model_kwargs)
    # verify dim from a real encode (not the config)
    probe = model.encode(["probe"], normalize_embeddings=True,
                         convert_to_numpy=True, batch_size=1)
    if probe.shape[1] != EMBED_DIM:
        raise RuntimeError(
            f"BGE-M3 embedding dim mismatch: expected {EMBED_DIM}, "
            f"got {probe.shape[1]}")
    return model, device, dtype


def encode_batch(model, texts: list[str], batch_size: int,
                 normalize: bool = True) -> np.ndarray:
    """Encode a list of texts to a (len(texts), 1024) float32 numpy array.

    ``sentence-transformers`` handles batching, padding, and pooling. With
    ``normalize_embeddings=True`` the rows have unit L2 norm (single
    normalization — do not re-normalize downstream). Output is float32 numpy
    regardless of the model's internal dtype, so the on-disk format and later
    dot-product math are dtype-stable across CPU/GPU runs.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    emb = model.encode(texts, batch_size=batch_size,
                       normalize_embeddings=normalize,
                       convert_to_numpy=True,
                       show_progress_bar=False)
    # sentence-transformers returns float32 numpy already when
    # convert_to_numpy=True (even if the model ran in fp16). Ensure dtype.
    return np.asarray(emb, dtype=np.float32)


def gpu_info(device: str) -> dict:
    """Return GPU info for reporting (empty dict on CPU)."""
    info = {"device": device}
    if device == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["capability"] = ".".join(str(c) for c in
                                      torch.cuda.get_device_capability(0))
    return info


__all__ = ["MODEL_NAME", "EMBED_DIM", "select_device", "load_embedder",
           "encode_batch", "gpu_info"]
