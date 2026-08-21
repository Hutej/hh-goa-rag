"""One-time serve-data bootstrap for deployed environments (HF Spaces).

At serve time the app needs ONLY (the 2.4G embeddings.npy is NOT needed — it is
used only for indexing, not retrieval):

    data/processed/qdrant/collection/hhgoa_adaptive/   (Qdrant index, ~373M)
    data/processed/bm25/adaptive/bm25.pkl              (~151M)
    data/processed/bm25/adaptive/metadata.parquet       (~57M)
    data/processed/chunks/adaptive.parquet             (~62M, BM25 rebuild fallback)

Locally these files already exist on disk. On a fresh deployment (e.g. a HF
Space's ephemeral/persistent volume) they do not, so this module downloads them
ONCE from a Hugging Face dataset repo into data/processed/, then the app runs
with unchanged PROJECT_ROOT-relative paths.

This is intentionally simple and idempotent: it only acts if the Qdrant index is
absent, and it never re-downloads files that already exist. It is a no-op in
local dev (data already present).

Config (env):
    HHGOA_DATA_REPO   HF dataset repo id holding the serve data, e.g.
                      "hutej/hh-goa-rag-data". If unset, bootstrap is skipped
                      (local dev, or data placed manually on the volume).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

# The presence of the Qdrant collection dir signals "serve data is ready".
SERVE_READY_MARKER = PROCESSED / "qdrant" / "collection" / "hhgoa_adaptive"

# Files that make up the serve-data set (relative to data/processed/). These are
# downloaded from the dataset repo on a fresh deployment.
SERVE_FILES = [
    "chunks/adaptive.parquet",
    "bm25/adaptive/bm25.pkl",
    "bm25/adaptive/metadata.parquet",
]
# The Qdrant collection is a directory of many small files; download the whole
# qdrant/ subtree from the repo instead of enumerating files here.
SERVE_DIRS = ["qdrant"]


def serve_data_ready() -> bool:
    """True if the Qdrant collection exists locally (serve data is in place)."""
    return SERVE_READY_MARKER.exists()


def bootstrap_serve_data(repo_id: str | None = None,
                        cache_dir: Path | str | None = None) -> bool:
    """Download serve data from a HF dataset repo if it is not already present.

    Returns True if data is present after the call (already there, or freshly
    downloaded). Returns False if no repo_id was given and data is missing (so
    the caller can decide whether to proceed or wait). Never raises on missing
    data — a missing repo is a configuration issue surfaced at /api/health, not
    a crash.
    """
    if serve_data_ready():
        return True

    repo_id = repo_id or os.environ.get("HHGOA_DATA_REPO")
    if not repo_id:
        return False  # local dev without data, or data placed manually later

    from huggingface_hub import snapshot_download

    target = Path(cache_dir) if cache_dir else PROCESSED
    target.mkdir(parents=True, exist_ok=True)
    # allow_patterns covers the serve set; qdrant/ is a subtree, bm25/chunks are
    # exact files. snapshot_download lays them out under `target` preserving
    # the repo's relative paths.
    allow = list(SERVE_FILES) + [f"{d}/*" for d in SERVE_DIRS]
    snapshot_download(
        repo_id=repo_id, repo_type="dataset",
        local_dir=str(target), allow_patterns=allow,
        cache_dir=os.environ.get("HF_HOME"),
    )
    return serve_data_ready()


__all__ = ["serve_data_ready", "bootstrap_serve_data", "PROCESSED"]
