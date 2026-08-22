"""Serve-artifact bootstrap for deployed environments (HF Spaces).

At serve time the app needs the encoder plus, per language, one dense index and
one sparse index. The 384-dim embedding matrices used to *build* those indexes
(`data/processed/embeddings/`, ~930 MB) are NOT needed and are never downloaded.

    data/models/<encoder>/onnx/model_int8.onnx        ~113 MB
    data/models/<encoder>/tokenizer.json              ~17 MB
    data/processed/dense/<strategy>/<lang>.hnsw       ~360 MB per language
    data/processed/dense/<strategy>/<lang>.meta.parquet
    data/processed/sparse/<strategy>/<lang>/          ~100 MB per language

Locally these already exist. On a fresh deployment they do not, so this module
fetches them once into the same PROJECT_ROOT-relative paths the app already uses,
so no code changes between local and deployed.

Two different sources, deliberately:

* The **encoder** comes straight from its public model repo on the Hub
  (`Xenova/multilingual-e5-small`). There is no reason to mirror a public model
  into a private data repo, and it keeps that repo to just the built indexes.
* The **indexes** are build outputs specific to this project, so they come from
  the dataset repo named by ``HHGOA_DATA_REPO`` (see scripts/push_serve_data.py).

Idempotent and non-fatal: it only fetches what is missing, and a missing or
misconfigured repo is surfaced through ``/api/health`` rather than crashing the
container on boot.

Config (env):
    HHGOA_DATA_REPO   HF dataset repo holding the built indexes, e.g.
                      "you/hh-goa-rag-data". Unset -> index download skipped.
    HF_TOKEN          only needed if that dataset repo is private.
"""

from __future__ import annotations

import os
from pathlib import Path

from backend.rag.config import CFG

PROJECT_ROOT = CFG.root
PROCESSED = CFG.processed_dir

# The ONNX encoder's public source repo, matching scripts/download_encoder.py.
ENCODER_REPO = "Xenova/multilingual-e5-small"
ENCODER_SUPPORT_FILES = [
    "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "config.json",
]


# --------------------------------------------------------------------------
# readiness
# --------------------------------------------------------------------------
def encoder_ready() -> bool:
    """True if the ONNX graph and its tokenizer are on disk."""
    return CFG.encoder_onnx_path().exists() and CFG.tokenizer_path.exists()


def indexes_ready(languages: list[str] | None = None) -> bool:
    """True if every configured language has both a dense and a sparse index."""
    for lang in (languages or CFG.languages):
        if not CFG.dense_index_path(lang).exists():
            return False
        if not CFG.dense_meta_path(lang).exists():
            return False
        if not (CFG.sparse_dir(lang) / "metadata.parquet").exists():
            return False
    return True


def serve_data_ready(languages: list[str] | None = None) -> bool:
    """True if everything needed to answer a query is present locally."""
    return encoder_ready() and indexes_ready(languages)


def missing_report(languages: list[str] | None = None) -> dict:
    """What is missing, for /api/health to report instead of failing opaquely."""
    langs = languages or CFG.languages
    return {
        "encoder": encoder_ready(),
        "languages": {
            lang: {
                "dense": CFG.dense_index_path(lang).exists(),
                "sparse": (CFG.sparse_dir(lang) / "metadata.parquet").exists(),
            }
            for lang in langs
        },
        "data_repo_configured": bool(CFG.data_repo),
    }


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
def bootstrap_encoder() -> bool:
    """Fetch the int8 ONNX encoder + tokenizer from its public model repo."""
    if encoder_ready():
        return True
    from huggingface_hub import hf_hub_download

    dest = CFG.encoder_dir
    dest.mkdir(parents=True, exist_ok=True)
    onnx_rel = f"onnx/{CFG.encoder_onnx_path().name}"
    for rel in ENCODER_SUPPORT_FILES + [onnx_rel]:
        if (dest / rel).exists():
            continue
        print(f"[bootstrap] encoder: {rel}", flush=True)
        hf_hub_download(repo_id=ENCODER_REPO, filename=rel,
                        local_dir=str(dest),
                        token=os.environ.get("HF_TOKEN") or None)
    return encoder_ready()


def bootstrap_indexes(repo_id: str | None = None,
                      languages: list[str] | None = None) -> bool:
    """Fetch the built dense + sparse indexes from the dataset repo."""
    if indexes_ready(languages):
        return True
    repo_id = repo_id or CFG.data_repo
    if not repo_id:
        return False

    from huggingface_hub import snapshot_download

    strategy = CFG.chunk_strategy
    langs = languages or CFG.languages
    # Only this strategy and these languages — the repo may hold more.
    allow: list[str] = []
    for lang in langs:
        allow += [
            f"dense/{strategy}/{lang}.hnsw",
            f"dense/{strategy}/{lang}.meta.parquet",
            f"dense/{strategy}/{lang}.info.json",
            f"sparse/{strategy}/{lang}/*",
        ]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"[bootstrap] indexes from {repo_id}: {', '.join(langs)} "
          f"({strategy})", flush=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset",
                      local_dir=str(PROCESSED), allow_patterns=allow,
                      token=os.environ.get("HF_TOKEN") or None)
    return indexes_ready(langs)


def bootstrap_serve_data(repo_id: str | None = None,
                         languages: list[str] | None = None) -> bool:
    """Ensure the encoder and indexes are present. Returns readiness.

    Never raises: a deployment misconfiguration should be visible at
    ``/api/health``, not a boot loop.
    """
    try:
        bootstrap_encoder()
    except Exception as e:                      # noqa: BLE001
        print(f"[bootstrap] encoder download failed: {e}", flush=True)
    try:
        bootstrap_indexes(repo_id, languages)
    except Exception as e:                      # noqa: BLE001
        print(f"[bootstrap] index download failed: {e}", flush=True)
    return serve_data_ready(languages)


__all__ = ["serve_data_ready", "encoder_ready", "indexes_ready",
           "missing_report", "bootstrap_serve_data", "bootstrap_encoder",
           "bootstrap_indexes", "PROCESSED", "ENCODER_REPO"]
