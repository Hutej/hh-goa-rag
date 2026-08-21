#!/usr/bin/env python3
"""One-time: push serve data to a Hugging Face dataset repo for deployment.

The retrieval serve data (Qdrant index + BM25 pickle/metadata + chunk text,
~643M) is gitignored (too large for the code repo). On a fresh deployment (e.g.
a HF Space) it is downloaded at startup from a HF dataset repo by
``backend/rag/bootstrap.py``. This script uploads that serve data ONCE, from
your local machine where the data already exists, into a dataset repo you own.

Usage:
    venv/bin/python scripts/push_serve_data.py --repo hutej/hh-goa-rag-data

Requires ``HF_TOKEN`` in the environment (write access to the target repo).
Run once after the data is built; re-run to refresh (it overwrites by path).
Only the serve set is uploaded — embeddings.npy (2.4G) is NOT included.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# Exact serve-data paths (mirrors backend/rag/bootstrap.py SERVE_FILES/SERVE_DIRS).
SERVE_PATHS = [
    PROCESSED / "chunks" / "adaptive.parquet",
    PROCESSED / "bm25" / "adaptive" / "bm25.pkl",
    PROCESSED / "bm25" / "adaptive" / "metadata.parquet",
]
SERVE_DIRS = [PROCESSED / "qdrant"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True,
                    help="HF dataset repo id, e.g. hutej/hh-goa-rag-data")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF write token (default: $HF_TOKEN)")
    args = ap.parse_args()

    if not args.token:
        print("ERROR: set HF_TOKEN (write access to the dataset repo).",
              file=sys.stderr)
        return 2

    missing = [str(p) for p in SERVE_PATHS if not p.exists()]
    missing += [str(d) for d in SERVE_DIRS if not d.exists()]
    if missing:
        print("ERROR: serve data missing locally — build it first:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo, repo_type="dataset",
                    private=False, exist_ok=True)

    total = 0
    for p in SERVE_PATHS:
        rel = p.relative_to(PROCESSED)
        print(f"uploading {rel} ({p.stat().st_size // (1024*1024)}M)...")
        api.upload_file(path_or_fileobj=str(p),
                        path_in_repo=str(rel).replace("\\", "/"),
                        repo_id=args.repo, repo_type="dataset")
        total += p.stat().st_size
    for d in SERVE_DIRS:
        rel = d.relative_to(PROCESSED)
        print(f"uploading {rel}/ (directory)...")
        api.upload_folder(folder_path=str(d),
                          path_in_repo=str(rel).replace("\\", "/"),
                          repo_id=args.repo, repo_type="dataset")
        for f in d.rglob("*"):
            if f.is_file():
                total += f.stat().st_size

    print(f"\nDone. Uploaded ~{total // (1024*1024)}M to dataset repo '{args.repo}'.")
    print(f"On the Space, set secret HHGOA_DATA_REPO={args.repo} so startup")
    print("downloads this data automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
