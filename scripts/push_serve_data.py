"""Upload the built serve indexes to a Hugging Face dataset repo for deployment.

The indexes are ~1.4 GB, far too large for a git repo, so they live in a HF
dataset repo and ``backend/rag/bootstrap.py`` fetches them on first boot.

Uploaded (per language, for one chunking strategy):

    dense/<strategy>/<lang>.hnsw            FAISS HNSW index    ~360 MB
    dense/<strategy>/<lang>.meta.parquet    chunk metadata + text
    dense/<strategy>/<lang>.info.json       build parameters
    sparse/<strategy>/<lang>/               bm25s index + metadata  ~100 MB

Deliberately NOT uploaded:

* ``embeddings/`` (~930 MB) — needed only to *build* the dense index, never to
  serve a query.
* the ONNX encoder — it is a public model, so bootstrap pulls it straight from
  ``Xenova/multilingual-e5-small`` rather than mirroring it here.
* ``passages/`` and ``chunks/`` — build inputs; the text needed at serve time is
  already inside the index metadata.

Usage:
    python scripts/push_serve_data.py --repo <you>/hh-goa-rag-data
    python scripts/push_serve_data.py --repo <you>/hh-goa-rag-data --dry-run

Requires ``HF_TOKEN`` with write access. Re-running overwrites by path, so it is
safe to use to refresh a rebuilt index.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag.config import CFG, LANGUAGES  # noqa: E402


def collect(languages: list[str], strategy: str
            ) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[str]]:
    """Return (files, dirs, missing) as (local_path, path_in_repo) pairs."""
    files: list[tuple[Path, str]] = []
    dirs: list[tuple[Path, str]] = []
    missing: list[str] = []

    for lang in languages:
        dense = CFG.dense_index_path(lang, strategy)
        meta = CFG.dense_meta_path(lang, strategy)
        info = dense.parent / f"{lang}.info.json"
        sparse = CFG.sparse_dir(lang, strategy)

        for p in (dense, meta):
            if p.exists():
                files.append((p, f"dense/{strategy}/{p.name}"))
            else:
                missing.append(str(p.relative_to(CFG.root)))
        if info.exists():
            files.append((info, f"dense/{strategy}/{info.name}"))

        if (sparse / "metadata.parquet").exists():
            dirs.append((sparse, f"sparse/{strategy}/{lang}"))
        else:
            missing.append(str(sparse.relative_to(CFG.root)))

    return files, dirs, missing


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True,
                    help="HF dataset repo id, e.g. you/hh-goa-rag-data")
    ap.add_argument("--languages", default=",".join(CFG.languages))
    ap.add_argument("--strategy", default=CFG.chunk_strategy)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--private", action="store_true",
                    help="create the repo private (bootstrap then needs HF_TOKEN)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be uploaded and exit")
    args = ap.parse_args()

    languages = [c.strip().lower() for c in args.languages.split(",") if c.strip()]
    bad = [c for c in languages if c not in LANGUAGES]
    if bad:
        print(f"ERROR: unknown language(s): {', '.join(bad)}", file=sys.stderr)
        return 2

    files, dirs, missing = collect(languages, args.strategy)
    if missing:
        print("ERROR: indexes missing locally — build them first with "
              "scripts/build_indexes.py:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1

    total = sum(size_of(p) for p, _ in files) + sum(size_of(d) for d, _ in dirs)

    print("=" * 70)
    print("PUSH SERVE INDEXES")
    print("=" * 70)
    print(f"Repo       : {args.repo} (dataset)")
    print(f"Strategy   : {args.strategy}")
    print(f"Languages  : {', '.join(languages)}")
    print(f"Total size : {total / (1024 ** 3):.2f} GB")
    print()
    for p, rel in files:
        print(f"  file  {rel:<44} {size_of(p) / (1024 ** 2):>7.0f} MB")
    for d, rel in dirs:
        print(f"  dir   {rel + '/':<44} {size_of(d) / (1024 ** 2):>7.0f} MB")
    print()
    print("NOT uploaded: embeddings/ (build-only), the ONNX encoder (public "
          "model, fetched from its own repo), passages/ and chunks/ (build "
          "inputs).")
    print()

    if args.dry_run:
        print("--dry-run: nothing uploaded.")
        return 0

    if not args.token:
        print("ERROR: set HF_TOKEN (needs write access to the dataset repo), "
              "or pass --token.", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo, repo_type="dataset",
                    private=args.private, exist_ok=True)

    for p, rel in files:
        print(f"uploading {rel} ({size_of(p) / (1024 ** 2):.0f} MB)...", flush=True)
        api.upload_file(path_or_fileobj=str(p), path_in_repo=rel,
                        repo_id=args.repo, repo_type="dataset")
    for d, rel in dirs:
        print(f"uploading {rel}/ ({size_of(d) / (1024 ** 2):.0f} MB)...", flush=True)
        api.upload_folder(folder_path=str(d), path_in_repo=rel,
                          repo_id=args.repo, repo_type="dataset")

    print()
    print("=" * 70)
    print(f"Uploaded {total / (1024 ** 3):.2f} GB to '{args.repo}'.")
    print()
    print("Now set these as Space secrets/variables:")
    print(f"  HHGOA_DATA_REPO = {args.repo}")
    print(f"  RAG_LANGUAGES   = {','.join(languages)}")
    print("  SARVAM_API_KEY  = <your Sarvam key>")
    print("  LLM_API_KEY     = <your provider key>")
    print(f"  LLM_PROVIDER    = {CFG.llm_provider}")
    print(f"  LLM_MODEL       = {CFG.llm_model}")
    if CFG.llm_base_url:
        print(f"  LLM_BASE_URL    = {CFG.llm_base_url}")
    if args.private:
        print("  HF_TOKEN        = <read token>  (repo is private)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
