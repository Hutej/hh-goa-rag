"""Download MSMARCO-XI shards from Hugging Face.

Two things make this much cheaper than it looks:

1. **Each shard carries BOTH languages.** Every row has
   ``passages.English_passages`` *and* ``passages.Translated_passages`` for the
   same document, with a shared ``passages.is_selected`` label. So the Hindi
   shard alone yields two serving languages (Hindi + English) *and* an aligned
   parallel corpus with free cross-lingual ground truth. Only a genuinely new
   language (e.g. Marathi) needs another download.

2. **The validation split is ~8x smaller than train** (~440 MB vs ~3.7 GB) and
   still holds the full MS MARCO dev query set — far more queries than this
   project's corpus needs. `validation` is therefore the default. Use
   ``--split train`` only if you specifically need the train shard.

Usage:
    python scripts/download_dataset.py                          # hin, validation
    python scripts/download_dataset.py --languages hin,mar
    python scripts/download_dataset.py --split train --languages hin
    python scripts/download_dataset.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESTINATION = PROJECT_ROOT / "data" / "raw" / "MSMARCO-XI"

REPO_ID = "ai4bharat/MSMARCO-XI"

# Shard codes published by the dataset (train has 13, validation has 14).
AVAILABLE = {
    "asm": "Assamese", "ben": "Bengali", "guj": "Gujarati", "hin": "Hindi",
    "kan": "Kannada", "mal": "Malayalam", "mar": "Marathi", "nep": "Nepali",
    "ori": "Odia", "pan": "Punjabi", "san": "Sanskrit", "tam": "Tamil",
    "tel": "Telugu", "urd": "Urdu",
}

# `tel` (Telugu) exists only in validation; `asm..urd` all exist in train.
TRAIN_ONLY_MISSING = {"tel"}

SPLITS = {"validation": "val", "train": "train"}

# Approximate on-Hub sizes, for the pre-download summary (bytes).
APPROX_SIZE = {"validation": 465_000_000, "train": 3_750_000_000}

DEFAULT_LANGUAGES = ["hin"]
DEFAULT_SPLIT = "validation"


def shard_path(lang: str, split: str) -> str:
    return f"{split}/{lang}{SPLITS[split]}.parquet"


def download(languages: list[str], split: str) -> int:
    DESTINATION.mkdir(parents=True, exist_ok=True)

    total = APPROX_SIZE[split] * len(languages) / (1024 ** 3)
    print("=" * 70)
    print("MSMARCO-XI DATASET DOWNLOAD")
    print("=" * 70)
    print(f"Destination: {DESTINATION}")
    print(f"Split:       {split}")
    print(f"Languages:   {', '.join(f'{l} ({AVAILABLE[l]})' for l in languages)}")
    print(f"Approx size: {total:.2f} GB")
    print()
    print("Each shard also contains the original English passages, aligned")
    print("row-for-row with the translations, so English needs no download.")
    print()

    if not os.environ.get("HF_TOKEN"):
        print("NOTE: HF_TOKEN is not set. Anonymous downloads are rate limited;")
        print("      setting a token in .env makes this noticeably faster.")
        print()

    for lang in languages:
        rel = shard_path(lang, split)
        local = DESTINATION / rel
        if local.exists():
            gb = local.stat().st_size / (1024 ** 3)
            print(f"SKIP {rel} — already present ({gb:.2f} GB)")
            continue
        print("-" * 70)
        print(f"Downloading: {rel}  ({AVAILABLE[lang]})")
        print("-" * 70)
        path = hf_hub_download(repo_id=REPO_ID, filename=rel,
                               repo_type="dataset", local_dir=str(DESTINATION))
        print(f"Downloaded: {path}")
        print()

    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    missing = 0
    for lang in languages:
        local = DESTINATION / shard_path(lang, split)
        if local.exists():
            gb = local.stat().st_size / (1024 ** 3)
            print(f"{shard_path(lang, split)}: {gb:.2f} GB")
        else:
            print(f"MISSING: {shard_path(lang, split)}")
            missing += 1
    return 1 if missing else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES),
                    help="comma-separated shard codes (default: hin)")
    ap.add_argument("--split", default=DEFAULT_SPLIT, choices=sorted(SPLITS),
                    help="dataset split (default: validation — 8x smaller)")
    ap.add_argument("--list", action="store_true",
                    help="list available language shards and exit")
    args = ap.parse_args()

    if args.list:
        print("Available shards:")
        for code, name in AVAILABLE.items():
            note = "" if code not in TRAIN_ONLY_MISSING else "  (validation only)"
            print(f"  {code}  {name}{note}")
        print("\nSizes: validation ~0.44 GB/shard, train ~3.7 GB/shard.")
        print("English (en) is included in every shard — no separate download.")
        return 0

    langs = [l.strip().lower() for l in args.languages.split(",") if l.strip()]
    bad = [l for l in langs if l not in AVAILABLE]
    if bad:
        print(f"ERROR: unknown language code(s): {', '.join(bad)}", file=sys.stderr)
        print(f"Valid codes: {', '.join(AVAILABLE)}", file=sys.stderr)
        return 2
    if args.split == "train":
        unavailable = [l for l in langs if l in TRAIN_ONLY_MISSING]
        if unavailable:
            print(f"ERROR: {', '.join(unavailable)} exist only in the validation "
                  f"split", file=sys.stderr)
            return 2
    if not langs:
        print("ERROR: no languages given", file=sys.stderr)
        return 2

    return download(langs, args.split)


if __name__ == "__main__":
    raise SystemExit(main())
