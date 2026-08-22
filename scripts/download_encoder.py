"""Download the ONNX query/corpus encoder — no torch required.

Why ONNX rather than sentence-transformers + torch:

* The int8 graph is ~118 MB against ~2.5 GB for a CUDA torch install, and it
  runs the same on the dev box and the deployed Space.
* One runtime serves both jobs. The corpus pass and the query pass use the
  *same* int8 weights, so quantization error cancels in the dot product instead
  of introducing query/document asymmetry.
* onnxruntime on CPU makes the deployed image ~500 MB instead of ~4 GB and
  removes a multi-GB cold start.

Model: ``Xenova/multilingual-e5-small`` — a pre-exported ONNX port of
``intfloat/multilingual-e5-small`` (384-dim, 12 layers, 100+ languages
including Hindi, Marathi and English). Mean pooling + L2 normalization are
applied by ``backend/rag/encoder.py``, not baked into the graph.

Usage:
    python scripts/download_encoder.py                 # int8 (default)
    python scripts/download_encoder.py --precision fp32
    python scripts/download_encoder.py --precision fp32 --precision-also int8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST = PROJECT_ROOT / "data" / "models" / "multilingual-e5-small"

REPO_ID = "Xenova/multilingual-e5-small"

# ONNX graph variants published by the repo, with real on-Hub sizes.
PRECISIONS = {
    "int8": ("onnx/model_int8.onnx", 118_054_593),
    "fp16": ("onnx/model_fp16.onnx", 235_336_732),
    "fp32": ("onnx/model.onnx", 470_268_533),
}

# Tokenizer + config. `tokenizers` loads tokenizer.json directly, so neither
# `transformers` nor torch is needed at any point.
SUPPORT_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
]


def fetch(rel: str) -> Path:
    local = DEST / rel
    if local.exists():
        mb = local.stat().st_size / (1024 ** 2)
        print(f"SKIP {rel} — already present ({mb:.1f} MB)")
        return local
    print(f"Downloading {rel} ...")
    path = hf_hub_download(repo_id=REPO_ID, filename=rel,
                           local_dir=str(DEST))
    mb = Path(path).stat().st_size / (1024 ** 2)
    print(f"  -> {path}  ({mb:.1f} MB)")
    return Path(path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", default="int8", choices=sorted(PRECISIONS),
                    help="ONNX graph precision to fetch (default: int8)")
    ap.add_argument("--precision-also", action="append", default=[],
                    choices=sorted(PRECISIONS),
                    help="fetch an additional precision (repeatable), e.g. to "
                         "compare int8 recall against fp32")
    args = ap.parse_args()

    wanted = [args.precision] + [p for p in args.precision_also
                                 if p != args.precision]

    DEST.mkdir(parents=True, exist_ok=True)
    total = sum(PRECISIONS[p][1] for p in wanted) / (1024 ** 2)

    print("=" * 70)
    print("ONNX ENCODER DOWNLOAD")
    print("=" * 70)
    print(f"Repo:        {REPO_ID}")
    print(f"Destination: {DEST}")
    print(f"Precisions:  {', '.join(wanted)}")
    print(f"Approx size: {total:.0f} MB (+ ~17 MB tokenizer)")
    print()

    for rel in SUPPORT_FILES:
        fetch(rel)
    print()
    for p in wanted:
        fetch(PRECISIONS[p][0])

    print()
    print("=" * 70)
    missing = [rel for rel in SUPPORT_FILES if not (DEST / rel).exists()]
    for p in wanted:
        rel = PRECISIONS[p][0]
        if not (DEST / rel).exists():
            missing.append(rel)
    if missing:
        print("INCOMPLETE — missing:")
        for m in missing:
            print(f"  {m}")
        return 1
    print("ENCODER READY")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
