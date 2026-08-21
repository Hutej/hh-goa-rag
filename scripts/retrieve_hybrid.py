"""Phase 5: hybrid retrieval (dense + BM25 via RRF).

Reuses the Phase 3B dense path and the Phase 4 BM25 index, fusing their rankings
with Reciprocal Rank Fusion. Prints dense / BM25 / hybrid results + per-stage
timing as JSON. No latency target claims.

Usage:
    venv/bin/python scripts/retrieve_hybrid.py --query "मैनहट्टन परियोजना" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.rag.bm25 import STRATEGIES  # noqa: E402
from backend.rag.embeddings import load_embedder  # noqa: E402
from backend.rag.hybrid import hybrid_search  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", required=True)
    ap.add_argument("--strategy", default="adaptive", choices=STRATEGIES)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--dense-k", type=int, default=20)
    ap.add_argument("--bm25-k", type=int, default=20)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--bm25-weight", type=float, default=1.0)
    ap.add_argument("--only-selected", action="store_true",
                    help="filter both retrievers to is_selected == 1")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default=None,
                    choices=["float16", "float32", "bfloat16"])
    args = ap.parse_args()

    print(f"loading BGE-M3...", flush=True)
    model, device, dtype = load_embedder(args.device, args.dtype)
    print(f"  loaded on {device} ({dtype})", flush=True)
    print(f"running hybrid search (dense_k={args.dense_k}, "
          f"bm25_k={args.bm25_k}, rrf_k={args.rrf_k}, "
          f"dense_w={args.dense_weight}, bm25_w={args.bm25_weight})...",
          flush=True)

    res = hybrid_search(
        model, args.query, strategy=args.strategy, top_k=args.top_k,
        dense_k=args.dense_k, bm25_k=args.bm25_k, rrf_k=args.rrf_k,
        dense_weight=args.dense_weight, bm25_weight=args.bm25_weight,
        only_selected=args.only_selected,
    )

    out = {
        "query": args.query,
        **res["config"],
        "timing": res["timing"],
        "dense_results": res["dense_results"],
        "bm25_results": res["bm25_results"],
        "hybrid_results": res["hybrid_results"],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
