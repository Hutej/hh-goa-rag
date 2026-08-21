"""Phase 4: BM25 lexical retrieval over the Adaptive chunk corpus.

Loads (or builds+persists) the BM25 index once, then queries it. Prints JSON
shaped like retrieve_dense.py (same result fields), with measured latency.
No latency target claims.

Usage:
    venv/bin/python scripts/retrieve_bm25.py --query "मैनहट्टन परियोजना" --top-k 5
    venv/bin/python scripts/retrieve_bm25.py --query "..." --only-selected
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

sys_path = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(sys_path))

from backend.rag.bm25 import BM25Index, STRATEGIES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="adaptive", choices=STRATEGIES,
                    help="chunking strategy (default adaptive)")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--only-selected", action="store_true",
                    help="filter to is_selected == 1")
    ap.add_argument("--rebuild", action="store_true",
                    help="force rebuild the index even if persisted")
    args = ap.parse_args()

    strategy = args.strategy
    t0 = time.time()
    if args.rebuild:
        from backend.rag.bm25 import build_index
        print(f"rebuilding BM25 index for {strategy}...", flush=True)
        bm = build_index(strategy)
    else:
        print(f"loading BM25 index for {strategy}...", flush=True)
        bm = BM25Index.load(strategy)
    load_ms = (time.time() - t0) * 1000
    print(f"  index ready: {bm.n} documents (load/build {load_ms:.0f} ms)",
          flush=True)

    t_s = time.time()
    results = bm.query(args.query, top_k=args.top_k, only_selected=args.only_selected)
    search_ms = (time.time() - t_s) * 1000
    latency_ms = (time.time() - t0) * 1000

    out = {
        "strategy": strategy,
        "query": args.query,
        "top_k": args.top_k,
        "only_selected": args.only_selected,
        "index_load_ms": round(load_ms, 1),
        "search_ms": round(search_ms, 1),
        "latency_ms": round(latency_ms, 1),
        "results": results,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
