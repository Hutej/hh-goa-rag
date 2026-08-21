"""Phase 6: retrieval evaluation (Recall@1/@5/@10) on the Adaptive corpus.

Relevance = dataset ``is_selected == 1`` passages. Dense ranking is exact cosine
bruteforce over the full local embeddings.npy (identical to Qdrant's cosine
ranking, so no Qdrant rebuild needed). BM25 and Hybrid reuse Phase 4 / Phase 5.

Usage:
    venv/bin/python scripts/evaluate_retrieval.py --n-queries 100
    venv/bin/python scripts/evaluate_retrieval.py --n-queries 100 --rankers dense bm25 hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.rag.bm25 import BM25Index  # noqa: E402
from backend.rag.embeddings import load_embedder, encode_batch  # noqa: E402
from backend.rag.evaluation import (  # noqa: E402
    K_VALUES, RANKERS, evaluate, load_eval_corpus, sample_queries,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="adaptive")
    ap.add_argument("--n-queries", type=int, default=100,
                    help="stratified sample size (0 = all evaluatable)")
    ap.add_argument("--rankers", nargs="+", default=RANKERS,
                    choices=RANKERS)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--bm25-weight", type=float, default=1.0)
    ap.add_argument("--dense-k", type=int, default=20)
    ap.add_argument("--bm25-k", type=int, default=20)
    args = ap.parse_args()

    print("loading eval corpus...", flush=True)
    corpus = load_eval_corpus(args.strategy)
    print(f"  evaluatable queries: {len(corpus)}", flush=True)

    n = args.n_queries if args.n_queries > 0 else len(corpus)
    qids = sample_queries(corpus, n=n, seed=args.seed)
    print(f"  sampled {len(qids)} queries (stratified by query_type)", flush=True)

    # query_type coverage of the sample
    from collections import Counter
    cov = Counter(corpus[q]["query_type"] for q in qids)
    print(f"  query_type coverage: {dict(cov)}", flush=True)

    print("loading BGE-M3 (for query encoding)...", flush=True)
    model, device, dtype = load_embedder(args.device, None)

    def encode_fn(text):
        return encode_batch(model, [text], batch_size=1)

    print("loading BM25 index...", flush=True)
    bm25 = BM25Index.load(args.strategy)

    print(f"evaluating rankers={args.rankers} strategy={args.strategy} ...",
          flush=True)
    t0 = time.time()
    res = evaluate(args.rankers, corpus, qids, bm25, encode_fn,
                   strategy=args.strategy, rrf_k=args.rrf_k,
                   dense_weight=args.dense_weight,
                   bm25_weight=args.bm25_weight,
                   dense_k=args.dense_k, bm25_k=args.bm25_k)
    elapsed = time.time() - t0

    print("\n=== RETRIEVAL EVALUATION ===", flush=True)
    print(f"queries evaluated: {len(qids)}", flush=True)
    print(f"rankers: {args.rankers}", flush=True)
    print(f"elapsed: {elapsed:.1f}s", flush=True)
    print(f"{'ranker':<10} " + " ".join(f"R@{k}" for k in K_VALUES), flush=True)
    for r in args.rankers:
        print(f"{r:<10} " + " ".join(f"{res[r][k]:.4f}" for k in K_VALUES),
              flush=True)

    out = {
        "strategy": args.strategy,
        "n_queries": len(qids),
        "seed": args.seed,
        "query_type_coverage": dict(cov),
        "rankers": args.rankers,
        "config": {"rrf_k": args.rrf_k, "dense_weight": args.dense_weight,
                   "bm25_weight": args.bm25_weight, "dense_k": args.dense_k,
                   "bm25_k": args.bm25_k},
        "recall": {r: {f"R@{k}": res[r][k] for k in K_VALUES} for r in args.rankers},
        "elapsed_s": round(elapsed, 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
