"""Experiment 1 — chunking strategy comparison (dense Recall@1/5/10).

Evaluates fixed / semantic / adaptive over the SAME deterministic stratified
query set, dense-only (exact cosine bruteforce over each strategy's full local
embeddings.npy == Qdrant cosine ranking, no Qdrant rebuild). BGE-M3 loaded once.

Saves results/evaluation/chunking_comparison.{json,md}.

Usage:
    venv/bin/python scripts/exp1_chunking.py --n-queries 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.rag.embeddings import load_embedder, encode_batch  # noqa: E402
from backend.rag.evaluation import (  # noqa: E402
    K_VALUES, dense_bruteforce, hit_at_k, load_eval_corpus, sample_queries,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "evaluation"
STRATEGIES = ["fixed", "semantic", "adaptive"]


def run_one(strategy, corpus, qids, encode_fn, top_k=10):
    t0 = time.time()
    hits = {k: [] for k in K_VALUES}
    for qid in qids:
        q = corpus[qid]
        cids = dense_bruteforce(strategy, encode_fn(q["query"]), top_k)
        for k in K_VALUES:
            hits[k].append(hit_at_k(cids, q["relevant"], k))
    elapsed = time.time() - t0
    recall = {f"R@{k}": round(sum(hits[k]) / len(hits[k]), 4) for k in K_VALUES}
    return {"recall": recall, "elapsed_s": round(elapsed, 1),
            "per_query_avg_ms": round(elapsed * 1000 / len(qids), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = ap.parse_args()

    print("loading corpora for all strategies...", flush=True)
    corpora = {s: load_eval_corpus(s) for s in STRATEGIES}
    # sample once from adaptive (all three share the same evaluatable query_ids)
    qids = sample_queries(corpora["adaptive"], n=args.n_queries, seed=args.seed)
    print(f"sampled {len(qids)} queries (stratified, seed={args.seed})",
          flush=True)
    from collections import Counter
    cov = Counter(corpora["adaptive"][q]["query_type"] for q in qids)
    print(f"query_type coverage: {dict(cov)}", flush=True)

    print("loading BGE-M3 ONCE (shared across strategies)...", flush=True)
    model, device, dtype = load_embedder(args.device, None)
    encode_fn = lambda t: encode_batch(model, [t], batch_size=1)

    results = {}
    for s in STRATEGIES:
        print(f"\n--- dense eval: {s} ---", flush=True)
        results[s] = run_one(s, corpora[s], qids, encode_fn)
        r = results[s]["recall"]
        print(f"  R@1={r['R@1']} R@5={r['R@5']} R@10={r['R@10']} "
              f"({results[s]['per_query_avg_ms']}ms/q, "
              f"{results[s]['elapsed_s']}s)", flush=True)

    # winner by R@10 (then R@5, R@1)
    def score(s):
        r = results[s]["recall"]
        return (r["R@10"], r["R@5"], r["R@1"])
    winner = max(STRATEGIES, key=score)

    out = {
        "n_queries": len(qids), "seed": args.seed,
        "query_type_coverage": dict(cov),
        "ranker": "dense (exact cosine bruteforce == Qdrant cosine)",
        "device": device, "dtype": dtype,
        "strategies": results,
        "winner": winner,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "chunking_comparison.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))

    md = ["# Experiment 1 — Chunking Comparison (dense Recall@k)",
          "",
          f"Same {len(qids)} deterministic stratified queries, dense-only "
          "(exact cosine == Qdrant ranking). BGE-M3 on {device} ({dtype}).",
          "",
          "| Strategy | R@1 | R@5 | R@10 | ms/query |",
          "|---|---:|---:|---:|---:|"]
    for s in STRATEGIES:
        r = results[s]["recall"]
        md.append(f"| {s} | {r['R@1']} | {r['R@5']} | {r['R@10']} | "
                  f"{results[s]['per_query_avg_ms']} |")
    md += ["", f"**Winner: {winner}** (by R@10, then R@5, then R@1).", ""]
    (RESULTS_DIR / "chunking_comparison.md").write_text("\n".join(md))

    print("\n=== RESULT ===", flush=True)
    print(f"winner: {winner}", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
