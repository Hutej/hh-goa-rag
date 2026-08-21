"""Experiment 2 — RRF weight sweep.

On a given strategy, sweeps RRF (dense_weight, bm25_weight) configurations and
measures Recall@1/5/10 over the SAME deterministic query set. Includes a dense-only
config (bm25_weight=0) so dense-only is a valid candidate. Does NOT force hybrid.

Saves results/evaluation/rrf_weights.{json,md}.

Usage:
    venv/bin/python scripts/exp2_rrf_weights.py --strategy adaptive --n-queries 100
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
    K_VALUES, dense_bruteforce, hit_at_k, load_eval_corpus, rrf_fuse,
    sample_queries,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "evaluation"

# (label, dense_weight, bm25_weight)
WEIGHTS = [
    ("dense_only", 1.0, 0.0),
    ("d1.0_b0.25", 1.0, 0.25),
    ("d1.0_b0.5", 1.0, 0.5),
    ("d1.0_b1.0", 1.0, 1.0),
    ("d2.0_b1.0", 2.0, 1.0),
]


def run_config(strategy, corpus, qids, encode_fn, bm25, dw, bw,
               top_k=10, dense_k=20, bm25_k=20, rrf_k=60):
    t0 = time.time()
    hits = {k: [] for k in K_VALUES}
    for qid in qids:
        q = corpus[qid]
        qvec = encode_fn(q["query"])
        dh = [{"rank": i + 1, "chunk_id": c} for i, c in
              enumerate(dense_bruteforce(strategy, qvec, dense_k))]
        bh = [{"rank": i + 1, "chunk_id": h["chunk_id"]} for i, h in
              enumerate(bm25.query(q["query"], top_k=bm25_k))]
        fused = rrf_fuse(dh, bh, rrf_k=rrf_k, dense_weight=dw, bm25_weight=bw)
        cids = [h["chunk_id"] for h in fused[:top_k] if h["chunk_id"] is not None]
        for k in K_VALUES:
            hits[k].append(hit_at_k(cids, q["relevant"], k))
    elapsed = time.time() - t0
    recall = {f"R@{k}": round(sum(hits[k]) / len(hits[k]), 4) for k in K_VALUES}
    return {"recall": recall, "elapsed_s": round(elapsed, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="adaptive")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = ap.parse_args()

    print("loading eval corpus...", flush=True)
    corpus = load_eval_corpus(args.strategy)
    qids = sample_queries(corpus, n=args.n_queries, seed=args.seed)
    print(f"sampled {len(qids)} queries (seed={args.seed})", flush=True)

    print("loading BGE-M3 + BM25...", flush=True)
    model, device, dtype = load_embedder(args.device, None)
    encode_fn = lambda t: encode_batch(model, [t], batch_size=1)
    bm25 = BM25Index.load(args.strategy)

    results = {}
    for label, dw, bw in WEIGHTS:
        print(f"--- {label} (d={dw}, b={bw}) ---", flush=True)
        r = run_config(args.strategy, corpus, qids, encode_fn, bm25, dw, bw)
        results[label] = {"dense_weight": dw, "bm25_weight": bw, **r}
        rc = r["recall"]
        print(f"  R@1={rc['R@1']} R@5={rc['R@5']} R@10={rc['R@10']} "
              f"({r['elapsed_s']}s)", flush=True)

    # best by R@10 then R@5 then R@1
    def score(lbl):
        r = results[lbl]["recall"]
        return (r["R@10"], r["R@5"], r["R@1"])
    best = max(results.keys(), key=score)

    out = {
        "strategy": args.strategy, "n_queries": len(qids), "seed": args.seed,
        "device": device, "dtype": dtype,
        "rrf_k": 60, "dense_k": 20, "bm25_k": 20,
        "configs": results, "best": best,
        "best_weights": {"dense_weight": results[best]["dense_weight"],
                         "bm25_weight": results[best]["bm25_weight"]},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "rrf_weights.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))

    md = ["# Experiment 2 — RRF Weight Sweep",
          "",
          f"Strategy: **{args.strategy}**. Same {len(qids)} deterministic "
          f"queries. BGE-M3 on {device} ({dtype}).",
          "",
          "| Config | dense_w | bm25_w | R@1 | R@5 | R@10 |",
          "|---|---:|---:|---:|---:|---:|"]
    for label, dw, bw in WEIGHTS:
        r = results[label]["recall"]
        md.append(f"| {label} | {dw} | {bw} | {r['R@1']} | {r['R@5']} | "
                  f"{r['R@10']} |")
    md += ["", f"**Best: {best}** (dense_w={results[best]['dense_weight']}, "
          f"bm25_w={results[best]['bm25_weight']}).", ""]
    (RESULTS_DIR / "rrf_weights.md").write_text("\n".join(md))

    print("\n=== RESULT ===", flush=True)
    print(f"best: {best}", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
