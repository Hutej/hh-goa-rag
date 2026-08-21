"""Phase 6: retrieval-stage latency benchmark (P50/P70/P100).

Measures per-query latency of the retrieval pipeline (encode -> dense -> BM25 ->
RRF) using the REAL Qdrant dense path, BM25, and RRF fusion. Reports cold-start
(first query) separately from warm per-query latency, and P50/P70/P100 for each
stage and the total.

This is a RETRIEVAL-STAGE benchmark only. It does NOT include STT or LLM answer
generation, and does NOT claim the official <200 ms end-to-end target is met.
The official target covers voice -> STT -> retrieval -> answer generation and
requires its own end-to-end measurement once those stages exist.

One-time model/index loading is performed (untimed) before warmup, so the timed
runs reflect steady-state per-query cost.

Usage:
    venv/bin/python scripts/benchmark.py --n-queries 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.rag.bm25 import BM25Index  # noqa: E402
from backend.rag.embeddings import load_embedder, encode_batch  # noqa: E402
from backend.rag.evaluation import load_eval_corpus, sample_queries  # noqa: E402
from backend.rag.qdrant_index import dense_search, get_client  # noqa: E402

STAGES = ["encode_ms", "dense_ms", "bm25_ms", "rrf_ms", "total_ms"]


def percentiles(values: list[float], ps: list[int] = (50, 70, 100)) -> dict:
    """Percentiles of a list of latencies (ms). P100 = max.

    Uses linear interpolation like numpy.percentile for P50/P70; P100 is the
    observed maximum (not interpolated, since the official target cares about the
    worst observed case). Empty input -> zeros.
    """
    if not values:
        return {f"P{p}": 0.0 for p in ps}
    arr = np.asarray(values, dtype=float)
    out = {}
    for p in ps:
        if p == 100:
            out[f"P{p}"] = round(float(arr.max()), 2)
        else:
            out[f"P{p}"] = round(float(np.percentile(arr, p)), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="adaptive")
    ap.add_argument("--n-queries", type=int, default=50)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--warmup", type=int, default=5,
                    help="warmup queries (untimed)")
    args = ap.parse_args()

    print("loading eval corpus + sampling queries...", flush=True)
    corpus = load_eval_corpus(args.strategy)
    qids = sample_queries(corpus, n=args.n_queries + args.warmup, seed=args.seed)
    print(f"  {len(qids)} queries sampled ({args.warmup} warmup, "
          f"{args.n_queries} timed)", flush=True)

    print("loading BGE-M3 (untimed)...", flush=True)
    model, device, dtype = load_embedder(args.device, None)
    print("loading BM25 index (untimed)...", flush=True)
    bm25 = BM25Index.load(args.strategy)

    client = get_client()
    try:
        # ---- warmup (untimed) ----
        print(f"warmup ({args.warmup} queries, untimed)...", flush=True)
        for i in range(args.warmup):
            q = corpus[qids[i]]["query"]
            qv = encode_batch(model, [q], batch_size=1)
            dh = dense_search(client, args.strategy, qv, top_k=20)
            bh = bm25.query(q, top_k=20)
            from backend.rag.hybrid import rrf_fuse
            rrf_fuse(dh, bh)

        # ---- timed runs ----
        timed_qids = qids[args.warmup:]
        per_q = {s: [] for s in STAGES}
        cold = None
        for i, qid in enumerate(timed_qids):
            q = corpus[qid]["query"]
            t_total = time.time()
            t = time.time()
            qv = encode_batch(model, [q], batch_size=1)
            encode_ms = (time.time() - t) * 1000

            t = time.time()
            dh = dense_search(client, args.strategy, qv, top_k=20)
            dense_ms = (time.time() - t) * 1000

            t = time.time()
            bh = bm25.query(q, top_k=20)
            bm25_ms = (time.time() - t) * 1000

            t = time.time()
            from backend.rag.hybrid import rrf_fuse
            rrf_fuse(dh, bh)
            rrf_ms = (time.time() - t) * 1000

            total_ms = (time.time() - t_total) * 1000
            if i == 0:
                cold = {"encode_ms": round(encode_ms, 2),
                        "dense_ms": round(dense_ms, 2),
                        "bm25_ms": round(bm25_ms, 2),
                        "rrf_ms": round(rrf_ms, 2),
                        "total_ms": round(total_ms, 2)}
            for s, v in [("encode_ms", encode_ms), ("dense_ms", dense_ms),
                         ("bm25_ms", bm25_ms), ("rrf_ms", rrf_ms),
                         ("total_ms", total_ms)]:
                per_q[s].append(v)
    finally:
        client.close()

    ps = (50, 70, 100)
    warm = {s: percentiles(per_q[s], ps) for s in STAGES}

    print("\n=== RETRIEVAL-STAGE LATENCY BENCHMARK ===", flush=True)
    print(f"strategy: {args.strategy}  device: {device}({dtype})", flush=True)
    print(f"timed queries: {len(timed_qids)}  (warmup: {args.warmup})",
          flush=True)
    print(f"\ncold-start (first timed query, ms): {cold}", flush=True)
    print(f"\nwarm per-query percentiles (ms):", flush=True)
    print(f"{'stage':<12} " + " ".join(f"{f'P{p}':<10}" for p in ps), flush=True)
    for s in STAGES:
        print(f"{s:<12} " + " ".join(f"{warm[s][f'P{p}']:<10}" for p in ps),
              flush=True)

    out = {
        "strategy": args.strategy, "device": device, "dtype": dtype,
        "timed_queries": len(timed_qids), "warmup": args.warmup,
        "cold_start_ms": cold, "warm_percentiles_ms": warm,
        "note": "retrieval-stage only; excludes STT + LLM; no <200ms claim",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

    # persist raw + summary + report to results/benchmark/ (small, git-friendly)
    from pathlib import Path
    RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "benchmark"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = {"strategy": args.strategy, "device": device, "dtype": dtype,
           "warmup": args.warmup, "timed_queries": len(timed_qids),
           "cold_start_ms": cold,
           "per_query_ms": {s: [round(v, 3) for v in per_q[s]] for s in STAGES}}
    (RESULTS_DIR / "retrieval_raw.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2))
    summary = {"strategy": args.strategy, "device": device, "dtype": dtype,
               "timed_queries": len(timed_qids), "warmup": args.warmup,
               "cold_start_ms": cold, "warm_percentiles_ms": warm,
               "target_ms": 200, "meets_target": (warm["total_ms"]["P50"] <= 200),
               "note": "retrieval-stage only; excludes STT + LLM; no <200ms claim"}
    (RESULTS_DIR / "retrieval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    md = ["# Experiment 3 — Retrieval Latency Benchmark",
          "",
          f"Strategy: {args.strategy}. Device: {device}({dtype}). "
          f"{len(timed_qids)} timed queries (warmup {args.warmup}). "
          "Warmup + model/index load untimed. Real Qdrant dense + BM25 + RRF.",
          "",
          "## Warm per-query percentiles (ms)",
          "",
          "| Stage | P50 | P70 | P100 |",
          "|---|---:|---:|---:|"]
    for s in STAGES:
        w = warm[s]
        md.append(f"| {s} | {w['P50']} | {w['P70']} | {w['P100']} |")
    md += ["", f"Cold-start (first timed query, ms): `{cold}`.", "",
           f"**Target (<200ms): {'MET' if summary['meets_target'] else 'NOT met'}** "
           "(retrieval-stage only — STT + LLM not included).", ""]
    (RESULTS_DIR / "retrieval_report.md").write_text("\n".join(md))
    print(f"\nsaved: {RESULTS_DIR}/retrieval_{{raw,summary}}.json + "
          f"retrieval_report.md", flush=True)


if __name__ == "__main__":
    main()
