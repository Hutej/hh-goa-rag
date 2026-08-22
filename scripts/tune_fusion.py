"""Sweep the RRF sparse weight and pick it from measurement.

The initial evaluation showed hybrid fusion roughly break-even against
dense-only (Hindi R@10 0.633 vs 0.627; English 0.880 vs 0.887). That is worth
investigating rather than shipping: word-level BM25 is weak on machine-translated
Devanagari, so a sparse vote weighted too heavily can displace correct dense hits,
while one weighted too lightly contributes nothing.

This sweeps ``sparse_weight`` over a grid on identical queries and reports recall
per language, so the shipped value is chosen from data. Dense retrieval is run
once per query and reused across all weights — fusion is a pure function of the
two ranked lists, so only the fusion step is repeated.

Usage:
    python scripts/tune_fusion.py
    python scripts/tune_fusion.py --queries 200 --weights 0,0.1,0.25,0.5,1.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag.config import CFG, LANGUAGES  # noqa: E402
from backend.rag.retrieval import get_retriever, rrf_fuse  # noqa: E402

SEED = 12345
KS = (1, 3, 5, 10)


def ground_truth(lang: str, n: int) -> list[dict]:
    path = CFG.passages_path(lang)
    if not path.exists():
        return []
    qcol = CFG.lang(lang).query_col
    tbl = pq.read_table(path, columns=["query_id", qcol, "document_id",
                                       "is_selected"])
    by_q: dict[int, dict] = {}
    for qid, q, doc, sel in zip(tbl.column("query_id").to_pylist(),
                               tbl.column(qcol).to_pylist(),
                               tbl.column("document_id").to_pylist(),
                               tbl.column("is_selected").to_pylist()):
        e = by_q.setdefault(qid, {"query": (q or "").strip(), "relevant": set()})
        if sel:
            e["relevant"].add(doc)
    pool = [e for e in by_q.values() if e["relevant"] and e["query"]]
    pool.sort(key=lambda e: e["query"])
    random.Random(SEED).shuffle(pool)
    return pool[:n]


def recall(hits: list[dict], relevant: set[str]) -> dict[int, int]:
    docs: list[str] = []
    for h in hits:
        d = h.get("document_id")
        if d not in docs:
            docs.append(d)
    return {k: int(any(x in relevant for x in docs[:k])) for k in KS}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=150)
    ap.add_argument("--weights", default="0,0.1,0.25,0.5,0.75,1.0")
    args = ap.parse_args()

    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    retr = get_retriever()

    print("=" * 78)
    print("RRF SPARSE-WEIGHT SWEEP")
    print("=" * 78)
    print(f"Queries : {args.queries}/language, seed {SEED}")
    print(f"Weights : {weights}  (dense weight fixed at {CFG.dense_weight})")
    print()

    results: dict[str, dict[str, dict[str, float]]] = {}

    for lang in CFG.languages:
        qs = ground_truth(lang, args.queries)
        if not qs:
            continue
        print(f"[{lang}] {LANGUAGES[lang].name} — {len(qs)} queries")

        # Retrieve once per query; re-fuse per weight.
        cached: list[tuple[list[dict], list[dict], set[str]]] = []
        for e in qs:
            res = retr.search(e["query"], top_k=max(KS), dense_k=30,
                              sparse_k=30, languages=[lang], use_cache=False)
            cached.append((res.dense_hits, res.sparse_hits, e["relevant"]))

        per_weight: dict[str, dict[str, float]] = {}
        for w in weights:
            totals = {k: 0 for k in KS}
            for dense_hits, sparse_hits, rel in cached:
                fused = rrf_fuse(dense_hits, sparse_hits,
                                 sparse_weight=w)[:max(KS)]
                retr.dense.hydrate(fused)
                got = recall(fused, rel)
                for k in KS:
                    totals[k] += got[k]
            n = max(len(cached), 1)
            per_weight[str(w)] = {f"R@{k}": round(totals[k] / n, 4) for k in KS}
            r = per_weight[str(w)]
            print(f"   sparse_weight={w:<5} " +
                  "  ".join(f"R@{k} {r[f'R@{k}']:.3f}" for k in KS))
        results[lang] = per_weight
        print()

    # Pick by mean R@10 across languages: the metric the answer layer depends on,
    # since generation sees the top few of a well-ordered list.
    print("-" * 78)
    print(f"{'sparse_weight':<16}" +
          "".join(f"{l:>10}" for l in results) + f"{'mean R@10':>12}")
    best = None
    for w in weights:
        row = [results[l][str(w)]["R@10"] for l in results]
        mean = sum(row) / len(row)
        print(f"{w:<16}" + "".join(f"{v:>10.3f}" for v in row) +
              f"{mean:>12.4f}")
        if best is None or mean > best[1]:
            best = (w, mean)
    print("-" * 78)
    print(f"BEST sparse_weight = {best[0]} (mean R@10 {best[1]:.4f})")
    print(f"Currently shipping RAG_SPARSE_WEIGHT={CFG.sparse_weight}")

    out = CFG.root / "results" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fusion_weights.json").write_text(json.dumps({
        "seed": SEED, "queries_per_language": args.queries,
        "dense_weight": CFG.dense_weight, "rrf_k": CFG.rrf_k,
        "weights": weights, "results": results,
        "best_sparse_weight": best[0], "best_mean_r_at_10": round(best[1], 4),
        "shipped_sparse_weight": CFG.sparse_weight,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {(out / 'fusion_weights.json').relative_to(CFG.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
