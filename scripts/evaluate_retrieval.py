"""Retrieval quality evaluation: monolingual recall, retriever ablation, and a
genuine cross-lingual test.

Ground truth
------------
MS MARCO marks relevant passages with ``is_selected=1``. A query is counted as
answered at rank k if any retrieved chunk in the top k belongs to a document that
was marked relevant for that query. Chunk-level hits are mapped back to
``document_id``, so a passage split into several chunks is not counted multiple
times.

The cross-lingual evaluation
----------------------------
This is the part the corpus construction bought for free. Every source row
carries the same document in English *and* in the Indic language, index-aligned,
sharing one ``is_selected`` label — so ``document_id`` identifies the same
document across all three languages.

That makes a real cross-lingual measurement possible: ask in Hindi, restrict
retrieval to the **English** index, and check whether the returned English
document is the one marked relevant for the Hindi query. Ground truth transfers
exactly, with no translation of labels and no manual annotation. Very few RAG
evaluations can do this, and it directly tests whether the shared embedding space
is doing real work rather than just matching surface forms.

Ablation
--------
Dense-only, sparse-only and fused are all measured on identical queries, which is
the only way to know whether hybrid fusion earns its cost and whether the sparse
weight is set sensibly.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --queries 200 --no-cross-lingual
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


def load_ground_truth(lang: str) -> list[dict]:
    """Queries with at least one relevant document, plus that relevant set."""
    path = CFG.passages_path(lang)
    if not path.exists():
        return []
    # Read the query column for THIS language: `query` holds the shard's Indic
    # query, `query_en` the original English one. Using `query` for English
    # would feed Hindi questions to the English index and quietly report a
    # cross-lingual result as monolingual.
    qcol = CFG.lang(lang).query_col
    tbl = pq.read_table(path, columns=["query_id", qcol, "document_id",
                                       "is_selected"])
    qids = tbl.column("query_id").to_pylist()
    queries = tbl.column(qcol).to_pylist()
    docs = tbl.column("document_id").to_pylist()
    sels = tbl.column("is_selected").to_pylist()

    by_q: dict[int, dict] = {}
    for qid, q, doc, sel in zip(qids, queries, docs, sels):
        entry = by_q.setdefault(qid, {"query_id": qid, "query": (q or "").strip(),
                                      "relevant": set()})
        if sel:
            entry["relevant"].add(doc)
    return [e for e in by_q.values() if e["relevant"] and e["query"]]


def sample(gt: list[dict], n: int) -> list[dict]:
    pool = sorted(gt, key=lambda e: e["query_id"])
    random.Random(SEED).shuffle(pool)
    return pool[:n]


def recall_at_k(hits: list[dict], relevant: set[str], ks=KS) -> dict[int, int]:
    """1 if any of the top-k chunks maps to a relevant document, else 0."""
    seen_docs: list[str] = []
    for h in hits:
        doc = h.get("document_id")
        if doc not in seen_docs:
            seen_docs.append(doc)
    return {k: int(any(d in relevant for d in seen_docs[:k])) for k in ks}


def evaluate_language(lang: str, queries: list[dict]) -> dict:
    """Dense-only / sparse-only / fused recall on the same queries."""
    retr = get_retriever()
    modes = {"dense": {k: 0 for k in KS},
             "sparse": {k: 0 for k in KS},
             "hybrid": {k: 0 for k in KS}}

    for entry in queries:
        q = entry["query"]
        rel = entry["relevant"]
        # Restrict to this language's own index: monolingual retrieval quality.
        res = retr.search(q, top_k=max(KS), dense_k=30, sparse_k=30,
                          languages=[lang], use_cache=False)

        dense_only = list(res.dense_hits)
        retr.dense.hydrate(dense_only)
        sparse_only = list(res.sparse_hits)

        for mode, hits in (("dense", dense_only),
                           ("sparse", sparse_only),
                           ("hybrid", res.hits)):
            got = recall_at_k(hits, rel)
            for k in KS:
                modes[mode][k] += got[k]

    n = max(len(queries), 1)
    return {mode: {f"R@{k}": round(v[k] / n, 4) for k in KS}
            for mode, v in modes.items()} | {"queries": len(queries)}


def evaluate_cross_lingual(src_lang: str, tgt_lang: str,
                           queries: list[dict]) -> dict:
    """Ask in ``src_lang``, retrieve only from ``tgt_lang``.

    Ground truth transfers because ``document_id`` is language-independent.
    """
    retr = get_retriever()
    if tgt_lang not in retr.dense.indexes:
        return {"skipped": f"{tgt_lang} index not loaded"}
    totals = {k: 0 for k in KS}
    for entry in queries:
        res = retr.search(entry["query"], top_k=max(KS), dense_k=30,
                          sparse_k=30, languages=[tgt_lang], use_cache=False)
        got = recall_at_k(res.hits, entry["relevant"])
        for k in KS:
            totals[k] += got[k]
    n = max(len(queries), 1)
    return {f"R@{k}": round(totals[k] / n, 4) for k in KS} | {"queries": n}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=150)
    ap.add_argument("--no-cross-lingual", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("RETRIEVAL EVALUATION")
    print("=" * 78)
    print(f"Ground truth : MS MARCO is_selected, matched at document_id level")
    print(f"Queries      : {args.queries}/language, seed {SEED}")
    print(f"Encoder      : {CFG.embed_model} ({CFG.embed_precision})")
    print(f"Fusion       : RRF k={CFG.rrf_k}, dense {CFG.dense_weight} / "
          f"sparse {CFG.sparse_weight}")
    print()

    get_retriever()
    per_lang: dict[str, dict] = {}
    samples: dict[str, list[dict]] = {}

    for lang in CFG.languages:
        gt = load_ground_truth(lang)
        if not gt:
            print(f"[{lang}] SKIP — no ground truth")
            continue
        qs = sample(gt, args.queries)
        samples[lang] = qs
        print(f"[{lang}] {LANGUAGES[lang].name} — {len(qs)} queries "
              f"({len(gt)} labelled available)")
        per_lang[lang] = evaluate_language(lang, qs)

    print()
    print("-" * 78)
    print("MONOLINGUAL RECALL (retriever ablation)")
    print("-" * 78)
    print(f"{'lang':<10}{'retriever':<10}" + "".join(f"{f'R@{k}':>9}" for k in KS))
    for lang, r in per_lang.items():
        for mode in ("dense", "sparse", "hybrid"):
            print(f"{lang:<10}{mode:<10}" +
                  "".join(f"{r[mode][f'R@{k}']:>9.3f}" for k in KS))
        print()

    cross: dict[str, dict] = {}
    if not args.no_cross_lingual:
        print("-" * 78)
        print("CROSS-LINGUAL RECALL (query language -> retrieval index)")
        print("-" * 78)
        pairs = [(s, t) for s in CFG.languages for t in CFG.languages if s != t]
        print(f"{'pair':<14}" + "".join(f"{f'R@{k}':>9}" for k in KS))
        for src, tgt in pairs:
            if src not in samples:
                continue
            res = evaluate_cross_lingual(src, tgt, samples[src])
            cross[f"{src}->{tgt}"] = res
            if "skipped" in res:
                print(f"{src+' -> '+tgt:<14}  SKIP: {res['skipped']}")
                continue
            print(f"{src+' -> '+tgt:<14}" +
                  "".join(f"{res[f'R@{k}']:>9.3f}" for k in KS))
        print()
        print("Ground truth transfers across languages because document_id is")
        print("language-independent: the same document exists in all three, with")
        print("one shared relevance label.")

    out = CFG.root / "results" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_quality.json").write_text(json.dumps({
        "seed": SEED, "queries_per_language": args.queries,
        "ks": list(KS),
        "config": {"encoder": CFG.embed_model, "precision": CFG.embed_precision,
                   "dim": CFG.embed_dim, "rrf_k": CFG.rrf_k,
                   "dense_weight": CFG.dense_weight,
                   "sparse_weight": CFG.sparse_weight,
                   "max_per_document": CFG.max_per_document},
        "monolingual": per_lang,
        "cross_lingual": cross,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("=" * 78)
    print(f"Wrote {(out / 'retrieval_quality.json').relative_to(CFG.root)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
