"""Build the serving indexes: FAISS HNSW (dense) + bm25s (sparse), per language.

Dense
-----
Reads the memmapped ``embeddings.npy`` and its positionally-aligned
``mapping.parquet`` and builds a FAISS ``IndexHNSWFlat`` with
``METRIC_INNER_PRODUCT``. Vectors are already L2-normalized by the encoder, so
inner product *is* cosine similarity — no extra normalization, no divide in the
inner loop.

With ``--verify-recall`` an exact ``IndexFlatIP`` is also built for a sample of
queries so the HNSW approximation error is measured rather than assumed.

Sparse
------
Tokenizes the chunk text with the Devanagari-safe tokenizer and builds a
``bm25s`` index per language. Separate indexes keep IDF meaningful per language;
merging corpora would blend document frequencies and degrade term weighting for
all three.

Usage:
    python scripts/build_indexes.py
    python scripts/build_indexes.py --languages hi --only dense
    python scripts/build_indexes.py --verify-recall 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag.config import CFG, LANGUAGES  # noqa: E402
from backend.rag.dense import META_FIELDS, DenseIndex  # noqa: E402
from backend.rag.sparse import SparseIndex  # noqa: E402


def build_dense(lang: str, strategy: str, verify: int) -> dict:
    emb_dir = CFG.embeddings_dir(lang, strategy)
    emb_path = emb_dir / "embeddings.npy"
    map_path = emb_dir / "mapping.parquet"
    if not emb_path.exists() or not map_path.exists():
        raise FileNotFoundError(
            f"missing embeddings for {lang!r}: {emb_dir}\n"
            f"Run: python scripts/embed_corpus.py --languages {lang}")

    vectors = np.load(emb_path, mmap_mode="r")
    meta = pq.read_table(map_path, columns=META_FIELDS)
    if vectors.shape[0] != meta.num_rows:
        raise RuntimeError(
            f"{lang}: {vectors.shape[0]} vectors vs {meta.num_rows} mapping rows")

    print(f"    vectors         : {vectors.shape[0]:,} x {vectors.shape[1]}")
    t0 = time.perf_counter()
    # FAISS needs a contiguous in-RAM array; ~310 MB per language here.
    dense = DenseIndex.build(lang, np.asarray(vectors), meta, strategy)
    build_s = time.perf_counter() - t0
    path = dense.save()
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"    built HNSW      : m={CFG.hnsw_m} "
          f"ef_construction={CFG.hnsw_ef_construction} in {build_s/60:.1f} min")
    print(f"    wrote           : {path.relative_to(CFG.root)} "
          f"({size_mb:.0f} MB)")

    report = {
        "lang": lang, "vectors": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "hnsw_m": CFG.hnsw_m,
        "ef_construction": CFG.hnsw_ef_construction,
        "ef_search": CFG.hnsw_ef_search,
        "build_seconds": round(build_s, 1),
        "index_mb": round(size_mb, 1),
    }

    if verify:
        # Measure ANN recall against exact search instead of assuming it.
        rng = np.random.default_rng(12345)
        n = vectors.shape[0]
        probe_rows = rng.choice(n, size=min(verify, n), replace=False)
        probes = np.ascontiguousarray(
            np.asarray(vectors[probe_rows], dtype=np.float32))

        exact = DenseIndex.build(lang, np.asarray(vectors), meta, strategy,
                                 exact=True)
        k = 10
        hits = 0
        total = 0
        t_ann = t_exact = 0.0
        for i in range(probes.shape[0]):
            q = probes[i]
            t = time.perf_counter()
            a = dense.search(q, top_k=k)
            t_ann += time.perf_counter() - t
            t = time.perf_counter()
            e = exact.search(q, top_k=k)
            t_exact += time.perf_counter() - t
            aset = {h["chunk_id"] for h in a}
            eset = {h["chunk_id"] for h in e}
            hits += len(aset & eset)
            total += len(eset)
        recall = hits / max(total, 1)
        report["ann_recall_at_10"] = round(recall, 4)
        report["ann_probe_queries"] = int(probes.shape[0])
        report["ann_ms_per_query"] = round(1000 * t_ann / probes.shape[0], 3)
        report["exact_ms_per_query"] = round(1000 * t_exact / probes.shape[0], 3)
        print(f"    ANN recall@10   : {recall:.4f} vs exact "
              f"({report['ann_ms_per_query']:.2f} ms ANN vs "
              f"{report['exact_ms_per_query']:.2f} ms exact)")
        del exact

    del dense
    return report


def build_sparse(lang: str, strategy: str) -> dict:
    print(f"    tokenizing + indexing ...")
    t0 = time.perf_counter()
    sparse = SparseIndex.build(lang, strategy)
    build_s = time.perf_counter() - t0
    d = sparse.save()
    size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 ** 2)
    print(f"    docs            : {sparse.n:,}")
    print(f"    wrote           : {d.relative_to(CFG.root)} "
          f"({size_mb:.0f} MB) in {build_s/60:.1f} min")
    return {"lang": lang, "docs": sparse.n,
            "build_seconds": round(build_s, 1),
            "index_mb": round(size_mb, 1), "backend": "bm25s"}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", default=None,
                    help="comma-separated codes (default: all embedded)")
    ap.add_argument("--strategy", default=CFG.chunk_strategy)
    ap.add_argument("--only", choices=["dense", "sparse"], default=None,
                    help="build only one index type")
    ap.add_argument("--verify-recall", type=int, default=0,
                    metavar="N",
                    help="measure HNSW recall@10 against exact search using N "
                         "probe queries (slow; 200 is plenty)")
    args = ap.parse_args()

    if args.languages:
        codes = [c.strip().lower() for c in args.languages.split(",") if c.strip()]
    else:
        codes = [c for c in CFG.languages
                 if (CFG.embeddings_dir(c, args.strategy) / "embeddings.npy").exists()]
    bad = [c for c in codes if c not in LANGUAGES]
    if bad:
        print(f"ERROR: unknown language(s): {', '.join(bad)}", file=sys.stderr)
        return 2
    if not codes:
        print("ERROR: no embedded languages found. Run "
              "scripts/embed_corpus.py first.", file=sys.stderr)
        return 1

    print("=" * 70)
    print("BUILD SERVING INDEXES")
    print("=" * 70)
    print(f"Strategy  : {args.strategy}")
    print(f"Languages : {', '.join(codes)}")
    print(f"Dense     : faiss HNSW m={CFG.hnsw_m} "
          f"ef_construction={CFG.hnsw_ef_construction} "
          f"ef_search={CFG.hnsw_ef_search}")
    print(f"Sparse    : bm25s")
    print()

    dense_reports, sparse_reports = [], []
    for code in codes:
        print(f"[{code}] {LANGUAGES[code].name}")
        try:
            if args.only != "sparse":
                dense_reports.append(
                    build_dense(code, args.strategy, args.verify_recall))
            if args.only != "dense":
                sparse_reports.append(build_sparse(code, args.strategy))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"    FAILED: {e}", file=sys.stderr)
            return 1
        print()

    out = CFG.root / "docs" / "index_build_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "strategy": args.strategy,
        "encoder": {"model": CFG.embed_model, "dim": CFG.embed_dim,
                    "precision": CFG.embed_precision},
        "dense": dense_reports, "sparse": sparse_reports,
    }, indent=2), encoding="utf-8")

    print("=" * 70)
    if dense_reports:
        print(f"dense  : {sum(r['vectors'] for r in dense_reports):,} vectors, "
              f"{sum(r['index_mb'] for r in dense_reports):.0f} MB")
    if sparse_reports:
        print(f"sparse : {sum(r['docs'] for r in sparse_reports):,} docs, "
              f"{sum(r['index_mb'] for r in sparse_reports):.0f} MB")
    print(f"Report : {out.relative_to(CFG.root)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
