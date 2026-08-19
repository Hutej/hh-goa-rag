"""Phase 0 — safe sample recon. Two modes, run separately so an OOM in one
cannot destroy the other's saved output.

  --mode short    : aggregate short scalar columns across ALL rows using
                    arrow-native value_counts + per-batch Counters. Bounded
                    RAM (only one small batch in Python at a time). Produces
                    global language-code / query_type distributions and
                    per-column null counts for the entire file. SAFE.

NOTE: a `--mode passages` path was removed. Both parquet files are written as
a SINGLE row group holding all ~778K rows; pyarrow/fastparquet must decode the
entire ~9 GB `passages` list<string> column chunk before returning any row,
which OOMs this 7 GB box. Passage TEXT samples are therefore NOT produced
here. Passage COUNTS and is_selected relevance stats (small int column) are
produced safely by scripts/inspect_passages.py. Real passage text sampling +
full ingestion streaming are deferred to Phase 1 (ingestion).

Run with the project venv:
    venv/bin/python scripts/inspect_sample.py --mode short
"""
from __future__ import annotations

import argparse
import json
import resource
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "hindi": PROJECT_ROOT / "data" / "raw" / "MSMARCO-XI" / "train" / "hintrain.parquet",
    "marathi": PROJECT_ROOT / "data" / "raw" / "MSMARCO-XI" / "train" / "martrain.parquet",
}
OUT = PROJECT_ROOT / "docs"
OUT.mkdir(parents=True, exist_ok=True)

SHORT_COLS = [
    "source_lang", "target_lang", "query_id", "query_type",
    "query", "Answer", "Eng_Query", "Eng_Answer", "meta",
]
BATCH = 8192
N_PASSAGE_ROWS = 8


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def mode_short() -> dict:
    out: dict = {}
    for label, path in FILES.items():
        print("=" * 70)
        print(f"[short] {label}: {path}")
        print("=" * 70)
        pf = pq.ParquetFile(path)
        n_rows = pf.metadata.num_rows

        dist: dict[str, Counter] = {c: Counter() for c in
                                    ("source_lang", "target_lang", "query_type")}
        nulls: dict[str, int] = {c: 0 for c in SHORT_COLS}
        # a handful of sample rows from the first batch (short cols only)
        sample_rows: list[dict] = []
        batches_seen = 0
        for batch in pf.iter_batches(batch_size=BATCH, columns=SHORT_COLS):
            batches_seen += 1
            for c in ("source_lang", "target_lang", "query_type"):
                col = batch.column(c)
                nulls[c] += col.null_count
                for v in col.to_pylist():
                    dist[c][str(v)] += 1
            # accumulate nulls for the long-string scalar cols without pylist
            for c in ("query_id", "query", "Answer", "Eng_Query", "Eng_Answer", "meta"):
                nulls[c] += batch.column(c).null_count
            if not sample_rows:  # capture first few rows from batch 0
                for r in range(min(4, batch.num_rows)):
                    row = {c: batch.column(c)[r].as_py() for c in SHORT_COLS}
                    for k in ("query", "Answer", "Eng_Query", "Eng_Answer"):
                        if isinstance(row.get(k), str) and len(row[k]) > 150:
                            row[k] = row[k][:150] + f"…(+{len(row[k])-150})"
                    sample_rows.append(row)

        rec = {
            "rows": n_rows,
            "batches_seen": batches_seen,
            "peak_rss_mb": peak_mb(),
            "source_lang_dist": dict(dist["source_lang"]),
            "target_lang_dist": dict(dist["target_lang"]),
            "query_type_dist": dict(dist["query_type"]),
            "null_counts": nulls,
            "sample_rows": sample_rows,
        }
        print(json.dumps({k: rec[k] for k in
              ("rows", "batches_seen", "peak_rss_mb",
               "source_lang_dist", "target_lang_dist",
               "query_type_dist", "null_counts")}, ensure_ascii=False, indent=2))
        print("sample_rows:", json.dumps(sample_rows, ensure_ascii=False)[:1500])
        out[label] = rec

    p = OUT / "phase0_short_aggregates.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["short"], default="short")
    args = ap.parse_args()
    if args.mode == "short":
        mode_short()


if __name__ == "__main__":
    main()
