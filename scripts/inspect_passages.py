"""Phase 0 — SAFE passage stats only. Reads ONLY passages.is_selected
(list<int64>) via fastparquet. This column is small ints (~60 MB for all
778K rows), so it cannot OOM the 7GB box the way the ~9GB passage-TEXT
columns do.

Produces:
  * per-row passage count distribution (len of is_selected list)
  * global is_selected value distribution (relevance labels)
  * how many rows have >=1 selected/relevant passage
  * rows where is_selected list length differs from a sane range

NOTE: passage TEXT samples are intentionally NOT read here. Decoding the
list<string> passage columns materializes the whole single-row-group chunk
(~9 GB) and OOMs this 7 GB box. Text samples + the full ingestion stream
are deferred to Phase 1 (ingestion), which will write a small subset to a
new chunked parquet using a row-by-row streaming reader.
"""
from __future__ import annotations

import json
import resource
from collections import Counter
from pathlib import Path

import fastparquet as fp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "hindi": PROJECT_ROOT / "data" / "raw" / "MSMARCO-XI" / "train" / "hintrain.parquet",
    "marathi": PROJECT_ROOT / "data" / "raw" / "MSMARCO-XI" / "train" / "martrain.parquet",
}
OUT = PROJECT_ROOT / "docs"
OUT.mkdir(parents=True, exist_ok=True)

COL = "passages.is_selected"  # flat name fastparquet expects


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    out: dict = {}
    for label, path in FILES.items():
        print("=" * 70)
        print(f"[is_selected] {label}: {path}")
        print("=" * 70)
        pf = fp.ParquetFile(path)
        print(f"  open peak_rss={peak_mb():.0f} MB")

        sel_val_dist: Counter[int] = Counter()
        passage_count_dist: Counter[int] = Counter()
        rows_with_any_selected = 0
        rows_total = 0
        empty_passage_rows = 0
        # iterate row groups (only 1 here) — read ONLY the int column
        for df in pf.iter_row_groups(columns=[COL]):
            col = df[COL]
            n = len(col)
            rows_total += n
            for v in col:
                if v is None:
                    empty_passage_rows += 1
                    passage_count_dist[0] += 1
                    continue
                m = len(v)
                passage_count_dist[m] += 1
                if any(int(x) == 1 for x in v):
                    rows_with_any_selected += 1
                for x in v:
                    sel_val_dist[int(x)] += 1
            print(f"  after group: rows_total={rows_total} peak_rss={peak_mb():.0f} MB")

        rec = {
            "rows_total": rows_total,
            "peak_rss_mb": peak_mb(),
            "empty_passage_rows": empty_passage_rows,
            "rows_with_any_selected": rows_with_any_selected,
            "passage_count_distribution": dict(sorted(passage_count_dist.items())),
            "is_selected_value_distribution": dict(sorted(sel_val_dist.items())),
        }
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        out[label] = rec

    p = OUT / "phase0_passages_stats.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
