"""Phase 2 prerequisite: measure the real token-length distribution.

Streams ``data/processed/hh_subset_hin.parquet`` row-group by row-group (so
peak RAM stays low — the host has ~1.5 GiB free) and counts content tokens of
each passage's ``text`` (Hindi) with the BGE-M3 tokenizer
(``add_special_tokens=False``). Prints the distribution and writes
``docs/chunking_length_stats.json``.

This is NOT chunking. It only establishes, on real data, the token-length
distribution that justifies the ``adaptive`` chunker's SHORT / MEDIUM / LONG
thresholds — instead of inventing them.

Usage:
    venv/bin/python scripts/measure_lengths.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.rag.chunkers.tokenizer import count_tokens, get_tokenizer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSET = PROJECT_ROOT / "data" / "processed" / "hh_subset_hin.parquet"
OUT = PROJECT_ROOT / "docs" / "chunking_length_stats.json"


def _pct(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    # Nearest-rank percentile (inclusive), 1-indexed.
    import math
    k = math.ceil(p / 100.0 * len(sorted_vals))
    k = max(1, min(k, len(sorted_vals)))
    return int(sorted_vals[k - 1])


def main() -> None:
    assert SUBSET.exists(), f"missing {SUBSET}"
    # Load tokenizer once (warm cache).
    print("loading BGE-M3 tokenizer (content-token convention, "
          "add_special_tokens=False)...")
    t0 = time.time()
    get_tokenizer()
    print(f"  tokenizer ready in {time.time()-t0:.1f}s")

    pf = pq.ParquetFile(SUBSET)
    n_rows = pf.metadata.num_rows
    n_rg = pf.metadata.num_row_groups
    print(f"subset: {n_rows} passages in {n_rg} row groups")

    lengths: list[int] = []
    empty = 0
    t_start = time.time()
    for rg in range(n_rg):
        tbl = pf.read_row_group(rg, columns=["text"])
        texts = tbl.column("text").to_pylist()
        for t in texts:
            if not t:
                empty += 1
                lengths.append(0)
                continue
            lengths.append(count_tokens(t))
        if (rg + 1) % 10 == 0 or rg == n_rg - 1:
            print(f"  rg {rg+1}/{n_rg}: {len(lengths)}/{n_rows} passages "
                  f"({time.time()-t_start:.1f}s)")
    elapsed = time.time() - t_start

    s = sorted(lengths)
    n = len(s)
    total = sum(s)
    # Coarse histogram by power-of-2-ish buckets relevant to chunking.
    buckets = [0, 1, 65, 129, 257, 385, 513, 1025, 2049, 4097]
    hist = Counter()
    for v in lengths:
        placed = False
        for i in range(len(buckets) - 1):
            if buckets[i] <= v < buckets[i + 1]:
                hist[f"{buckets[i]}-{buckets[i+1]-1}"] += 1
                placed = True
                break
        if not placed:
            hist[f"{buckets[-1]}+"] += 1

    stats = {
        "passages": n,
        "empty_passages": empty,
        "min": int(s[0]) if s else 0,
        "max": int(s[-1]) if s else 0,
        "mean": round(total / n, 2) if n else 0,
        "median": int(_pct(s, 50)),
        "p25": int(_pct(s, 25)),
        "p50": int(_pct(s, 50)),
        "p75": int(_pct(s, 75)),
        "p90": int(_pct(s, 90)),
        "p95": int(_pct(s, 95)),
        "p99": int(_pct(s, 99)),
        "le_128": sum(1 for v in lengths if v <= 128),
        "129_to_512": sum(1 for v in lengths if 129 <= v <= 512),
        "gt_512": sum(1 for v in lengths if v > 512),
        "le_256": sum(1 for v in lengths if v <= 256),
        "257_to_384": sum(1 for v in lengths if 257 <= v <= 384),
        "gt_384": sum(1 for v in lengths if v > 384),
        "histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0].split("-")[0].replace("+", "999999")))),
        "seconds": round(elapsed, 2),
        "throughput_passages_per_s": round(n / max(elapsed, 1e-9), 1),
        "tokenizer": "BAAI/bge-m3 (XLM-RoBERTa, content tokens, add_special_tokens=False)",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("\n=== token-length distribution (content tokens) ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
