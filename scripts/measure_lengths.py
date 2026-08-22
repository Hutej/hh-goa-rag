"""Measure per-language passage token-length distributions and derive the
adaptive chunking thresholds from them.

Why per-language thresholds are not optional
--------------------------------------------
The adaptive strategy routes a passage by token count: short passages stay
whole, medium ones get sentence-aware grouping, long ones get overlapping
windows. Those band edges were originally calibrated on Hindi alone.

The same token budget does not mean the same amount of text across scripts. A
sentencepiece vocabulary fragments Devanagari far more than Latin, so a Hindi
passage and its English translation — literally the same document — land at
different token counts. Reusing one threshold set would route the two
languages' identical content into different bands, which is exactly the kind of
silent inconsistency that makes cross-lingual comparison meaningless.

This script measures the real distribution per language and emits thresholds
placed at distribution landmarks rather than round numbers:

    short_max  = P80 of the distribution, clamped to a sane range
    medium_max = P99.5, rounded up
    semantic_max = short_max * 1.5, so grouping has headroom above the short band

Output:
    docs/chunking_length_stats.json   full stats + derived thresholds per language

The thresholds are consumed by ``scripts/chunk_corpus.py``. Nothing here writes
chunks; it only measures.

Usage:
    python scripts/measure_lengths.py
    python scripts/measure_lengths.py --languages hi,en --sample 50000
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
from backend.rag.chunkers.tokenizer import (  # noqa: E402
    MODEL_NAME, count_tokens_batch,
)

BATCH = 2000

# Clamps keep derived thresholds inside values the strategies behave well at.
SHORT_MIN, SHORT_MAX_CLAMP = 64, 256
MEDIUM_MIN, MEDIUM_MAX_CLAMP = 256, 1024


def percentiles(a: np.ndarray) -> dict:
    ps = [1, 5, 25, 50, 75, 80, 90, 95, 99, 99.5]
    out = {f"P{p:g}": round(float(np.percentile(a, p)), 1) for p in ps}
    out["min"] = int(a.min())
    out["max"] = int(a.max())
    out["mean"] = round(float(a.mean()), 1)
    return out


def derive_thresholds(a: np.ndarray) -> dict:
    short_max = int(np.clip(round(float(np.percentile(a, 80))),
                            SHORT_MIN, SHORT_MAX_CLAMP))
    medium_max = int(np.clip(round(float(np.percentile(a, 99.5))),
                             max(MEDIUM_MIN, short_max * 2), MEDIUM_MAX_CLAMP))
    semantic_max = int(min(short_max * 1.5, medium_max))
    return {"short_max": short_max, "medium_max": medium_max,
            "semantic_max": semantic_max}


def band_shares(a: np.ndarray, t: dict) -> dict:
    n = a.size
    short = int((a <= t["short_max"]).sum())
    long_ = int((a > t["medium_max"]).sum())
    medium = n - short - long_
    return {
        "short": {"count": short, "pct": round(100 * short / n, 2)},
        "medium": {"count": medium, "pct": round(100 * medium / n, 2)},
        "long": {"count": long_, "pct": round(100 * long_ / n, 2)},
    }


def measure(lang: str, sample: int | None) -> dict:
    path = CFG.passages_path(lang)
    if not path.exists():
        raise FileNotFoundError(
            f"missing passages for {lang!r}: {path}\n"
            f"Run: python scripts/extract_subset.py --languages {lang}")

    pf = pq.ParquetFile(path)
    counts: list[int] = []
    chars: list[int] = []
    t0 = time.perf_counter()
    for batch in pf.iter_batches(batch_size=BATCH, columns=["text"]):
        texts = batch.column("text").to_pylist()
        counts.extend(count_tokens_batch(texts))
        chars.extend(len(t or "") for t in texts)
        if sample and len(counts) >= sample:
            break
    elapsed = time.perf_counter() - t0

    a = np.asarray(counts, dtype=np.int32)
    c = np.asarray(chars, dtype=np.int32)
    thresholds = derive_thresholds(a)

    stats = {
        "lang": lang,
        "name": LANGUAGES[lang].name,
        "script": LANGUAGES[lang].script,
        "passages_measured": int(a.size),
        "tokens": percentiles(a),
        "chars": percentiles(c),
        # The headline cross-language number: how much text one token buys.
        "chars_per_token": round(float(c.sum() / max(a.sum(), 1)), 2),
        "thresholds": thresholds,
        "bands": band_shares(a, thresholds),
        "measure_seconds": round(elapsed, 1),
    }

    t = thresholds
    print(f"  passages        : {a.size:,}")
    print(f"  tokens          : median {stats['tokens']['P50']:.0f}  "
          f"P80 {stats['tokens']['P80']:.0f}  P99.5 {stats['tokens']['P99.5']:.0f}  "
          f"max {stats['tokens']['max']}")
    print(f"  chars per token : {stats['chars_per_token']}")
    print(f"  -> thresholds   : short<={t['short_max']}  "
          f"medium<={t['medium_max']}  semantic_max={t['semantic_max']}")
    b = stats["bands"]
    print(f"  -> band shares  : short {b['short']['pct']}%  "
          f"medium {b['medium']['pct']}%  long {b['long']['pct']}%")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", default=None,
                    help="comma-separated codes (default: all extracted)")
    ap.add_argument("--sample", type=int, default=None,
                    help="measure only the first N passages per language")
    args = ap.parse_args()

    if args.languages:
        codes = [c.strip().lower() for c in args.languages.split(",") if c.strip()]
    else:
        codes = [c for c in CFG.languages if CFG.passages_path(c).exists()]
    if not codes:
        print("ERROR: no extracted passage files found. Run "
              "scripts/extract_subset.py first.", file=sys.stderr)
        return 1

    print("=" * 70)
    print("MEASURE PASSAGE TOKEN LENGTHS")
    print("=" * 70)
    print(f"Tokenizer : {MODEL_NAME}")
    print(f"Languages : {', '.join(codes)}")
    print()

    results = {}
    for code in codes:
        print(f"[{code}] {LANGUAGES[code].name} ({LANGUAGES[code].script})")
        try:
            results[code] = measure(code, args.sample)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}", file=sys.stderr)
        print()

    if not results:
        return 1

    # The comparison that justifies per-language thresholds.
    print("-" * 70)
    print("Cross-language comparison (same documents, different scripts):")
    print(f"{'lang':<6}{'median tok':>12}{'chars/tok':>12}{'short_max':>12}")
    for code, s in results.items():
        print(f"{code:<6}{s['tokens']['P50']:>12.0f}"
              f"{s['chars_per_token']:>12.2f}{s['thresholds']['short_max']:>12}")

    if len(results) > 1:
        med = {c: s["tokens"]["P50"] for c, s in results.items()}
        lo = min(med, key=med.get)
        hi = max(med, key=med.get)
        if med[lo] > 0:
            print(f"\n{hi} passages need {med[hi] / med[lo]:.2f}x the tokens of "
                  f"{lo} for equivalent content — which is why one shared "
                  f"threshold set would misroute at least one language.")

    out = CFG.root / "docs" / "chunking_length_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tokenizer": MODEL_NAME,
        "add_special_tokens": False,
        "languages": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"Wrote {out.relative_to(CFG.root)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
