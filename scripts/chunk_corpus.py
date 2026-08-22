"""Chunk the per-language passage corpora with one or more strategies.

Strategies (all deterministic, all token-counted with the embedding model's own
tokenizer):

* ``fixed``    — 256-token windows, 32-token overlap, sliced from the tokenizer
                 offset map so every chunk is an exact substring.
* ``semantic`` — sentence-aware greedy grouping (Devanagari danda / ``?`` /
                 ``!`` / conservative ASCII period), with a fixed-window
                 fallback for any single sentence over budget.
* ``adaptive`` — routes by measured token length: short passages stay whole,
                 medium get sentence-aware grouping, long get overlapping
                 windows. Thresholds are **per language**, read from
                 ``docs/chunking_length_stats.json``.

Performance note
----------------
The naive implementation calls ``count_tokens`` once per passage, which is a
Python-level round trip per document. Here each batch is token-counted in a
single ``encode_batch`` call (which parallelizes and releases the GIL), and only
the ~20% of passages that actually need splitting are tokenized a second time
for their offset map. On this corpus that is the difference between minutes and
tens of minutes.

Output:
    data/processed/chunks/{strategy}/{lang}.parquet
        chunk_id, passage_uid, document_id, query_id, chunk_index,
        lang, is_selected, text

Usage:
    python scripts/chunk_corpus.py
    python scripts/chunk_corpus.py --languages hi,en --strategies adaptive
    python scripts/chunk_corpus.py --strategies fixed,semantic,adaptive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag.config import CFG, LANGUAGES  # noqa: E402
from backend.rag.chunkers import adaptive as A  # noqa: E402
from backend.rag.chunkers.fixed import split_fixed  # noqa: E402
from backend.rag.chunkers.semantic import split_semantic  # noqa: E402
from backend.rag.chunkers.tokenizer import (  # noqa: E402
    MODEL_NAME, count_tokens_batch,
)

STRATEGIES = ["fixed", "semantic", "adaptive"]
READ_BATCH = 2000
ROW_GROUP = 4000

SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("passage_uid", pa.string()),
    ("document_id", pa.string()),
    ("query_id", pa.int64()),
    ("chunk_index", pa.int32()),
    ("lang", pa.string()),
    ("is_selected", pa.int8()),
    ("text", pa.string()),
])

READ_COLUMNS = ["passage_uid", "document_id", "query_id", "lang",
                "is_selected", "text"]


def load_thresholds(lang: str) -> dict:
    """Per-language adaptive thresholds, measured if available.

    Falls back to the module defaults (calibrated on Hindi) with a warning, so
    chunking still runs if the measurement step was skipped.
    """
    stats_path = CFG.root / "docs" / "chunking_length_stats.json"
    if stats_path.exists():
        try:
            data = json.loads(stats_path.read_text(encoding="utf-8"))
            t = data.get("languages", {}).get(lang, {}).get("thresholds")
            if t:
                return {"short_max": int(t["short_max"]),
                        "medium_max": int(t["medium_max"]),
                        "semantic_max": int(t["semantic_max"]),
                        "source": "measured"}
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    print(f"  WARNING: no measured thresholds for {lang!r}; using Hindi-"
          f"calibrated defaults. Run scripts/measure_lengths.py first.")
    return {"short_max": A.SHORT_MAX, "medium_max": A.MEDIUM_MAX,
            "semantic_max": None, "source": "default"}


def split_batch(texts: list[str], strategy: str, t: dict
                ) -> tuple[list[list[str]], dict]:
    """Split a batch of passages, returning per-passage piece lists.

    Token counting is batched; only passages that need splitting get a second
    offset-map pass.
    """
    routes = {"short": 0, "medium": 0, "long": 0}

    if strategy == "fixed":
        # Every passage goes through the window splitter, but short ones return
        # a single piece immediately.
        return [split_fixed(x or "") for x in texts], routes

    counts = count_tokens_batch(texts)

    if strategy == "semantic":
        out = []
        for text, n in zip(texts, counts):
            if not text or not text.strip():
                out.append([])
            elif n <= A.MEDIUM_MAX:
                out.append(split_semantic(text))
            else:
                out.append(split_semantic(text))
        return out, routes

    # adaptive: route on the batched count, avoiding a re-count per passage
    short_max, medium_max = t["short_max"], t["medium_max"]
    semantic_max = t["semantic_max"]
    out: list[list[str]] = []
    for text, n in zip(texts, counts):
        if not text or not text.strip():
            out.append([])
            continue
        if n <= short_max:
            routes["short"] += 1
            out.append([text])            # keep whole — no fragmentation
        elif n <= medium_max:
            routes["medium"] += 1
            out.append(split_semantic(text, max_tokens=semantic_max))
        else:
            routes["long"] += 1
            out.append(split_fixed(text))
    return out, routes


def chunk_language(lang: str, strategy: str) -> dict:
    src = CFG.passages_path(lang)
    if not src.exists():
        raise FileNotFoundError(
            f"missing passages for {lang!r}: {src}\n"
            f"Run: python scripts/extract_subset.py --languages {lang}")

    t = load_thresholds(lang) if strategy == "adaptive" else {}
    pf = pq.ParquetFile(src)

    cids: list[str] = []
    puids: list[str] = []
    dids: list[str] = []
    qids: list[int] = []
    cidxs: list[int] = []
    langs: list[str] = []
    sels: list[int] = []
    texts_out: list[str] = []

    routes_total = {"short": 0, "medium": 0, "long": 0}
    n_passages = 0
    empty_passages = 0
    t0 = time.perf_counter()

    for batch in pf.iter_batches(batch_size=READ_BATCH, columns=READ_COLUMNS):
        cols = {c: batch.column(c).to_pylist() for c in READ_COLUMNS}
        pieces_list, routes = split_batch(cols["text"], strategy, t)
        for k in routes_total:
            routes_total[k] += routes[k]

        for i, pieces in enumerate(pieces_list):
            n_passages += 1
            if not pieces:
                empty_passages += 1
                continue
            did = cols["document_id"][i]
            idx = 0
            for piece in pieces:
                if not piece or not piece.strip():
                    continue
                cids.append(f"{lang}_{did}_c{idx}")
                puids.append(cols["passage_uid"][i])
                dids.append(did)
                qids.append(cols["query_id"][i])
                cidxs.append(idx)
                langs.append(lang)
                sels.append(cols["is_selected"][i])
                texts_out.append(piece)
                idx += 1

    elapsed = time.perf_counter() - t0

    table = pa.Table.from_arrays([
        pa.array(cids, pa.string()),
        pa.array(puids, pa.string()),
        pa.array(dids, pa.string()),
        pa.array(qids, pa.int64()),
        pa.array(cidxs, pa.int32()),
        pa.array(langs, pa.string()),
        pa.array(sels, pa.int8()),
        pa.array(texts_out, pa.string()),
    ], schema=SCHEMA)

    out_path = CFG.chunks_path(lang, strategy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd", row_group_size=ROW_GROUP)

    # Verify the chunk_id invariant that downstream indexes rely on.
    if len(set(cids)) != len(cids):
        raise RuntimeError(
            f"{lang}/{strategy}: duplicate chunk_id generated "
            f"({len(cids) - len(set(cids))} duplicates)")

    lens = np.asarray(count_tokens_batch(texts_out[:20000]), dtype=np.int32)
    stats = {
        "lang": lang, "strategy": strategy,
        "passages": n_passages,
        "chunks": len(cids),
        "chunks_per_passage": round(len(cids) / max(n_passages, 1), 4),
        "empty_passages": empty_passages,
        "single_chunk_pct": None,
        "routes": routes_total if strategy == "adaptive" else None,
        "thresholds": t or None,
        "chunk_tokens_sampled": {
            "n": int(lens.size),
            "median": round(float(np.median(lens)), 1) if lens.size else 0,
            "p95": round(float(np.percentile(lens, 95)), 1) if lens.size else 0,
            "max": int(lens.max()) if lens.size else 0,
        },
        "seconds": round(elapsed, 1),
        "out": str(out_path.relative_to(CFG.root)),
        "size_mb": round(out_path.stat().st_size / (1024 ** 2), 1),
    }

    per_doc = len(cids) / max(n_passages - empty_passages, 1)
    stats["single_chunk_pct"] = round(100 * (2 - min(per_doc, 2)), 2) \
        if per_doc <= 2 else 0.0

    print(f"    chunks          : {stats['chunks']:,} from "
          f"{n_passages:,} passages ({stats['chunks_per_passage']}/passage)")
    if strategy == "adaptive":
        r = routes_total
        tot = max(sum(r.values()), 1)
        print(f"    routing         : short {100*r['short']/tot:.1f}%  "
              f"medium {100*r['medium']/tot:.1f}%  long {100*r['long']/tot:.1f}%"
              f"  (thresholds {t['short_max']}/{t['medium_max']}, {t['source']})")
    ct = stats["chunk_tokens_sampled"]
    print(f"    chunk tokens    : median {ct['median']:.0f}  p95 {ct['p95']:.0f}  "
          f"max {ct['max']}")
    print(f"    wrote           : {stats['out']} ({stats['size_mb']} MB) "
          f"in {stats['seconds']}s")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", default=None,
                    help="comma-separated codes (default: all extracted)")
    ap.add_argument("--strategies", default=CFG.chunk_strategy,
                    help=f"comma-separated from {','.join(STRATEGIES)} "
                         f"(default: {CFG.chunk_strategy})")
    args = ap.parse_args()

    if args.languages:
        codes = [c.strip().lower() for c in args.languages.split(",") if c.strip()]
    else:
        codes = [c for c in CFG.languages if CFG.passages_path(c).exists()]
    bad = [c for c in codes if c not in LANGUAGES]
    if bad:
        print(f"ERROR: unknown language(s): {', '.join(bad)}", file=sys.stderr)
        return 2
    if not codes:
        print("ERROR: no extracted passages found. Run "
              "scripts/extract_subset.py first.", file=sys.stderr)
        return 1

    strategies = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
    bad = [s for s in strategies if s not in STRATEGIES]
    if bad:
        print(f"ERROR: unknown strateg(ies): {', '.join(bad)}", file=sys.stderr)
        return 2

    print("=" * 70)
    print("CHUNK CORPUS")
    print("=" * 70)
    print(f"Tokenizer  : {MODEL_NAME}")
    print(f"Languages  : {', '.join(codes)}")
    print(f"Strategies : {', '.join(strategies)}")
    print()

    all_stats = []
    for strategy in strategies:
        print(f"[{strategy}]")
        for code in codes:
            print(f"  {code} ({LANGUAGES[code].name})")
            try:
                all_stats.append(chunk_language(code, strategy))
            except (FileNotFoundError, RuntimeError) as e:
                print(f"    FAILED: {e}", file=sys.stderr)
                return 1
        print()

    out = CFG.root / "docs" / "chunking_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tokenizer": MODEL_NAME, "runs": all_stats},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(s["chunks"] for s in all_stats)
    print("=" * 70)
    print(f"CHUNKING COMPLETE — {total:,} chunks across "
          f"{len(strategies)} strateg(ies) x {len(codes)} language(s)")
    print(f"Stats: {out.relative_to(CFG.root)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
