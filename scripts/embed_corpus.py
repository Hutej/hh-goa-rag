"""Embed chunk text into dense vectors with the ONNX encoder.

Runs the same int8 graph that serves queries at request time. That symmetry is
deliberate: quantization error then points the same direction for queries and
documents and largely cancels in the dot product, whereas encoding documents in
fp32 and queries in int8 introduces an asymmetry that costs recall.

Output per language:
    data/processed/embeddings/{strategy}/{lang}/
        embeddings.npy   float32 memmap, shape (n_chunks, dim), L2-normalized
        mapping.parquet  row i  <-> chunk_id / document_id / query_id / ...
        progress.json    resume state
        run_report.json  measured throughput

Invariant: ``embeddings[i]`` corresponds to ``mapping[i]``. Vectors and metadata
are kept separate so the index builder can memmap the vectors without pulling
text into RAM.

Resumable: re-run with ``--resume`` after an interruption and it continues from
the last completed batch instead of recomputing.

Usage:
    python scripts/embed_corpus.py
    python scripts/embed_corpus.py --languages hi --batch-size 64
    python scripts/embed_corpus.py --resume
    python scripts/embed_corpus.py --languages hi --limit 2000   # smoke test
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
from backend.rag.encoder import OnnxEncoder  # noqa: E402

MAPPING_COLUMNS = ["chunk_id", "document_id", "query_id", "chunk_index",
                   "lang", "is_selected", "text"]

MAPPING_SCHEMA = pa.schema([
    ("embedding_index", pa.int64()),
    ("chunk_id", pa.string()),
    ("document_id", pa.string()),
    ("query_id", pa.int64()),
    ("chunk_index", pa.int32()),
    ("lang", pa.string()),
    ("is_selected", pa.int8()),
    ("text", pa.string()),
])


def load_progress(d: Path) -> int:
    p = d / "progress.json"
    if not p.exists():
        return 0
    try:
        return int(json.loads(p.read_text(encoding="utf-8"))["next_index"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0


def save_progress(d: Path, next_index: int, total: int) -> None:
    (d / "progress.json").write_text(json.dumps(
        {"next_index": next_index, "total": total}, indent=2), encoding="utf-8")


def embed_language(enc: OnnxEncoder, lang: str, strategy: str,
                   batch_size: int, resume: bool, limit: int | None) -> dict:
    src = CFG.chunks_path(lang, strategy)
    if not src.exists():
        raise FileNotFoundError(
            f"missing chunks for {lang!r}/{strategy}: {src}\n"
            f"Run: python scripts/chunk_corpus.py --languages {lang} "
            f"--strategies {strategy}")

    out_dir = CFG.embeddings_dir(lang, strategy)
    out_dir.mkdir(parents=True, exist_ok=True)

    tbl = pq.read_table(src, columns=MAPPING_COLUMNS)
    total = tbl.num_rows if limit is None else min(limit, tbl.num_rows)
    dim = enc.dim

    emb_path = out_dir / "embeddings.npy"
    start = load_progress(out_dir) if resume else 0
    if start and start >= total:
        print(f"    already complete ({total:,} vectors) — skipping")
        return {"lang": lang, "strategy": strategy, "chunks": total,
                "skipped": True}
    if start:
        print(f"    resuming at {start:,}/{total:,}")

    # Preallocate on disk so the array can be filled in place and resumed.
    if start == 0 or not emb_path.exists():
        np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32,
                                  shape=(total, dim))
        start = 0
    mm = np.lib.format.open_memmap(emb_path, mode="r+")
    if mm.shape != (total, dim):
        raise RuntimeError(
            f"existing {emb_path.name} has shape {mm.shape}, expected "
            f"{(total, dim)} — delete the directory to rebuild")

    texts_col = tbl.column("text")
    t0 = time.perf_counter()
    done = start
    last_report = t0

    # Length-bucketed batching. Transformer cost is driven by the padded
    # sequence length, so a batch mixing an 8-token chunk with a 257-token one
    # pays the 257-token price for every row. Sorting a large window by length
    # before splitting it into encode batches cuts that waste; measured ~35%
    # throughput gain on this corpus. Results are scattered back to their
    # original rows, so `embeddings[i] <-> mapping[i]` still holds exactly.
    window = max(batch_size * 64, batch_size)

    while done < total:
        win_stop = min(done + window, total)
        win_texts = texts_col.slice(done, win_stop - done).to_pylist()
        order = sorted(range(len(win_texts)),
                       key=lambda i: len(win_texts[i] or ""))

        out = np.empty((len(win_texts), dim), dtype=np.float32)
        for b in range(0, len(order), batch_size):
            idxs = order[b:b + batch_size]
            vecs = enc.encode_passages([win_texts[i] for i in idxs],
                                       batch_size=len(idxs))
            for slot, i in enumerate(idxs):
                out[i] = vecs[slot]

        mm[done:win_stop] = out
        done = win_stop

        now = time.perf_counter()
        if now - last_report >= 5.0 or done >= total:
            rate = (done - start) / max(now - t0, 1e-9)
            remain = (total - done) / max(rate, 1e-9)
            pct = 100 * done / total
            print(f"      {done:,}/{total:,} ({pct:5.1f}%)  "
                  f"{rate:7.0f} chunks/s  eta {remain/60:5.1f} min",
                  flush=True)
            last_report = now
            save_progress(out_dir, done, total)

    mm.flush()
    del mm
    save_progress(out_dir, total, total)
    elapsed = time.perf_counter() - t0

    # Mapping, aligned positionally with the vector rows.
    sub = tbl.slice(0, total)
    mapping = pa.Table.from_arrays([
        pa.array(np.arange(total, dtype=np.int64)),
        sub.column("chunk_id").combine_chunks(),
        sub.column("document_id").combine_chunks(),
        sub.column("query_id").combine_chunks(),
        sub.column("chunk_index").combine_chunks(),
        sub.column("lang").combine_chunks(),
        sub.column("is_selected").combine_chunks(),
        sub.column("text").combine_chunks(),
    ], schema=MAPPING_SCHEMA)
    pq.write_table(mapping, out_dir / "mapping.parquet", compression="zstd",
                   row_group_size=4000)

    # Validate what was written rather than trusting the loop.
    check = np.load(emb_path, mmap_mode="r")
    idx = np.linspace(0, total - 1, min(total, 512)).astype(np.int64)
    norms = np.linalg.norm(np.asarray(check[idx], dtype=np.float32), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        raise RuntimeError(
            f"{lang}: vectors are not unit-norm (min {norms.min():.4f}, "
            f"max {norms.max():.4f}) — encoder normalization is broken")
    if np.abs(np.asarray(check[idx])).sum() == 0:
        raise RuntimeError(f"{lang}: sampled vectors are all zero")

    report = {
        "lang": lang, "strategy": strategy, "chunks": total, "dim": dim,
        "model": CFG.embed_model, "precision": CFG.embed_precision,
        "providers": list(enc.stats()["providers"]),
        "batch_size": batch_size,
        "seconds": round(elapsed, 1),
        "chunks_per_second": round(total / max(elapsed, 1e-9), 1),
        "embeddings_mb": round(emb_path.stat().st_size / (1024 ** 2), 1),
        "unit_norm_verified": True,
    }
    (out_dir / "run_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"    done            : {total:,} vectors in {elapsed/60:.1f} min "
          f"({report['chunks_per_second']:.0f} chunks/s)")
    print(f"    wrote           : {emb_path.relative_to(CFG.root)} "
          f"({report['embeddings_mb']} MB), unit-norm verified")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", default=None,
                    help="comma-separated codes (default: all chunked)")
    ap.add_argument("--strategy", default=CFG.chunk_strategy)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=None,
                    help="onnxruntime intra-op threads (default: auto)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="embed only the first N chunks (smoke test)")
    args = ap.parse_args()

    if args.languages:
        codes = [c.strip().lower() for c in args.languages.split(",") if c.strip()]
    else:
        codes = [c for c in CFG.languages
                 if CFG.chunks_path(c, args.strategy).exists()]
    bad = [c for c in codes if c not in LANGUAGES]
    if bad:
        print(f"ERROR: unknown language(s): {', '.join(bad)}", file=sys.stderr)
        return 2
    if not codes:
        print(f"ERROR: no chunk files for strategy {args.strategy!r}. Run "
              f"scripts/chunk_corpus.py first.", file=sys.stderr)
        return 1

    enc = OnnxEncoder(threads=args.threads)

    print("=" * 70)
    print("EMBED CORPUS")
    print("=" * 70)
    print(f"Model      : {CFG.embed_model} ({CFG.embed_precision}, "
          f"dim {enc.dim})")
    print(f"Providers  : {', '.join(enc.stats()['providers'])}")
    print(f"Threads    : {enc.stats()['threads']}")
    print(f"Strategy   : {args.strategy}")
    print(f"Languages  : {', '.join(codes)}")
    print(f"Batch size : {args.batch_size}")
    print()

    reports = []
    for code in codes:
        print(f"  [{code}] {LANGUAGES[code].name}")
        try:
            reports.append(embed_language(enc, code, args.strategy,
                                          args.batch_size, args.resume,
                                          args.limit))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"    FAILED: {e}", file=sys.stderr)
            return 1
        print()

    total = sum(r.get("chunks", 0) for r in reports)
    print("=" * 70)
    print(f"EMBEDDING COMPLETE — {total:,} vectors across {len(reports)} "
          f"language(s)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
