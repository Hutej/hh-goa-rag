"""Phase 2: run all three chunkers over the real 199,590-passage Hindi subset.

Streams ``data/processed/hh_subset_hin.parquet`` row-group by row-group (so
peak RAM stays bounded on the ~1.5 GiB-free host), applies each chunking
strategy, and writes one many-small-row-group parquet per strategy:

    data/processed/chunks/fixed.parquet
    data/processed/chunks/semantic.parquet
    data/processed/chunks/adaptive.parquet

Per strategy it measures REAL statistics (no fabrication):

  * input passages, output chunks
  * avg / median / min / max chunks per passage
  * avg / median / P50 / P90 / P95 / max TOKEN count per chunk
  * number of chunks exceeding the configured maximum
  * number of empty chunks
  * processing time, throughput (passages/s, chunks/s), output file size
  * peak RSS
  * % documents split into multiple chunks, % remaining as one chunk

Token counts are the chunk's content-token count computed from the ORIGINAL
passage encoding (NOT by re-tokenizing the chunk text — re-tokenizing gives a
different count due to BPE context). To keep memory bounded we do not keep
all chunk token counts in memory: we maintain running sums + a reservoir of
recent counts for percentiles via a sorted-sample approximation is avoided in
favour of a simple two-pass-free exact approach: we DO keep all per-chunk
token counts in a flat list (ints, ~one int per chunk — fixed produces the
most chunks; even at ~2x passages that is <400k ints = a few MB). This is
cheap and exact, so percentiles are real.

Outputs:
    data/processed/chunks/{fixed,semantic,adaptive}.parquet
    docs/chunking_stats.json
    docs/chunking_report.md

Usage:
    venv/bin/python scripts/chunk_subset.py
"""

from __future__ import annotations

import json
import math
import resource
import statistics
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag.chunkers.adaptive import AdaptiveChunker  # noqa: E402
from backend.rag.chunkers.base import CHUNK_FIELDS  # noqa: E402
from backend.rag.chunkers.fixed import FixedChunker  # noqa: E402
from backend.rag.chunkers.semantic import SemanticChunker  # noqa: E402
from backend.rag.chunkers.tokenizer import (  # noqa: E402
    count_tokens, encode_with_offsets, get_tokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSET = PROJECT_ROOT / "data" / "processed" / "hh_subset_hin.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
DOCS = PROJECT_ROOT / "docs"

# Small output row groups (NOT a giant single row group).
ROW_GROUP_CHUNKS = 4000

# pyarrow schema for chunk parquet. The chunk fields in canonical order;
# types match the normalized subset where possible (int64 query_id, int8
# is_selected, bool answerable) with chunk-specific fields added.
CHUNK_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("chunk_index", pa.int32()),
    ("chunk_strategy", pa.string()),
    ("document_id", pa.string()),
    ("query_id", pa.int64()),
    ("passage_idx", pa.int32()),
    ("text", pa.string()),
    ("text_en", pa.string()),
    ("query", pa.string()),
    ("query_en", pa.string()),
    ("answer", pa.string()),
    ("answer_en", pa.string()),
    ("language", pa.string()),
    ("source_lang_code", pa.string()),
    ("target_lang_code", pa.string()),
    ("is_selected", pa.int8()),
    ("query_type", pa.string()),
    ("source", pa.string()),
    ("source_file", pa.string()),
    ("answerable", pa.bool_()),
])

# Per-strategy configured maximum (for "chunks exceeding max" reporting).
STRATEGY_MAX_TOKENS = {
    "fixed": 256,        # + overlap-1 tolerance for the merged final window
    "semantic": 384,
    "adaptive": 384,     # medium path caps at 384; long path 256+overlap
}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _pct(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    k = math.ceil(p / 100.0 * len(sorted_vals))
    k = max(1, min(k, len(sorted_vals)))
    return int(sorted_vals[k - 1])


def chunk_token_counts_for_passage(
    passage_text: str, chunk_texts: list[str]
) -> list[int]:
    """Exact content-token counts for a passage's chunks, from ONE encoding.

    Tokenizes ``passage_text`` once, then for each chunk counts how many token
    offset-spans are covered by the chunk's character span. Robust to XLM-R's
    leading/trailing whitespace trimming. Falls back to re-counting a chunk if
    it can't be located as a substring (rare; only fixed-fallback overlap
    pieces on passages with leading/trailing whitespace quirks).
    """
    if not chunk_texts:
        return []
    if not passage_text:
        return [0] * len(chunk_texts)
    ids, offs = encode_with_offsets(passage_text)
    if not offs:
        return [count_tokens(c) for c in chunk_texts]
    out: list[int] = []
    cursor = 0
    for chunk_text in chunk_texts:
        if not chunk_text:
            out.append(0)
            continue
        try:
            schar = passage_text.index(chunk_text, cursor)
        except ValueError:
            try:
                schar = passage_text.index(chunk_text)
            except ValueError:
                out.append(count_tokens(chunk_text))
                continue
        echar = schar + len(chunk_text)
        cnt = 0
        for (a, b) in offs:
            if a >= schar and b <= echar and a < echar and b > schar:
                cnt += 1
        out.append(cnt if cnt > 0 else count_tokens(chunk_text))
        cursor = schar + 1
    return out


def run_strategy(name: str, chunker, limit_rg: int = 0) -> dict:
    """Run one chunker over the whole subset, write its parquet, return stats.

    ``limit_rg`` > 0 caps the number of row groups processed (smoke test).
    """
    print(f"\n=== chunker: {name} ===")
    pf = pq.ParquetFile(SUBSET)
    n_rg_total = pf.metadata.num_row_groups
    n_rg = min(n_rg_total, limit_rg) if limit_rg else n_rg_total
    out_path = OUT_DIR / f"{name}.parquet"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    writer = pq.ParquetWriter(out_path, CHUNK_SCHEMA, compression="zstd")

    input_passages = 0
    total_chunks = 0
    passages_with_multiple = 0
    passages_with_zero = 0
    empty_chunks = 0
    chunks_over_max = 0
    chunk_token_counts: list[int] = []
    chunks_per_passage: list[int] = []

    buf: list[dict] = []
    t_start = time.time()
    for rg in range(n_rg):
        tbl = pf.read_row_group(rg)
        cols = {c: tbl.column(c).to_pylist() for c in tbl.schema.names}
        nrows = tbl.num_rows
        for r in range(nrows):
            doc = {c: cols[c][r] for c in tbl.schema.names}
            input_passages += 1
            chunks = chunker.chunk(doc)
            nch = len(chunks)
            chunks_per_passage.append(nch)
            if nch == 0:
                passages_with_zero += 1
            if nch > 1:
                passages_with_multiple += 1
            # token counts for all chunks of this passage from ONE encoding
            tcs = chunk_token_counts_for_passage(
                doc["text"], [c["text"] for c in chunks])
            for c, tc in zip(chunks, tcs):
                total_chunks += 1
                if not c["text"] or not c["text"].strip():
                    empty_chunks += 1
                chunk_token_counts.append(tc)
                if tc > STRATEGY_MAX_TOKENS[name]:
                    chunks_over_max += 1
                buf.append(c)
                if len(buf) >= ROW_GROUP_CHUNKS:
                    table = pa.Table.from_pylist(buf, schema=CHUNK_SCHEMA)
                    writer.write_table(table)
                    buf = []
        if (rg + 1) % 10 == 0 or rg == n_rg - 1:
            print(f"  rg {rg+1}/{n_rg}: passages={input_passages} "
                  f"chunks={total_chunks} ({time.time()-t_start:.1f}s, "
                  f"peak_rss={peak_rss_mb():.0f} MB)")
    if buf:
        table = pa.Table.from_pylist(buf, schema=CHUNK_SCHEMA)
        writer.write_table(table)
    writer.close()

    elapsed = time.time() - t_start
    out_size = out_path.stat().st_size

    # sort for percentiles
    stc = sorted(chunk_token_counts)
    scp = sorted(chunks_per_passage)
    stats = {
        "strategy": name,
        "input_passages": input_passages,
        "output_chunks": total_chunks,
        "chunks_per_passage": {
            "avg": round(statistics.mean(chunks_per_passage), 4) if chunks_per_passage else 0,
            "median": int(statistics.median(chunks_per_passage)) if chunks_per_passage else 0,
            "min": int(scp[0]) if scp else 0,
            "max": int(scp[-1]) if scp else 0,
        },
        "chunk_token_count": {
            "avg": round(statistics.mean(chunk_token_counts), 2) if chunk_token_counts else 0,
            "median": int(statistics.median(chunk_token_counts)) if chunk_token_counts else 0,
            "min": int(stc[0]) if stc else 0,
            "max": int(stc[-1]) if stc else 0,
            "p50": _pct(stc, 50),
            "p90": _pct(stc, 90),
            "p95": _pct(stc, 95),
        },
        "chunks_exceeding_max": chunks_over_max,
        "configured_max_tokens": STRATEGY_MAX_TOKENS[name],
        "empty_chunks": empty_chunks,
        "passages_split_into_multiple": passages_with_multiple,
        "pct_split": round(100.0 * passages_with_multiple / input_passages, 3),
        "passages_one_chunk": input_passages - passages_with_multiple - passages_with_zero,
        "pct_one_chunk": round(100.0 * (input_passages - passages_with_multiple - passages_with_zero) / input_passages, 3),
        "passages_with_zero_chunks": passages_with_zero,
        "processing_seconds": round(elapsed, 2),
        "throughput_passages_per_s": round(input_passages / max(elapsed, 1e-9), 1),
        "throughput_chunks_per_s": round(total_chunks / max(elapsed, 1e-9), 1),
        "output_path": str(out_path),
        "output_size_bytes": out_size,
        "output_size_human": _human(out_size),
        "row_group_chunks": ROW_GROUP_CHUNKS,
        "peak_rss_mb": round(peak_rss_mb(), 0),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def _human(n: int) -> str:
    x = float(n)
    for u in ("B", "KiB", "MiB", "GiB"):
        if x < 1024 or u == "GiB":
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{n} B"


def main() -> None:
    assert SUBSET.exists(), f"missing {SUBSET}"
    print("loading BGE-M3 tokenizer...")
    t0 = time.time()
    get_tokenizer()
    print(f"  ready in {time.time()-t0:.1f}s")
    print(f"input: {SUBSET}")
    print(f"row groups: {pq.ParquetFile(SUBSET).metadata.num_row_groups}")

    results = {}
    # optional smoke test: process only the first N row groups
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-rg", type=int, default=0,
                    help="process only the first N row groups (smoke test)")
    args = ap.parse_args()
    for name, chunker in [
        ("fixed", FixedChunker()),
        ("semantic", SemanticChunker()),
        ("adaptive", AdaptiveChunker()),
    ]:
        results[name] = run_strategy(name, chunker, limit_rg=args.limit_rg)

    # machine-readable
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "chunking_stats.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(DOCS / "chunking_report.md", results)
    print(f"\nwrote {DOCS/'chunking_stats.json'}")
    print(f"wrote {DOCS/'chunking_report.md'}")


def _write_report(path: Path, results: dict) -> None:
    lines = [
        "# Phase 2 — Chunking Report",
        "",
        "Measured over the real 199,590-passage Hindi subset "
        "(`data/processed/hh_subset_hin.parquet`). Token counts are BGE-M3 "
        "content tokens (`add_special_tokens=False`). All numbers measured, "
        "not fabricated.",
        "",
        "| Strategy | Input passages | Output chunks | Avg tokens/chunk | "
        "P50 tokens | P90 tokens | P95 tokens | Max tokens | "
        "Chunks > max | Empty | % split | % one-chunk | Time (s) | "
        "Passages/s | Peak RSS (MB) | Size |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["fixed", "semantic", "adaptive"]:
        r = results[name]
        tk = r["chunk_token_count"]
        lines.append(
            f"| {name} | {r['input_passages']} | {r['output_chunks']} | "
            f"{tk['avg']} | {tk['p50']} | {tk['p90']} | {tk['p95']} | "
            f"{tk['max']} | {r['chunks_exceeding_max']} | "
            f"{r['empty_chunks']} | {r['pct_split']} | {r['pct_one_chunk']} | "
            f"{r['processing_seconds']} | {r['throughput_passages_per_s']} | "
            f"{r['peak_rss_mb']} | {r['output_size_human']} |"
        )
    lines += [
        "",
        "## Strategy details",
        "",
    ]
    for name in ["fixed", "semantic", "adaptive"]:
        r = results[name]
        lines += [
            f"### {name}",
            f"- input passages: {r['input_passages']}",
            f"- output chunks: {r['output_chunks']}",
            f"- chunks/passage: avg {r['chunks_per_passage']['avg']}, "
            f"median {r['chunks_per_passage']['median']}, "
            f"min {r['chunks_per_passage']['min']}, "
            f"max {r['chunks_per_passage']['max']}",
            f"- tokens/chunk: avg {r['chunk_token_count']['avg']}, "
            f"median {r['chunk_token_count']['median']}, "
            f"P50 {r['chunk_token_count']['p50']}, "
            f"P90 {r['chunk_token_count']['p90']}, "
            f"P95 {r['chunk_token_count']['p95']}, "
            f"max {r['chunk_token_count']['max']}",
            f"- chunks exceeding configured max "
            f"({r['configured_max_tokens']}): {r['chunks_exceeding_max']}",
            f"- empty chunks: {r['empty_chunks']}",
            f"- passages split into multiple chunks: "
            f"{r['passages_split_into_multiple']} ({r['pct_split']}%)",
            f"- passages kept as one chunk: "
            f"{r['passages_one_chunk']} ({r['pct_one_chunk']}%)",
            f"- passages producing zero chunks: {r['passages_with_zero_chunks']}",
            f"- processing time: {r['processing_seconds']} s",
            f"- throughput: {r['throughput_passages_per_s']} passages/s, "
            f"{r['throughput_chunks_per_s']} chunks/s",
            f"- peak RSS: {r['peak_rss_mb']} MB",
            f"- output: {r['output_path']} ({r['output_size_human']})",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
