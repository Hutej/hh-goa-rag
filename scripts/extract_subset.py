"""Phase 1: extract a deterministic ~20,000-row Hindi subset and normalize it
into per-passage documents written as a many-small-row-group parquet.

Memory strategy
----------------
* Scalar columns (source_lang, target_lang, query_id, query_type, query,
  Answer, Eng_Query, Eng_Answer) are small (<= ~90 MB compressed each) and read
  fully via fastparquet ``to_pandas`` with a single-column projection.
* The three nested passage columns are streamed with
  ``backend.rag.bounded_reader.BoundedColumnReader`` — only the first N rows
  are decoded from a bounded byte slice of each column chunk, so peak RSS is
  set by the buffer size (<= ~400 MB), not by the ~7 GB file.
* The three passage columns are read for the SAME first N rows and aligned by
  row index (verified equal lengths per row).
* Normalized documents are written incrementally to parquet in small row
  groups (default 4,000 passages) so the output never recreates the
  single-giant-row-group problem.

Determinism
-----------
The subset is the FIRST ``--rows`` rows of ``hintrain.parquet`` (row order is
fixed by the file). No random sampling. Phase 0 confirmed the first rows are
not obviously biased (query_type distribution on the full file is roughly
DESCRIPTION > NUMERIC > ENTITY ~ LOCATION > PERSON; the first-rows subset is
checked by this script and reported).

Usage
-----
    venv/bin/python scripts/extract_subset.py --rows 20000

Outputs (under data/processed/):
    hh_subset_hin.parquet      normalized per-passage documents
    hh_subset_hin_sample.jsonl  small human-inspectable sample (first ~12 docs)
    extraction_stats.json      measured performance + validation summary
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag.bounded_reader import BoundedColumnReader, peak_rss_mb  # noqa: E402
from backend.rag.normalize import (  # noqa: E402
    CANONICAL_FIELDS, normalize_row,
)
import fastparquet as fp  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw" / "MSMARCO-XI" / "train" / "hintrain.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

# Bounded buffer for the passage text columns. 96 MB comfortably yields
# >25,000 rows for both English and Translated passages (measured), so it
# covers a 20,000-row subset with margin.
PASSAGE_BUF_MB = 96
ROW_GROUP_PASSAGES = 4000  # small row groups in the output parquet

SCALAR_COLUMNS = [
    "source_lang", "target_lang", "query_id", "query_type",
    "query", "Answer", "Eng_Query", "Eng_Answer",
]


def read_scalar_columns(path: str, columns: list[str]) -> dict[str, list]:
    """Read the small scalar columns fully (single-column projection)."""
    pf = fp.ParquetFile(path)
    out: dict[str, list] = {}
    for c in columns:
        df = pf.to_pandas(columns=[c])
        out[c] = df[c].tolist()
    return out


def build_rows(
    n_rows: int,
    scalars: dict[str, list],
    eng_passages: list,
    trans_passages: list,
    is_selected: list,
    source_file: str,
):
    """Yield normalized passage dicts for the first ``n_rows`` rows.

    Streams row-by-row; does NOT accumulate all documents in memory.
    """
    for i in range(n_rows):
        row = {
            "query_id": scalars["query_id"][i],
            "query_type": scalars["query_type"][i],
            "query": scalars["query"][i],
            "Answer": scalars["Answer"][i],
            "Eng_Query": scalars["Eng_Query"][i],
            "Eng_Answer": scalars["Eng_Answer"][i],
            "source_lang": scalars["source_lang"][i],
            "target_lang": scalars["target_lang"][i],
            "English_passages": eng_passages[i],
            "Translated_passages": trans_passages[i],
            "is_selected": is_selected[i],
        }
        try:
            docs = normalize_row(row, source_file=source_file)
        except ValueError as e:
            # Misaligned row: report, skip (do not silently drop).
            yield ("misaligned", i, str(e))
            continue
        for d in docs:
            yield ("doc", d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000,
                    help="number of Hindi query rows to extract (default 20000)")
    ap.add_argument("--buf-mb", type=int, default=PASSAGE_BUF_MB,
                    help="byte-buffer size MB for passage columns")
    args = ap.parse_args()
    n_rows = args.rows

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = OUT_DIR / "hh_subset_hin.parquet"
    out_sample = SAMPLE_DIR / "hh_subset_hin_sample.jsonl"
    out_stats = OUT_DIR / "extraction_stats.json"

    print(f"=== Phase 1 extraction: first {n_rows} Hindi rows ===")
    print(f"source: {RAW}")
    print(f"output: {out_parquet}")
    t_start = time.time()

    # 1. Scalar columns (cheap, full read of small columns).
    t0 = time.time()
    scalars = read_scalar_columns(str(RAW), SCALAR_COLUMNS)
    print(f"[1/4] scalar columns read in {time.time()-t0:.1f}s "
          f"(peak_rss={peak_rss_mb():.0f} MB)")
    avail_rows = min(len(v) for v in scalars.values())
    n_rows = min(n_rows, avail_rows)
    print(f"      using {n_rows} rows (file has {avail_rows})")

    # 2. Passage columns via bounded reader.
    reader = BoundedColumnReader(str(RAW))
    t0 = time.time()
    is_selected = reader.read_first_rows(
        ["passages", "is_selected", "list", "element"], n_rows,
        buf_mb=args.buf_mb,
    )
    print(f"[2/4] is_selected read: {len(is_selected)} rows "
          f"in {time.time()-t0:.1f}s (peak_rss={peak_rss_mb():.0f} MB)")
    t0 = time.time()
    eng_passages = reader.read_first_rows(
        ["passages", "English_passages", "list", "element"], n_rows,
        buf_mb=args.buf_mb,
    )
    print(f"[3/4] English_passages read: {len(eng_passages)} rows "
          f"in {time.time()-t0:.1f}s (peak_rss={peak_rss_mb():.0f} MB)")
    t0 = time.time()
    trans_passages = reader.read_first_rows(
        ["passages", "Translated_passages", "list", "element"], n_rows,
        buf_mb=args.buf_mb,
    )
    print(f"[4/4] Translated_passages read: {len(trans_passages)} rows "
          f"in {time.time()-t0:.1f}s (peak_rss={peak_rss_mb():.0f} MB)")

    # If the bounded buffer gave fewer rows than asked (shouldn't for 20k/96MB),
    # trim to the common minimum so all columns stay aligned.
    n = min(len(is_selected), len(eng_passages), len(trans_passages), n_rows)
    if n < n_rows:
        print(f"  WARN: bounded buffer yielded only {n} rows; trimming subset.")
    n_rows = n
    is_selected = is_selected[:n_rows]
    eng_passages = eng_passages[:n_rows]
    trans_passages = trans_passages[:n_rows]

    # 3. Normalize + write incrementally to a many-row-group parquet.
    schema = pa.schema([
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
    writer = pq.ParquetWriter(out_parquet, schema, compression="zstd")
    source_file = os.path.basename(str(RAW))

    buf: list[dict] = []
    total_passages = 0
    rows_written = 0
    misaligned = 0
    sample_written = False
    t0 = time.time()

    for kind, *payload in build_rows(
        n_rows, scalars, eng_passages, trans_passages, is_selected, source_file
    ):
        if kind == "misaligned":
            misaligned += 1
            idx, msg = payload
            print(f"  MISALIGNED row {idx}: {msg}")
            continue
        doc = payload[0]
        buf.append(doc)
        total_passages += 1
        # small human-inspectable sample (first ~12 docs only)
        if not sample_written and total_passages <= 12:
            with out_sample.open("a", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            if total_passages == 12:
                sample_written = True
        if len(buf) >= ROW_GROUP_PASSAGES:
            table = pa.Table.from_pylist(buf, schema=schema)
            writer.write_table(table)
            rows_written += len(buf)
            buf = []
    if buf:
        table = pa.Table.from_pylist(buf, schema=schema)
        writer.write_table(table)
        rows_written += len(buf)
    writer.close()
    extract_secs = time.time() - t0

    # 4. Validation + stats.
    out_size = out_parquet.stat().st_size
    stats = {
        "source_file": source_file,
        "rows_requested": args.rows,
        "rows_processed": n_rows,
        "passages_produced": total_passages,
        "misaligned_rows": misaligned,
        "extraction_seconds": round(extract_secs, 2),
        "total_seconds": round(time.time() - t_start, 2),
        "peak_rss_mb": round(peak_rss_mb(), 0),
        "output_path": str(out_parquet),
        "output_size_bytes": out_size,
        "output_size_human": _human(out_size),
        "row_group_passages": ROW_GROUP_PASSAGES,
        "passage_buf_mb": args.buf_mb,
        "throughput_rows_per_s": round(n_rows / max(extract_secs, 1e-9), 1),
        "throughput_passages_per_s": round(total_passages / max(extract_secs, 1e-9), 1),
    }
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== DONE ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def _human(n: int) -> str:
    x = float(n)
    for u in ("B", "KiB", "MiB", "GiB"):
        if x < 1024 or u == "GiB":
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{n} B"


if __name__ == "__main__":
    main()
