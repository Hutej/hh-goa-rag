"""Phase 3A: generate BGE-M3 embeddings for one chunking strategy's chunks.

Streams ``data/processed/chunks/{strategy}.parquet`` row-group by row-group,
embeds the ``text`` field in batches, and writes:

    data/processed/embeddings/{strategy}/
        embeddings.npy      # memory-mapped float32 array (n_chunks, 1024)
        mapping.parquet     # compact embedding_index -> chunk metadata
        progress.json       # resume state (next_index, config)
        run_report.json     # measured performance for the run

Output format — memory-mapped NumPy ``.npy``:
* The full (n_chunks, 1024) float32 array is preallocated on disk and written
  row-by-row via ``np.memmap`` (mode='r+'). This avoids holding all embeddings
  in a Python list, supports random access by index for later Qdrant indexing,
  and is the natural format for large sequential vector storage (the task's
  stated preference). float32 on disk keeps the format stable across CPU/GPU
  runs (the model runs in fp16 on CUDA, fp32 on CPU, but encode_batch returns
  float32 numpy either way).
* Mapping is a separate compact parquet (embedding_index + the 7 required
  metadata fields) so chunk metadata and vectors stay separate, ready for
  Qdrant upsert. The invariant embedding[i] <-> mapping[i] holds by
  construction (both written in chunk-parquet row order, same index).

Resume mechanism:
* ``progress.json`` records ``next_index`` (number of chunk rows already
  embedded). On ``--resume`` the script reads next_index, reopens the
  preallocated memmap, seeks the chunk parquet past the completed rows, and
  continues writing at row next_index. Deterministic because chunk-parquet
  row order is fixed and the memmap is indexed by that order. No partial
  output is deleted.

Embedding input: the chunk ``text`` ONLY. No query/answer/text_en/metadata
is concatenated — the corpus vector represents the chunk itself.

Automatic batch-size reduction: on a CUDA OOM during encode, the batch size
is halved and the failed batch retried (down to a minimum of 1). Repeated
OOM at batch_size 1 is reported, not hidden.

Usage:
    venv/bin/python scripts/embed_chunks.py --strategy adaptive --batch-size 16
    venv/bin/python scripts/embed_chunks.py --strategy adaptive --resume
    venv/bin/python scripts/embed_chunks.py --strategy adaptive --limit 100 --batch-size 8
    venv/bin/python scripts/embed_chunks.py --strategy fixed --device cpu --dtype float32
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.rag.embeddings import (  # noqa: E402
    EMBED_DIM, MODEL_NAME, gpu_info, load_embedder, select_device,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
OUT_ROOT = PROJECT_ROOT / "data" / "processed" / "embeddings"

# Metadata fields kept in the mapping (the 7 required + a couple useful for
# retrieval; deliberately compact — not the whole chunk record).
MAPPING_FIELDS = [
    "embedding_index", "chunk_id", "document_id", "query_id",
    "chunk_index", "chunk_strategy", "language", "is_selected",
]
MAPPING_SCHEMA = pa.schema([
    ("embedding_index", pa.int64()),
    ("chunk_id", pa.string()),
    ("document_id", pa.string()),
    ("query_id", pa.int64()),
    ("chunk_index", pa.int32()),
    ("chunk_strategy", pa.string()),
    ("language", pa.string()),
    ("is_selected", pa.int8()),
])

MIN_BATCH = 1


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _human(n: int) -> str:
    x = float(n)
    for u in ("B", "KiB", "MiB", "GiB"):
        if x < 1024 or u == "GiB":
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{n} B"


def validate_chunk_parquet(path: Path, strategy: str) -> tuple[int, list[str]]:
    """Validate the chunk parquet before embedding. Returns (n_rows, columns)."""
    assert path.exists(), f"missing chunk parquet: {path}"
    pf = pq.ParquetFile(path)
    cols = pf.schema_arrow.names
    required = {"chunk_id", "text", "chunk_strategy"}
    missing = required - set(cols)
    assert not missing, f"chunk parquet missing required columns: {missing}"
    # no empty text, strategy matches (scan cheaply with a column projection)
    n_rows = pf.metadata.num_rows
    empty = 0
    bad_strategy = 0
    # stream to keep RAM bounded
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["text", "chunk_strategy"])
        texts = tbl.column("text").to_pylist()
        strs = tbl.column("chunk_strategy").to_pylist()
        for t, s in zip(texts, strs):
            if not t or not str(t).strip():
                empty += 1
            if s != strategy:
                bad_strategy += 1
    if empty:
        print(f"  WARNING: {empty} chunks have empty text (will embed as zero/CLS)")
    if bad_strategy:
        raise ValueError(
            f"{bad_strategy} chunks have chunk_strategy != '{strategy}' "
            f"— refusing to embed the wrong strategy")
    return n_rows, cols


def build_mapping(chunk_path: Path, n_rows: int) -> list[dict]:
    """Build the complete embedding_index -> metadata mapping for the first
    ``n_rows`` rows of the chunk parquet, in parquet row order.

    The mapping is rebuilt (not accumulated) so it stays complete after a
    resume — embedding rows written in a previous run are still linked.
    Cheap because it carries only the 7 metadata fields, no vectors.
    """
    pf = pq.ParquetFile(chunk_path)
    rows: list[dict] = []
    cur = 0
    for rg in range(pf.metadata.num_row_groups):
        if cur >= n_rows:
            break
        tbl = pf.read_row_group(rg, columns=[
            "chunk_id", "document_id", "query_id", "chunk_index",
            "chunk_strategy", "language", "is_selected"])
        rids = tbl.column("chunk_id").to_pylist()
        dids = tbl.column("document_id").to_pylist()
        qids = tbl.column("query_id").to_pylist()
        cidxs = tbl.column("chunk_index").to_pylist()
        cstrs = tbl.column("chunk_strategy").to_pylist()
        langs = tbl.column("language").to_pylist()
        sels = tbl.column("is_selected").to_pylist()
        take = min(tbl.num_rows, n_rows - cur)
        for k in range(take):
            rows.append({
                "embedding_index": cur,
                "chunk_id": rids[k],
                "document_id": dids[k],
                "query_id": qids[k],
                "chunk_index": cidxs[k],
                "chunk_strategy": cstrs[k],
                "language": langs[k],
                "is_selected": sels[k],
            })
            cur += 1
    return rows


def encode_with_oom_retry(model, texts, batch_size, dtype, device):
    """Encode with automatic batch-size halving on CUDA OOM. Returns
    (embeddings np.ndarray, final_batch_size)."""
    bs = batch_size
    last_exc = None
    while bs >= MIN_BATCH:
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            emb = model.encode(texts, batch_size=bs,
                               normalize_embeddings=True,
                               convert_to_numpy=True,
                               show_progress_bar=False)
            return np.asarray(emb, dtype=np.float32), bs
        except torch.cuda.OutOfMemoryError as e:
            last_exc = e
            if bs == MIN_BATCH:
                # don't hide it
                raise
            print(f"  CUDA OOM at batch_size={bs}; halving to {bs//2} and retrying",
                  flush=True)
            torch.cuda.empty_cache()
            bs //= 2
        except RuntimeError as e:
            # some torch versions raise plain RuntimeError for OOM
            if "out of memory" in str(e).lower():
                last_exc = e
                if bs == MIN_BATCH:
                    raise
                print(f"  CUDA OOM at batch_size={bs}; halving to {bs//2} and retrying",
                      flush=True)
                torch.cuda.empty_cache()
                bs //= 2
            else:
                raise
    raise last_exc  # unreachable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", required=True,
                    choices=["fixed", "semantic", "adaptive"])
    ap.add_argument("--batch-size", type=int, default=16,
                    help="encode batch size (default 16, conservative for 4GB GPUs)")
    ap.add_argument("--dtype", default=None,
                    choices=["float16", "float32", "bfloat16"],
                    help="inference dtype (default: fp16 cuda, fp32 cpu)")
    ap.add_argument("--device", default=None,
                    choices=["cuda", "cpu"],
                    help="override device (default: auto)")
    ap.add_argument("--limit", type=int, default=0,
                    help="embed only the first N chunks (test mode; 0 = all)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from progress.json without recomputing")
    args = ap.parse_args()

    strategy = args.strategy
    chunk_path = CHUNK_DIR / f"{strategy}.parquet"
    out_dir = OUT_ROOT / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npy"
    map_path = out_dir / "mapping.parquet"
    prog_path = out_dir / "progress.json"
    report_path = out_dir / "run_report.json"

    # ---- input validation ----
    print(f"=== embedding strategy: {strategy} ===", flush=True)
    print(f"input: {chunk_path}", flush=True)
    n_rows, cols = validate_chunk_parquet(chunk_path, strategy)

    # ---- resume state (read BEFORE computing total so the memmap shape
    # matches the original run: if a limit was used originally, the on-disk
    # array has `total=limit` rows; resuming without --limit would otherwise
    # compute total=n_rows and mis-open the memmap) ----
    next_index = 0
    stored_total = None
    if args.resume and prog_path.exists():
        prog = json.loads(prog_path.read_text())
        next_index = prog.get("next_index", 0)
        stored_total = prog.get("total")
        print(f"  resume: prior run total={stored_total}, "
              f"next_index={next_index}", flush=True)
    elif args.resume:
        print("  resume requested but no progress.json; starting fresh",
              flush=True)

    # total = explicit limit, else the prior run's stored total (resume),
    # else the full file size.
    if args.limit:
        total = min(args.limit, n_rows)
    elif stored_total is not None:
        total = min(stored_total, n_rows)
    else:
        total = n_rows
    print(f"chunks to embed: {total} (of {n_rows} in file)", flush=True)

    if args.resume and prog_path.exists():
        if next_index > total:
            print(f"  resume: next_index {next_index} > total {total}; "
                  f"starting fresh", flush=True)
            next_index = 0
        elif next_index > 0:
            print(f"  resume: skipping first {next_index} chunks, "
                  f"continuing from index {next_index}", flush=True)

    remaining = total - next_index
    if remaining <= 0:
        print(f"  nothing to do (already completed {next_index}/{total})",
              flush=True)
    else:
        # ---- device + model ----
        device = select_device(args.device)
        dtype = args.dtype or ("float16" if device == "cuda" else "float32")
        print(f"loading {MODEL_NAME} on {device} ({dtype})...", flush=True)
        t0 = time.time()
        model, loaded_device, loaded_dtype = load_embedder(args.device, args.dtype)
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        info = gpu_info(loaded_device)
        print(f"  device: {loaded_device}", flush=True)
        if "gpu_name" in info:
            print(f"  GPU: {info['gpu_name']} (CUDA {info.get('cuda_version')}, "
                  f"cap {info.get('capability')})", flush=True)
        print(f"  model: {MODEL_NAME}", flush=True)
        print(f"  embedding dim: {EMBED_DIM}", flush=True)
        print(f"  dtype: {loaded_dtype}", flush=True)
        print(f"  batch size: {args.batch_size}", flush=True)

        # ---- preallocate memmap for the full run (or limit) ----
        # If resuming, the memmap already exists with the right shape.
        if next_index == 0 or not emb_path.exists():
            mmap = np.lib.format.open_memmap(
                emb_path, mode="w+", dtype=np.float32,
                shape=(total, EMBED_DIM))
        else:
            # reopen existing (header-aware) and verify shape matches the run
            mmap = np.load(emb_path, mmap_mode="r+")
            if mmap.shape != (total, EMBED_DIM):
                raise RuntimeError(
                    f"resume shape mismatch: on-disk {mmap.shape} vs "
                    f"expected {(total, EMBED_DIM)} (progress total={total}). "
                    f"Delete the output dir or run with --limit {total} to "
                    f"match the prior run.")

        # ---- stream chunks + embed + write ----
        # Stream row-group by row-group (bounded RAM). For each row group we
        # read its rows (text + the 7 mapping metadata fields) into flat lists,
        # then batch-encode with a clean while-index. Resume skips whole rows
        # before next_index by tracking a global cursor.
        pf = pq.ParquetFile(chunk_path)
        rg_sizes = [pf.metadata.row_group(i).num_rows
                    for i in range(pf.metadata.num_row_groups)]
        batch_size = args.batch_size
        final_batch_size = batch_size
        mapping_rows: list[dict] = []

        t_start = time.time()
        encode_time = 0.0
        written = 0
        cur_index = 0  # global chunk row index (matches embedding row index)
        done = False
        for rg in range(pf.metadata.num_row_groups):
            if done:
                break
            rg_size = rg_sizes[rg]
            rg_start = cur_index
            rg_end = cur_index + rg_size
            # fast-skip a whole row group if entirely before next_index
            if rg_end <= next_index:
                cur_index = rg_end
                continue
            # read this row group's needed columns
            tbl = pf.read_row_group(rg, columns=[
                "chunk_id", "document_id", "query_id", "chunk_index",
                "chunk_strategy", "language", "is_selected", "text"])
            rids = tbl.column("chunk_id").to_pylist()
            dids = tbl.column("document_id").to_pylist()
            qids = tbl.column("query_id").to_pylist()
            cidxs = tbl.column("chunk_index").to_pylist()
            cstrs = tbl.column("chunk_strategy").to_pylist()
            langs = tbl.column("language").to_pylist()
            sels = tbl.column("is_selected").to_pylist()
            texts = tbl.column("text").to_pylist()
            # local offset within this row group where processing starts
            local = 0
            if cur_index < next_index:
                local = next_index - cur_index
                # advance the global cursor past the rows we are skipping
                # (resume), so buf_idx / cur_index align with embedding rows.
                cur_index += local
            if cur_index >= total:
                done = True
                break
            j = local
            while j < rg_size and cur_index < total:
                buf_idx = cur_index
                n_take = min(batch_size, rg_size - j, total - buf_idx)
                batch_texts = [texts[k] if texts[k] is not None else ""
                               for k in range(j, j + n_take)]
                te = time.time()
                emb, used_bs = encode_with_oom_retry(
                    model, batch_texts, batch_size, loaded_dtype, loaded_device)
                encode_time += time.time() - te
                final_batch_size = used_bs
                if used_bs < batch_size:
                    # an OOM forced a smaller effective batch; keep it smaller
                    batch_size = used_bs
                # write embeddings to memmap at rows [buf_idx, buf_idx+n)
                mmap[buf_idx:buf_idx + emb.shape[0]] = emb
                written += emb.shape[0]
                cur_index += emb.shape[0]
                j += emb.shape[0]
                # periodic flush + progress
                if written % (batch_size * 20) < batch_size:
                    mmap.flush()
                    prog_path.write_text(json.dumps({
                        "strategy": strategy,
                        "next_index": cur_index,
                        "total": total,
                        "model": MODEL_NAME,
                        "embed_dim": EMBED_DIM,
                        "dtype": loaded_dtype,
                        "device": loaded_device,
                        "batch_size": final_batch_size,
                    }, indent=2))
                    print(f"  progress {cur_index}/{total} "
                          f"({time.time()-t_start:.1f}s, "
                          f"peak_rss={peak_rss_mb():.0f} MB)", flush=True)
            # NOTE: do NOT jump cur_index to rg_end here. cur_index already
            # advances by emb.shape[0] per batch and is the global row cursor
            # (used for resume skip logic AND the embedding row index). A jump
            # would desync the mapping/progress from the actual embedded count
            # when --limit stops mid-row-group.
        mmap.flush()
        # ---- write the COMPLETE mapping for all embedded rows ----
        # The mapping must cover embedding_index 0..cur_index-1 (the
        # embedding[i] <-> mapping[i] invariant), including rows embedded in
        # a previous (resumed) run. We rebuild it from the chunk parquet for
        # exactly that range — it's cheap (metadata only, no vectors) and
        # keeps the mapping complete after every run, resume or fresh.
        print(f"  writing complete mapping for {cur_index} rows...", flush=True)
        mapping_rows = build_mapping(chunk_path, cur_index)
        if mapping_rows:
            mtbl = pa.Table.from_pylist(mapping_rows, schema=MAPPING_SCHEMA)
            pq.write_table(mtbl, map_path, compression="zstd")
        # final progress
        prog_path.write_text(json.dumps({
            "strategy": strategy,
            "next_index": cur_index,
            "total": total,
            "model": MODEL_NAME,
            "embed_dim": EMBED_DIM,
            "dtype": loaded_dtype,
            "device": loaded_device,
            "batch_size": final_batch_size,
        }, indent=2))

        elapsed = time.time() - t_start
        # ---- report ----
        peak_gpu_mb = None
        if loaded_device == "cuda" and torch.cuda.is_available():
            peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6
        report = {
            "strategy": strategy,
            "model": MODEL_NAME,
            "implementation": "sentence-transformers",
            "device": loaded_device,
            "gpu_name": info.get("gpu_name"),
            "cuda_version": info.get("cuda_version"),
            "dtype": loaded_dtype,
            "batch_size_requested": args.batch_size,
            "batch_size_used": final_batch_size,
            "embedding_dim": EMBED_DIM,
            "chunks_embedded": written,
            "total_time_s": round(elapsed, 2),
            "encoding_time_s": round(encode_time, 2),
            "chunks_per_s": round(written / max(elapsed, 1e-9), 1),
            "peak_gpu_memory_mb": round(peak_gpu_mb, 1) if peak_gpu_mb else None,
            "peak_rss_mb": round(peak_rss_mb(), 0),
            "output_embeddings": str(emb_path),
            "output_mapping": str(map_path),
            "output_size_bytes": emb_path.stat().st_size,
            "output_size_human": _human(emb_path.stat().st_size),
            "resumed_from": next_index,
        }
        report_path.write_text(json.dumps(report, indent=2))
        print("\n=== RUN REPORT ===", flush=True)
        print(json.dumps(report, indent=2), flush=True)
        print(f"\nfinal batch size used: {final_batch_size}", flush=True)


if __name__ == "__main__":
    main()
