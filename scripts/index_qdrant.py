"""Phase 3B: index Phase 3A embeddings into local Qdrant (dense).

Streams ``data/processed/embeddings/{strategy}/embeddings.npy`` (memmap, sliced)
+ ``chunks/{strategy}.parquet`` (text + metadata, positional) into the strategy's
Qdrant collection as batched upserts. Point id = embedding_index, so re-running is
idempotent. Resume is just ``count()`` — no progress file.

Usage:
    venv/bin/python scripts/index_qdrant.py --strategy adaptive --limit 100 --recreate
    venv/bin/python scripts/index_qdrant.py --strategy adaptive
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.rag.qdrant_index import (  # noqa: E402
    EMBED_DIM, STRATEGIES, collection_name, ensure_collection, get_client,
    stream_points, validate_artifacts,
)


def index(client, name, strategy, start, total, batch_size):
    """Upsert rows [start, total) in batches. Returns (n_indexed, elapsed)."""
    batch = []
    t0 = time.time()
    indexed = 0
    for pid, vec, payload in stream_points(strategy, start, total):
        from qdrant_client.http.models import PointStruct
        batch.append(PointStruct(id=pid, vector=vec, payload=payload))
        if len(batch) >= batch_size:
            client.upsert(name, points=batch)
            indexed += len(batch)
            batch.clear()
            if indexed % (batch_size * 20) < batch_size:
                print(f"  progress {start + indexed}/{total} "
                      f"({time.time() - t0:.1f}s)", flush=True)
    if batch:
        client.upsert(name, points=batch)
        indexed += len(batch)
    return indexed, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0,
                    help="index only the first N vectors (test mode; 0 = all)")
    ap.add_argument("--recreate", action="store_true",
                    help="drop and rebuild the collection")
    args = ap.parse_args()

    strategy = args.strategy
    name = collection_name(strategy)

    print(f"=== indexing strategy: {strategy} ===", flush=True)
    n_total = validate_artifacts(strategy)
    print(f"  artifacts: {n_total} vectors, dim={EMBED_DIM} (validation OK)",
          flush=True)

    total = min(args.limit, n_total) if args.limit else n_total
    print(f"  target: {total} vectors (batch_size={args.batch_size})",
          flush=True)

    client = get_client()
    try:
        if args.recreate and client.collection_exists(name):
            client.delete_collection(name)
            print(f"  dropped existing collection {name}", flush=True)
        ensure_collection(client, strategy)

        existing = client.count(name).count
        if existing == total:
            print(f"  collection already has {existing} points; nothing to do",
                  flush=True)
        elif existing > total:
            print(f"  WARNING: collection has {existing} > target {total}; "
                  f"re-run with --recreate to rebuild", flush=True)
        else:
            start = existing
            if start:
                print(f"  resuming from {start} (existing count)", flush=True)
            print(f"  indexing {total - start} vectors...", flush=True)
            indexed, elapsed = index(client, name, strategy, start, total,
                                     args.batch_size)
            final = client.count(name).count
            print(f"\n=== SUMMARY ===", flush=True)
            print(f"strategy:      {strategy}", flush=True)
            print(f"collection:     {name}", flush=True)
            print(f"vector dim:     {EMBED_DIM}", flush=True)
            print(f"distance:       cosine", flush=True)
            print(f"points indexed: {indexed}", flush=True)
            print(f"final count:    {final}", flush=True)
            print(f"index time:     {elapsed:.1f}s "
                  f"({indexed / max(elapsed, 1e-9):.1f} vectors/s)", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
