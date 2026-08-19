"""Phase 3B: dense retrieval over a Qdrant collection.

Encodes the query with the existing BGE-M3 utilities (Phase 3A — ``encode_batch``
is the single normalization), queries the strategy's Qdrant collection, and prints
top-k results as JSON with measured latency. No latency target claims.

Usage:
    venv/bin/python scripts/retrieve_dense.py --strategy adaptive \\
        --query "मैनहट्टन परियोजना" --top-k 5
    venv/bin/python scripts/retrieve_dense.py --strategy adaptive --query "..." --only-selected
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.rag.embeddings import load_embedder, encode_batch  # noqa: E402
from backend.rag.qdrant_index import (  # noqa: E402
    EMBED_DIM, STRATEGIES, collection_name, get_client,
)
from qdrant_client.http.models import FieldCondition, Filter, MatchValue  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--only-selected", action="store_true",
                    help="filter to is_selected == 1")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default=None,
                    choices=["float16", "float32", "bfloat16"])
    args = ap.parse_args()

    strategy = args.strategy
    name = collection_name(strategy)

    print(f"loading BGE-M3...", flush=True)
    t0 = time.time()
    model, device, dtype = load_embedder(args.device, args.dtype)
    encode_ms = 0.0
    t_q = time.time()
    qvec = encode_batch(model, [args.query], batch_size=1)  # (1, 1024), unit-norm
    encode_ms = (time.time() - t_q) * 1000
    print(f"  query encoded on {device} ({dtype}) in {encode_ms:.0f} ms, "
          f"dim={qvec.shape[1]}", flush=True)

    client = get_client()
    try:
        if not client.collection_exists(name):
            raise SystemExit(f"collection {name} does not exist — run index_qdrant.py")
        qfilter = None
        if args.only_selected:
            qfilter = Filter(must=[FieldCondition(key="is_selected",
                                                 match=MatchValue(value=1))])
        t_s = time.time()
        res = client.query_points(
            name, query=qvec[0].tolist(), limit=args.top_k,
            query_filter=qfilter, with_payload=True, with_vectors=False)
        search_ms = (time.time() - t_s) * 1000
        latency_ms = (time.time() - t0) * 1000

        results = []
        for r, p in enumerate(res.points, start=1):
            pl = p.payload or {}
            results.append({
                "rank": r,
                "chunk_id": pl.get("chunk_id"),
                "document_id": pl.get("document_id"),
                "score": round(float(p.score), 6),
                "text": pl.get("text"),
                "query_id": pl.get("query_id"),
                "chunk_index": pl.get("chunk_index"),
                "is_selected": pl.get("is_selected"),
                "strategy": strategy,
            })
        out = {
            "strategy": strategy,
            "query": args.query,
            "top_k": args.top_k,
            "only_selected": args.only_selected,
            "encode_ms": round(encode_ms, 1),
            "search_ms": round(search_ms, 1),
            "latency_ms": round(latency_ms, 1),
            "results": results,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
