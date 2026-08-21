"""Phase 7: grounded answer generation over hybrid retrieval.

query -> hybrid retrieval -> top-k chunks -> LLM -> grounded answer + sources.

Usage:
    venv/bin/python scripts/answer_query.py --query "मैनहट्टन परियोजना क्या थी?"
    venv/bin/python scripts/answer_query.py --query "..." --top-k 5 --strategy adaptive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Progress messages go to stderr so stdout holds only the final JSON result.
_log = lambda *m: print(*m, file=sys.stderr, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.rag.bm25 import STRATEGIES  # noqa: E402
from backend.rag.embeddings import load_embedder  # noqa: E402
from backend.rag.generation import (  # noqa: E402
    Source, GenerationError, generate_answer, get_provider,
)
from backend.rag.hybrid import hybrid_search  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", required=True)
    ap.add_argument("--strategy", default="adaptive", choices=STRATEGIES)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--dense-k", type=int, default=20)
    ap.add_argument("--bm25-k", type=int, default=20)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = ap.parse_args()

    t_total = time.time()

    # --- retrieval (Phase 5 hybrid) ---
    _log("loading BGE-M3...")
    model, device, dtype = load_embedder(args.device, None)
    _log("running hybrid retrieval...")
    t_r = time.time()
    res = hybrid_search(
        model, args.query, strategy=args.strategy, top_k=args.top_k,
        dense_k=args.dense_k, bm25_k=args.bm25_k, rrf_k=args.rrf_k,
    )
    retrieval_latency_ms = (time.time() - t_r) * 1000

    sources = [Source(
        chunk_id=h["chunk_id"], document_id=h.get("document_id"),
        score=h.get("rrf_score") or h.get("score"), text=h.get("text"),
    ) for h in res["hybrid_results"]]

    # --- generation ---
    _log("loading LLM provider...")
    try:
        provider = get_provider()
    except GenerationError as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False), flush=True)
        sys.exit(1)
    _log(f"  provider: {provider.name}")

    t_g = time.time()
    try:
        answer = generate_answer(args.query, sources, provider=provider)
    except GenerationError as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False), flush=True)
        sys.exit(1)
    generation_latency_ms = (time.time() - t_g) * 1000
    total_latency_ms = (time.time() - t_total) * 1000

    out = {
        "query": args.query,
        "answer": answer,
        "sources": [s.to_dict() for s in sources],
        "retrieval_latency_ms": round(retrieval_latency_ms, 1),
        "generation_latency_ms": round(generation_latency_ms, 1),
        "total_latency_ms": round(total_latency_ms, 1),
        "provider": provider.name,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
