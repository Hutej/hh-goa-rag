"""Three-tier latency benchmark: P50 / P70 / P100 across all served languages.

Why three tiers
---------------
Reporting one number for "the pipeline" hides the only thing that matters: which
parts are under our control and which are a third party's queue. So each tier is
measured and reported separately, with the boundary stated explicitly.

    Tier A  retrieval core       encode + dense + sparse + fuse + hydrate
    Tier B  first grounded answer  Tier A + guardrails + extractive answer
    Tier C  full generated answer  Tier B + LLM round trip

Tier B is the headline latency claim: at that point the user has a grounded,
verbatim answer on screen. Tier C is reported honestly alongside it, because a
hosted LLM's time-to-first-token cannot be controlled from here — and hiding it
would be the dishonest version of this report.

Speech-to-text is measured separately and excluded from the tiers, per the task
clarification that STT latency is not counted. Its measured cost is still
reported so the full voice-path number can be reconstructed.

Methodology
-----------
* Queries are sampled **deterministically** (seed 12345) from the labelled
  ground-truth set, so the same queries are used on every run and across
  languages.
* The query-embedding cache is **disabled**. With it on, a benchmark that cycles
  a fixed query list would report cache-hit latency, which would be meaningless.
* Warmup iterations run first and are excluded, then cold-start is reported
  separately from the warm percentiles — a single cold number does not represent
  steady state, and a warm number alone hides deployment reality.
* P100 is the true maximum, not a smoothed p99: with a hard latency target the
  worst observed case is the honest tail.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --queries 150 --llm-queries 20
    python scripts/benchmark.py --no-llm            # tiers A and B only
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag import guardrails as G  # noqa: E402
from backend.rag.config import CFG, LANGUAGES  # noqa: E402
from backend.rag.extractive import extract_answer  # noqa: E402
from backend.rag.llm import GenerationError, Source  # noqa: E402
from backend.rag.pipeline import Components, warmup  # noqa: E402
from backend.rag.retrieval import get_retriever  # noqa: E402

SEED = 12345
TARGET_MS = 200


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"P50": None, "P70": None, "P100": None, "mean": None, "n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "P50": round(float(np.percentile(a, 50)), 2),
        "P70": round(float(np.percentile(a, 70)), 2),
        "P100": round(float(a.max()), 2),
        "mean": round(float(a.mean()), 2),
        "n": int(a.size),
    }


def sample_queries(n_per_lang: int) -> dict[str, list[str]]:
    """Deterministic query sample per language, drawn from labelled queries.

    Only queries that have at least one ``is_selected`` passage are used, so
    every benchmark query is one the corpus can actually answer — otherwise the
    guardrails would refuse and Tier C would measure refusals instead of work.
    """
    rng = random.Random(SEED)
    out: dict[str, list[str]] = {}
    for code in CFG.languages:
        path = CFG.passages_path(code)
        if not path.exists():
            continue
        # Query column per language: `query_en` for English, `query` (the
        # shard's Indic text) otherwise — see config.Language.query_col.
        qcol = CFG.lang(code).query_col
        tbl = pq.read_table(path, columns=["query_id", qcol, "is_selected"])
        seen: dict[int, str] = {}
        qids = tbl.column("query_id").to_pylist()
        queries = tbl.column(qcol).to_pylist()
        sel = tbl.column("is_selected").to_pylist()
        for qid, q, s in zip(qids, queries, sel):
            if s and qid not in seen and (q or "").strip():
                seen[qid] = q.strip()
        pool = sorted(seen.items())
        rng2 = random.Random(SEED)
        rng2.shuffle(pool)
        out[code] = [q for _, q in pool[:n_per_lang]]
    return out


def bench_retrieval(queries: dict[str, list[str]], warmup_n: int) -> dict:
    """Tiers A and B, per language and pooled."""
    retr = get_retriever()
    flat = [(lang, q) for lang, qs in queries.items() for q in qs]

    # Warm the code paths (page-ins, first-call allocations) and discard.
    for _, q in flat[:warmup_n]:
        retr.search(q, use_cache=False)

    stages = ["encode_ms", "dense_ms", "sparse_ms", "fuse_ms", "hydrate_ms",
              "retrieval_ms", "guard_ms", "extractive_ms", "first_answer_ms"]
    per_lang: dict[str, dict[str, list[float]]] = {
        lang: {s: [] for s in stages} for lang in queries}
    cold: dict | None = None

    for i, (lang, q) in enumerate(flat):
        t0 = time.perf_counter()
        result = retr.search(q, use_cache=False)
        t_guard = time.perf_counter()
        G.check_input(q)
        v = G.check_retrieval(result)
        guard_ms = (time.perf_counter() - t_guard) * 1000
        t_ext = time.perf_counter()
        extract_answer(q, result.hits)
        ext_ms = (time.perf_counter() - t_ext) * 1000
        first_ms = (time.perf_counter() - t0) * 1000

        row = dict(result.timing)
        row["guard_ms"] = guard_ms
        row["extractive_ms"] = ext_ms
        row["first_answer_ms"] = first_ms
        if i == 0:
            cold = {k: round(v2, 2) for k, v2 in row.items() if k in stages}
        for s in stages:
            if s in row:
                per_lang[lang][s].append(row[s])

    pooled = {s: [] for s in stages}
    for lang, d in per_lang.items():
        for s in stages:
            pooled[s].extend(d[s])

    return {
        "cold_start_ms": cold,
        "pooled": {s: percentiles(pooled[s]) for s in stages},
        "per_language": {lang: {s: percentiles(d[s]) for s in stages}
                         for lang, d in per_lang.items()},
    }


def bench_generation(queries: dict[str, list[str]], n: int) -> dict:
    """Tier C: full generated answer, including the LLM round trip."""
    client = Components.llm()
    if client is None:
        return {"skipped": "llm not configured"}
    retr = get_retriever()
    client.warmup()

    flat = [(lang, q) for lang, qs in queries.items() for q in qs]
    rng = random.Random(SEED)
    rng.shuffle(flat)
    flat = flat[:n]

    gen_ms: list[float] = []
    total_ms: list[float] = []
    per_lang: dict[str, list[float]] = {}
    modes = {"generated": 0, "refused": 0, "failed": 0}

    for lang, q in flat:
        t0 = time.perf_counter()
        result = retr.search(q, use_cache=False)
        srcs = [Source(chunk_id=h["chunk_id"], document_id=h.get("document_id"),
                       score=h.get("rrf_score"), text=h.get("text") or "",
                       lang=h.get("lang"))
                for h in result.hits[:CFG.generation_k]]
        t_gen = time.perf_counter()
        try:
            answer, meta = client.complete(q, srcs)
        except GenerationError:
            modes["failed"] += 1
            continue
        g = (time.perf_counter() - t_gen) * 1000
        tot = (time.perf_counter() - t0) * 1000
        gen_ms.append(g)
        total_ms.append(tot)
        per_lang.setdefault(lang, []).append(tot)
        modes["generated" if answer.sufficient else "refused"] += 1

    return {
        "generation_ms": percentiles(gen_ms),
        "total_ms": percentiles(total_ms),
        "per_language_total_ms": {k: percentiles(v) for k, v in per_lang.items()},
        "outcomes": modes,
        "provider": client.describe(),
    }


def bench_stt(sample: Path | None) -> dict:
    """Speech-to-text cost, reported separately (excluded from the tiers)."""
    if sample is None or not sample.exists():
        return {"skipped": "no sample audio; pass --audio <file>"}
    stt = Components.stt()
    if stt is None:
        return {"skipped": "stt not configured"}
    times: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        try:
            stt.transcribe(str(sample))
        except Exception as e:
            return {"error": str(e)[:180]}
        times.append((time.perf_counter() - t0) * 1000)
    return {"file": sample.name, "latency_ms": percentiles(times)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=50,
                    help="queries per language for tiers A/B (default 50)")
    ap.add_argument("--llm-queries", type=int, default=20,
                    help="total queries for tier C (default 20)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--audio", default="data/samples/voice_hi_manhattan.wav")
    args = ap.parse_args()

    print("=" * 78)
    print("LATENCY BENCHMARK — P50 / P70 / P100")
    print("=" * 78)

    w = warmup()
    queries = sample_queries(args.queries)
    total_q = sum(len(v) for v in queries.values())
    if not total_q:
        print("ERROR: no labelled queries found; run the data pipeline first",
              file=sys.stderr)
        return 1

    print(f"Encoder    : {CFG.embed_model} ({CFG.embed_precision}, "
          f"dim {CFG.embed_dim}, {CFG.embed_backend})")
    print(f"Dense      : faiss HNSW m={CFG.hnsw_m} ef_search={CFG.hnsw_ef_search}")
    print(f"Sparse     : bm25s")
    print(f"Languages  : {', '.join(queries)}")
    print(f"Queries    : {total_q} ({args.queries}/language), seed {SEED}, "
          f"embedding cache DISABLED")
    print(f"Target     : {TARGET_MS} ms")
    print()

    retrieval = bench_retrieval(queries, args.warmup)

    print("-" * 78)
    print("TIER A — retrieval core   |   TIER B — first grounded answer")
    print("-" * 78)
    print(f"{'stage':<20}{'P50':>10}{'P70':>10}{'P100':>10}{'mean':>10}")
    order = ["encode_ms", "dense_ms", "sparse_ms", "fuse_ms", "hydrate_ms",
             "retrieval_ms", "guard_ms", "extractive_ms", "first_answer_ms"]
    for s in order:
        p = retrieval["pooled"][s]
        if p["n"] == 0:
            continue
        label = s
        if s == "retrieval_ms":
            label = "TIER A total"
        elif s == "first_answer_ms":
            label = "TIER B total"
        print(f"{label:<20}{p['P50']:>10.2f}{p['P70']:>10.2f}"
              f"{p['P100']:>10.2f}{p['mean']:>10.2f}")

    print()
    print(f"{'per language':<20}{'tier A P50':>12}{'tier B P50':>12}"
          f"{'tier B P100':>13}")
    for lang, d in retrieval["per_language"].items():
        a = d["retrieval_ms"]
        b = d["first_answer_ms"]
        if a["n"] == 0:
            continue
        print(f"{LANGUAGES[lang].name:<20}{a['P50']:>12.2f}{b['P50']:>12.2f}"
              f"{b['P100']:>13.2f}")

    print()
    print(f"cold start (first query): {json.dumps(retrieval['cold_start_ms'])}")

    generation = {"skipped": "--no-llm"} if args.no_llm else \
        bench_generation(queries, args.llm_queries)

    if "skipped" not in generation:
        print()
        print("-" * 78)
        print("TIER C — full generated answer")
        print("-" * 78)
        g = generation["generation_ms"]
        t = generation["total_ms"]
        print(f"{'stage':<20}{'P50':>10}{'P70':>10}{'P100':>10}")
        print(f"{'llm round trip':<20}{g['P50']:>10.2f}{g['P70']:>10.2f}"
              f"{g['P100']:>10.2f}")
        print(f"{'TIER C total':<20}{t['P50']:>10.2f}{t['P70']:>10.2f}"
              f"{t['P100']:>10.2f}")
        print(f"outcomes: {generation['outcomes']}  "
              f"(n={t['n']}, provider={generation['provider']['model']})")

    stt = bench_stt(Path(args.audio) if args.audio else None)

    tier_a = retrieval["pooled"]["retrieval_ms"]
    tier_b = retrieval["pooled"]["first_answer_ms"]
    tier_c = generation.get("total_ms") if "skipped" not in generation else None

    print()
    print("=" * 78)
    print("VERDICT against the 200 ms target")
    print("=" * 78)
    print(f"Tier A  retrieval core        P50 {tier_a['P50']:>8.2f} ms  "
          f"P100 {tier_a['P100']:>8.2f} ms  "
          f"{'PASS' if tier_a['P100'] <= TARGET_MS else 'FAIL'}")
    print(f"Tier B  first grounded answer P50 {tier_b['P50']:>8.2f} ms  "
          f"P100 {tier_b['P100']:>8.2f} ms  "
          f"{'PASS' if tier_b['P100'] <= TARGET_MS else 'FAIL'}")
    if tier_c:
        print(f"Tier C  generated answer      P50 {tier_c['P50']:>8.2f} ms  "
              f"P100 {tier_c['P100']:>8.2f} ms  "
              f"{'PASS' if tier_c['P50'] <= TARGET_MS else 'over target'}")
        print("        Tier C is dominated by the hosted LLM round trip, which "
              "is not locally controllable.")
    if "latency_ms" in stt:
        print(f"STT     (excluded per task)    P50 "
              f"{stt['latency_ms']['P50']:>8.2f} ms")

    report = {
        "target_ms": TARGET_MS,
        "seed": SEED,
        "cache_disabled": True,
        "queries_per_language": args.queries,
        "config": CFG.describe(),
        "warmup": w,
        "tier_a_and_b": retrieval,
        "tier_c": generation,
        "stt_excluded_from_tiers": stt,
        "verdict": {
            "tier_a_p50": tier_a["P50"], "tier_a_p100": tier_a["P100"],
            "tier_b_p50": tier_b["P50"], "tier_b_p100": tier_b["P100"],
            "tier_b_meets_target_at_p100": tier_b["P100"] <= TARGET_MS,
            "tier_c_p50": tier_c["P50"] if tier_c else None,
        },
    }
    out_dir = CFG.root / "results" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latency.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"Wrote {(out_dir / 'latency.json').relative_to(CFG.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
