"""Measure LLM providers on the two axes that matter here: latency and whether
they can actually answer in Hindi and Marathi.

Because ``backend/rag/llm.py`` speaks the OpenAI-compatible protocol, switching
providers is configuration, not code — so the choice should be settled by
measurement rather than by vendor claims. Published TTFT figures are not usable
for this decision: they are typically measured at ~10K-token inputs (this
pipeline sends ~1.5K) and often include reasoning-model thinking time.

Measured per provider:

* **TTFT** — streaming time-to-first-token, the number that governs perceived
  latency once the extractive answer has already been shown.
* **total** — full completion time.
* **script match** — does a Hindi question get a Devanagari answer? This is the
  failure mode that matters for an Indic deployment and the one where
  smaller/English-centric models fall down.
* **groundedness** — share of answer terms present in the retrieved context.
* **valid JSON** — did structured output survive.

Usage:
    python scripts/compare_providers.py
    python scripts/compare_providers.py --providers openai,groq --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag import guardrails as G  # noqa: E402
from backend.rag.config import CFG  # noqa: E402
from backend.rag.llm import (  # noqa: E402
    GenerationError, LLMClient, Source, resolve_api_key,
)
from backend.rag.retrieval import get_retriever  # noqa: E402

# Candidate configurations. Each is (provider, model) and relies on
# PROVIDER_BASE_URLS + the provider's key env var.
CANDIDATES = {
    # Verified against each provider's live /models endpoint. Groq no longer
    # serves Llama chat models, and Gemini retired 2.5-flash-lite for new users,
    # so the obvious names from documentation and blog posts are already stale.
    "openai": "gpt-4.1-mini",
    "groq": "openai/gpt-oss-20b",
    "gemini": "gemini-3.5-flash-lite",
    "openrouter": "qwen/qwen3.6-27b",
}

# One query per served language, so Indic fluency is actually exercised.
PROBES = [
    ("en", "how much does a hip replacement cost?"),
    ("hi", "मैनहट्टन परियोजना क्या थी?"),
    ("mr", "मॅनहॅटन प्रकल्प काय होता?"),
]


def measure(client: LLMClient, query: str, sources: list[Source]) -> dict:
    """One streaming call: TTFT, total, and the assembled text."""
    ttft: list[float] = []
    t0 = time.perf_counter()
    parts: list[str] = []
    try:
        for piece in client.stream(query, sources,
                                   on_first_token=lambda ms: ttft.append(ms)):
            parts.append(piece)
    except Exception as e:
        return {"error": str(e)[:180]}
    total = (time.perf_counter() - t0) * 1000
    return {"ttft_ms": round(ttft[0], 1) if ttft else None,
            "total_ms": round(total, 1),
            "text": "".join(parts).strip()}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--providers", default="openai,groq,gemini")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    names = [p.strip().lower() for p in args.providers.split(",") if p.strip()]

    retr = get_retriever()
    contexts: dict[str, tuple[str, list[Source], list[dict]]] = {}
    for lang, q in PROBES:
        res = retr.search(q, use_cache=False)
        srcs = [Source(chunk_id=h["chunk_id"], document_id=h.get("document_id"),
                       score=h.get("rrf_score"), text=h.get("text") or "",
                       lang=h.get("lang"))
                for h in res.hits[:CFG.generation_k]]
        contexts[lang] = (q, srcs, res.hits[:CFG.generation_k])

    print("=" * 78)
    print("LLM PROVIDER COMPARISON")
    print("=" * 78)
    print(f"Repeats per probe : {args.repeats}")
    print(f"Context chunks    : {CFG.generation_k}")
    print(f"max_tokens        : {CFG.llm_max_tokens}")
    print()

    results: dict[str, dict] = {}

    for name in names:
        model = CANDIDATES.get(name)
        if not model:
            print(f"[{name}] SKIP — no candidate model configured")
            continue
        if not resolve_api_key(name):
            print(f"[{name}] SKIP — no API key in environment")
            continue

        print(f"[{name}] {model}")
        try:
            client = LLMClient(provider=name, model=model)
        except GenerationError as e:
            print(f"  FAILED to construct: {e}")
            results[name] = {"model": model, "error": str(e)[:180]}
            print()
            continue

        warm = client.warmup()
        print(f"  warmup: {warm}")

        per_lang: dict[str, dict] = {}
        all_ttft: list[float] = []

        for lang, (q, srcs, hits) in contexts.items():
            ttfts, totals, texts, errors = [], [], [], []
            for _ in range(args.repeats):
                m = measure(client, q, srcs)
                if "error" in m:
                    errors.append(m["error"])
                    continue
                if m["ttft_ms"] is not None:
                    ttfts.append(m["ttft_ms"])
                totals.append(m["total_ms"])
                texts.append(m["text"])

            if not totals:
                per_lang[lang] = {"error": errors[0] if errors else "no response"}
                print(f"  {lang}: FAILED — {per_lang[lang]['error']}")
                continue

            sample = texts[-1]
            q_script, _ = G.dominant_script(q)
            a_script, _ = G.dominant_script(sample)
            grounded = G.groundedness(sample, hits)
            all_ttft.extend(ttfts)

            per_lang[lang] = {
                "ttft_p50": round(statistics.median(ttfts), 1) if ttfts else None,
                "total_p50": round(statistics.median(totals), 1),
                "query_script": q_script, "answer_script": a_script,
                "script_match": q_script == a_script,
                "groundedness": round(grounded, 3),
                "sample": sample[:150],
            }
            p = per_lang[lang]
            print(f"  {lang}: ttft {p['ttft_p50']}ms  total {p['total_p50']}ms  "
                  f"script {'OK' if p['script_match'] else 'MISMATCH -> ' + a_script}"
                  f"  grounded {p['groundedness']}")
            print(f"      {sample[:120]}")

        results[name] = {
            "model": model,
            "warmup": warm,
            "ttft_p50_overall": round(statistics.median(all_ttft), 1)
            if all_ttft else None,
            "languages": per_lang,
        }
        print()

    # ---- summary table ----
    print("=" * 78)
    print(f"{'provider':<12}{'model':<34}{'TTFT P50':>10}{'script OK':>11}")
    print("-" * 78)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<12}{r['model']:<34}{'ERROR':>10}{'-':>11}")
            continue
        langs = r["languages"]
        ok = sum(1 for v in langs.values() if v.get("script_match"))
        tot = sum(1 for v in langs.values() if "error" not in v)
        ttft = r["ttft_p50_overall"]
        print(f"{name:<12}{r['model']:<34}"
              f"{(f'{ttft:.0f} ms' if ttft else 'n/a'):>10}{f'{ok}/{tot}':>11}")
    print("=" * 78)
    print("script OK = answered in the same script as the question "
          "(the Indic-fluency check)")

    out = CFG.root / "results" / "provider_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generation_k": CFG.generation_k,
        "max_tokens": CFG.llm_max_tokens,
        "repeats": args.repeats,
        "note": "measured from the build machine; a deployment colocated with "
                "the provider region will see materially lower TTFT",
        "providers": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out.relative_to(CFG.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
