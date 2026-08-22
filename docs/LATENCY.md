# Latency: what was measured, how, and what it cost to get there

All numbers here come from this repository. Raw outputs are committed under
`results/`. Reproduce with:

```bash
python scripts/benchmark.py            # three-tier P50/P70/P100
python -m benchmarks.benchmark 60      # the organizers' script, unmodified
```

---

## Measurement methodology

- **150 queries**, 50 per language, sampled deterministically (seed 12345) from
  the labelled ground-truth set so every query is one the corpus can answer.
- **The query embedding cache is disabled.** With it enabled, a benchmark that
  cycles a fixed query list measures cache hits, and the percentiles become
  meaningless. `app/retriever.py` also passes `use_cache=False` for the same
  reason.
- **Warmup iterations are discarded**, then cold start is reported separately
  from warm percentiles. A single cold number misrepresents steady state; a warm
  number alone hides deployment reality.
- **P100 is the true maximum**, not a smoothed p99. With a hard target, the worst
  observed case is the honest tail.
- Timings are taken **server-side inside the request path**, so they exclude the
  user's network round trip to the service.

## The three tiers

Reporting one number for "the pipeline" would hide the only thing that matters:
which parts are under our control and which are a third party's queue.

| Tier | Contains | P50 | P70 | P100 |
|---|---|---:|---:|---:|
| **A** retrieval core | encode, dense, sparse, fusion, hydration | 22.05 | 25.18 | 71.73 |
| **B** first grounded answer | Tier A + 2 guardrail layers + extractive answer | **22.57** | **25.65** | **72.09** |
| **C** full generated answer | Tier B + LLM round trip | 949.97 | 999.83 | 1125.23 |

Speech-to-text is **excluded from all tiers** per the organizers' clarification
that STT latency is not counted. Measured separately: **485.9 ms P50** (Sarvam
`saaras:v4`, batch upload). The full voice path can be reconstructed by adding it.

### Stage breakdown (Tier A / B)

| Stage | P50 | P70 | P100 |
|---|---:|---:|---:|
| encode (int8 ONNX, 384-dim) | 4.37 | 4.84 | 9.70 |
| dense (FAISS HNSW × 3 languages) | 2.61 | 2.84 | 5.90 |
| sparse (bm25s, script-routed) | 14.38 | 16.85 | 62.72 |
| fusion (weighted RRF) | 0.18 | 0.20 | 0.49 |
| hydration (text for final hits) | 0.08 | 0.09 | 0.25 |
| **Tier A total** | **22.05** | **25.18** | **71.73** |
| guardrails (input + retrieval) | 0.12 | 0.13 | 0.38 |
| extractive answer | 0.22 | 0.25 | 0.71 |
| **Tier B total** | **22.57** | **25.65** | **72.09** |

Cold start (first query after load): 20.42 ms — the indexes are memory-resident
after warmup, so there is no meaningful cold-start penalty on the retrieval path.
The real cold cost is process startup: ~6.5 s to load three HNSW indexes, three
BM25 indexes and the encoder, plus ~1.8 s for the LLM connection warm-up ping.

### Per language

| Language | Tier A P50 | Tier B P50 | Tier B P100 |
|---|---:|---:|---:|
| English | 12.78 | 13.08 | 29.56 |
| Marathi | 23.50 | 23.87 | 72.09 |
| Hindi | 25.18 | 25.65 | 66.18 |

English is roughly half the cost of the Indic languages, for a structural reason:
Latin script routes sparse retrieval to **one** index, while Devanagari cannot be
resolved to Hindi or Marathi by script alone, so both are searched and fused.

---

## Where the 42x came from

The first implementation measured **929 ms P50** on retrieval alone. The
improvement was not one change:

| Change | Before | After |
|---|---:|---:|
| Encoder: BGE-M3 fp32 torch → int8 ONNX e5-small | 220 ms | 4.4 ms |
| Dense: embedded Qdrant → in-process FAISS HNSW | 279 ms | 2.6 ms |
| Sparse: `rank_bm25` → `bm25s` | 398 ms | 14.4 ms |
| Metadata access: Arrow `ChunkedArray.take` → Python lists | 4.4 ms | ~0 |
| Text hydration: chunked take → contiguous array, final hits only | 9.3 ms | 0.08 ms |
| **Total** | **929 ms** | **22.05 ms** |

### The two findings that were not obvious

**Embedded Qdrant has no ANN at all.** `QdrantClient(path=...)` is a pure-Python
local mode that brute-force scans every vector and silently ignores
`hnsw_config`. The original code also opened and closed the client on every
request, so each query re-acquired the store lock. That is the most likely cause
of dense latency swinging between 279 ms and 1162 ms in the original numbers.

**Arrow metadata lookup cost 20x more than the actual search.** Profiling the
dense path showed FAISS returning in **0.319 ms** while materializing metadata
for those same 20 rows took **4.4 ms** — 93% of dense time was not retrieval.
`ChunkedArray.take` resolves chunk boundaries across the whole array, so it is
O(array size) rather than O(k). Two fixes: small fields are materialized once as
Python lists for O(1) indexing, and the large `text` column is collapsed with
`combine_chunks()` and read only for the handful of chunks that survive fusion.
Text hydration went from 9.26 ms P50 (76 ms P100) to 0.08 ms.

The lesson generalises: at these latencies the data-access layer, not the
vector search, is where the time goes.

---

## Why Tier B exists

No hosted LLM's time-to-first-token can be relied on to fit a 200 ms budget.
Measured from this machine, across three providers, TTFT ranged 572–743 ms, and
published figures were not usable — they are typically measured at ~10K-token
inputs where this pipeline sends ~1.5K, and often include reasoning time.

So the design does not bet the latency target on a third party. The moment
retrieval finishes, the grounded text is already in hand: sentence selection over
the top chunks produces a verbatim answer in **0.22 ms**. That answer is more
grounded than a generated one, not less, because it is copied from the retrieved
evidence.

Tier C is then reported alongside it rather than hidden. On the streaming
endpoint the sequence is visible directly:

```
[  391 ms] retrieval    5 sources, confidence high
[  392 ms] extractive   grounded answer on screen
[ 1226 ms] done         generated answer complete, ttft 829 ms
```

The extractive answer arrives roughly 30x sooner than the first generated token.

---

## Deployment notes affecting latency

- **Colocate with the LLM provider.** Tier C here is measured from India against
  US-hosted providers. A Space in `us-east-1` should see materially lower TTFT.
- **Connection pooling matters more than it looks.** The original implementation
  measured 9,995 ms cold against 112 ms warm on identical work — a 90x spread
  that was DNS, TLS and connection setup, not inference. The client now holds one
  pooled keep-alive HTTP/2 connection and pings it at startup.
- **Rate limiting is a latency problem too.** ~25% of rapid sequential requests to
  Gemini's free tier return HTTP 429. Rate-limit-aware jittered backoff took
  observed failures from 5/20 to 0/15. When all attempts fail the harness serves
  the extractive answer, so a 429 degrades quality rather than failing the request.
- **Thread pinning.** `OMP_NUM_THREADS=4` and `EMBED_THREADS=4` are set in the
  Dockerfile. onnxruntime and FAISS each default to one thread per core and
  oversubscribe a shared Space CPU, which worsens tail latency. Measured on the
  build machine: 8 threads beat 4, and pinning beat letting each library guess.
- **Single worker on purpose.** The indexes are ~1.4 GB resident and held
  in-process, so extra workers multiply memory rather than throughput. Request
  handlers are sync `def` so FastAPI dispatches them to a thread pool; declaring
  them `async` (as the original did) ran blocking work on the event loop and made
  one in-flight query block every other request, including health checks.
