---
title: Voice RAG - Hindi, Marathi, English
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Voice-enabled multilingual RAG over MSMARCO-XI, 22ms retrieval
---

<!-- The block above is Hugging Face Spaces configuration (Docker SDK, port
     7860). GitHub renders it as a table; it is required for the live deploy. -->

# Voice-Enabled Multilingual RAG — HH Goa 2026

Ask a question out loud in **Hindi, Marathi or English**. The system transcribes
it, retrieves grounded evidence from a 632,668-chunk index built from
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
and answers — in the language you asked, or refuses when the corpus cannot
support an answer.

```
voice ─▶ Sarvam STT ─▶ [input guard] ─▶ hybrid retrieval ─▶ [retrieval guard]
                                              │
                                              ├─▶ extractive answer   22 ms
                                              │
                                              └─▶ LLM ─▶ [self-report guard]
                                                          ─▶ [grounding guard]
```

Everything below is measured on this repository, not estimated. Every number is
reproducible with the commands given, and the raw outputs are committed under
`results/`.

---

## Headline results

| Metric | Value |
|---|---|
| **Retrieval core** (encode + dense + sparse + fuse) | **22.05 ms** P50 / 71.73 ms P100 |
| **First grounded answer** (adds guardrails + extractive) | **22.57 ms** P50 / 72.09 ms P100 |
| Full LLM-generated answer | 949.97 ms P50 |
| Corpus | 632,668 chunks / 599,652 passages / 3 languages |
| Retrieval quality | R@10 — English 0.887, Hindi 0.627, Marathi 0.487 |
| Cross-lingual retrieval | en→hi R@10 0.547, hi→en 0.447 |
| ANN approximation loss | 0.8% (recall@10 0.992 vs exact search) |
| Latency improvement over the first implementation | **929 ms → 22 ms P50 (42x)** |
| Tests | 182 passing |

The organizers' own benchmark script runs against this repo unmodified:

```
$ python -m benchmarks.benchmark 60
stage            avg     p50     p95     p99   (ms)
embed           4.70    4.45    5.95   10.20
search         10.10    7.50   15.96   41.42
total          14.88   12.13   20.72   50.05

Latency budget: 200ms | p95 total: 20.72ms
PASS: within budget
```

---

## On the 200 ms target: what is being claimed

Full voice-to-generated-answer under 200 ms is not achievable with any hosted
STT and any hosted LLM. Network round trip alone exceeds it. Rather than pick a
flattering boundary and stay quiet about it, the pipeline is measured in three
tiers with the boundary stated:

| Tier | Contains | P50 | P70 | P100 |
|---|---|---:|---:|---:|
| **A** retrieval core | encode, dense, sparse, fusion, hydration | 22.05 | 25.18 | 71.73 |
| **B** first grounded answer | Tier A + 2 guardrail layers + extractive answer | **22.57** | **25.65** | **72.09** |
| **C** full generated answer | Tier B + LLM round trip | 949.97 | 999.83 | 1125.23 |

*150 queries (50/language), seed 12345, embedding cache disabled, `results/benchmark/latency.json`.*

**Tier B is the claim.** At 22.57 ms the user has a grounded answer on screen —
verbatim text from retrieved evidence, so hallucination is impossible by
construction. It clears 200 ms at P100, not just P50.

**Tier C is reported honestly.** Speech-to-text is excluded from all tiers per
the organizers' clarification that STT latency is not counted; its measured cost
(485.9 ms P50) is reported separately so the full voice path can be
reconstructed. Tier C is dominated by the hosted LLM, measured from India — a
deployment colocated with the provider region will be materially lower.

Per language, Tier B P50: **English 13.08 ms · Marathi 23.87 ms · Hindi 25.65 ms.**

---

## Architecture

| Stage | Choice | Why |
|---|---|---|
| STT | Sarvam `saaras:v4` | Indic-specialised, handles code-mixing; latency not counted so accuracy governs |
| Encoder | `multilingual-e5-small`, **int8 ONNX**, 384-dim | 3.96 ms/query on CPU. No torch anywhere |
| Dense | FAISS `IndexHNSWFlat`, M=32, efSearch=64 | 0.32 ms raw search; 99.2% recall vs exact |
| Sparse | `bm25s`, one index per language | Replaced `rank_bm25`, the measured #1 bottleneck |
| Fusion | Weighted RRF, k=60 | Rank-based, so incomparable score scales never mix |
| Generation | Gemini 3.5 Flash-Lite (provider-agnostic) | Chosen by measurement, see below |
| Serving | FastAPI, single worker, sync handlers | Indexes are ~1.4 GB resident and held in-process |

### No torch, anywhere

The encoder is a pre-exported int8 ONNX graph
(`Xenova/multilingual-e5-small`, 113 MB) run by `onnxruntime`, used for **both**
corpus and query embedding. The tokenizer loads from `tokenizer.json` via
`tokenizers`. Consequences:

- Deployed image is **~500 MB instead of ~4 GB**, with no multi-GB cold start.
- Query encoding is **3.96 ms** on CPU.
- Using identical weights for documents and queries means quantization error
  points the same way on both sides and largely cancels in the dot product,
  rather than introducing query/document asymmetry.

`requirements.txt` is serve-only; indexing and evaluation dependencies live in
`requirements-build.txt` and are never installed on the deploy target.

---

## Chunking

Three strategies, all deterministic and all token-counted with the embedding
model's own tokenizer: **fixed** (256-token windows, 32 overlap, sliced from the
offset map so each chunk is an exact substring), **semantic** (sentence-aware
greedy grouping with a Devanagari-aware splitter), and **adaptive** (routes by
measured length — short passages stay whole, medium get sentence grouping, long
get overlapping windows).

**The thresholds are calibrated per language, and the measurement proves that is
necessary.** Because the corpus is parallel, the *same documents* exist in all
three languages:

| Language | Median tokens | Chars per token | `short_max` | Routing (short/medium/long) |
|---|---:|---:|---:|---|
| Hindi | 86 | 3.29 | 113 | 80.2% / 19.4% / 0.4% |
| English | 71 | 4.07 | 92 | 80.1% / 19.8% / 0.0% |
| Marathi | 84 | 3.29 | 112 | 80.3% / 19.2% / 0.5% |

Hindi needs **1.21x the tokens of English for identical content** — the
sentencepiece vocabulary fragments Devanagari more aggressively. A single shared
threshold set would route the same document into different bands depending on
language. Calibrating per language produces near-identical routing behaviour
across all three, which is the point.

Reproduce: `python scripts/measure_lengths.py` → `docs/chunking_length_stats.json`

---

## Retrieval quality

150 queries per language, seed 12345, ground truth = MS MARCO `is_selected`
matched at `document_id` level.

| Language | Retriever | R@1 | R@3 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|
| English | dense | 0.407 | 0.673 | 0.807 | **0.887** |
| English | sparse | 0.207 | 0.480 | 0.580 | 0.727 |
| Hindi | dense | 0.200 | 0.387 | 0.513 | **0.627** |
| Hindi | sparse | 0.147 | 0.307 | 0.407 | 0.473 |
| Marathi | dense | 0.153 | 0.300 | 0.427 | **0.487** |
| Marathi | sparse | 0.113 | 0.227 | 0.293 | 0.367 |

English is 26–40 points ahead because the English passages *and* queries are
original MS MARCO text, while Hindi and Marathi are machine translations of
both — translation noise on both sides of the retrieval.

### Cross-lingual retrieval

The corpus construction makes this measurable for free. Every source row carries
the same document in English and in the Indic language, index-aligned, sharing
one `is_selected` label — so `document_id` identifies the same document across
all three languages and ground truth transfers with **no** label translation.

| Query → Index | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| en → hi | 0.213 | 0.473 | 0.547 |
| hi → en | 0.167 | 0.360 | 0.447 |
| mr → hi | 0.127 | 0.367 | 0.480 |
| hi → mr | 0.120 | 0.420 | 0.507 |

Ask in Hindi, retrieve an English passage, and the answer still comes back in
Hindi. All three languages share one embedding space, so this needs no
translation step.

### Hybrid fusion is close to break-even — and that is reported, not hidden

A sweep of the sparse weight over 150 queries × 3 languages:

| `sparse_weight` | mean R@1 | mean R@3 | mean R@5 | mean R@10 |
|---|---:|---:|---:|---:|
| 0.0 (dense only) | **0.280** | **0.520** | 0.620 | 0.698 |
| **0.1 (shipped)** | 0.278 | 0.500 | **0.627** | 0.702 |
| 0.25 | 0.264 | 0.485 | 0.602 | 0.698 |
| 1.0 | 0.220 | 0.447 | 0.580 | **0.707** |

The effect is monotonic and mostly *negative*: more sparse weight buys a little
R@10 while steadily degrading the top of the ranking. Since only 3 chunks reach
the model, **R@3 is the governing objective** — and dense-only wins it. Shipping
0.1: indistinguishable from dense-only at R@3, best at R@5, and it retains
exact-match capability for entity and number queries that a natural-language
question set under-represents.

Honest summary: on this corpus, word-level BM25 does not earn a large place in
the top of the ranking. Character n-gram sparse indexing is the obvious next
step for morphologically rich Devanagari.

Reproduce: `python scripts/evaluate_retrieval.py` and `python scripts/tune_fusion.py`

---

## Scored by the organizers' own harness

[`rag-local-eval-loop`](https://github.com/BeaconBandhu/rag-local-eval-loop) runs
against this repo with **zero configuration** — no env vars, no HTTP config file
— because `app/embedder.py` and `app/generator.py` match its default module
names and exact interface. Drop its `eval/` folder and `run.ps1` in and run.

```
FAITHFULNESS / HALLUCINATION   (reference-free, LLM-as-judge)
  Faithful rate           1.000   PERFECT
  Hallucination rate      0.000   PERFECT
  Self-report precision   1.000   PERFECT

CORRECTNESS  (reference-based, vs MSMARCO-XI Eng_Answer)
  Correct rate            1.000   PERFECT

RETRIEVAL  (vs is_selected labels, its own throwaway index)
  Recall@1 0.560 · Recall@3 0.800 · Recall@5 0.880 · MRR 0.693

LATENCY
  retrieval p95  7.81 ms  vs 200 ms budget  ->  PASS
```

**Self-report precision 1.000** is the one worth pointing at: every answer this
system flagged `grounded=True` was independently confirmed faithful by the
judge. That is the check for whether a system's own confidence signal can be
trusted, and it is the metric the suite's README describes catching a real bug in
its reference target.

### Reliability, reported straight

The "lying factor" is the check most able to catch overselling, so here is the
full picture rather than the best run. Across repeated 25+25 runs at one worker:

| | measured | suite's own reference target |
|---|---:|---:|
| False confidence (fabricated on unanswerable) | 0.400 – 0.480 | 0.667 |
| False refusal (declined an answerable) | 0.280 – 0.360 | 0.267 |

Two things behind those numbers, both verified rather than asserted:

**Roughly 20% of generation calls never reached the model.** Gemini's free tier
returned HTTP 429 for 10 of 50 calls even at a single worker. A failed call
returns `grounded=False`, which the harness correctly scores as a refusal — so
false refusal here is inflated by quota, not by judgement. Separating them with a
per-decision reason log gives the underlying behaviour:

| | answered | wrongly declined | 429 |
|---|---:|---:|---:|
| answerable (25) | 16 | **4** | 5 |
| unanswerable (25) | 10 | 10 | 5 |

So genuine over-refusal is 4 of the 20 answerable queries that completed. Honouring
the provider's own `retryDelay` instead of guessing a backoff cut failures from
36% to 12% and moved false refusal from 0.600 to 0.280.

**Fabrication on unanswerable rows is the real remaining weakness**, at 10 of 20
completed. These are MS MARCO rows where no annotator marked any passage
relevant, but the retrieved passages are often topically adjacent, and the model
answers from them. Tightening the prompt was tried and is documented below as a
failed experiment.

To reproduce, including the reason log:

```powershell
$env:HHGOA_DECISION_LOG = "results\decisions.jsonl"
.\run.ps1 --num-answerable 25 --num-unanswerable 25 --workers 1
```

Note the suite's judge needs its own credential and its generation checks are
English-only. `app/config.py` declares `GENERATION_BACKEND = "api"` so the suite
does not clamp itself to one worker, and `LATENCY_BUDGET_MS = 200` per the task
brief — retrieval clears the suite's stricter 50 ms default too.

### A failed experiment, left in the record

Seeing false confidence at 0.333 on a first small run, I rewrote the system
prompt to demand the context *specifically* answer the question. Result:

| Prompt | False refusal | False confidence |
|---|---:|---:|
| original | 0.000 (n=3) | 0.333 (n=3) |
| tightened | **0.840** | 0.120 |
| rebalanced (shipped) | 0.280 – 0.360 | 0.400 – 0.480 |

Declining 84% of answerable questions is far worse than the problem it fixed. Two
mistakes produced it: comparing a 3-query run at one worker against a 25-query run
at three workers, so rate-limit failures masqueraded as a prompt effect; and
optimising one side of a two-sided metric. Both rates are now always measured
together, at fixed sample size and worker count.

## Guardrails: four layers, and two measured negative results

| Layer | Checks | Blocks on |
|---|---|---|
| **Input** | length, script, unsafe content, prompt injection | before any compute is spent |
| **Retrieval** | empty results, degenerate scores; advisory confidence | genuinely degenerate retrieval only |
| **Generation** | the model's structured `sufficient` flag and `confidence` | model says the passages don't answer it |
| **Answer** | groundedness, citation validity, answer language | unsupported answer or fabricated citation |

Verified behaviour: injection blocked in English *and* Hindi, unsafe content
blocked in both, unsupported scripts declined, a fabricated answer scored 0.300
groundedness against 0.778 for a supported one, and a hallucinated `chunk_id`
rejected.

### The similarity threshold that does not work

The obvious guardrail is a cosine floor: refuse when nothing retrieved is similar
enough. It was built, measured against 80 in-corpus and 13 off-topic queries, and
**it does not separate the classes**:

```
in-corpus best cosine   min 0.848   P50 0.915
off-topic best cosine   min 0.838   P50 0.866   max 0.896
```

Two real causes: E5 vectors occupy a narrow cone (anisotropy), so absolute cosine
carries little information; and MS MARCO is web-scale, so "capital of France"
genuinely retrieves French geography. Rank-shape signals were then scored by
Youden's J:

| Signal | J | Keeps in-corpus | Admits off-topic |
|---|---:|---:|---:|
| dense score std (top 20) | 0.648 | 72% | 8% |
| dense gap (top1 − top10) | 0.598 | 68% | 8% |
| max BM25 score | 0.236 | 31% | 8% |
| dense/sparse rank overlap | 0.188 | 65% | 46% |

The best still costs ~28% false refusals. So the design follows the evidence:
the retrieval layer blocks only degenerate results, the shape signals are surfaced
as **advisory confidence** (high/medium/low), and the real refusal weight moves to
the model's grounded self-assessment — it has seen the actual passages and judges
"these do not answer this" far better than any threshold — plus post-hoc
groundedness verification of what it wrote.

### The model's own confidence score does not work either

The next obvious knob is the `confidence` the model returns alongside its answer.
Measured over 50 MS MARCO rows split evenly answerable/unanswerable, it carries
no usable signal — on every case it chose to answer, it reported the same narrow
high band regardless of whether the question was answerable:

```
answerable,   answered (n=16):  0.9 - 1.0
unanswerable, answered (n=10):  0.9 - 1.0
```

Sweeping the threshold is consequently flat: false refusal plus false confidence
sums to 0.760 at every value from 0.25 to 0.90, then *worsens* to 0.840 at 0.95
where the only cases it begins rejecting are correct answers. The floor stays low
to catch a degenerate `confidence: 0`; it is not a tuning knob, and presenting it
as one would be false precision.

Both calibration attempts landed the same way — the plausible scalar signal
(retrieval cosine, then model confidence) proved uninformative, and what actually
works is the model's binary `sufficient` judgement plus lexical verification of
its answer against the evidence.

---

## The harness

`backend/rag/pipeline.py` is staged, not a single call. Each stage is timed,
individually failable, and wrapped in a guardrail decision. Every response
carries a `trace_id`, per-stage timings, and the full guardrail audit trail with
the signals each layer measured, so a refusal can always be explained.

**Degradation is layered, not binary:**

- LLM errors or times out → serve the extractive answer, mark `degraded`. An
  outage never becomes a failed request.
- LLM answer fails the grounding check → serve the extractive answer, because
  verbatim retrieved text is strictly better than both a hallucination and a
  refusal.
- Only genuine "the corpus cannot answer this" refuses outright.

The extractive answer costs **0.24 ms** and does three jobs: the sub-millisecond
grounded answer, the circuit-breaker fallback, and a reference for judging
whether generation drifted from the evidence.

Also in the harness: structured JSON output validated against a schema with a
retry on malformed output, per-stage timeouts, capped `max_tokens`, and
rate-limit-aware jittered backoff. That last one came from measurement — ~25% of
rapid sequential requests to Gemini's free tier return HTTP 429; the retry policy
took observed failures from **5/20 to 0/15**.

---

## LLM provider: chosen by measurement

`generation.py`/`llm.py` speak the OpenAI-compatible protocol, so the provider is
configuration, not code. Published TTFT figures were not usable — they are
typically measured at ~10K-token inputs (this pipeline sends ~1.5K) and often
include reasoning time. So all three were measured directly:

| Provider | Model | TTFT P50 | Hindi total | Script match | Groundedness |
|---|---|---:|---:|---:|---:|
| **Gemini** | gemini-3.5-flash-lite | 743 ms | **774 ms** | **3/3** | **0.90–1.0** |
| OpenAI | gpt-4.1-mini | 661 ms | 2133 ms | 3/3 | 0.77–0.96 |
| Groq | openai/gpt-oss-20b | **572 ms** | 557 ms | **1/3** | 0.0 (hi/mr) |

Gemini ships: it completes a Hindi answer ~3x faster than OpenAI with equal script
fidelity and the best grounding. Groq is fastest to first token but **cannot answer
in Devanagari at all**, which disqualifies it for an Indic deployment regardless of
speed. The others stay wired up as documented failover.

Worth noting for anyone reproducing this: the obvious model names from
documentation are already stale. Groq no longer serves any Llama chat model, and
`gemini-2.5-flash-lite` is retired for new users. Verify against each provider's
`/models` endpoint.

Reproduce: `python scripts/compare_providers.py`

---

## Reproducing from scratch

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements-build.txt   # Windows
# source venv/bin/activate && pip install -r requirements-build.txt

cp .env.example .env        # add SARVAM_API_KEY and LLM_API_KEY

python scripts/download_encoder.py                      # 130 MB, int8 ONNX
python scripts/download_dataset.py --languages hin,mar  # 880 MB, validation split
python scripts/extract_subset.py --languages hi,en,mr --rows 20000
python scripts/measure_lengths.py                       # calibrate thresholds
python scripts/chunk_corpus.py --strategies adaptive
python scripts/embed_corpus.py --languages hi,en,mr     # ~2 h CPU, resumable
python scripts/build_indexes.py --verify-recall 200

python -m uvicorn backend.app:app --port 7860
```

**The validation split, not train.** Validation shards are ~440 MB against
~3.7 GB for train — 8x smaller — and still contain the full MS MARCO dev query
set (97,941 queries), far more than this corpus needs.

**English costs no extra download.** Every shard carries
`passages.English_passages` alongside `passages.Translated_passages`, so the
Hindi shard yields two serving languages *and* the aligned parallel corpus the
cross-lingual evaluation depends on.

### Verification

```bash
python -m pytest tests/ -q              # 182 passing
python -m benchmarks.benchmark 60       # organizers' script, unmodified
python scripts/benchmark.py             # three-tier P50/P70/P100
python scripts/evaluate_retrieval.py    # recall + cross-lingual
```

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | state, full serving config, guardrail config, live index stats |
| `POST /api/query` | text and/or audio → answer, sources, timings, guardrail trail |
| `POST /api/query/stream` | SSE: `retrieval` → `extractive` → `token`* → `done` |
| `POST /api/transcribe` | audio → transcript |

`POST /api/query` with `use_llm=false` returns the extractive answer only —
**12.6 ms** end to end through HTTP, useful for demonstrating the retrieval core
without a provider round trip.

The streaming endpoint emits events in the order they become available, so the
grounded extractive answer arrives roughly 30x sooner than the first generated
token.

---

## What I would do next

- **Character n-gram sparse index.** Word-level BM25 is weak on
  machine-translated Devanagari (Marathi R@10 0.367). Char 3–5 grams typically
  beat it substantially on morphologically rich script, at ~2 ms with the same
  `bm25s` machinery.
- **Reduce sparse latency.** At 14.4 ms it is now 65% of Tier A, having started
  as the single worst stage at 398 ms.
- **Trained multilingual moderation** in place of the pattern-based unsafe-content
  filter, which is demo-grade and easy to defeat (stated plainly in
  `guardrails.py`).
- **Colocate with the LLM region** to cut Tier C, which is currently dominated by
  an India→US round trip.

## Known limitations

- Retrieval quality on Hindi and Marathi is bounded by machine-translation noise
  in the dataset itself; English R@10 is 0.887 against Hindi 0.627 on identical
  documents.
- The unsafe-content filter is pattern-based, not a trained classifier.
- Streaming gives up the structured `sufficient` flag, so the generation-layer
  guardrail cannot run on the streamed path. Input and retrieval guards still
  apply, and groundedness is verified on the completed text.
- No authentication or rate limiting on the API. It is a public read-only demo
  over a fixed corpus, storing no user data — but it should not be deployed as-is
  in front of a paid LLM key without a rate limit.
- Single worker by design: the indexes are ~1.4 GB resident, so extra workers
  would multiply memory rather than throughput.

---

## Layout

```
backend/rag/
  config.py       env-driven config, language registry
  encoder.py      int8 ONNX encoder, query cache
  dense.py        FAISS HNSW per language
  sparse.py       bm25s per language
  retrieval.py    RRF fusion, script routing, diversity cap
  extractive.py   verbatim grounded answer (0.24 ms)
  guardrails.py   four layers + measured evidence for the design
  llm.py          pooled/streaming/structured provider client
  pipeline.py     the harness
  stt.py          Sarvam
backend/app.py    FastAPI
app/              compatibility shim for the organizers' benchmark
scripts/          data pipeline, indexing, evaluation, benchmarks
results/          committed measurement outputs
```

Datasets and models used: [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
via the [`Xenova`](https://huggingface.co/Xenova/multilingual-e5-small) ONNX export.

`#RAGInGoa`
