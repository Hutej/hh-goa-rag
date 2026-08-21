# Phase 7 — Grounded Answer Generation

query → hybrid retrieval → top-k chunks → LLM → grounded answer + sources.

The LLM answers **only** from retrieved context; if context is insufficient it
says so rather than inventing an answer. Source `chunk_id`s are preserved for
traceability (no fabricated citations).

## Model / provider

Configurable via environment variables (no hardcoded keys):

| env var | default | meaning |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` (OpenAI-compatible chat) or `echo` (offline mock, for tests) |
| `LLM_MODEL` | `ANTHROPIC_DEFAULT_SONNET_MODEL` if set, else `gpt-4o-mini` | model id |
| `LLM_BASE_URL` | `ANTHROPIC_BASE_URL` if set | OpenAI-compatible base URL |
| `LLM_API_KEY` | `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` if set | API key |

The default **reuses the project's already-configured OpenAI-compatible endpoint**
(no new provider introduced). The `openai` library (3.1.0) is used. For STT, the
`sarvamai` library is installed (a later phase).

## System prompt (grounding rules)

The model is told it is a grounded multilingual QA assistant that must:
1. not invent facts, 2. say context is insufficient if so, 3. prefer the question's
language, 4. be concise, 5. not expose retrieval internals, 6. use the provided
source chunk IDs (the caller attaches them).

## Context format

```
[Source 1]
chunk_id: ...
text: ...

[Source 2]
chunk_id: ...
text: ...
```

## Output (JSON)

```json
{
  "query": "...",
  "answer": "...",
  "sources": [{"chunk_id": "...", "document_id": "...", "score": ..., "text": "..."}],
  "retrieval_latency_ms": ...,
  "generation_latency_ms": ...,
  "total_latency_ms": ...,
  "provider": "..."
}
```

Source `score` is the hybrid RRF score. Sources are exactly the retrieved chunks —
never fabricated.

## CLI

```bash
venv/bin/python scripts/answer_query.py --query "मैनहट्टन परियोजना क्या थी?"
venv/bin/python scripts/answer_query.py --query "..." --top-k 5 --strategy adaptive --device cpu
```

Flags: `--query`, `--strategy` (adaptive), `--top-k` (5), `--dense-k` (20),
`--bm25-k` (20), `--rrf-k` (60), `--device`.

## Latency

Reports `retrieval_latency_ms`, `generation_latency_ms`, `total_latency_ms`.
LLM generation is expected to dominate. This is **not** a claim against the
official `<200 ms` end-to-end target (STT not yet integrated; final benchmark
comes after the demo pipeline exists).

## Guardrails (minimum)

- Empty/insufficient context → model returns an explicit insufficient-context
  message (does not hallucinate).
- Model/API failure → controlled `GenerationError` (CLI prints an error JSON,
  exits non-zero).
- Source traceability → retrieved `chunk_id`s are always preserved and returned.

A more complete guardrail layer can be added after STT integration if time permits.
