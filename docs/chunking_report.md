# Phase 2 — Chunking Report

Measured over the real 199,590-passage Hindi subset (`data/processed/hh_subset_hin.parquet`). Token counts are BGE-M3 content tokens (`add_special_tokens=False`). All numbers measured, not fabricated.

| Strategy | Input passages | Output chunks | Avg tokens/chunk | P50 tokens | P90 tokens | P95 tokens | Max tokens | Chunks > max | Empty | % split | % one-chunk | Time (s) | Passages/s | Peak RSS (MB) | Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 199590 | 204932 | 100.43 | 89 | 162 | 194 | 258 | 223 | 0 | 0.483 | 99.517 | 88.76 | 2248.6 | 1247.0 | 61.57 MiB |
| semantic | 199590 | 203621 | 100.67 | 89 | 160 | 191 | 384 | 0 | 0 | 0.212 | 99.788 | 93.09 | 2144.0 | 1247.0 | 61.63 MiB |
| adaptive | 199590 | 204390 | 100.61 | 89 | 162 | 194 | 382 | 0 | 0 | 0.212 | 99.788 | 98.46 | 2027.2 | 1247.0 | 61.54 MiB |

## Strategy details

### fixed
- input passages: 199590
- output chunks: 204932
- chunks/passage: avg 1.0268, median 1, min 1, max 37
- tokens/chunk: avg 100.43, median 89, P50 89, P90 162, P95 194, max 258
- chunks exceeding configured max (256): 223
- empty chunks: 0
- passages split into multiple chunks: 964 (0.483%)
- passages kept as one chunk: 198626 (99.517%)
- passages producing zero chunks: 0
- processing time: 88.76 s
- throughput: 2248.6 passages/s, 2308.8 chunks/s
- peak RSS: 1247.0 MB
- output: /home/hutej/System hang/hh-goa-rag/data/processed/chunks/fixed.parquet (61.57 MiB)

### semantic
- input passages: 199590
- output chunks: 203621
- chunks/passage: avg 1.0202, median 1, min 1, max 38
- tokens/chunk: avg 100.67, median 89, P50 89, P90 160, P95 191, max 384
- chunks exceeding configured max (384): 0
- empty chunks: 0
- passages split into multiple chunks: 423 (0.212%)
- passages kept as one chunk: 199167 (99.788%)
- passages producing zero chunks: 0
- processing time: 93.09 s
- throughput: 2144.0 passages/s, 2187.3 chunks/s
- peak RSS: 1247.0 MB
- output: /home/hutej/System hang/hh-goa-rag/data/processed/chunks/semantic.parquet (61.63 MiB)

### adaptive
- input passages: 199590
- output chunks: 204390
- chunks/passage: avg 1.024, median 1, min 1, max 37
- tokens/chunk: avg 100.61, median 89, P50 89, P90 162, P95 194, max 382
- chunks exceeding configured max (384): 0
- empty chunks: 0
- passages split into multiple chunks: 423 (0.212%)
- passages kept as one chunk: 199167 (99.788%)
- passages producing zero chunks: 0
- processing time: 98.46 s
- throughput: 2027.2 passages/s, 2075.9 chunks/s
- peak RSS: 1247.0 MB
- output: /home/hutej/System hang/hh-goa-rag/data/processed/chunks/adaptive.parquet (61.54 MiB)
