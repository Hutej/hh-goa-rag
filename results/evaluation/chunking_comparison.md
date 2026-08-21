# Experiment 1 — Chunking Comparison (dense Recall@k)

Same 100 deterministic stratified queries, dense-only (exact cosine == Qdrant ranking). BGE-M3 on cuda (float16).

| Strategy | R@1 | R@5 | R@10 | ms/query |
|---|---:|---:|---:|---:|
| fixed | 0.26 | 0.64 | 0.72 | 295.5 |
| semantic | 0.26 | 0.64 | 0.72 | 347.9 |
| adaptive | 0.26 | 0.64 | 0.72 | 320.5 |

**Result: all three strategies tie on dense Recall@1/5/10.** The selected
passage's text is found via cosine similarity regardless of how it is chunked
(overlapping chunks preserve the relevant content). The `ms/query` column is
bruteforce latency (dominated by the 204k×1024 dot product), not chunking cost,
so it is not a meaningful chunking differentiator here.

**Recommended: adaptive.** Recall is tied, and adaptive chunking is the
semantically most principled strategy (sentence-aware boundaries) — chosen as
the serving strategy for downstream experiments and the demo. The tie means
chunking choice is not a retrieval-quality lever on this subset.
