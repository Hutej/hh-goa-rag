# Experiment 2 — RRF Weight Sweep

Strategy: **adaptive**. Same 100 deterministic queries. BGE-M3 on cuda (float16).

| Config | dense_w | bm25_w | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|
| dense_only | 1.0 | 0.0 | 0.26 | 0.64 | 0.72 |
| d1.0_b0.25 | 1.0 | 0.25 | 0.27 | 0.61 | 0.74 |
| d1.0_b0.5 | 1.0 | 0.5 | 0.26 | 0.6 | 0.73 |
| d1.0_b1.0 | 1.0 | 1.0 | 0.24 | 0.56 | 0.72 |
| d2.0_b1.0 | 2.0 | 1.0 | 0.26 | 0.6 | 0.73 |

**Best: d1.0_b0.25** (dense_w=1.0, bm25_w=0.25).
