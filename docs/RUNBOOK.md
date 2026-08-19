# Friend Runbook — from clone to embeddings

Reproduce the full Phase 1 → 2 → 3A pipeline on a fresh machine. Everything is
CLI/config driven — no source edits needed to change GPU, batch size, paths, or
strategy. The dataset is downloaded from Hugging Face (`ai4bharat/MSMARCO-XI`),
so no manual dataset setup is required.

## 0. What you need
* Python 3.11+ (3.14 verified on the dev box)
* An NVIDIA GPU is recommended but NOT required — the same code runs on CPU.
* ~25 GB free disk: ~7 GB raw dataset, ~0.5 GB processed chunks, ~3 GB BGE-M3
  model cache, ~0.8 GB embeddings per strategy (×3).
* The first BGE-M3 run downloads ~2.3 GB of model weights (cached after).

## 1. Clone + virtual env + dependencies
```bash
git clone https://github.com/Hutej/hh-goa-rag.git
cd hh-goa-rag

python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Verify CUDA is seen:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
If CUDA shows False on an NVIDIA machine, reinstall torch with the CUDA index:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

## 2. Download the dataset (Hugging Face — ~3.4 GB, first time only)
```bash
python scripts/download_dataset.py
```
Downloads `train/hintrain.parquet` + `train/martrain.parquet` from
`ai4bharat/MSMARCO-XI` into `data/raw/MSMARCO-XI/`. Deterministic — the raw
files are byte-identical to the dev box, so the first-20k subset reproduces
exactly.

## 3. Phase 1 — extract + validate the 20k Hindi subset (~5 min, ~1.3 GB RAM)
```bash
python scripts/extract_subset.py --rows 20000
python scripts/validate_subset.py
```
Output: `data/processed/hh_subset_hin.parquet` (199,590 passages). The
validate script must print `ALL CHECKS PASSED`.

## 4. Phase 2 — chunk all three strategies (~5 min total, ~1.3 GB RAM)
```bash
python scripts/chunk_subset.py
python scripts/validate_chunks.py
```
Output: `data/processed/chunks/{fixed,semantic,adaptive}.parquet`
(204,932 / 203,621 / 204,390 chunks). The validate script must print
`ALL STRATEGIES PASS` (42/42 checks).

## 5. (optional) Run the tests — fast, no model download
```bash
python -m pytest -v
# expect: 51 passed, 2 skipped (the 2 skipped need BGE-M3; see step 6)
```

## 6. Phase 3A — TEST MODE FIRST (100 chunks, ~30 s, verifies everything)
ALWAYS do this before the full run. Loads BGE-M3 (~2.3 GB download, first time
only), embeds the first 100 adaptive chunks, writes a small artifact:
```bash
python scripts/embed_chunks.py --strategy adaptive --limit 100 --batch-size 16
```
Verify the output:
```bash
python - <<'PY'
import numpy as np, pyarrow.parquet as pq, json
d="data/processed/embeddings/adaptive"
e=np.load(d+"/embeddings.npy", mmap_mode="r"); m=pq.read_table(d+"/mapping.parquet")
print("shape:", e.shape, "| mapping rows:", m.num_rows)
print("all unit-norm:", all(abs(float(np.linalg.norm(e[i]))-1.0)<1e-3 for i in range(e.shape[0])))
print("progress:", json.loads(open(d+"/progress.json").read())["next_index"])
PY
# expect: shape (100, 1024) | mapping rows 100 | all unit-norm True | progress 100
```

## 7. Phase 3A — FULL RUN, all three strategies
The full job on each strategy is long. Run each separately, or one at a time. 
**It is resumable** — if it
dies or you Ctrl-C, re-run the same command with `--resume` and it continues
without recomputing finished batches.
```bash
# adaptive (recommended default for retrieval)
python scripts/embed_chunks.py --strategy adaptive  --batch-size 16

# fixed + semantic (for the benchmark comparison)
python scripts/embed_chunks.py --strategy fixed     --batch-size 16
python scripts/embed_chunks.py --strategy semantic  --batch-size 16
```
To resume any of them after an interruption:
```bash
python scripts/embed_chunks.py --strategy adaptive --resume
```
If you ever hit a CUDA OOM (unlikely on an 8 GB RTX 4050), halve the batch:
```bash
python scripts/embed_chunks.py --strategy adaptive --batch-size 8
```

## 8. Output layout (all under data/processed/embeddings/, git-ignored)
```
data/processed/embeddings/{fixed,semantic,adaptive}/
  embeddings.npy     # memmap float32 (n_chunks, 1024)
  mapping.parquet    # embedding_index -> chunk_id, document_id, query_id,
                     #   chunk_index, chunk_strategy, language, is_selected
  progress.json      # resume state
  run_report.json    # measured performance for the run
```
The invariant is `embeddings[i] ↔ mapping[i]` (same index). Vectors and
metadata are kept separate, ready for the next phase (Qdrant indexing).

## 9. Real-model tests (optional, ~35 s, loads BGE-M3)
```bash
RUN_REAL_EMBED_TESTS=1 python -m pytest tests/test_embeddings.py::TestRealModel -v
```

## Notes
* **No source edits** are needed to change GPU, batch size, input/output path,
  or strategy — all via CLI flags.
* The embedding input is the chunk `text` ONLY (no query/answer/metadata
  concatenated), so the corpus vector represents the chunk itself.
* BGE-M3 loads once per process; don't load it per-chunk.
* Embeddings are L2-normalized (unit norm) by the pipeline — do NOT re-normalize.
* Nothing large is committed — embeddings, model cache, and raw data are all
  git-ignored.
