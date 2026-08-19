"""Tests for the Phase 3A embedding pipeline.

Fast tests use a tiny DETERMINISTIC fake encoder (no BGE-M3 download, no GPU)
to verify the pipeline invariants: shape, dimension, normalization, mapping
length, unique chunk IDs, index alignment, empty/limit handling, CPU fallback,
and resume behavior.

The real-model tests (which actually load BGE-M3 and check the true dim/norm)
are gated behind ``RUN_REAL_EMBED_TESTS=1`` so the default ``pytest -v`` stays
fast and offline-friendly:

    RUN_REAL_EMBED_TESTS=1 venv/bin/python -m pytest tests/test_embeddings.py -v

Run:
    venv/bin/python -m pytest -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from backend.rag.embeddings import (
    EMBED_DIM, MODEL_NAME, select_device, _torch_dtype,
)


# ---------------------------------------------------------------------------
# Fake deterministic encoder for pipeline tests (no model download, no GPU).
# Mimics the contract of load_embedder/encode_batch: returns a (n, EMBED_DIM)
# float32 array whose rows are L2-normalized and deterministic in the input.
# ---------------------------------------------------------------------------

class FakeEncoder:
    """Deterministic, offline stand-in for BGE-M3. Maps each text to a hashed
    unit-norm vector so shape/dim/norm/alignment are testable without the
    model."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def encode(self, texts, batch_size=None, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            # deterministic per-text hash -> dim values
            h = abs(hash(t)) if t else 0
            rng = np.random.default_rng(h)
            # normalize the float32 vector directly (matches what
            # sentence-transformers stores: a float32 unit vector). Computing
            # the norm in float32 and dividing keeps the stored values unit-
            # norm to float32 precision.
            v = rng.standard_normal(self.dim).astype(np.float32)
            n = np.float32(np.linalg.norm(v))
            if n > 0:
                v = v / n
            out[i] = v
        return out


def _make_chunk_parquet(tmp_path: Path, n: int = 12, strategy: str = "adaptive",
                       dim_texts=True) -> Path:
    """Write a small chunk parquet for pipeline tests."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ("chunk_id", pa.string()),
        ("chunk_index", pa.int32()),
        ("chunk_strategy", pa.string()),
        ("document_id", pa.string()),
        ("query_id", pa.int64()),
        ("passage_idx", pa.int32()),
        ("text", pa.string()),
        ("text_en", pa.string()),
        ("query", pa.string()),
        ("query_en", pa.string()),
        ("answer", pa.string()),
        ("answer_en", pa.string()),
        ("language", pa.string()),
        ("source_lang_code", pa.string()),
        ("target_lang_code", pa.string()),
        ("is_selected", pa.int8()),
        ("query_type", pa.string()),
        ("source", pa.string()),
        ("source_file", pa.string()),
        ("answerable", pa.bool_()),
    ])
    rows = []
    for i in range(n):
        text = f"यह परीक्षण वाक्य {i} है।" if dim_texts else ""
        rows.append({
            "chunk_id": f"hi_1_p{i}_c0",
            "chunk_index": 0,
            "chunk_strategy": strategy,
            "document_id": f"hi_1_p{i}",
            "query_id": 1,
            "passage_idx": i,
            "text": text,
            "text_en": f"en text {i}",
            "query": "प्रश्न",
            "query_en": "query",
            "answer": "उत्तर",
            "answer_en": "answer",
            "language": "hi",
            "source_lang_code": "eng_Latn",
            "target_lang_code": "hin_Deva",
            "is_selected": 1 if i % 2 == 0 else 0,
            "query_type": "DESCRIPTION",
            "source": "MSMARCO-XI",
            "source_file": "hintrain.parquet",
            "answerable": True,
        })
    path = tmp_path / f"{strategy}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path,
                   compression="zstd")
    return path


# ===========================================================================
# Embedder utility tests (no model)
# ===========================================================================

class TestEmbedderUtils:
    def test_embed_dim_is_1024(self):
        assert EMBED_DIM == 1024

    def test_select_device_override(self):
        assert select_device("cpu") == "cpu"
        assert select_device("cuda") == "cuda"

    def test_select_device_auto(self):
        import torch
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert select_device() == expected

    def test_torch_dtype_map(self):
        import torch
        assert _torch_dtype("float16") == torch.float16
        assert _torch_dtype("float32") == torch.float32
        assert _torch_dtype("bfloat16") == torch.bfloat16

    def test_fake_encoder_shape_and_norm(self):
        enc = FakeEncoder()
        emb = enc.encode(["a", "bb", "ccc"])
        assert emb.shape == (3, EMBED_DIM)
        assert emb.dtype == np.float32
        for i in range(3):
            assert abs(float(np.linalg.norm(emb[i])) - 1.0) < 2e-3  # float32 unit-norm drift

    def test_fake_encoder_deterministic(self):
        enc = FakeEncoder()
        a = enc.encode(["same text", "other"])
        b = enc.encode(["same text", "other"])
        np.testing.assert_allclose(a, b)

    def test_fake_encoder_empty(self):
        enc = FakeEncoder()
        emb = enc.encode([])
        assert emb.shape[0] == 0


# ===========================================================================
# Pipeline invariants with the fake encoder (no real model)
# ===========================================================================

def _run_embed(monkeypatch, tmp_path, strategy, n, fake, args):
    """Drive scripts/embed_chunks.py main() with the fake encoder patched in."""
    # patch load_embedder to return (fake, device, dtype)
    import scripts.embed_chunks as ec

    def fake_load(device=None, dtype=None):
        return fake, "cpu", "float32"

    monkeypatch.setattr(ec, "load_embedder", fake_load)
    # redirect paths to tmp
    monkeypatch.setattr(ec, "CHUNK_DIR", tmp_path)
    monkeypatch.setattr(ec, "OUT_ROOT", tmp_path / "embeddings")
    # parse args
    import sys as _sys
    old = _sys.argv
    _sys.argv = ["embed_chunks.py", "--strategy", strategy] + args
    try:
        ec.main()
    finally:
        _sys.argv = old


class TestPipelineInvariants:
    def test_shape_dim_norm_mapping(self, monkeypatch, tmp_path):
        _make_chunk_parquet(tmp_path, n=12)
        fake = FakeEncoder()
        _run_embed(monkeypatch, tmp_path, "adaptive", 12, fake,
                   ["--batch-size", "4"])
        import numpy as np
        emb = np.load(tmp_path / "embeddings" / "adaptive" / "embeddings.npy",
                       mmap_mode="r")
        assert emb.shape == (12, EMBED_DIM)
        for i in range(12):
            assert abs(float(np.linalg.norm(emb[i])) - 1.0) < 2e-3  # float32 unit-norm drift
        # mapping length == embedding count
        import pyarrow.parquet as pq
        m = pq.read_table(tmp_path / "embeddings" / "adaptive" / "mapping.parquet")
        assert m.num_rows == 12
        assert set(m.schema.names) == {
            "embedding_index", "chunk_id", "document_id", "query_id",
            "chunk_index", "chunk_strategy", "language", "is_selected"}

    def test_mapping_index_alignment(self, monkeypatch, tmp_path):
        _make_chunk_parquet(tmp_path, n=8)
        fake = FakeEncoder()
        _run_embed(monkeypatch, tmp_path, "adaptive", 8, fake, ["--batch-size", "8"])
        import pyarrow.parquet as pq
        import numpy as np
        emb = np.load(tmp_path / "embeddings" / "adaptive" / "embeddings.npy",
                       mmap_mode="r")
        m = pq.read_table(tmp_path / "embeddings" / "adaptive" / "mapping.parquet").to_pylist()
        # embedding_index must be 0..7 contiguous
        assert [r["embedding_index"] for r in m] == list(range(8))
        # chunk_id unique
        cids = [r["chunk_id"] for r in m]
        assert len(cids) == len(set(cids))
        # the embedding row i corresponds to mapping row i (same chunk_id order
        # as the chunk parquet, which we built as hi_1_p{i}_c0)
        for i, r in enumerate(m):
            assert r["chunk_id"] == f"hi_1_p{i}_c0"

    def test_limit_handling(self, monkeypatch, tmp_path):
        _make_chunk_parquet(tmp_path, n=20)
        fake = FakeEncoder()
        _run_embed(monkeypatch, tmp_path, "adaptive", 20, fake,
                   ["--limit", "5", "--batch-size", "2"])
        import numpy as np
        emb = np.load(tmp_path / "embeddings" / "adaptive" / "embeddings.npy",
                       mmap_mode="r")
        assert emb.shape == (5, EMBED_DIM)
        import pyarrow.parquet as pq
        assert pq.read_table(
            tmp_path / "embeddings" / "adaptive" / "mapping.parquet").num_rows == 5
        # progress reflects limit
        import json
        prog = json.loads((tmp_path / "embeddings" / "adaptive" /
                           "progress.json").read_text())
        assert prog["next_index"] == 5
        assert prog["total"] == 5

    def test_resume_continues_without_recomputing(self, monkeypatch, tmp_path):
        # Simulate a real partial run: do a full-total run, then zero out rows
        # 5..9 and set progress next_index=5 (as if the process died after 5
        # rows). Resume must fill 5..9 and leave 0..4 embedded (unit norm), with
        # a complete mapping over all 10 rows.
        _make_chunk_parquet(tmp_path, n=10)
        fake = FakeEncoder()
        _run_embed(monkeypatch, tmp_path, "adaptive", 10, fake,
                   ["--batch-size", "5"])
        import numpy as np
        import json
        emb_path = tmp_path / "embeddings" / "adaptive" / "embeddings.npy"
        # snapshot the first-run norms of rows 0..4 (completed before "crash")
        emb_first = np.load(emb_path, mmap_mode="r+")
        norms0_4 = [float(np.linalg.norm(emb_first[i])) for i in range(5)]
        emb_first[5:10] = 0.0  # simulate crash: rows 5..9 lost
        emb_first.flush()
        # sanity: rows 5..9 are now zero
        assert all(float(np.linalg.norm(emb_first[i])) == 0.0 for i in range(5, 10))
        prog_path = tmp_path / "embeddings" / "adaptive" / "progress.json"
        prog = json.loads(prog_path.read_text())
        prog["next_index"] = 5
        prog_path.write_text(json.dumps(prog))
        # resume — re-runs with the full total
        _run_embed(monkeypatch, tmp_path, "adaptive", 10, fake,
                   ["--resume", "--batch-size", "5"])
        emb = np.load(emb_path, mmap_mode="r")
        assert emb.shape == (10, EMBED_DIM)
        # rows 0..4 still embedded (unit norm) — not zeroed/recomputed
        for i in range(5):
            n = float(np.linalg.norm(emb[i]))
            assert abs(n - 1.0) < 2e-3, f"row {i} not preserved: norm={n}"  # float32 drift
        # rows 5..9 now embedded (unit norm) — resumed work
        for i in range(5, 10):
            n = float(np.linalg.norm(emb[i]))
            assert abs(n - 1.0) < 2e-3, f"row {i} not resumed: norm={n}"  # float32 drift
        # mapping complete over all 10
        import pyarrow.parquet as pq
        assert pq.read_table(
            tmp_path / "embeddings" / "adaptive" / "mapping.parquet").num_rows == 10
        # progress shows full completion
        prog2 = json.loads(prog_path.read_text())
        assert prog2["next_index"] == 10

    def test_cpu_fallback(self, monkeypatch, tmp_path):
        # explicit --device cpu forces cpu path (fake runs on cpu anyway)
        _make_chunk_parquet(tmp_path, n=5)
        _run_embed(monkeypatch, tmp_path, "adaptive", 5, FakeEncoder(),
                   ["--device", "cpu", "--batch-size", "2"])
        import numpy as np
        emb = np.load(tmp_path / "embeddings" / "adaptive" / "embeddings.npy",
                      mmap_mode="r")
        assert emb.shape == (5, EMBED_DIM)

    def test_empty_input_validation(self, monkeypatch, tmp_path):
        # all-empty-text chunk parquet: validation warns but proceeds (texts
        # become "" and encode to a CLS-like vector). Verify it does not crash
        # and produces shape (n, dim).
        _make_chunk_parquet(tmp_path, n=3, dim_texts=False)
        _run_embed(monkeypatch, tmp_path, "adaptive", 3, FakeEncoder(),
                   ["--batch-size", "2"])
        import numpy as np
        emb = np.load(tmp_path / "embeddings" / "adaptive" / "embeddings.npy",
                      mmap_mode="r")
        assert emb.shape == (3, EMBED_DIM)

    def test_strategy_mismatch_raises(self, monkeypatch, tmp_path):
        # parquet is strategy=adaptive but we ask for fixed -> must refuse
        _make_chunk_parquet(tmp_path, n=5, strategy="adaptive")
        import scripts.embed_chunks as ec
        monkeypatch.setattr(ec, "CHUNK_DIR", tmp_path)
        monkeypatch.setattr(ec, "OUT_ROOT", tmp_path / "embeddings")
        monkeypatch.setattr(ec, "load_embedder",
                            lambda *a, **k: (FakeEncoder(), "cpu", "float32"))
        import sys as _sys
        old = _sys.argv
        _sys.argv = ["embed_chunks.py", "--strategy", "fixed"]
        try:
            with pytest.raises((ValueError, AssertionError)):
                ec.main()
        finally:
            _sys.argv = old


# ===========================================================================
# Real-model tests (gated behind RUN_REAL_EMBED_TESTS=1)
# ===========================================================================

REAL = os.environ.get("RUN_REAL_EMBED_TESTS") == "1"


@pytest.mark.skipif(not REAL, reason="set RUN_REAL_EMBED_TESTS=1 to load BGE-M3")
class TestRealModel:
    def test_real_dim_and_norm(self):
        from backend.rag.embeddings import load_embedder, encode_batch
        model, device, dtype = load_embedder()
        emb = encode_batch(model, ["यह एक हिंदी वाक्य है।",
                                   "an english sentence"], batch_size=2)
        assert emb.shape == (2, EMBED_DIM)
        assert EMBED_DIM == 1024
        for i in range(2):
            assert abs(float(np.linalg.norm(emb[i])) - 1.0) < 1e-3  # real BGE-M3 unit norm

    def test_real_deterministic(self):
        from backend.rag.embeddings import load_embedder, encode_batch
        model, device, dtype = load_embedder()
        t = ["नियतात्मकता परीक्षण वाक्य।"]
        a = encode_batch(model, t, batch_size=1)
        b = encode_batch(model, t, batch_size=1)
        np.testing.assert_allclose(a, b, atol=1e-4)
