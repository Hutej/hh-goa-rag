"""Tests for Phase 3B Qdrant indexing + dense retrieval.

Minimal, synthetic (no 200k indexing, no real BGE-M3). A tiny 20-vector / 8-dim
fixture is written to tmp_path and the Qdrant store is pointed there. Covers the
essentials: indexing + count, payload round-trip, retrieval ordering, only-
selected filter, and idempotent re-upsert.

Run:
    venv/bin/python -m pytest tests/test_qdrant.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Legacy module: superseded by backend/rag/dense.py (in-process FAISS HNSW).
# Embedded Qdrant turned out to be a pure-Python brute-force scan that ignores
# hnsw_config, and the client was reopened per request. `qdrant-client` is now a
# build-time-only dependency, so skip rather than fail when it is absent.
pytest.importorskip("qdrant_client",
                    reason="legacy dependency; superseded by dense.py (FAISS)")

import backend.rag.qdrant_index as qi


# ---------------------------------------------------------------------------
# Tiny synthetic fixture: 20 vectors, 8-dim, written to tmp_path.
# ---------------------------------------------------------------------------

DIM = 8
N = 20


def _unit(rng_seed: int) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v).astype(np.float32)


def _write_fixture(tmp_path: Path) -> dict:
    """Write embeddings.npy + mapping.parquet + chunks.parquet under tmp_path.

    Returns the dirs and the deterministic vectors so tests can build queries.
    """
    emb_dir = tmp_path / "embeddings" / "adaptive"
    chunk_dir = tmp_path / "chunks"
    qdrant_dir = tmp_path / "qdrant"
    emb_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    vecs = np.stack([_unit(i) for i in range(N)])
    np.save(emb_dir / "embeddings.npy", vecs)

    mapping = {
        "embedding_index": pa.array(list(range(N)), pa.int64()),
        "chunk_id": pa.array([f"hi_{i}_p0_c0" for i in range(N)], pa.string()),
        "document_id": pa.array([f"hi_{i}_p0" for i in range(N)], pa.string()),
        "query_id": pa.array(list(range(100, 100 + N)), pa.int64()),
        "chunk_index": pa.array([0] * N, pa.int32()),
        "chunk_strategy": pa.array(["adaptive"] * N, pa.string()),
        "language": pa.array(["hi"] * N, pa.string()),
        "is_selected": pa.array([1 if i % 2 == 0 else 0 for i in range(N)], pa.int8()),
    }
    pq.write_table(pa.table(mapping), emb_dir / "mapping.parquet")

    chunk = {
        "chunk_id": mapping["chunk_id"],
        "document_id": mapping["document_id"],
        "query_id": mapping["query_id"],
        "chunk_index": mapping["chunk_index"],
        "chunk_strategy": mapping["chunk_strategy"],
        "language": mapping["language"],
        "is_selected": mapping["is_selected"],
        "text": pa.array([f"पाठ {i}" for i in range(N)], pa.string()),
    }
    pq.write_table(pa.table(chunk), chunk_dir / "adaptive.parquet")
    return {"vecs": vecs}


@pytest.fixture
def fx(monkeypatch, tmp_path):
    """Write the synthetic fixture and repoint qdrant_index path constants at it."""
    f = _write_fixture(tmp_path)
    monkeypatch.setattr(qi, "EMB_ROOT", tmp_path / "embeddings")
    monkeypatch.setattr(qi, "CHUNK_DIR", tmp_path / "chunks")
    monkeypatch.setattr(qi, "QDRANT_DIR", tmp_path / "qdrant")
    monkeypatch.setattr(qi, "EMBED_DIM", DIM)  # allow the 8-dim fixture
    return f


def _index_all(f):
    """Index all N fixture points into the adaptive collection. Returns (client, name)."""
    client = qi.get_client()
    try:
        qi.ensure_collection(client, "adaptive", dim=DIM)
        name = qi.collection_name("adaptive")
        from qdrant_client.http.models import PointStruct
        for pid, vec, payload in qi.stream_points("adaptive", 0, N):
            client.upsert(name, points=[PointStruct(id=pid, vector=vec, payload=payload)])
        return client, name
    except Exception:
        client.close()
        raise


class TestIndexing:

    def test_index_count_matches(self, fx):
        client, name = _index_all(fx)
        try:
            assert client.count(name).count == N
        finally:
            client.close()

    def test_payload_roundtrip(self, fx):
        client, name = _index_all(fx)
        try:
            pts, _ = client.scroll(name, limit=N, with_payload=True, with_vectors=False)
            by_id = {p.id: p.payload for p in pts}
            assert by_id[0]["chunk_id"] == "hi_0_p0_c0"
            assert by_id[0]["document_id"] == "hi_0_p0"
            assert by_id[0]["query_id"] == 100
            assert by_id[0]["chunk_index"] == 0
            assert by_id[0]["chunk_strategy"] == "adaptive"
            assert by_id[0]["language"] == "hi"
            assert by_id[0]["is_selected"] == 1
            assert by_id[0]["text"] == "पाठ 0"
        finally:
            client.close()

    def test_idempotent_reupsert(self, fx):
        client, name = _index_all(fx)
        try:
            from qdrant_client.http.models import PointStruct
            # re-upsert id=5 with a different text -> count stable, payload overwritten
            new_payload = {k: "x" for k in qi.PAYLOAD_FIELDS}
            new_payload["text"] = "CHANGED"
            client.upsert(name, points=[PointStruct(
                id=5, vector=fx["vecs"][5].tolist(), payload=new_payload)])
            assert client.count(name).count == N  # no duplicate points
            pts, _ = client.scroll(name, limit=N, with_payload=True)
            by_id = {p.id: p.payload for p in pts}
            assert by_id[5]["text"] == "CHANGED"
        finally:
            client.close()


class TestRetrieval:

    def test_retrieval_ordering(self, fx):
        client, name = _index_all(fx)
        try:
            # query with point 0's exact vector -> point 0 rank 0, score ~1.0
            res = client.query_points(name, query=fx["vecs"][0].tolist(),
                                      limit=3, with_payload=True)
            assert res.points[0].id == 0
            assert abs(float(res.points[0].score) - 1.0) < 2e-3
            assert res.points[0].payload["chunk_id"] == "hi_0_p0_c0"
        finally:
            client.close()

    def test_only_selected_filter(self, fx):
        client, name = _index_all(fx)
        try:
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue
            f = Filter(must=[FieldCondition(key="is_selected", match=MatchValue(value=1))])
            res = client.query_points(name, query=fx["vecs"][0].tolist(),
                                      limit=N, query_filter=f, with_payload=True)
            assert all(p.payload["is_selected"] == 1 for p in res.points)
            assert len(res.points) == N // 2
        finally:
            client.close()


class TestValidation:

    def test_validate_artifacts_ok(self, fx):
        n = qi.validate_artifacts("adaptive")
        assert n == N
