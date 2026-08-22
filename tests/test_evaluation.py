"""Tests for Phase 6 evaluation (Recall@k, hit detection, sampling).

Pure/offline — no Qdrant, no BGE-M3. Builds a tiny in-memory corpus and checks
the evaluation primitives.

Run:
    venv/bin/python -m pytest tests/test_evaluation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Legacy module: evaluation.py targets the old single-language Qdrant/BM25 stack.
# Current evaluation lives in scripts/evaluate_retrieval.py (multilingual +
# cross-lingual). Skip when the legacy dependencies are not installed.
pytest.importorskip("qdrant_client",
                    reason="legacy dependency; see requirements-build.txt")
pytest.importorskip("rank_bm25",
                    reason="legacy dependency; see requirements-build.txt")

import backend.rag.evaluation as ev


# ---------------------------------------------------------------------------
# hit_at_k + recall_at_k
# ---------------------------------------------------------------------------

class TestHitAndRecall:

    def test_hit_at_k_found_in_topk(self):
        assert ev.hit_at_k(["a", "b", "c"], {"b"}, 5) == 1

    def test_hit_at_k_not_found(self):
        assert ev.hit_at_k(["a", "b", "c"], {"z"}, 3) == 0

    def test_hit_at_k_only_counts_topk(self):
        # relevant is at rank 6, beyond k=5 -> miss
        assert ev.hit_at_k(["a", "b", "c", "d", "e", "r"], {"r"}, 5) == 0
        assert ev.hit_at_k(["a", "b", "c", "d", "e", "r"], {"r"}, 6) == 1

    def test_hit_multiple_relevant_one_hit_suffices(self):
        assert ev.hit_at_k(["a", "x"], {"a", "y"}, 2) == 1

    def test_recall_at_k_mean(self):
        # 3 queries: hit, hit, miss -> recall = 2/3
        assert abs(ev.recall_at_k([1, 1, 0]) - 2 / 3) < 1e-9

    def test_recall_at_k_empty(self):
        assert ev.recall_at_k([]) == 0.0

    def test_recall_at_k_all_hit(self):
        assert ev.recall_at_k([1, 1, 1]) == 1.0


# ---------------------------------------------------------------------------
# load_eval_corpus (only evaluatable queries kept)
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus_fx(monkeypatch, tmp_path):
    """Write a tiny chunks parquet with one query that has a selected passage
    and one that has none."""
    rows = []
    # query 1: chunks a,b,c ; 'b' selected -> evaluatable
    for i, (cid, sel) in enumerate([("c1_a", 0), ("c1_b", 1), ("c1_c", 0)]):
        rows.append({"query_id": 1, "is_selected": sel, "query": "क्वेरी एक",
                     "query_en": "query one", "query_type": "DESCRIPTION",
                     "chunk_id": cid})
    # query 2: chunks d,e ; none selected -> NOT evaluatable
    for cid in ["c2_d", "c2_e"]:
        rows.append({"query_id": 2, "is_selected": 0, "query": "क्वेरी दो",
                     "query_en": "query two", "query_type": "NUMERIC",
                     "chunk_id": cid})
    schema = pa.schema([("query_id", pa.int64()), ("is_selected", pa.int8()),
                        ("query", pa.string()), ("query_en", pa.string()),
                        ("query_type", pa.string()), ("chunk_id", pa.string())])
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                   chunk_dir / "adaptive.parquet")
    monkeypatch.setattr(ev, "CHUNK_DIR", chunk_dir)
    return {"dir": chunk_dir}


class TestEvalCorpus:

    def test_only_evaluatable_queries_kept(self, corpus_fx):
        c = ev.load_eval_corpus("adaptive")
        assert 1 in c
        assert 2 not in c  # no selected passage
        assert c[1]["relevant"] == {"c1_b"}
        assert c[1]["query"] == "क्वेरी एक"
        assert c[1]["query_type"] == "DESCRIPTION"

    def test_none_hindi_query_falls_back_to_english(self, monkeypatch, tmp_path):
        # query 3: Hindi query is None, English present, one selected -> kept,
        # and the usable query text is the English fallback
        rows = [{"query_id": 3, "is_selected": 1, "query": None,
                 "query_en": "what is a gaucho?", "query_type": "ENTITY",
                 "chunk_id": "c3_x"}]
        chunk_dir = tmp_path / "chunks"; chunk_dir.mkdir()
        schema = pa.schema([("query_id", pa.int64()), ("is_selected", pa.int8()),
                           ("query", pa.string()), ("query_en", pa.string()),
                           ("query_type", pa.string()), ("chunk_id", pa.string())])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                       chunk_dir / "adaptive.parquet")
        monkeypatch.setattr(ev, "CHUNK_DIR", chunk_dir)
        c = ev.load_eval_corpus("adaptive")
        assert 3 in c
        assert c[3]["query"] == "what is a gaucho?"
        assert c[3]["query_hi"] is None
        assert c[3]["query_en"] == "what is a gaucho?"

    def test_query_with_no_text_at_all_is_dropped(self, monkeypatch, tmp_path):
        # both query and query_en empty -> not usable (dense can't encode None)
        rows = [{"query_id": 4, "is_selected": 1, "query": None,
                 "query_en": None, "query_type": "ENTITY", "chunk_id": "c4_x"}]
        chunk_dir = tmp_path / "chunks"; chunk_dir.mkdir()
        schema = pa.schema([("query_id", pa.int64()), ("is_selected", pa.int8()),
                           ("query", pa.string()), ("query_en", pa.string()),
                           ("query_type", pa.string()), ("chunk_id", pa.string())])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                       chunk_dir / "adaptive.parquet")
        monkeypatch.setattr(ev, "CHUNK_DIR", chunk_dir)
        c = ev.load_eval_corpus("adaptive")
        assert 4 not in c

    def test_sample_queries_deterministic(self, corpus_fx):
        c = ev.load_eval_corpus("adaptive")
        s1 = ev.sample_queries(c, n=1, seed=99)
        s2 = ev.sample_queries(c, n=1, seed=99)
        assert s1 == s2  # same seed -> same sample

    def test_sample_all_when_n_large(self, corpus_fx):
        c = ev.load_eval_corpus("adaptive")
        s = ev.sample_queries(c, n=1000, seed=1)
        assert s == sorted(c.keys())

    def test_empty_corpus_sample(self, monkeypatch, tmp_path):
        # no rows at all -> no evaluatable queries
        chunk_dir = tmp_path / "chunks"; chunk_dir.mkdir()
        schema = pa.schema([("query_id", pa.int64()), ("is_selected", pa.int8()),
                            ("query", pa.string()), ("query_en", pa.string()),
                            ("query_type", pa.string()), ("chunk_id", pa.string())])
        pq.write_table(pa.Table.from_pylist([], schema=schema),
                       chunk_dir / "adaptive.parquet")
        monkeypatch.setattr(ev, "CHUNK_DIR", chunk_dir)
        c = ev.load_eval_corpus("adaptive")
        assert c == {}
        assert ev.sample_queries(c, n=10, seed=1) == []


# ---------------------------------------------------------------------------
# dense_bruteforce correctness (exact cosine == top-k by dot product)
# ---------------------------------------------------------------------------

class TestBruteforce:

    def test_exact_match_returns_same_vector_first(self, monkeypatch, tmp_path):
        import numpy as np
        emb_dir = tmp_path / "embeddings" / "adaptive"; emb_dir.mkdir(parents=True)
        # 3 unit vectors; query == row 1 -> row 1 ranks first
        rng = np.random.default_rng(0)
        v = rng.standard_normal((3, 4)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        np.save(emb_dir / "embeddings.npy", v)
        # mapping: row index -> chunk_id
        mp = pa.table({"embedding_index": [0, 1, 2],
                       "chunk_id": ["c0", "c1", "c2"]})
        pq.write_table(mp, emb_dir / "mapping.parquet")
        monkeypatch.setattr(ev, "EMB_ROOT", tmp_path / "embeddings")
        ids = ev.dense_bruteforce("adaptive", v[1], top_k=3)
        assert ids[0] == "c1"
        assert set(ids) == {"c0", "c1", "c2"}
