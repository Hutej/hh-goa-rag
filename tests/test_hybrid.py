"""Tests for Phase 5 hybrid retrieval (RRF fusion).

Tests the pure ``rrf_fuse`` function with synthetic ranked lists — no Qdrant,
no BGE-M3, no real corpus. Fast and offline.

Run:
    venv/bin/python -m pytest tests/test_hybrid.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.rag.hybrid import rrf_fuse


def _hit(rank, cid, text=None, is_selected=0, document_id="d"):
    return {
        "rank": rank, "chunk_id": cid, "document_id": document_id,
        "text": text if text is not None else f"text-{cid}",
        "query_id": 1, "chunk_index": 0, "is_selected": is_selected,
        "score": 0.5, "strategy": "adaptive",
    }


class TestRRFFusion:

    def test_combines_two_ranked_lists(self):
        dense = [_hit(1, "a"), _hit(2, "b"), _hit(3, "c")]
        bm25 = [_hit(1, "b"), _hit(2, "a"), _hit(3, "d")]
        fused = rrf_fuse(dense, bm25, rrf_k=60)
        ids = {f["chunk_id"] for f in fused}
        assert ids == {"a", "b", "c", "d"}
        # a: dense rank1 + bm25 rank2; b: dense rank2 + bm25 rank1
        # both have the same RRF score (symmetric) -> tie, but both rank above c/d
        top2 = {f["chunk_id"] for f in fused[:2]}
        assert top2 == {"a", "b"}

    def test_doc_high_in_both_ranks_first(self):
        # 'a' is rank1 in both retrievers -> must be the overall winner
        dense = [_hit(1, "a"), _hit(2, "b")]
        bm25 = [_hit(1, "a"), _hit(2, "c")]
        fused = rrf_fuse(dense, bm25, rrf_k=60)
        assert fused[0]["chunk_id"] == "a"
        # rrf_score of a = 1/(60+1) + 1/(60+1) = 2/61 (rounded to 6 dp)
        assert abs(fused[0]["rrf_score"] - round(2 / 61, 6)) < 1e-9

    def test_missing_from_one_retriever_handled(self):
        # 'd' only in BM25 -> only the BM25 term contributes, dense_rank is None
        dense = [_hit(1, "a"), _hit(2, "b")]
        bm25 = [_hit(1, "a"), _hit(2, "d")]
        fused = rrf_fuse(dense, bm25, rrf_k=60)
        by_id = {f["chunk_id"]: f for f in fused}
        assert by_id["d"]["dense_rank"] is None
        assert by_id["d"]["bm25_rank"] == 2
        # d's score is just the bm25 term: 1/(60+2) (rounded to 6 dp)
        assert abs(by_id["d"]["rrf_score"] - round(1 / 62, 6)) < 1e-9
        # a (in both, rank1 both) still beats d (only bm25 rank2)
        assert fused[0]["chunk_id"] == "a"

    def test_weights_affect_ranking(self):
        # 'a' favored by dense, 'b' favored by bm25; equal ranks -> weight decides
        dense = [_hit(1, "a"), _hit(2, "b")]
        bm25 = [_hit(1, "b"), _hit(2, "a")]
        # up-weight dense -> a wins
        fused_d = rrf_fuse(dense, bm25, dense_weight=2.0, bm25_weight=1.0)
        assert fused_d[0]["chunk_id"] == "a"
        # up-weight bm25 -> b wins
        fused_b = rrf_fuse(dense, bm25, dense_weight=1.0, bm25_weight=2.0)
        assert fused_b[0]["chunk_id"] == "b"

    def test_metadata_and_text_preserved(self):
        dense = [_hit(1, "a", text="hello", is_selected=1, document_id="doc-a")]
        bm25 = [_hit(1, "a", text="hello", is_selected=1, document_id="doc-a")]
        fused = rrf_fuse(dense, bm25, rrf_k=60)
        assert len(fused) == 1
        f = fused[0]
        assert f["chunk_id"] == "a"
        assert f["text"] == "hello"
        assert f["document_id"] == "doc-a"
        assert f["is_selected"] == 1
        assert f["query_id"] == 1
        assert f["chunk_index"] == 0
        assert f["strategy"] == "adaptive"
        assert f["dense_rank"] == 1
        assert f["bm25_rank"] == 1
        assert f["rank"] == 1

    def test_empty_inputs(self):
        assert rrf_fuse([], []) == []
        assert len(rrf_fuse([_hit(1, "a")], [])) == 1

    def test_rrf_k_changes_score_magnitude(self):
        dense = [_hit(1, "a")]
        bm25 = [_hit(1, "a")]
        small_k = rrf_fuse(dense, bm25, rrf_k=1)[0]["rrf_score"]
        big_k = rrf_fuse(dense, bm25, rrf_k=60)[0]["rrf_score"]
        # smaller k -> larger score (1/(1+1)+1/(1+1)=1.0 vs 2/61)
        assert small_k > big_k
        assert abs(small_k - 1.0) < 1e-9
