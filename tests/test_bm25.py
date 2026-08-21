"""Tests for Phase 4 BM25 lexical retrieval.

Minimal, synthetic, in-process (no real corpus, no heavy deps). Writes a tiny
chunk parquet to tmp_path and points the BM25 path constants at it.

Run:
    venv/bin/python -m pytest tests/test_bm25.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import backend.rag.bm25 as bm25mod
from backend.rag.bm25 import BM25Index, tokenize

# ---------------------------------------------------------------------------
# Tiny synthetic corpus (6 docs). Hindi + English so the tokenizer's
# Devanagari handling is exercised.
# ---------------------------------------------------------------------------

DOCS = [
    "मैनहट्टन परियोजना द्वितीय विश्व युद्ध के दौरान परमाणु बम विकसित करने की परियोजना थी।",
    "परमाणु ऊर्जा का शांतिपूर्ण उपयोग विज्ञान के लिए महत्वपूर्ण है।",
    "The Manhattan Project developed the first atomic bombs during World War II.",
    "भारत की स्वतंत्रता 1947 में हुई थी।",
    "द्वितीय विश्व युद्ध 1939 से 1945 तक चला।",
    "सूर्य हमारे सौर मंडल के केंद्र में है।",
]


@pytest.fixture
def fx(monkeypatch, tmp_path):
    """Write a tiny adaptive chunk parquet and repoint BM25 path constants."""
    chunk_dir = tmp_path / "chunks"
    bm_dir = tmp_path / "bm25"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, text in enumerate(DOCS):
        rows.append({
            "chunk_id": f"hi_{i}_p0_c0",
            "document_id": f"hi_{i}_p0",
            "query_id": 100 + i,
            "chunk_index": 0,
            "is_selected": 1 if i % 2 == 0 else 0,
            "text": text,
        })
    schema = pa.schema([
        ("chunk_id", pa.string()),
        ("document_id", pa.string()),
        ("query_id", pa.int64()),
        ("chunk_index", pa.int32()),
        ("is_selected", pa.int8()),
        ("text", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                   chunk_dir / "adaptive.parquet")
    monkeypatch.setattr(bm25mod, "CHUNK_DIR", chunk_dir)
    monkeypatch.setattr(bm25mod, "BM25_DIR", bm_dir)
    return {"bm_dir": bm_dir}


@pytest.fixture
def idx(fx):
    return BM25Index.load("adaptive")


class TestTokenizer:

    def test_hindi_kept_whole(self):
        toks = tokenize("मैनहट्टन परियोजना।")
        assert "मैनहट्टन" in toks
        assert "परियोजना" in toks

    def test_lowercase_and_strip_punct(self):
        toks = tokenize("The Manhattan-Project, 1942.")
        assert "the" in toks
        # internal hyphen is preserved (hyphenated compounds are lexical units)
        assert "manhattan-project" in toks
        assert "1942" in toks

    def test_empty_and_short_dropped(self):
        assert tokenize("") == []
        assert tokenize("a । b") == []


class TestBM25:

    def test_index_built_and_count(self, idx):
        assert idx.n == len(DOCS)

    def test_exact_keyword_ranks_matching_first(self, idx):
        hits = idx.query("मैनहट्टन परियोजना", top_k=3)
        assert hits, "expected hits"
        assert hits[0]["chunk_id"] == "hi_0_p0_c0"
        assert hits[0]["score"] > 0

    def test_top_k_length(self, idx):
        hits = idx.query("विश्व युद्ध", top_k=2)
        assert len(hits) <= 2
        assert len(hits) >= 1

    def test_metadata_and_text_returned(self, idx):
        hits = idx.query("Manhattan Project", top_k=1)
        assert hits
        h = hits[0]
        assert h["chunk_id"] == "hi_2_p0_c0"
        assert h["document_id"] == "hi_2_p0"
        assert h["query_id"] == 102
        assert h["chunk_index"] == 0
        assert "Manhattan" in h["text"]
        assert h["strategy"] == "adaptive"
        assert "rank" in h and h["rank"] == 1

    def test_only_selected_filter(self, idx):
        hits = idx.query("विश्व युद्ध", top_k=10, only_selected=True)
        assert all(h["is_selected"] == 1 for h in hits)

    def test_no_hits_for_unmatched_query(self, idx):
        hits = idx.query("zzzznomatch", top_k=5)
        assert hits == []

    def test_persistence_and_reload(self, fx):
        # build once
        BM25Index.load("adaptive")
        # second load uses the persisted files (no rebuild)
        idx2 = BM25Index.load("adaptive")
        assert idx2.n == len(DOCS)
        h = idx2.query("मैनहट्टन", top_k=1)
        assert h and h[0]["chunk_id"] == "hi_0_p0_c0"
        # persisted files exist
        assert (fx["bm_dir"] / "adaptive" / "bm25.pkl").exists()
        assert (fx["bm_dir"] / "adaptive" / "metadata.parquet").exists()
