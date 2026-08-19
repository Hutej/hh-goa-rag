"""Unit tests for the canonical normalization logic (backend.rag.normalize).

Tests cover the pure functions: language mapping, answerable derivation,
document-id generation, passage alignment, variable passage counts, and the
normalized schema. These run with the project venv and need no large data:

    venv/bin/python -m pytest tests/test_normalize.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.rag.normalize import (
    CANONICAL_FIELDS,
    is_answerable,
    map_language,
    normalize_row,
)


def test_map_language():
    assert map_language("hin_Deva") == "hi"
    assert map_language("mar_Deva") == "mr"
    assert map_language("eng_Latn") == "en"
    assert map_language("xxx_Zzz") == "und"
    assert map_language(None) == "und"


def test_is_answerable():
    assert is_answerable([0, 0, 0]) is False
    assert is_answerable([0, 1, 0]) is True
    assert is_answerable([1, 1, 1]) is True
    assert is_answerable([]) is False
    assert is_answerable(None) is False


def _base_row(**over):
    row = {
        "query_id": 1185869,
        "query_type": "DESCRIPTION",
        "query": "कुछ प्रश्न",
        "Answer": "कुछ उत्तर",
        "Eng_Query": "some question",
        "Eng_Answer": "some answer",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "English_passages": ["eng0", "eng1"],
        "Translated_passages": ["hi0", "hi1"],
        "is_selected": [0, 1],
    }
    row.update(over)
    return row


def test_normalize_row_basic():
    docs = normalize_row(_base_row(), source_file="hintrain.parquet")
    assert len(docs) == 2
    d0 = docs[0]
    # schema exactly the canonical fields
    assert set(d0.keys()) == set(CANONICAL_FIELDS)
    # identity / derived fields
    assert d0["document_id"] == "hi_1185869_p0"
    assert d0["query_id"] == 1185869
    assert d0["passage_idx"] == 0
    assert d0["language"] == "hi"
    assert d0["source_lang_code"] == "eng_Latn"
    assert d0["target_lang_code"] == "hin_Deva"
    assert d0["source"] == "MSMARCO-XI"
    assert d0["source_file"] == "hintrain.parquet"
    assert d0["query_type"] == "DESCRIPTION"
    assert d0["answerable"] is True  # is_selected=[0,1] -> answerable
    # retrievable text mapping
    assert d0["text"] == "hi0"
    assert d0["text_en"] == "eng0"
    # query side carried through
    assert d0["query"] == "कुछ प्रश्न"
    assert d0["query_en"] == "some question"
    assert d0["answer"] == "कुछ उत्तर"
    assert d0["answer_en"] == "some answer"
    # relevance preserved
    assert docs[0]["is_selected"] == 0
    assert docs[1]["is_selected"] == 1
    assert docs[1]["passage_idx"] == 1
    assert docs[1]["document_id"] == "hi_1185869_p1"


def test_normalize_row_not_answerable():
    docs = normalize_row(_base_row(is_selected=[0, 0]),
                         source_file="hintrain.parquet")
    assert all(d["answerable"] is False for d in docs)
    assert all(d["is_selected"] == 0 for d in docs)


def test_normalize_row_variable_passage_counts():
    # row with 3 passages (not the typical 10)
    docs = normalize_row(_base_row(
        English_passages=["e0", "e1", "e2"],
        Translated_passages=["h0", "h1", "h2"],
        is_selected=[1, 0, 0],
    ), source_file="hintrain.parquet")
    assert len(docs) == 3
    assert [d["passage_idx"] for d in docs] == [0, 1, 2]
    assert [d["document_id"] for d in docs] == [
        "hi_1185869_p0", "hi_1185869_p1", "hi_1185869_p2"]
    assert docs[0]["answerable"] is True


def test_normalize_row_single_passage():
    docs = normalize_row(_base_row(
        English_passages=["only"],
        Translated_passages=["केवल"],
        is_selected=[1],
    ), source_file="hintrain.parquet")
    assert len(docs) == 1
    assert docs[0]["passage_idx"] == 0
    assert docs[0]["text"] == "केवल"
    assert docs[0]["answerable"] is True


def test_normalize_row_alignment_violation_raises():
    with pytest.raises(ValueError):
        normalize_row(_base_row(
            English_passages=["e0", "e1"],
            Translated_passages=["h0"],  # length mismatch
            is_selected=[0, 1],
        ), source_file="hintrain.parquet")
    with pytest.raises(ValueError):
        normalize_row(_base_row(
            English_passages=["e0", "e1"],
            Translated_passages=["h0", "h1"],
            is_selected=[0],  # length mismatch
        ), source_file="hintrain.parquet")


def test_normalize_row_empty_passages():
    docs = normalize_row(_base_row(
        English_passages=[], Translated_passages=[], is_selected=[],
    ), source_file="hintrain.parquet")
    assert docs == []
    # answerable should be False for an empty-passage row
    assert is_answerable([]) is False


def test_normalize_row_marathi():
    docs = normalize_row(_base_row(
        target_lang="mar_Deva",
        query="काही प्रश्न",
    ), source_file="martrain.parquet")
    assert docs[0]["language"] == "mr"
    assert docs[0]["source_file"] == "martrain.parquet"
    assert docs[0]["document_id"].startswith("mr_")


def test_document_id_unique_across_rows():
    d1 = normalize_row(_base_row(query_id=1), source_file="hintrain.parquet")
    d2 = normalize_row(_base_row(query_id=2), source_file="hintrain.parquet")
    ids = [d["document_id"] for d in d1 + d2]
    assert len(ids) == len(set(ids)), "document_ids must be unique across rows"


def test_is_selected_preserved_as_int():
    docs = normalize_row(_base_row(is_selected=[0, 1]),
                         source_file="hintrain.parquet")
    assert all(isinstance(d["is_selected"], int) for d in docs)
