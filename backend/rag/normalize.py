"""Canonical normalization of MSMARCO-XI rows into per-passage RAG documents.

One dataset *row* = one query + ~10 candidate passages. The retrieval unit is
ONE passage, so each row expands into ~10 normalized documents. This module
holds the pure normalization logic; the streaming I/O lives in
``scripts/extract_subset.py``.

The canonical schema is derived from the REAL parquet schema (see
``docs/phase0_metadata.json``), not assumed. Field roles:

  SOURCE  (copied verbatim from the parquet row):
    query_id, query, Eng_Query, Answer, Eng_Answer, query_type,
    passages.English_passages[i], passages.Translated_passages[i],
    passages.is_selected[i], source_lang, target_lang

  DERIVED (computed here):
    document_id   f"{lang}_{query_id}_p{passage_idx}"
    language      map target_lang -> ISO 639-1 ("hin_Deva"->"hi", "mar_Deva"->"mr")
    source        constant "MSMARCO-XI"
    source_file   basename of the originating parquet
    answerable     bool: row has >=1 passage with is_selected==1
    passage_idx   position of the passage within the row's passage list

``answer`` / ``answer_en`` are retained as metadata for evaluation/debugging
only — they are model-generated translations (meta.model_name), NOT gold.
``text`` is the local-language passage (the user speaks Hindi/Marathi);
``text_en`` is the parallel English passage, kept for cross-lingual fallback.
"""

from __future__ import annotations

from typing import Any

# target_lang (BCP-47 script code) -> ISO 639-1, the value stored on documents.
# Derived from the actual data: Hindi file -> "hin_Deva", Marathi file -> "mar_Deva".
LANG_CODE_MAP = {
    "hin_Deva": "hi",
    "mar_Deva": "mr",
    "eng_Latn": "en",
}


def map_language(target_lang: str | None) -> str:
    """Map a target_lang code to an ISO 639-1 code; unknown -> 'und'."""
    if target_lang is None:
        return "und"
    return LANG_CODE_MAP.get(target_lang, "und")


def is_answerable(is_selected: list[int] | None) -> bool:
    """A row is answerable if at least one passage is selected (relevance=1)."""
    if not is_selected:
        return False
    return any(int(v) == 1 for v in is_selected)


def normalize_row(
    row: dict[str, Any],
    source_file: str,
) -> list[dict[str, Any]]:
    """Expand one dataset row into a list of per-passage documents.

    Validates that the three passage lists are parallel; raises ValueError if
    they are not (the caller decides whether to skip or report the row).

    Each passage document has exactly the canonical fields documented above.
    """
    eng_pass = row.get("English_passages") or []
    trans_pass = row.get("Translated_passages") or []
    is_sel = row.get("is_selected") or []

    # Alignment invariant for the MSMARCO-XI schema.
    if not (len(eng_pass) == len(trans_pass) == len(is_sel)):
        raise ValueError(
            f"passage lists not parallel for query_id={row.get('query_id')}: "
            f"eng={len(eng_pass)} trans={len(trans_pass)} sel={len(is_sel)}"
        )

    query_id = row.get("query_id")
    target_lang = row.get("target_lang")
    language = map_language(target_lang)
    answerable = is_answerable(is_sel)
    query_type = row.get("query_type")

    docs: list[dict[str, Any]] = []
    for idx in range(len(eng_pass)):
        docs.append(
            {
                # identity (derived)
                "document_id": f"{language}_{query_id}_p{idx}",
                "query_id": query_id,
                "passage_idx": idx,
                # retrievable text (source)
                "text": trans_pass[idx],
                "text_en": eng_pass[idx],
                # query side of the row (source), carried for eval/join
                "query": row.get("query"),
                "query_en": row.get("Eng_Query"),
                "answer": row.get("Answer"),
                "answer_en": row.get("Eng_Answer"),
                # language (derived)
                "language": language,
                "source_lang_code": row.get("source_lang"),
                "target_lang_code": target_lang,
                # relevance (source)
                "is_selected": int(is_sel[idx]) if is_sel[idx] is not None else None,
                "query_type": query_type,
                # provenance (derived)
                "source": "MSMARCO-XI",
                "source_file": source_file,
                "answerable": answerable,
            }
        )
    return docs


CANONICAL_FIELDS = [
    "document_id", "query_id", "passage_idx",
    "text", "text_en",
    "query", "query_en",
    "answer", "answer_en",
    "language", "source_lang_code", "target_lang_code",
    "is_selected", "query_type",
    "source", "source_file", "answerable",
]


__all__ = [
    "LANG_CODE_MAP", "map_language", "is_answerable",
    "normalize_row", "CANONICAL_FIELDS",
]
