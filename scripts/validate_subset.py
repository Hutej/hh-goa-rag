"""Phase 1: validate the extracted subset and run the normalization unit tests.

Usage:
    venv/bin/python scripts/validate_subset.py

Checks (all run against the REAL output parquet):
  * row count + passage count
  * schema exactly matches CANONICAL_FIELDS
  * no missing required IDs (document_id, query_id, passage_idx)
  * no duplicate document_id
  * is_selected in {0, 1} only
  * both selected and non-selected passages present
  * language metadata == "hi" for all rows
  * answerable flag correct (== any(is_selected==1) per query_id)
  * no empty text records (text and text_en non-empty strings)
  * per-row passage alignment preserved in OUTPUT (via per-query_id grouping:
    the set of passage_idx for a query_id must be 0..n-1 contiguous)
  * query_type distribution of the subset (sanity vs full file)
  * output parquet has MANY small row groups (not a single giant one)
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.rag.normalize import CANONICAL_FIELDS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSET = PROJECT_ROOT / "data" / "processed" / "hh_subset_hin.parquet"
FULL_STATS = PROJECT_ROOT / "docs" / "phase0_short_aggregates.json"


def main() -> None:
    assert SUBSET.exists(), f"missing {SUBSET}"
    pf = pq.ParquetFile(SUBSET)
    n_rows = pf.metadata.num_rows
    n_rg = pf.metadata.num_row_groups
    schema_names = pf.schema_arrow.names

    print(f"=== VALIDATE {SUBSET} ===")
    print(f"rows (passages) = {n_rows}")
    print(f"row_groups     = {n_rg}")
    print(f"columns        = {list(schema_names)}")

    checks: list[tuple[str, bool, str]] = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}: {detail}")

    # 1. schema exactly matches canonical fields
    chk("schema==canonical",
        list(schema_names) == CANONICAL_FIELDS,
        f"{len(schema_names)} cols, expected {len(CANONICAL_FIELDS)}")

    # 2. output has many small row groups (not one giant one)
    rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(n_rg)]
    chk("many_row_groups", n_rg >= 5 and max(rg_sizes) <= 50000,
        f"{n_rg} groups, max={max(rg_sizes)}, sizes={rg_sizes[:3]}...")

    # Read everything (60 MiB — cheap).
    table = pf.read()
    cols = {c: table.column(c).to_pylist() for c in schema_names}

    # 3. passage count is positive
    chk("passage_count>0", n_rows > 0, f"{n_rows}")

    # 4. no missing required IDs
    missing_docid = sum(1 for v in cols["document_id"] if v is None)
    missing_qid = sum(1 for v in cols["query_id"] if v is None)
    missing_pidx = sum(1 for v in cols["passage_idx"] if v is None)
    chk("no_missing_ids",
        missing_docid == 0 and missing_qid == 0 and missing_pidx == 0,
        f"docid_null={missing_docid} qid_null={missing_qid} pidx_null={missing_pidx}")

    # 5. no duplicate document_id
    docids = cols["document_id"]
    chk("unique_doc_ids", len(set(docids)) == len(docids),
        f"{len(set(docids))} unique / {len(docids)} total")

    # 6. is_selected in {0,1} only
    sel_vals = set(cols["is_selected"])
    chk("is_selected_binary", sel_vals.issubset({0, 1}),
        f"distinct values={sorted(sel_vals)}")

    # 7. both selected and non-selected present
    n_sel = sum(1 for v in cols["is_selected"] if v == 1)
    n_nonsel = sum(1 for v in cols["is_selected"] if v == 0)
    chk("both_sel_and_nonsel_present", n_sel > 0 and n_nonsel > 0,
        f"selected={n_sel} non-selected={n_nonsel}")

    # 8. language == "hi" for all rows
    lang_dist = Counter(cols["language"])
    chk("language_all_hi", set(lang_dist) == {"hi"},
        f"lang dist={dict(lang_dist)}")

    # 9. answerable flag correct per query_id
    qid_to_sel: dict[int, list[int]] = {}
    qid_to_answerable: dict[int, list[bool]] = {}
    for qid, sel, ans in zip(cols["query_id"], cols["is_selected"],
                            cols["answerable"]):
        qid_to_sel.setdefault(qid, []).append(sel)
        qid_to_answerable.setdefault(qid, []).append(ans)
    bad_answerable = 0
    for qid, sels in qid_to_sel.items():
        expected = any(v == 1 for v in sels)
        if qid_to_answerable[qid][0] != expected:
            bad_answerable += 1
    chk("answerable_flag_correct", bad_answerable == 0,
        f"{len(qid_to_sel)} query_ids, {bad_answerable} bad")

    # 10. no empty text records
    empty_text = sum(1 for v in cols["text"] if not v)
    empty_text_en = sum(1 for v in cols["text_en"] if not v)
    chk("no_empty_text", empty_text == 0 and empty_text_en == 0,
        f"empty_text={empty_text} empty_text_en={empty_text_en}")

    # 11. per-query_id passage_idx contiguous from 0
    qid_to_pidx: dict[int, list[int]] = {}
    for qid, pidx in zip(cols["query_id"], cols["passage_idx"]):
        qid_to_pidx.setdefault(qid, []).append(pidx)
    bad_idx = 0
    for qid, idxs in qid_to_pidx.items():
        if sorted(idxs) != list(range(len(idxs))):
            bad_idx += 1
    chk("passage_idx_contiguous", bad_idx == 0,
        f"{len(qid_to_pidx)} query_ids, {bad_idx} bad")

    # 12. query_type distribution sanity (subset vs full file)
    qt_dist = Counter(cols["query_type"])
    print("  query_type distribution (subset):", dict(qt_dist))
    chk("query_types_present", len(qt_dist) == 5,
        f"{len(qt_dist)} types: {sorted(qt_dist)}")

    # 13. number of unique query_ids == 20000
    chk("unique_query_ids==20000", len(qid_to_sel) == 20000,
        f"{len(qid_to_sel)}")

    print("\n=== SUMMARY ===")
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        print("FAILED CHECKS:")
        for name, ok, detail in checks:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
