"""Extract a deterministic per-language passage corpus from MSMARCO-XI shards.

Each source row is one query with ~10 candidate passages, and every row carries
the passages in **both** English and the shard's Indic language, aligned
index-for-index with a shared ``is_selected`` relevance label:

    passages.English_passages[i]    <-> passages.Translated_passages[i]
    passages.is_selected[i]         (1 = human-marked relevant)

Two consequences this script relies on:

* Hindi and English come from the *same* shard, so two serving languages cost
  one download.
* Because index ``i`` refers to the same document in both languages and the
  label is shared, the extracted corpora form an **aligned parallel corpus with
  free cross-lingual ground truth** — you can ask in Hindi, retrieve an English
  passage, and still know whether the hit was correct.

Determinism: rows are taken in file order (first ``--rows``), never sampled, so
the corpus reproduces byte-identically on any machine.

Output (one file per language):
    data/processed/passages/{lang}.parquet
        passage_uid, document_id, query_id, passage_index, lang,
        is_selected, query, query_en, text

``document_id`` is language-independent (``q{query_id}_p{i}``) so the same
document across languages is trivially joinable — that is what makes the
cross-lingual evaluation possible.

Usage:
    python scripts/extract_subset.py                        # all CFG.languages
    python scripts/extract_subset.py --languages hi,en --rows 20000
    python scripts/extract_subset.py --rows 5000 --out-dir data/processed/passages
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag.config import CFG, LANGUAGES  # noqa: E402

SCHEMA = pa.schema([
    ("passage_uid", pa.string()),
    ("document_id", pa.string()),
    ("query_id", pa.int64()),
    ("passage_index", pa.int32()),
    ("lang", pa.string()),
    ("is_selected", pa.int8()),
    ("query", pa.string()),
    ("query_en", pa.string()),
    ("text", pa.string()),
])

READ_COLUMNS = ["query_id", "query", "Eng_Query", "passages"]
READ_BATCH = 2000  # rows per Arrow batch; bounds peak RAM on a 1-row-group file


def dominant_script(text: str, sample: int = 400) -> str:
    """Most common Unicode script among the letters of ``text``.

    Used as a data sanity check (translated Hindi text really is Devanagari) and
    later for zero-cost sparse-retrieval routing.
    """
    counts: Counter[str] = Counter()
    for ch in text[:sample]:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        counts[name.split()[0]] += 1
    return counts.most_common(1)[0][0] if counts else "UNKNOWN"


def extract_language(lang_code: str, rows: int, out_dir: Path) -> dict:
    lang = LANGUAGES[lang_code]
    shard = CFG.shard_path(lang.shard)
    if not shard.exists():
        raise FileNotFoundError(
            f"missing shard for {lang_code} ({lang.name}): {shard}\n"
            f"Run: python scripts/download_dataset.py "
            f"--languages {lang.shard} --split {CFG.split}")

    column = lang.passage_col
    pf = pq.ParquetFile(shard)

    uids: list[str] = []
    docids: list[str] = []
    qids: list[int] = []
    pidxs: list[int] = []
    langs: list[str] = []
    sels: list[int] = []
    queries: list[str] = []
    queries_en: list[str] = []
    texts: list[str] = []

    seen_rows = 0
    skipped_empty = 0
    misaligned = 0
    # A handful of query_ids genuinely repeat in the source file, so query_id
    # alone does not identify a row. Suffix repeats with their occurrence number
    # to keep document_id unique. Because every language is read from the same
    # rows in the same order, the numbering is identical across languages and
    # the cross-lingual document_id join still holds exactly.
    qid_seen: Counter[int] = Counter()
    duplicate_qids = 0

    for batch in pf.iter_batches(batch_size=READ_BATCH, columns=READ_COLUMNS):
        if seen_rows >= rows:
            break
        d = batch.to_pylist()
        for r in d:
            if seen_rows >= rows:
                break
            seen_rows += 1
            ps = r.get("passages") or {}
            passages = ps.get(column) or []
            selected = ps.get("is_selected") or []
            english = ps.get("English_passages") or []
            translated = ps.get("Translated_passages") or []

            # The parallel-corpus guarantee only holds when all three lists
            # line up; count violations rather than silently mislabelling.
            if not (len(english) == len(translated) == len(selected)):
                misaligned += 1

            qid = int(r.get("query_id") or 0)
            q = (r.get("query") or "").strip()
            q_en = (r.get("Eng_Query") or "").strip()

            qid_seen[qid] += 1
            occurrence = qid_seen[qid]
            if occurrence == 1:
                doc_stem = f"q{qid}"
            else:
                doc_stem = f"q{qid}d{occurrence}"
                duplicate_qids += 1

            for i, raw in enumerate(passages):
                text = (raw or "").strip()
                if not text:
                    skipped_empty += 1
                    continue
                uids.append(f"{lang_code}_{doc_stem}_p{i}")
                docids.append(f"{doc_stem}_p{i}")
                qids.append(qid)
                pidxs.append(i)
                langs.append(lang_code)
                sels.append(int(selected[i]) if i < len(selected) else 0)
                queries.append(q)
                queries_en.append(q_en)
                texts.append(text)

    if not texts:
        raise RuntimeError(f"extracted zero passages for {lang_code}")

    # Uniqueness is an invariant every downstream index depends on; fail here
    # rather than letting duplicate ids silently collide during chunking.
    if len(set(docids)) != len(docids):
        raise RuntimeError(
            f"{lang_code}: document_id is not unique "
            f"({len(docids) - len(set(docids))} collisions)")

    table = pa.Table.from_arrays([
        pa.array(uids, pa.string()),
        pa.array(docids, pa.string()),
        pa.array(qids, pa.int64()),
        pa.array(pidxs, pa.int32()),
        pa.array(langs, pa.string()),
        pa.array(sels, pa.int8()),
        pa.array(queries, pa.string()),
        pa.array(queries_en, pa.string()),
        pa.array(texts, pa.string()),
    ], schema=SCHEMA)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{lang_code}.parquet"
    pq.write_table(table, out_path, compression="zstd", row_group_size=4000)

    n_selected = sum(1 for s in sels if s)
    queries_with_selected = len({q for q, s in zip(qids, sels) if s})
    script = dominant_script(" ".join(texts[:50]))

    stats = {
        "lang": lang_code,
        "name": lang.name,
        "shard": str(shard.relative_to(CFG.root)),
        "passage_column": column,
        "source_rows": seen_rows,
        "passages": len(texts),
        "avg_passages_per_row": round(len(texts) / max(seen_rows, 1), 2),
        "selected_passages": n_selected,
        "queries_with_selected": queries_with_selected,
        "unique_queries": len(set(qids)),
        "unique_documents": len(set(docids)),
        "skipped_empty": skipped_empty,
        "misaligned_rows": misaligned,
        "repeated_query_id_rows": duplicate_qids,
        "dominant_script": script,
        "expected_script": lang.script,
        "script_ok": script.upper().startswith(lang.script.upper()[:4]),
        "out": str(out_path.relative_to(CFG.root)),
        "size_mb": round(out_path.stat().st_size / (1024 ** 2), 1),
    }

    print(f"  passages          : {stats['passages']:,}")
    print(f"  from source rows  : {stats['source_rows']:,} "
          f"({stats['avg_passages_per_row']} passages/row)")
    print(f"  selected (relevant): {stats['selected_passages']:,} across "
          f"{stats['queries_with_selected']:,} queries")
    print(f"  script            : {script} (expected {lang.script}) "
          f"{'OK' if stats['script_ok'] else 'MISMATCH'}")
    if misaligned:
        print(f"  WARNING: {misaligned} rows had misaligned EN/translated/label "
              f"lists — cross-lingual pairing is not exact for those")
    print(f"  wrote             : {stats['out']} ({stats['size_mb']} MB)")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", default=",".join(CFG.languages),
                    help=f"comma-separated language codes "
                         f"(default: {','.join(CFG.languages)})")
    ap.add_argument("--rows", type=int, default=CFG.subset_rows,
                    help=f"source query rows per language "
                         f"(default: {CFG.subset_rows})")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: data/processed/passages)")
    args = ap.parse_args()

    codes = [c.strip().lower() for c in args.languages.split(",") if c.strip()]
    bad = [c for c in codes if c not in LANGUAGES]
    if bad:
        print(f"ERROR: unknown language(s): {', '.join(bad)}", file=sys.stderr)
        print(f"Known: {', '.join(LANGUAGES)}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else \
        CFG.processed_dir / "passages"

    print("=" * 70)
    print("EXTRACT PASSAGE CORPUS")
    print("=" * 70)
    print(f"Split     : {CFG.split}")
    print(f"Languages : {', '.join(codes)}")
    print(f"Rows/lang : {args.rows:,}")
    print(f"Output    : {out_dir}")
    print()

    all_stats = []
    for code in codes:
        print(f"[{code}] {LANGUAGES[code].name} "
              f"<- {LANGUAGES[code].passage_col}")
        try:
            all_stats.append(extract_language(code, args.rows, out_dir))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            return 1
        print()

    # Cross-lingual alignment check: languages drawn from the same shard must
    # produce identical document_id sets, which is what the parallel evaluation
    # depends on.
    print("-" * 70)
    doc_sets: dict[str, set[str]] = {}
    for s in all_stats:
        tbl = pq.read_table(out_dir / f"{s['lang']}.parquet",
                            columns=["document_id"])
        doc_sets[s["lang"]] = set(tbl.column("document_id").to_pylist())
    codes_present = list(doc_sets)
    for i, a in enumerate(codes_present):
        for b in codes_present[i + 1:]:
            shared = len(doc_sets[a] & doc_sets[b])
            print(f"parallel documents {a} <-> {b}: {shared:,} shared "
                  f"document_ids ({len(doc_sets[a]):,} / {len(doc_sets[b]):,})")

    report = CFG.root / "docs" / "corpus_stats.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "split": CFG.split, "rows_per_language": args.rows,
        "languages": all_stats,
        "parallel_overlap": {
            f"{a}|{b}": len(doc_sets[a] & doc_sets[b])
            for i, a in enumerate(codes_present) for b in codes_present[i + 1:]
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"EXTRACTION COMPLETE — {sum(s['passages'] for s in all_stats):,} "
          f"passages across {len(all_stats)} language(s)")
    print(f"Stats: {report.relative_to(CFG.root)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
