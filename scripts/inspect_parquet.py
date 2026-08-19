"""Phase 0 reconnaissance — STEP A: metadata-only, reads NOTHING but the footer.

No row data is materialized. Prints: file size, num_rows, num_row_groups,
full schema (incl. nested struct/list), and created_by. Cheap & safe.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "hindi": PROJECT_ROOT / "data" / "raw" / "MSMARCO-XI" / "train" / "hintrain.parquet",
    "marathi": PROJECT_ROOT / "data" / "raw" / "MSMARCO-XI" / "train" / "martrain.parquet",
}


def human_size(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{n} B"


def describe_field(field: pa.Field, indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines = [f"{pad}- {field.name}: {field.type}  (nullable={field.nullable})"]
    t = field.type
    if pa.types.is_struct(t):
        for i in range(t.num_fields):
            lines.extend(describe_field(t.field(i), indent + 1))
    elif pa.types.is_list(t):
        lines.append(f"{pad}    [list of {t.value_type}]")
        if pa.types.is_struct(t.value_type):
            for i in range(t.value_type.num_fields):
                lines.extend(describe_field(t.value_type.field(i), indent + 2))
    return lines


def main() -> None:
    out: dict = {}
    for label, path in FILES.items():
        print("=" * 78)
        print(f"{label.upper()}: {path}")
        print("=" * 78)
        assert path.exists(), f"MISSING {path}"
        size = path.stat().st_size
        pf = pq.ParquetFile(path)  # reads only the footer metadata
        fmeta = pf.metadata
        schema = pf.schema_arrow

        lines = []
        lines.append(f"size_bytes={size} ({human_size(size)})")
        lines.append(f"num_rows={fmeta.num_rows}")
        lines.append(f"num_row_groups={fmeta.num_row_groups}")
        lines.append(f"created_by={fmeta.created_by!r}")
        # row-group sizes (metadata only) to see how big each group is
        rg_sizes = []
        for rgi in range(fmeta.num_row_groups):
            rg = fmeta.row_group(rgi)
            rg_sizes.append(
                {
                    "rg": rgi,
                    "rows": rg.num_rows,
                    "total_byte_size": rg.total_byte_size,
                    "total_byte_size_human": human_size(rg.total_byte_size),
                    "num_columns": rg.num_columns,
                }
            )
        lines.append("row_groups:")
        for rg in rg_sizes:
            lines.append(
                f"  rg{rg['rg']}: rows={rg['rows']} "
                f"bytes={rg['total_byte_size_human']} cols={rg['num_columns']}"
            )

        lines.append("schema:")
        for name in schema.names:
            lines.extend("  " + l for l in describe_field(schema.field(name)))

        text = "\n".join(lines)
        print(text)
        out[label] = {
            "path": str(path),
            "size_bytes": size,
            "num_rows": fmeta.num_rows,
            "num_row_groups": fmeta.num_row_groups,
            "created_by": fmeta.created_by,
            "row_groups": rg_sizes,
            "columns": list(schema.names),
            "schema_text": text,
        }
        print()

    outpath = PROJECT_ROOT / "docs" / "phase0_metadata.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
