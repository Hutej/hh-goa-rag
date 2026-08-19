"""Memory-bounded reader for the single-row-group MSMARCO-XI parquet files.

Why this exists
---------------
Both ``hintrain.parquet`` and ``martrain.parquet`` are written as ONE row
group holding every row (``parquet-cpp-arrow 19.0.1``). Every "normal" reader
API (pyarrow ``read_table``/``read_row_group``/``iter_batches``, fastparquet
``to_pandas``/``iter_row_groups``/``head``) reads the *entire* compressed
column chunk before returning any row:

* ``passages.Translated_passages`` compressed chunk = ~2.05 GiB
* ``passages.English_passages``     compressed chunk = ~1.39 GiB

Decoded, those become ~6.8 GiB and ~2.6 GiB. On a ~7 GB box with only ~2 GB
free, that OOMs every time (verified — peak RSS hits ~2 GB then SIGKILL).

This reader does NOT load a whole column chunk. It reads a bounded byte slice
of a column's chunk, decodes data pages one at a time with fastparquet's
``core.read_data_page`` (which returns the definition/repetition level
arrays + the per-page values), and reconstructs per-row lists in *pure
Python* from the def/rep levels. It stops as soon as it has produced
``max_rows`` top-level rows. Peak RSS is set by the buffer size
(``buf_mb``), not by the file size.

Only the public ``fastparquet.core.read_data_page`` / ``read_dictionary_page``
and ``ThriftObject.from_buffer`` entry points are used — no C internals, no
monkey-patching, no library edits.

Limitations / honest scope
--------------------------
* Reads the FIRST ``max_rows`` rows of a column. Random row access by index
  is not supported (the column is one row group; there is no per-row
  statistics offset index).
* The byte slice is taken from the start of the column chunk. We stop before
  the slice end with a safety margin so a page header never straddles the
  buffer boundary.
* ``buf_mb`` must be large enough to hold enough pages for ``max_rows``;
  if it is too small we simply get fewer rows (the caller checks the count).
"""

from __future__ import annotations

import resource
from typing import Any, Iterable

import numpy as np

import fastparquet as fp
import fastparquet.core as core
import fastparquet.encoding as encoding
from fastparquet.cencoding import ThriftObject
import fastparquet.parquet_thrift as pt

# Safety margin: stop reading new pages when fewer than this many bytes remain
# in the slice, so a page header + body never straddles the slice boundary.
_PAGE_MARGIN_BYTES = 4 * 1024 * 1024


def peak_rss_mb() -> float:
    """Peak RSS of this process in MiB (Linux: ru_maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class BoundedColumnReader:
    """Stream the first ``max_rows`` top-level rows of one nested column."""

    def __init__(self, parquet_path: str):
        self.path = parquet_path
        self._pf = fp.ParquetFile(parquet_path)
        self.schema = self._pf.schema
        self.row_group = self._pf.row_groups[0]
        self.num_rows = self._pf.count()

    def _find_col(self, parts: list[str]):
        for c in self.row_group.columns:
            if list(c.meta_data.path_in_schema) == list(parts):
                return c
        raise KeyError(parts)

    def read_first_rows(
        self,
        path_in_schema: list[str],
        max_rows: int,
        buf_mb: int = 64,
    ) -> list[Any]:
        """Return the per-row list values for the first ``max_rows`` rows.

        ``path_in_schema`` is the dotted parquet path as a list, e.g.
        ``['passages', 'English_passages', 'list', 'element']``.

        Reconstructs per-row lists in pure Python from def/rep levels.
        """
        col = self._find_col(path_in_schema)
        cmd = col.meta_data
        se = self.schema.schema_element(path_in_schema)
        max_def = self.schema.max_definition_level(cmd.path_in_schema)
        chunk_end = cmd.data_page_offset + cmd.total_compressed_size
        off = min(
            x for x in (
                getattr(cmd, "dictionary_page_offset", None),
                cmd.data_page_offset,
            ) if x is not None
        )
        cap = min(buf_mb * 1024 * 1024, chunk_end - off)
        # The safety margin must not exceed the buffer, otherwise we stop
        # before reading any page (e.g. the 891 KB is_selected column).
        margin = min(_PAGE_MARGIN_BYTES, cap // 2)

        with open(self.path, "rb") as raw:
            raw.seek(off)
            chunk = raw.read(cap)
        infile = encoding.NumpyIO(chunk)

        # Dictionary: decode as plain bytes (utf=False) so values are plain
        # Python/numpy strings, not an ArrowStringArray.
        dic: Any = None

        # Reconstruct per-row lists in pure Python.
        # For a struct<...list<string>> the def levels encode:
        #   def == max_def           -> a leaf value present
        #   def == max_def - 1        -> list present, element null
        #   def <  max_def - 1        -> outer struct/list null at some level
        # rep == 0 marks the start of a new top-level row.
        rows: list[list] = []
        cur: list = []
        started = False

        while len(rows) < max_rows:
            if infile.tell() + margin > cap:
                break
            try:
                ph = ThriftObject.from_buffer(infile, "PageHeader")
            except Exception:
                break  # end of slice

            if ph.type == pt.PageType.DICTIONARY_PAGE:
                dic = core.read_dictionary_page(
                    infile, self.schema, ph, cmd, utf=False
                )
                dic = core.convert(dic, se)
                continue

            try:
                defi, rep, val = core.read_data_page(
                    infile, self.schema, ph, cmd, skip_nulls=False
                )
            except Exception:
                break  # slice ended mid-page

            is_dict_enc = ph.data_page_header.encoding in (
                pt.Encoding.PLAIN_DICTIONARY, pt.Encoding.RLE_DICTIONARY,
            )
            n = len(rep) if rep is not None else len(val)
            for i in range(n):
                r = int(rep[i]) if rep is not None else 0
                d = int(defi[i]) if defi is not None else max_def
                if r == 0:
                    # new top-level row boundary
                    if started:
                        rows.append(cur)
                        cur = []
                    started = True
                if d == max_def:
                    # leaf value present
                    if is_dict_enc:
                        v = dic[int(val[i])] if dic is not None else val[i]
                    else:
                        v = val[i]
                    if isinstance(v, (bytes, bytearray)):
                        v = bytes(v).decode("utf-8", "replace")
                    cur.append(v)
                elif d == max_def - 1:
                    # element slot present but null
                    cur.append(None)
                # d < max_def - 1 -> null at an outer level; no element added
            if len(rows) >= max_rows:
                break
        if started and cur and len(rows) < max_rows:
            rows.append(cur)
        return rows[:max_rows]


def scalar_column(parquet_path: str, column: str) -> list[Any]:
    """Read a flat scalar column fully (cheap: these are small columns).

    Uses fastparquet ``to_pandas`` with a single-column projection. Safe
    because the scalar columns' compressed chunks are at most tens of MB.
    """
    pf = fp.ParquetFile(parquet_path)
    df = pf.to_pandas(columns=[column])
    return df[column].tolist()


__all__ = ["BoundedColumnReader", "scalar_column", "peak_rss_mb"]
