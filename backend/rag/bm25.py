"""BM25 lexical retrieval over the Adaptive chunk corpus (Phase 4).

Lightweight: uses ``rank_bm25.BM25Okapi``. Tokenization is 
whitespace + punctuation-strip + lowercase,which keeps 
Devanagari words whole (unlike ``\\w+``, which fragments combiningmatras). 
BM25 handles exact lexical surface forms; dense retrieval (Phase 3B)
handles semantic fuzz — they are complementary for later hybrid/RRF.

Build the index ONCE (tokenize the whole corpus), persist it to
``data/processed/bm25/{strategy}/`` as a pickle (``BM25Okapi`` is picklable) plus a
metadata parquet mapping BM25 row-index -> payload (chunk_id, document_id,
query_id, chunk_index, is_selected, text). ``get_scores`` returns an array
indexed by original document order, so BM25 row i <-> metadata row i.

Kept intentionally simple: no custom search engine, no per-query re-tokenization
of the corpus.

Usage (via scripts/retrieve_bm25.py):
    bm = BM25Index.load("adaptive")      # builds+persists on first call
    hits = bm.query("मैनहट्टन परियोजना", top_k=5)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
BM25_DIR = PROJECT_ROOT / "data" / "processed" / "bm25"

STRATEGIES = ["adaptive", "fixed", "semantic"]

# Metadata written alongside the index (mirrors dense retrieval output + text).
META_FIELDS = ["chunk_id", "document_id", "query_id", "chunk_index", "is_selected", "text"]


def tokenize(text: str) -> list[str]:
    """Whitespace + punctuation-strip + lowercase. Keeps Devanagari whole.

    ``\\w+`` regex fragments Hindi (combining matras are not matched by ``\\w``),
    so a naive regex tokenizer turns ``मैनहट्टन`` into garbage. Whitespace split
    on already-chunked text yields clean whole-word tokens — sufficient for BM25
    lexical matching where the dense retriever covers semantic fuzz.
    """
    if not text:
        return []
    out = []
    for w in text.split():
        w = w.strip("।.,;:!?\"'()[]{}|/\\-").lower()
        if len(w) >= 2:
            out.append(w)
    return out


def _paths(strategy: str) -> tuple[Path, Path, Path]:
    d = BM25_DIR / strategy
    return d, d / "bm25.pkl", d / "metadata.parquet"


def build_index(strategy: str) -> "BM25Index":
    """Tokenize the strategy chunk parquet and build+persist the BM25 index.

    Streams row-group by row-group (bounded RAM). Returns a ready ``BM25Index``.
    """
    chunk_path = CHUNK_DIR / f"{strategy}.parquet"
    if not chunk_path.exists():
        raise FileNotFoundError(f"missing chunk parquet: {chunk_path}")
    d, pkl_path, meta_path = _paths(strategy)
    d.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(chunk_path)
    corpus: list[list[str]] = []
    meta_rows: list[dict] = []
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=META_FIELDS)
        texts = tbl.column("text").to_pylist()
        cids = tbl.column("chunk_id").to_pylist()
        dids = tbl.column("document_id").to_pylist()
        qids = tbl.column("query_id").to_pylist()
        cidxs = tbl.column("chunk_index").to_pylist()
        sels = tbl.column("is_selected").to_pylist()
        for i in range(tbl.num_rows):
            corpus.append(tokenize(texts[i] or ""))
            meta_rows.append({
                "row_index": len(meta_rows),
                "chunk_id": cids[i],
                "document_id": dids[i],
                "query_id": int(qids[i]),
                "chunk_index": int(cidxs[i]),
                "is_selected": int(sels[i]),
                "text": texts[i] or "",
            })

    bm = BM25Okapi(corpus)
    with open(pkl_path, "wb") as f:
        pickle.dump(bm, f)
    schema = pa.schema([
        ("row_index", pa.int64()),
        ("chunk_id", pa.string()),
        ("document_id", pa.string()),
        ("query_id", pa.int64()),
        ("chunk_index", pa.int32()),
        ("is_selected", pa.int8()),
        ("text", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(meta_rows, schema=schema), meta_path)
    return BM25Index(strategy, bm, meta_rows)


class BM25Index:
    """Wraps a built/loaded BM25Okapi + the row-aligned metadata."""

    def __init__(self, strategy: str, bm: BM25Okapi, meta_rows: list[dict]):
        self.strategy = strategy
        self.bm = bm
        self.meta = meta_rows
        self.n = len(meta_rows)

    @classmethod
    def load(cls, strategy: str) -> "BM25Index":
        """Load a persisted index, building+persisting it on first use."""
        _, pkl_path, meta_path = _paths(strategy)
        if not pkl_path.exists() or not meta_path.exists():
            return build_index(strategy)
        with open(pkl_path, "rb") as f:
            bm = pickle.load(f)
        tbl = pq.read_table(meta_path)
        meta_rows = tbl.to_pylist()
        return cls(strategy, bm, meta_rows)

    def query(self, query: str, top_k: int = 10,
              only_selected: bool = False) -> list[dict]:
        """Return top_k hits as dicts with the dense-retrieval-compatible fields.

        score = BM25 score (unbounded positive; not a similarity in [-1, 1]).
        """
        q_tokens = tokenize(query)
        if not q_tokens or self.n == 0:
            return []
        scores = self.bm.get_scores(q_tokens)  # (n,) float64, row-index aligned
        order = np.argsort(-scores)
        out = []
        for idx in order:
            if len(out) >= top_k:
                break
            si = scores[int(idx)]
            if si <= 0:  # no term overlap -> not a real lexical hit
                break
            m = self.meta[int(idx)]
            if only_selected and not m["is_selected"]:
                continue
            out.append({
                "rank": len(out) + 1,
                "chunk_id": m["chunk_id"],
                "document_id": m["document_id"],
                "query_id": m["query_id"],
                "chunk_index": m["chunk_index"],
                "text": m["text"],
                "is_selected": m["is_selected"],
                "score": round(float(scores[int(idx)]), 6),
                "strategy": self.strategy,
            })
        return out


__all__ = ["tokenize", "build_index", "BM25Index", "STRATEGIES",
           "BM25_DIR", "META_FIELDS"]
