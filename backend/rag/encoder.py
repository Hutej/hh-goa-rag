"""ONNX multilingual sentence encoder — no torch, no transformers.

Serves both jobs from one int8 graph:

* **query encoding** on the hot path (target ~10-15 ms on CPU), and
* **corpus encoding** during the offline indexing pass.

Using the *same* weights for both is deliberate. Quantization error then points
the same way for queries and documents, so it largely cancels in the dot
product; encoding documents in fp32 and queries in int8 would introduce an
asymmetry that costs recall.

Model contract (``intfloat/multilingual-e5-small`` family):

* E5 models are **prefix-conditioned**. Queries MUST be prefixed ``"query: "``
  and documents ``"passage: "``. Omitting the prefixes measurably degrades
  retrieval — this is the single easiest way to silently lose recall.
* Pooling is **mean over unmasked tokens**, not CLS.
* Vectors are L2-normalized here, exactly once. Cosine similarity therefore
  reduces to a dot product downstream. Do NOT re-normalize.

Threading: onnxruntime defaults to one thread per core, which oversubscribes
when several requests arrive at once. ``EMBED_THREADS`` pins it.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from backend.rag.config import CFG

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class EncoderError(RuntimeError):
    """Encoder could not be constructed or run (missing artifact, bad graph)."""


class OnnxEncoder:
    """Mean-pooled, L2-normalized sentence encoder over an ONNX graph.

    Thread-safe: onnxruntime sessions support concurrent ``run`` calls, and the
    tokenizer is used without mutation. The query cache is mutex-guarded.
    """

    def __init__(self, onnx_path: Path | None = None,
                 tokenizer_path: Path | None = None,
                 dim: int | None = None,
                 threads: int | None = None,
                 query_cache_size: int = 512):
        self.onnx_path = Path(onnx_path or CFG.encoder_onnx_path())
        self.tokenizer_path = Path(tokenizer_path or CFG.tokenizer_path)
        self.dim = int(dim or CFG.embed_dim)
        self._threads = CFG.embed_threads if threads is None else int(threads)

        if not self.onnx_path.exists():
            raise EncoderError(
                f"ONNX encoder missing: {self.onnx_path}\n"
                f"Run: python scripts/download_encoder.py")
        if not self.tokenizer_path.exists():
            raise EncoderError(
                f"tokenizer.json missing: {self.tokenizer_path}\n"
                f"Run: python scripts/download_encoder.py")

        self._session = self._build_session()
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._output_name = self._session.get_outputs()[0].name

        # One tokenizer instance per truncation length, each configured once at
        # construction and never mutated afterwards.
        #
        # This is not a micro-optimization. `tokenizers.Tokenizer` is a Rust
        # object behind a RefCell: `enable_truncation` / `enable_padding` take a
        # mutable borrow, while `encode_batch` takes an immutable one. Calling
        # the setters per encode (the obvious implementation) races as soon as
        # two threads encode at once and fails with
        # `RuntimeError: Already borrowed`. Measured before this fix: 53 of 64
        # concurrent calls failed. That matters here because FastAPI dispatches
        # sync handlers to a thread pool, so two simultaneous HTTP requests hit
        # it — as does any eval harness that parallelizes across examples.
        self._tokenizers: dict[int, object] = {}
        self._tok_lock = threading.Lock()

        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = int(query_cache_size)
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

        self._verify_dim()

    # -- construction -----------------------------------------------------
    def _build_session(self):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - dependency guard
            raise EncoderError(
                "onnxruntime is not installed (pip install -r requirements.txt)"
            ) from e

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Latency, not throughput: a single short query should not fan out.
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if self._threads > 0:
            opts.intra_op_num_threads = self._threads
            opts.inter_op_num_threads = 1
        opts.log_severity_level = 3  # warnings and above only

        providers = ["CPUExecutionProvider"]
        if CFG.embed_device in ("cuda", "auto"):
            available = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif CFG.embed_device == "cuda":
                raise EncoderError(
                    "EMBED_DEVICE=cuda but onnxruntime has no CUDA provider "
                    "(install onnxruntime-gpu, or use EMBED_DEVICE=cpu)")
        try:
            return ort.InferenceSession(str(self.onnx_path), sess_options=opts,
                                        providers=providers)
        except Exception as e:
            raise EncoderError(
                f"failed to load ONNX graph {self.onnx_path}: {e}") from e

    def _build_tokenizer(self, max_tokens: int):
        """Load a tokenizer and fix its truncation/padding once, immutably."""
        try:
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover - dependency guard
            raise EncoderError(
                "tokenizers is not installed (pip install -r requirements.txt)"
            ) from e
        try:
            tok = Tokenizer.from_file(str(self.tokenizer_path))
        except Exception as e:
            raise EncoderError(
                f"failed to load tokenizer {self.tokenizer_path}: {e}") from e
        tok.enable_truncation(max_length=max_tokens)
        tok.enable_padding()  # pad to the longest sequence in each batch
        return tok

    def _tokenizer_for(self, max_tokens: int):
        """Return the immutable tokenizer for ``max_tokens``, building it once.

        Only construction is locked; ``encode_batch`` on an unmutated tokenizer
        is safe to call concurrently.
        """
        tok = self._tokenizers.get(max_tokens)
        if tok is not None:
            return tok
        with self._tok_lock:
            tok = self._tokenizers.get(max_tokens)
            if tok is None:
                tok = self._build_tokenizer(max_tokens)
                self._tokenizers[max_tokens] = tok
            return tok

    def _verify_dim(self) -> None:
        """Confirm the real output width instead of trusting config."""
        v = self._encode_raw(["probe"], max_tokens=8)
        got = int(v.shape[1])
        if got != self.dim:
            raise EncoderError(
                f"encoder dim mismatch: graph produces {got}, config says "
                f"{self.dim}. Set EMBED_DIM={got} or fetch the matching model.")

    # -- tokenization -----------------------------------------------------
    def _tokenize(self, texts: list[str], max_tokens: int
                  ) -> tuple[np.ndarray, np.ndarray]:
        # No mutation here: the instance is already configured for this length.
        encs = self._tokenizer_for(max_tokens).encode_batch(texts)
        ids = np.asarray([e.ids for e in encs], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
        return ids, mask

    # -- inference --------------------------------------------------------
    def _encode_raw(self, texts: list[str], max_tokens: int) -> np.ndarray:
        ids, mask = self._tokenize(texts, max_tokens)
        feeds: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            feeds["input_ids"] = ids
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = mask
        # XLM-R has no token_type_ids, but some exports still declare it.
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        missing = self._input_names - set(feeds)
        if missing:
            raise EncoderError(
                f"ONNX graph expects unsupported input(s): {sorted(missing)}")

        out = self._session.run([self._output_name], feeds)[0]
        hidden = np.asarray(out, dtype=np.float32)
        if hidden.ndim != 3:
            raise EncoderError(
                f"expected (batch, seq, hidden) from the graph, got {hidden.shape}")

        # mean pool over unmasked tokens, then L2 normalize (once, here)
        m = mask.astype(np.float32)[:, :, None]
        summed = (hidden * m).sum(axis=1)
        counts = np.clip(m.sum(axis=1), 1e-9, None)
        vecs = summed / counts
        norms = np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        return (vecs / norms).astype(np.float32)

    def _encode(self, texts: list[str], prefix: str, max_tokens: int,
                batch_size: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefixed = [prefix + (t or "") for t in texts]
        if len(prefixed) <= batch_size:
            return self._encode_raw(prefixed, max_tokens)
        parts = [self._encode_raw(prefixed[i:i + batch_size], max_tokens)
                 for i in range(0, len(prefixed), batch_size)]
        return np.vstack(parts)

    # -- public API -------------------------------------------------------
    def encode_queries(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Encode search queries (adds the required ``query: `` prefix)."""
        return self._encode(texts, QUERY_PREFIX,
                            CFG.embed_query_max_tokens, batch_size)

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode corpus chunks (adds the required ``passage: `` prefix)."""
        return self._encode(texts, PASSAGE_PREFIX,
                            CFG.embed_passage_max_tokens, batch_size)

    def encode_query(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Encode one query to a (dim,) unit vector, with a small LRU cache.

        The cache exists so repeated identical queries during a demo do not pay
        full cost. Benchmarks must run with ``use_cache=False`` — reporting
        percentiles over cache hits would be meaningless.
        """
        key = (text or "").strip()
        if use_cache and key:
            with self._cache_lock:
                hit = self._cache.get(key)
                if hit is not None:
                    self._cache.move_to_end(key)
                    self._cache_hits += 1
                    return hit
                self._cache_misses += 1
        vec = self.encode_queries([key], batch_size=1)[0]
        if use_cache and key:
            with self._cache_lock:
                self._cache[key] = vec
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return vec

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0

    def stats(self) -> dict:
        with self._cache_lock:
            hits, misses = self._cache_hits, self._cache_misses
            size = len(self._cache)
        total = hits + misses
        return {
            "model": CFG.embed_model,
            "precision": CFG.embed_precision,
            "dim": self.dim,
            "providers": list(self._session.get_providers()),
            "threads": self._threads or "auto",
            "cache": {"size": size, "hits": hits, "misses": misses,
                      "hit_rate": round(hits / total, 4) if total else 0.0},
        }


# --------------------------------------------------------------------------
# process-wide singleton
# --------------------------------------------------------------------------
_encoder: OnnxEncoder | None = None
_encoder_lock = threading.Lock()


def get_encoder() -> OnnxEncoder:
    """Return the process-wide encoder, constructing it once."""
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                _encoder = OnnxEncoder()
    return _encoder


def reset_encoder() -> None:
    """Drop the singleton (tests / config changes)."""
    global _encoder
    with _encoder_lock:
        _encoder = None


__all__ = ["OnnxEncoder", "EncoderError", "get_encoder", "reset_encoder",
           "QUERY_PREFIX", "PASSAGE_PREFIX"]
