"""Central, env-driven configuration for the serving path.

Single source of truth for paths, model identity, index layout, retrieval
parameters and guardrail thresholds. Everything is overridable by environment
variable (or `.env`) so no source edit is needed to change device, backend,
language set or thresholds.

Import this instead of hardcoding constants:

    from backend.rag.config import CFG
    CFG.embed_dim, CFG.languages, CFG.dense_index_path("hi")

Secrets are NOT exposed here — provider modules read their own keys so that a
config dump (``CFG.describe()``, ``/api/health``) can never leak a key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------------------------------
# .env loading (real environment always wins)
# --------------------------------------------------------------------------
def load_env(env_path: Path | None = None) -> None:
    """Load bare ``KEY=VALUE`` lines from `.env` into ``os.environ``.

    Idempotent. Real environment variables take precedence, so a Space secret
    is never overwritten by a stale committed default. Blank lines, ``#``
    comments and surrounding quotes are handled; no python-dotenv dependency.
    """
    p = env_path or (PROJECT_ROOT / ".env")
    if not p.exists():
        return
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and v and k not in os.environ:
            os.environ[k] = v


load_env()


# --------------------------------------------------------------------------
# small typed env helpers
# --------------------------------------------------------------------------
def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env_str(name, "1" if default else "0").lower() in \
        ("1", "true", "yes", "on")


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = _env_str(name, ",".join(default))
    out = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return out or default


# --------------------------------------------------------------------------
# language registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Language:
    """A served language.

    ``code``        short serving code used in artifact paths ("hi").
    ``shard``       MSMARCO-XI shard code the text came from ("hin"); English is
                    read from the ``English_passages`` column of any shard.
    ``script``      dominant Unicode script, used for reliable BM25 routing.
    ``stt_code``    Sarvam ``language_code`` for this language.
    ``passage_col`` which passage column of the shard supplies the passage text.
    ``query_col``   which column of the extracted corpus holds the query *in
                    this language*. This is easy to get wrong: the extractor
                    stores the shard's Indic query as ``query`` and the original
                    English one as ``query_en`` for every language, so reading
                    ``query`` for English yields Hindi text and silently turns a
                    monolingual evaluation into a cross-lingual one.
    """
    code: str
    name: str
    shard: str
    script: str
    stt_code: str
    passage_col: str
    query_col: str


# Hindi and Marathi share the Devanagari script, so script detection cannot
# separate them — sparse retrieval unions both and STT's returned language_code
# is preferred when available. This is handled in the retrieval layer, not here.
LANGUAGES: dict[str, Language] = {
    "hi": Language("hi", "Hindi", "hin", "Devanagari", "hi-IN",
                   "Translated_passages", "query"),
    "en": Language("en", "English", "hin", "Latin", "en-IN",
                   "English_passages", "query_en"),
    "mr": Language("mr", "Marathi", "mar", "Devanagari", "mr-IN",
                   "Translated_passages", "query"),
}


@dataclass
class Config:
    # ---- paths ----
    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"

    # ---- corpus / languages ----
    languages: list[str] = field(
        default_factory=lambda: _env_list("RAG_LANGUAGES", ["hi", "en", "mr"]))
    split: str = field(default_factory=lambda: _env_str("RAG_SPLIT", "validation"))
    # Number of source query rows to build the corpus from, per language.
    subset_rows: int = field(default_factory=lambda: _env_int("RAG_SUBSET_ROWS", 20000))

    # ---- chunking ----
    chunk_strategy: str = field(
        default_factory=lambda: _env_str("RAG_CHUNK_STRATEGY", "adaptive"))

    # ---- encoder ----
    embed_model: str = field(default_factory=lambda: _env_str(
        "EMBED_MODEL", "intfloat/multilingual-e5-small"))
    embed_dim: int = field(default_factory=lambda: _env_int("EMBED_DIM", 384))
    embed_backend: str = field(
        default_factory=lambda: _env_str("EMBED_BACKEND", "onnx").lower())
    embed_precision: str = field(
        default_factory=lambda: _env_str("EMBED_PRECISION", "int8").lower())
    embed_device: str = field(
        default_factory=lambda: _env_str("EMBED_DEVICE", "auto").lower())
    # Queries are short; clamping the sequence length cuts encode time with no
    # quality cost. Corpus chunks use the longer limit.
    embed_query_max_tokens: int = field(
        default_factory=lambda: _env_int("EMBED_QUERY_MAX_TOKENS", 64))
    embed_passage_max_tokens: int = field(
        default_factory=lambda: _env_int("EMBED_PASSAGE_MAX_TOKENS", 512))
    # onnxruntime intra-op threads. 0 = let ORT decide.
    embed_threads: int = field(
        default_factory=lambda: _env_int("EMBED_THREADS", 0))

    # ---- dense index ----
    index_backend: str = field(
        default_factory=lambda: _env_str("INDEX_BACKEND", "hnswlib").lower())
    hnsw_m: int = field(default_factory=lambda: _env_int("HNSW_M", 32))
    hnsw_ef_construction: int = field(
        default_factory=lambda: _env_int("HNSW_EF_CONSTRUCTION", 200))
    hnsw_ef_search: int = field(
        default_factory=lambda: _env_int("HNSW_EF_SEARCH", 64))
    qdrant_url: str = field(default_factory=lambda: _env_str(
        "QDRANT_URL", "http://127.0.0.1:6333"))
    qdrant_grpc_port: int = field(
        default_factory=lambda: _env_int("QDRANT_GRPC_PORT", 6334))

    # ---- retrieval / fusion (verified serving config) ----
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    dense_k: int = field(default_factory=lambda: _env_int("RAG_DENSE_K", 20))
    sparse_k: int = field(default_factory=lambda: _env_int("RAG_SPARSE_K", 20))
    rrf_k: int = field(default_factory=lambda: _env_int("RAG_RRF_K", 60))
    dense_weight: float = field(
        default_factory=lambda: _env_float("RAG_DENSE_WEIGHT", 1.0))
    # Chosen from a measured sweep (results/evaluation/fusion_weights.json,
    # 150 queries x 3 languages). The effect is monotonic and mostly negative:
    # more sparse weight improves R@10 slightly but degrades R@1/R@3 steadily
    # (mean R@3: 0.520 at weight 0.0, 0.500 at 0.1, 0.447 at 1.0). Since only
    # `generation_k` chunks reach the model, R@3 is the governing objective.
    # 0.1 is statistically indistinguishable from dense-only at R@3, best at
    # R@5, and retains exact-match capability for entity/number queries that a
    # natural-language question set under-represents. Honest summary: on this
    # corpus hybrid fusion is close to break-even, not a large win.
    sparse_weight: float = field(
        default_factory=lambda: _env_float("RAG_SPARSE_WEIGHT", 0.1))
    # How many chunks are actually handed to the LLM (fewer prefill tokens =
    # lower time-to-first-token). Display still shows top_k sources.
    generation_k: int = field(
        default_factory=lambda: _env_int("RAG_GENERATION_K", 3))
    # Cap chunks per document_id in the fused list, for source diversity.
    max_per_document: int = field(
        default_factory=lambda: _env_int("RAG_MAX_PER_DOCUMENT", 2))

    # ---- guardrail thresholds ----
    min_relevance: float = field(
        default_factory=lambda: _env_float("RAG_MIN_RELEVANCE", 0.005))
    # Degenerate-retrieval floor on raw cosine. NOT an off-topic detector:
    # measured in-corpus minimum is 0.848 while off-topic queries reach 0.896,
    # so absolute cosine cannot separate them (evidence in guardrails.py). This
    # only trips when the encoder or index is malfunctioning.
    min_cosine: float = field(
        default_factory=lambda: _env_float("RAG_MIN_COSINE", 0.80))
    min_groundedness: float = field(
        default_factory=lambda: _env_float("RAG_MIN_GROUNDEDNESS", 0.45))
    min_query_chars: int = field(
        default_factory=lambda: _env_int("RAG_MIN_QUERY_CHARS", 3))
    max_query_chars: int = field(
        default_factory=lambda: _env_int("RAG_MAX_QUERY_CHARS", 512))

    # ---- generation ----
    # Defaults chosen by measurement, not by vendor claims — see
    # results/provider_comparison.json. Gemini Flash-Lite completed a Hindi
    # answer in 774 ms against OpenAI's 2133 ms with equal script fidelity.
    llm_provider: str = field(
        default_factory=lambda: _env_str("LLM_PROVIDER", "gemini").lower())
    llm_model: str = field(
        default_factory=lambda: _env_str("LLM_MODEL", "gemini-3.5-flash-lite"))
    llm_base_url: str = field(default_factory=lambda: _env_str("LLM_BASE_URL", ""))
    llm_max_tokens: int = field(
        default_factory=lambda: _env_int("LLM_MAX_TOKENS", 160))
    llm_timeout_s: float = field(
        default_factory=lambda: _env_float("LLM_TIMEOUT_S", 12.0))
    # Retries exist mainly to absorb provider rate limiting, which measured at
    # 36% of calls on Gemini's free tier even at a single worker. Backoff honours
    # the provider's own suggested delay when it supplies one, which is far more
    # effective than guessing (see LLMClient._retry_after_seconds).
    llm_max_retries: int = field(
        default_factory=lambda: _env_int("LLM_MAX_RETRIES", 4))
    llm_max_backoff_s: float = field(
        default_factory=lambda: _env_float("LLM_MAX_BACKOFF_S", 20.0))
    llm_stream: bool = field(default_factory=lambda: _env_bool("LLM_STREAM", True))

    # ---- STT ----
    stt_provider: str = field(
        default_factory=lambda: _env_str("STT_PROVIDER", "sarvam").lower())
    stt_language: str = field(
        default_factory=lambda: _env_str("STT_LANGUAGE", "hi-IN"))

    # ---- deployment ----
    data_repo: str = field(default_factory=lambda: _env_str("HHGOA_DATA_REPO", ""))

    # ----------------------------------------------------------------
    # derived paths
    # ----------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data / "raw" / "MSMARCO-XI"

    @property
    def processed_dir(self) -> Path:
        return self.data / "processed"

    @property
    def encoder_dir(self) -> Path:
        return self.data / "models" / self.embed_model.split("/")[-1]

    def encoder_onnx_path(self, precision: str | None = None) -> Path:
        p = (precision or self.embed_precision).lower()
        name = {"int8": "model_int8.onnx", "fp16": "model_fp16.onnx",
                "fp32": "model.onnx"}.get(p, "model_int8.onnx")
        return self.encoder_dir / "onnx" / name

    @property
    def tokenizer_path(self) -> Path:
        return self.encoder_dir / "tokenizer.json"

    def shard_path(self, shard: str) -> Path:
        suffix = "val" if self.split == "validation" else "train"
        return self.raw_dir / self.split / f"{shard}{suffix}.parquet"

    def passages_path(self, lang: str) -> Path:
        return self.processed_dir / "passages" / f"{lang}.parquet"

    def chunks_path(self, lang: str, strategy: str | None = None) -> Path:
        s = strategy or self.chunk_strategy
        return self.processed_dir / "chunks" / s / f"{lang}.parquet"

    def embeddings_dir(self, lang: str, strategy: str | None = None) -> Path:
        s = strategy or self.chunk_strategy
        return self.processed_dir / "embeddings" / s / lang

    def dense_index_path(self, lang: str, strategy: str | None = None) -> Path:
        s = strategy or self.chunk_strategy
        return self.processed_dir / "dense" / s / f"{lang}.hnsw"

    def dense_meta_path(self, lang: str, strategy: str | None = None) -> Path:
        s = strategy or self.chunk_strategy
        return self.processed_dir / "dense" / s / f"{lang}.meta.parquet"

    def sparse_dir(self, lang: str, strategy: str | None = None) -> Path:
        s = strategy or self.chunk_strategy
        return self.processed_dir / "sparse" / s / lang

    # ----------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------
    def lang(self, code: str) -> Language:
        try:
            return LANGUAGES[code]
        except KeyError:
            raise ValueError(
                f"unknown language {code!r}; known: {', '.join(LANGUAGES)}"
            ) from None

    @property
    def active_languages(self) -> list[Language]:
        return [self.lang(c) for c in self.languages]

    @property
    def required_shards(self) -> list[str]:
        """Distinct source shards needed for the active language set."""
        seen: list[str] = []
        for l in self.active_languages:
            if l.shard not in seen:
                seen.append(l.shard)
        return seen

    def languages_for_script(self, script: str) -> list[str]:
        """Active language codes whose script matches — used to route sparse
        retrieval. Devanagari legitimately returns both hi and mr."""
        return [l.code for l in self.active_languages if l.script == script]

    def describe(self) -> dict:
        """Serializable, secret-free config summary for /api/health."""
        return {
            "languages": self.languages,
            "split": self.split,
            "chunk_strategy": self.chunk_strategy,
            "encoder": {"model": self.embed_model, "dim": self.embed_dim,
                        "backend": self.embed_backend,
                        "precision": self.embed_precision,
                        "device": self.embed_device},
            "dense": {"backend": self.index_backend, "m": self.hnsw_m,
                      "ef_search": self.hnsw_ef_search},
            "retrieval": {"top_k": self.top_k, "dense_k": self.dense_k,
                          "sparse_k": self.sparse_k, "rrf_k": self.rrf_k,
                          "dense_weight": self.dense_weight,
                          "sparse_weight": self.sparse_weight,
                          "generation_k": self.generation_k,
                          "max_per_document": self.max_per_document},
            "guardrails": {"min_relevance": self.min_relevance,
                           "min_cosine": self.min_cosine,
                           "min_groundedness": self.min_groundedness},
            "generation": {"provider": self.llm_provider, "model": self.llm_model,
                           "max_tokens": self.llm_max_tokens,
                           "timeout_s": self.llm_timeout_s,
                           "stream": self.llm_stream},
            "stt": {"provider": self.stt_provider, "language": self.stt_language},
        }


CFG = Config()

__all__ = ["CFG", "Config", "Language", "LANGUAGES", "PROJECT_ROOT", "load_env"]
