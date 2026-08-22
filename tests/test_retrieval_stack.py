"""Tests for the current retrieval stack: encoder, sparse, dense, fusion.

Split by dependency so the suite is useful in every environment:

* Pure-logic tests (fusion maths, tokenization, script routing) always run.
* Tests needing the ONNX encoder skip if it has not been downloaded.
* Tests needing built indexes skip if the data pipeline has not been run.

That way a fresh clone gets meaningful signal without a 1.4 GB artifact set.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from backend.rag.config import CFG, LANGUAGES
from backend.rag.retrieval import detect_script, route_languages, rrf_fuse
from backend.rag.sparse import tokenize


# ---------------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------------
def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_drops_single_characters():
    # Single chars carry almost no lexical signal and inflate the vocabulary.
    assert tokenize("a bb c ddd") == ["bb", "ddd"]


def test_tokenize_keeps_devanagari_words_intact():
    """The reason a ``\\w+`` regex tokenizer is not used.

    Python's ``\\w`` does not match Devanagari combining vowel signs, so a regex
    tokenizer splits words at the matras. Whitespace splitting keeps them whole.
    """
    tokens = tokenize("मैनहट्टन परियोजना क्या थी?")
    assert "मैनहट्टन" in tokens
    assert "परियोजना" in tokens
    # danda and question mark stripped, no fragments
    assert all("?" not in t and "।" not in t for t in tokens)


def test_tokenize_strips_devanagari_danda():
    assert tokenize("परियोजना। थी।") == ["परियोजना", "थी"]


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ") == []


# ---------------------------------------------------------------------------
# script detection / routing
# ---------------------------------------------------------------------------
def test_detect_script():
    assert detect_script("what is a corporation?") == "Latin"
    assert detect_script("मैनहट्टन परियोजना क्या थी?") == "Devanagari"
    assert detect_script("12345 !!!") == "Unknown"


def test_route_languages_latin_selects_english_only():
    assert route_languages("what is a corporation?") == ["en"]


def test_route_languages_devanagari_cannot_separate_hindi_from_marathi():
    """Hindi and Marathi share Devanagari, so both must be searched.

    This is a deliberate design property, not a limitation being papered over:
    script routing is free and never wrong, and fusion resolves the ambiguity.
    """
    routed = route_languages("मैनहट्टन परियोजना क्या थी?")
    assert set(routed) == {"hi", "mr"}


def test_route_languages_honours_pinned_language():
    # Sarvam returns a language_code; when present it beats script inference.
    assert route_languages("मैनहट्टन परियोजना क्या थी?", pinned="mr-IN") == ["mr"]


def test_route_languages_unknown_script_searches_everything():
    assert set(route_languages("12345")) == set(CFG.languages)


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------
def _hit(cid, rank, doc=None, score=0.9, lang="en"):
    return {"chunk_id": cid, "rank": rank, "document_id": doc or f"doc_{cid}",
            "score": score, "lang": lang, "text": f"text {cid}", "row": rank}


def test_rrf_fuse_orders_by_combined_rank():
    dense = [_hit("a", 1), _hit("b", 2)]
    sparse = [_hit("b", 1), _hit("a", 2)]
    fused = rrf_fuse(dense, sparse, dense_weight=1.0, sparse_weight=1.0)
    assert {h["chunk_id"] for h in fused} == {"a", "b"}
    # Symmetric input with equal weights: scores tie, both present, ranks assigned
    assert [h["rank"] for h in fused] == [1, 2]


def test_rrf_fuse_respects_weights():
    """A dense-only hit must outrank a sparse-only hit when dense is weighted up."""
    dense = [_hit("dense_only", 1)]
    sparse = [_hit("sparse_only", 1)]
    fused = rrf_fuse(dense, sparse, dense_weight=1.0, sparse_weight=0.1)
    assert fused[0]["chunk_id"] == "dense_only"


def test_rrf_fuse_zero_sparse_weight_excludes_sparse_only_hits_from_top():
    dense = [_hit("d1", 1), _hit("d2", 2)]
    sparse = [_hit("s1", 1)]
    fused = rrf_fuse(dense, sparse, sparse_weight=0.0)
    # s1 still appears (rank information is preserved) but scores 0 and sorts last
    assert fused[0]["chunk_id"] == "d1"
    assert fused[-1]["chunk_id"] == "s1"


def test_rrf_fuse_records_both_ranks():
    dense = [_hit("a", 3)]
    sparse = [_hit("a", 7)]
    fused = rrf_fuse(dense, sparse)
    assert fused[0]["dense_rank"] == 3
    assert fused[0]["sparse_rank"] == 7


def test_rrf_fuse_caps_chunks_per_document():
    """Adjacent chunks of one passage are near-duplicates; without a cap they
    crowd out genuinely different evidence."""
    dense = [_hit(f"c{i}", i, doc="same_doc") for i in range(1, 6)]
    fused = rrf_fuse(dense, [], max_per_document=2)
    assert len(fused) == 2


def test_rrf_fuse_cap_disabled_when_zero():
    dense = [_hit(f"c{i}", i, doc="same_doc") for i in range(1, 6)]
    fused = rrf_fuse(dense, [], max_per_document=0)
    assert len(fused) == 5


def test_rrf_fuse_empty_inputs():
    assert rrf_fuse([], []) == []


def test_rrf_fuse_skips_hits_without_chunk_id():
    fused = rrf_fuse([{"rank": 1, "document_id": "d"}], [])
    assert fused == []


# ---------------------------------------------------------------------------
# encoder (needs the downloaded ONNX graph)
# ---------------------------------------------------------------------------
encoder_available = pytest.mark.skipif(
    not CFG.encoder_onnx_path().exists() or not CFG.tokenizer_path.exists(),
    reason="ONNX encoder not downloaded; run scripts/download_encoder.py")


@encoder_available
def test_encoder_produces_unit_norm_vectors_of_configured_dim():
    from backend.rag.encoder import get_encoder
    enc = get_encoder()
    v = enc.encode_queries(["what is a corporation?", "मैनहट्टन परियोजना"])
    assert v.shape == (2, CFG.embed_dim)
    # Normalization happens exactly once, in the encoder; cosine == dot product.
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


@encoder_available
def test_encoder_query_and_passage_prefixes_differ():
    """E5 models are prefix-conditioned; the two must not produce identical
    vectors, or the required prefixes are not being applied."""
    from backend.rag.encoder import get_encoder
    enc = get_encoder()
    q = enc.encode_queries(["hip replacement cost"])[0]
    p = enc.encode_passages(["hip replacement cost"])[0]
    assert not np.allclose(q, p, atol=1e-5)


@encoder_available
def test_encoder_places_translations_near_each_other():
    """The property the whole cross-lingual design rests on."""
    from backend.rag.encoder import get_encoder
    enc = get_encoder()
    v = enc.encode_queries(["What was the Manhattan Project?",
                            "मैनहट्टन परियोजना क्या थी?",
                            "how do I bake sourdough bread"])
    same = float(v[0] @ v[1])
    different = float(v[0] @ v[2])
    assert same > different


@encoder_available
def test_encoder_cache_returns_identical_vector_and_counts_hits():
    from backend.rag.encoder import OnnxEncoder
    enc = OnnxEncoder()
    a = enc.encode_query("what is a corporation?", use_cache=True)
    b = enc.encode_query("what is a corporation?", use_cache=True)
    assert np.array_equal(a, b)
    assert enc.stats()["cache"]["hits"] >= 1


@encoder_available
def test_encoder_cache_can_be_bypassed_for_benchmarking():
    from backend.rag.encoder import OnnxEncoder
    enc = OnnxEncoder()
    enc.encode_query("q", use_cache=False)
    enc.encode_query("q", use_cache=False)
    assert enc.stats()["cache"]["hits"] == 0


@encoder_available
def test_encoder_is_thread_safe_under_mixed_query_and_passage_load():
    """Regression test for `RuntimeError: Already borrowed`.

    The encoder used to call `enable_truncation` / `enable_padding` on a shared
    `tokenizers.Tokenizer` before every encode. That takes a mutable borrow of a
    Rust RefCell while a concurrent `encode_batch` holds an immutable one, so
    simultaneous calls raced: 53 of 64 failed, and an eval harness silently lost
    28 of 30 examples.

    It matters in production because FastAPI dispatches sync handlers to a thread
    pool, so two overlapping HTTP requests are enough to trigger it. Queries and
    passages must be interleaved here — they use different truncation lengths,
    which is what made the mutation observable.
    """
    from concurrent.futures import ThreadPoolExecutor

    from backend.rag.encoder import get_encoder
    enc = get_encoder()
    queries = ["what is a corporation?", "मैनहट्टन परियोजना क्या थी?",
               "hip replacement cost"] * 6
    passages = ["A corporation is an independent legal entity. " * 20] * 6

    errors: list[str] = []

    def do_query(t):
        try:
            return enc.encode_query(t, use_cache=False)
        except Exception as e:                    # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")
            return None

    def do_passage(t):
        try:
            return enc.encode_passages([t])
        except Exception as e:                    # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = ([pool.submit(do_query, t) for t in queries]
                   + [pool.submit(do_passage, t) for t in passages])
        outputs = [f.result() for f in futures]

    assert not errors, f"concurrent encoding failed: {errors[:3]}"
    assert all(o is not None for o in outputs)

    # Concurrency must not perturb the result, only its timing.
    reference = enc.encode_query(queries[0], use_cache=False)
    for out, text in zip(outputs, queries):
        if text == queries[0] and out is not None:
            assert np.allclose(out, reference, atol=1e-6)


# ---------------------------------------------------------------------------
# indexes (need the built artifacts)
# ---------------------------------------------------------------------------
def _indexes_ready() -> bool:
    return all(CFG.dense_index_path(l).exists()
               and (CFG.sparse_dir(l) / "metadata.parquet").exists()
               for l in CFG.languages)


indexes_available = pytest.mark.skipif(
    not _indexes_ready(),
    reason="indexes not built; run the data pipeline + scripts/build_indexes.py")


@indexes_available
def test_retriever_returns_hydrated_hits():
    from backend.rag.retrieval import get_retriever
    res = get_retriever().search("what is a corporation?", use_cache=False)
    assert res.hits
    # Text is hydrated lazily for final hits only; every returned hit must have it.
    assert all(h["text"] for h in res.hits)
    assert all(h["chunk_id"] for h in res.hits)


@indexes_available
def test_retriever_reports_per_stage_timing():
    from backend.rag.retrieval import get_retriever
    res = get_retriever().search("what is a corporation?", use_cache=False)
    for key in ("encode_ms", "dense_ms", "sparse_ms", "fuse_ms",
                "hydrate_ms", "retrieval_ms"):
        assert key in res.timing


@indexes_available
def test_retriever_can_restrict_to_one_language():
    from backend.rag.retrieval import get_retriever
    res = get_retriever().search("what is a corporation?", languages=["en"],
                                 use_cache=False)
    assert {h["lang"] for h in res.hits} == {"en"}


@indexes_available
def test_cross_lingual_retrieval_finds_english_from_hindi_query():
    """A Hindi question restricted to the English index must still retrieve."""
    from backend.rag.retrieval import get_retriever
    res = get_retriever().search("मैनहट्टन परियोजना क्या थी?",
                                 languages=["en"], use_cache=False)
    assert res.hits
    assert {h["lang"] for h in res.hits} == {"en"}


@indexes_available
def test_dense_scores_are_valid_cosines():
    from backend.rag.retrieval import get_retriever
    res = get_retriever().search("what is a corporation?", use_cache=False)
    for h in res.dense_hits:
        assert -1.0001 <= h["score"] <= 1.0001


@indexes_available
def test_retrieval_result_exposes_guardrail_signals():
    from backend.rag.retrieval import get_retriever
    res = get_retriever().search("what is a corporation?", use_cache=False)
    assert 0.0 < res.best_cosine <= 1.0
    assert res.best_rrf > 0
    assert res.score_margin >= 0
