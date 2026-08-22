"""Tests for the serve-artifact bootstrap (deployment provisioning).

These cover the decision logic — is each artifact present, what is missing, and
what happens with no repo configured — without any network access. The actual
Hub download is integration-only and is not unit-tested here.

Readiness is driven by real paths from `CFG`, so the tests redirect `CFG` at a
tmp tree rather than monkeypatching module constants; that keeps them honest
about the paths the deployed app will actually check.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.rag import bootstrap as B
from backend.rag.config import CFG


@pytest.fixture
def fake_root(monkeypatch, tmp_path):
    """Point CFG's derived paths at an empty tmp tree."""
    monkeypatch.setattr(CFG, "data", tmp_path, raising=False)
    monkeypatch.setattr(CFG, "languages", ["hi", "en"], raising=False)
    monkeypatch.setattr(B, "PROCESSED", tmp_path / "processed", raising=False)
    return tmp_path


def _make_encoder(tmp_path):
    onnx = CFG.encoder_onnx_path()
    onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.write_bytes(b"fake onnx graph")
    tok = CFG.tokenizer_path
    tok.parent.mkdir(parents=True, exist_ok=True)
    tok.write_text("{}", encoding="utf-8")


def _make_indexes(languages):
    for lang in languages:
        dense = CFG.dense_index_path(lang)
        dense.parent.mkdir(parents=True, exist_ok=True)
        dense.write_bytes(b"fake hnsw")
        CFG.dense_meta_path(lang).write_bytes(b"fake parquet")
        sparse = CFG.sparse_dir(lang)
        sparse.mkdir(parents=True, exist_ok=True)
        (sparse / "metadata.parquet").write_bytes(b"fake parquet")


# ---------------------------------------------------------------------------
# readiness
# ---------------------------------------------------------------------------
def test_nothing_ready_on_empty_tree(fake_root):
    assert B.encoder_ready() is False
    assert B.indexes_ready() is False
    assert B.serve_data_ready() is False


def test_encoder_ready_needs_both_graph_and_tokenizer(fake_root):
    onnx = CFG.encoder_onnx_path()
    onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.write_bytes(b"fake onnx graph")
    # graph without tokenizer is not usable
    assert B.encoder_ready() is False
    CFG.tokenizer_path.write_text("{}", encoding="utf-8")
    assert B.encoder_ready() is True


def test_indexes_ready_requires_every_configured_language(fake_root):
    _make_indexes(["hi"])
    # 'en' is configured but missing
    assert B.indexes_ready() is False
    _make_indexes(["en"])
    assert B.indexes_ready() is True


def test_indexes_ready_requires_sparse_not_just_dense(fake_root):
    for lang in ("hi", "en"):
        dense = CFG.dense_index_path(lang)
        dense.parent.mkdir(parents=True, exist_ok=True)
        dense.write_bytes(b"fake hnsw")
        CFG.dense_meta_path(lang).write_bytes(b"fake parquet")
    assert B.indexes_ready() is False


def test_serve_data_ready_when_everything_present(fake_root):
    _make_encoder(fake_root)
    _make_indexes(["hi", "en"])
    assert B.serve_data_ready() is True


def test_indexes_ready_accepts_explicit_language_subset(fake_root):
    _make_indexes(["hi"])
    assert B.indexes_ready(["hi"]) is True
    assert B.indexes_ready(["hi", "en"]) is False


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def test_missing_report_pinpoints_the_gap(fake_root):
    """A deploy failure should be one line to diagnose, not an opaque 500."""
    _make_encoder(fake_root)
    _make_indexes(["hi"])
    report = B.missing_report()
    assert report["encoder"] is True
    assert report["languages"]["hi"] == {"dense": True, "sparse": True}
    assert report["languages"]["en"] == {"dense": False, "sparse": False}


def test_missing_report_flags_unconfigured_repo(fake_root, monkeypatch):
    monkeypatch.setattr(CFG, "data_repo", "", raising=False)
    assert B.missing_report()["data_repo_configured"] is False
    monkeypatch.setattr(CFG, "data_repo", "someone/some-data", raising=False)
    assert B.missing_report()["data_repo_configured"] is True


# ---------------------------------------------------------------------------
# no-network behaviour
# ---------------------------------------------------------------------------
def test_bootstrap_is_a_noop_when_already_ready(fake_root, monkeypatch):
    _make_encoder(fake_root)
    _make_indexes(["hi", "en"])

    def explode(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("bootstrap tried to download despite being ready")

    monkeypatch.setattr(B, "bootstrap_encoder", explode)
    monkeypatch.setattr(B, "bootstrap_indexes", explode)
    # serve_data_ready() short-circuits before either download path
    assert B.serve_data_ready() is True


def test_bootstrap_indexes_returns_false_without_a_repo(fake_root, monkeypatch):
    monkeypatch.setattr(CFG, "data_repo", "", raising=False)
    monkeypatch.delenv("HHGOA_DATA_REPO", raising=False)
    assert B.bootstrap_indexes() is False


def test_bootstrap_never_raises_on_download_failure(fake_root, monkeypatch):
    """A misconfigured deployment must surface at /api/health, not crash boot."""
    def boom(*a, **k):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(B, "bootstrap_encoder", boom)
    monkeypatch.setattr(B, "bootstrap_indexes", boom)
    assert B.bootstrap_serve_data() is False


def test_encoder_repo_is_the_public_model_source():
    """The encoder is public, so it is not mirrored into the private data repo."""
    assert "multilingual-e5-small" in B.ENCODER_REPO
