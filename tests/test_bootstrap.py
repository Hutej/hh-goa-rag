"""Tests for the serve-data bootstrap (deployment data provisioning).

Covers the no-op-when-present path and the no-repo path without any network.
The actual HF download path is integration-only (needs a private dataset repo)
and is not unit-tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.rag import bootstrap as B


def test_serve_data_ready_true_when_qdrant_present(monkeypatch, tmp_path):
    # point PROCESSED at a tmp tree with the Qdrant marker dir
    marker = tmp_path / "qdrant" / "collection" / "hhgoa_adaptive"
    marker.mkdir(parents=True)
    monkeypatch.setattr(B, "PROCESSED", tmp_path)
    monkeypatch.setattr(B, "SERVE_READY_MARKER", marker)
    assert B.serve_data_ready() is True


def test_serve_data_ready_false_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "PROCESSED", tmp_path)
    monkeypatch.setattr(B, "SERVE_READY_MARKER", tmp_path / "qdrant" / "collection" / "hhgoa_adaptive")
    assert B.serve_data_ready() is False


def test_bootstrap_noop_when_ready(monkeypatch, tmp_path):
    marker = tmp_path / "qdrant" / "collection" / "hhgoa_adaptive"
    marker.mkdir(parents=True)
    monkeypatch.setattr(B, "PROCESSED", tmp_path)
    monkeypatch.setattr(B, "SERVE_READY_MARKER", marker)
    # must not touch the network
    assert B.bootstrap_serve_data() is True


def test_bootstrap_returns_false_when_no_repo_and_data_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "PROCESSED", tmp_path)
    monkeypatch.setattr(B, "SERVE_READY_MARKER", tmp_path / "qdrant" / "collection" / "hhgoa_adaptive")
    monkeypatch.delenv("HHGOA_DATA_REPO", raising=False)
    assert B.bootstrap_serve_data() is False
