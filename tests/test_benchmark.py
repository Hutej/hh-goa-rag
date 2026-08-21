"""Tests for Phase 6 benchmark percentile helper.

Pure/offline — checks the percentile math only.

Run:
    venv/bin/python -m pytest tests/test_benchmark.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import the percentiles helper from the benchmark script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import importlib
bench = importlib.import_module("benchmark")
percentiles = bench.percentiles


class TestPercentiles:

    def test_p100_is_max(self):
        p = percentiles([1.0, 5.0, 3.0, 9.0, 2.0])
        assert p["P100"] == 9.0

    def test_p50_median_odd(self):
        p = percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
        assert p["P50"] == 3.0

    def test_p50_median_even(self):
        p = percentiles([1.0, 2.0, 3.0, 4.0])
        # numpy linear interpolation between 2nd and 3rd of sorted -> 2.5
        assert p["P50"] == 2.5

    def test_p70_between(self):
        # 10 values 1..10; P70 ~ 7.3 (numpy interpolation)
        p = percentiles([float(i) for i in range(1, 11)])
        assert 7.0 <= p["P70"] <= 8.0

    def test_empty_returns_zeros(self):
        p = percentiles([])
        assert p["P50"] == 0.0
        assert p["P70"] == 0.0
        assert p["P100"] == 0.0

    def test_single_value(self):
        p = percentiles([7.5])
        assert p["P50"] == 7.5
        assert p["P100"] == 7.5

    def test_keys_present(self):
        p = percentiles([1.0, 2.0])
        assert set(p.keys()) == {"P50", "P70", "P100"}
