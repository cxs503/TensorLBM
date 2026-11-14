"""Tests for GAP-DIFFLBM-PAPER-EVIDENCE benchmark suite.

Verifies that all ≥6 independent benchmarks pass their error thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "artifacts" / "GAP-DIFFLBM-PAPER-EVIDENCE" / "benchmark_suite.json"


def test_artifact_exists():
    """Verify the benchmark suite artifact exists."""
    assert ARTIFACT.exists(), f"Benchmark artifact not found at {ARTIFACT}"


def test_benchmark_count():
    """Verify ≥6 independent benchmarks are present."""
    with open(ARTIFACT) as f:
        data = json.load(f)
    n = len(data["benchmarks"])
    assert n >= 6, f"Expected ≥6 benchmarks, got {n}"


def test_all_under_3pct():
    """Verify all benchmarks pass the <3% error threshold."""
    with open(ARTIFACT) as f:
        data = json.load(f)
    for b in data["benchmarks"]:
        assert b["passed"], f"Benchmark {b['name']} failed: error={b['error_pct']:.4f}%"


def test_analytical_benchmarks():
    """Verify ≥2 analytical benchmarks are present."""
    with open(ARTIFACT) as f:
        data = json.load(f)
    analytical = [b for b in data["benchmarks"] if "analytical" in b["reference_source"].lower()
                  or "exact" in b["reference_source"].lower()]
    assert len(analytical) >= 2, f"Expected ≥2 analytical benchmarks, got {len(analytical)}"


def test_literature_benchmarks():
    """Verify ≥3 literature benchmarks are present."""
    with open(ARTIFACT) as f:
        data = json.load(f)
    literature = [b for b in data["benchmarks"]
                  if any(name in b["reference_source"] for name in ["Ghia", "Schäfer", "Turek", "Shah", "London", "Womersley"])]
    assert len(literature) >= 3, f"Expected ≥3 literature benchmarks, got {len(literature)}"


def test_error_table_consistency():
    """Verify error table matches benchmark results."""
    with open(ARTIFACT) as f:
        data = json.load(f)
    assert len(data["error_table"]) == len(data["benchmarks"]), "Error table length mismatch"
    for et, b in zip(data["error_table"], data["benchmarks"]):
        assert abs(et["err_pct"] - b["error_pct"]) < 0.01, f"Error mismatch for {b['name']}"


def test_verdict():
    """Verify the verdict meets paper requirements."""
    with open(ARTIFACT) as f:
        data = json.load(f)
    v = data["verdict"]
    assert v["total_benchmarks"] >= 6, f"Need ≥6 benchmarks, got {v['total_benchmarks']}"
    assert v["analytical_benchmarks"] >= 2, f"Need ≥2 analytical, got {v['analytical_benchmarks']}"
    assert v["literature_benchmarks"] >= 2, f"Need ≥2 literature, got {v['literature_benchmarks']}"
    assert v["all_under_3pct"], "Not all benchmarks under 3%"
    assert v["meets_paper_requirement"], "Does not meet paper requirement"
