"""Benchmark-level tests for the Rayleigh--Taylor and Ritter drivers."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.dam_break_ritter.self_similar import run_ritter_suite  # noqa: E402
from benchmarks.rayleigh_taylor.growth_rate import run_growth_rate_suite  # noqa: E402


def test_rayleigh_taylor_suite_fast(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    report = run_growth_rate_suite(
        lambdas=(32, 64), fast=True, output_root=tmp_path / "runs", result_path=result_path
    )
    assert result_path.exists()
    on_disk = json.loads(result_path.read_text(encoding="utf-8"))
    assert on_disk["benchmark"] == "rayleigh_taylor_growth_rate"
    assert len(report["cases"]) == 2
    for case in report["cases"]:
        assert case["sigma_analytic"] > 0.0
        assert math.isfinite(case["sigma_sim"])
        assert case["sigma_sim"] > 0.0
        assert case["fit_r2"] > 0.9
        assert case["phase_mass_drift_pct"] < 1.0
    assert "observed_orders" in report["grid_convergence"]
    assert report["verdict"]["status"] in {"PASS", "FAIL"}
    assert (tmp_path / "runs" / "lam32" / "growth_history.csv").exists()


def test_ritter_suite_fast(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    report = run_ritter_suite(
        resolutions=(200, 400),
        fast=True,
        output_root=tmp_path / "runs",
        result_path=result_path,
    )
    assert result_path.exists()
    assert len(report["cases"]) == 2
    for case in report["cases"]:
        assert case["mass_drift_pct"] < 1e-6  # conservative to round-off
        assert case["gate_mass_pass"] is True
        assert math.isfinite(case["front_err_mass_pct_max"])
        assert len(case["front_series"]) > 0
    assert report["verdict"]["status"] in {"PASS", "FAIL"}
    assert report["verdict"]["declared_reference_resolution"] == 400