"""Tests for the contact-angle inversion closure audit.

Fast (no LBM): exercises the pure post-processing that decides whether a
calibration-based inversion was actually validated.

The important test here is :func:`test_round1_numbers_are_tautological`: it
reproduces the round-1 ("8/9 within 15%") error arithmetic from the published
calibration table and shows the errors are zero *by construction*, i.e. the
published pass count measured self-consistency of an interpolant, not physics.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.contact_angle_inversion.closure import (  # noqa: E402
    CRITERION_PCT,
    closure_metrics,
    estimator_disagreement,
    piecewise_linear_forward,
    piecewise_linear_invert,
    self_interpolation_error,
    stage_verdict,
)

# Round-1 (benchmarks/xval_huang2007_contact_angle_inversion/results.json)
# coarse-grid calibration table, f_w -> measured theta.
ROUND1_COARSE = [
    (0.40, 132.03),
    (0.45, 118.92),
    (0.50, 105.73),
    (0.55, 91.76),
    (0.65, 61.31),
    (0.70, 41.35),
]


def test_piecewise_linear_invert_roundtrips_knots():
    inv = [(th, fw) for fw, th in ROUND1_COARSE]
    for fw, th in ROUND1_COARSE:
        assert piecewise_linear_invert(inv, th) == pytest.approx(fw, abs=1e-9)


def test_piecewise_linear_invert_is_monotone_in_angle():
    inv = [(th, fw) for fw, th in ROUND1_COARSE]
    fws = [piecewise_linear_invert(inv, th) for th in (50, 70, 90, 110, 130)]
    # Larger contact angle (more hydrophobic) needs a weaker wall.
    assert all(a > b for a, b in zip(fws, fws[1:]))


def test_forward_fit_clamps_outside_range():
    fwd = list(ROUND1_COARSE)
    assert piecewise_linear_forward(fwd, 0.10) == pytest.approx(132.03)
    assert piecewise_linear_forward(fwd, 0.95) == pytest.approx(41.35)


def test_round1_numbers_are_tautological():
    """In-grid targets round-trip to machine precision => the pass count is empty."""
    targets = [60.0, 75.0, 90.0, 105.0, 120.0]
    rows = self_interpolation_error(ROUND1_COARSE, targets)
    for row in rows:
        assert not row["clamped"]
        assert row["err_pct"] < 1e-9
    # A target outside the measured range cannot be inverted at all: the
    # routine returns the end knot, so any "pass" it produces there is an
    # accident of where the knot happens to sit.  The flag exists so such rows
    # are visible instead of silently counted.
    clamped = self_interpolation_error(ROUND1_COARSE, [30.0, 150.0])
    assert all(r["clamped"] for r in clamped)
    # 150 deg clamps to the 132.03 deg knot => 12% "error" even though nothing
    # was inverted, which is exactly why clamped rows must not be scored.
    assert clamped[1]["err_pct"] < CRITERION_PCT
    assert clamped[1]["theta_roundtrip"] == pytest.approx(132.03)


def test_closure_metrics_recompute_errors_from_raw_numbers():
    rows = [
        {"label": "in_grid_70", "theta_target": 70.0, "theta_measured": 74.0},
        {"label": "in_grid_100", "theta_target": 100.0, "theta_measured": 70.0},
        {"label": "in_grid_130", "theta_target": 130.0, "theta_measured": float("nan"), "status": "no_measurement"},
    ]
    m = closure_metrics(rows)
    assert m["n_rows"] == 3
    assert m["n_measured"] == 2
    assert m["rows"][0]["err_pct"] == pytest.approx(4.0 / 70.0 * 100.0)
    assert m["rows"][1]["err_pct"] == pytest.approx(30.0)
    assert m["n_passed"] == 1
    assert m["max_err_pct"] == pytest.approx(30.0)
    assert m["all_passed"] is False


def test_closure_metrics_refuses_to_pass_without_measurements():
    m = closure_metrics([{"theta_target": 90.0, "theta_measured": float("nan"), "status": "no_measurement"}])
    assert m["n_measured"] == 0
    assert m["all_passed"] is False
    v = stage_verdict({"in_grid": m})
    assert v["stages"]["in_grid"] == "inconclusive"
    assert v["verdict"] == "INCONCLUSIVE"


def test_stage_verdict_rules():
    good = closure_metrics([{"theta_target": 90.0, "theta_measured": 91.0}])
    bad = closure_metrics([{"theta_target": 90.0, "theta_measured": 40.0}])
    assert stage_verdict({"in_grid": good})["verdict"] == "PASS"
    # A failing transfer stage must sink the overall verdict.
    v = stage_verdict({"in_grid": good, "transfer_gridB": bad})
    assert v["stages"]["transfer_gridB"] == "fail"
    assert v["verdict"] == "FAIL"


def test_estimator_disagreement_flags_large_delta():
    ok = estimator_disagreement(91.0, 93.5)
    assert ok["agree"] and ok["status"] == "ok"
    bad = estimator_disagreement(91.0, 120.0)
    assert not bad["agree"] and bad["status"] == "estimators_disagree"
    nan = estimator_disagreement(float("nan"), 90.0)
    assert nan["status"] == "no_measurement"


def _closure_results() -> dict | None:
    path = REPO / "artifacts" / "xval_huang2007_contact_angle_inversion" / "closure_result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def test_committed_closure_results_are_self_consistent():
    """If a closure run is committed, its verdict must follow from its raw rows."""
    data = _closure_results()
    if data is None:
        pytest.skip("no closure_result.json committed yet")
    assert "stages" in data and "verdict" in data
    for name, stage in data["stages"].items():
        recomputed = closure_metrics(stage["rows"])
        assert recomputed["n_measured"] == stage["metrics"]["n_measured"]
        assert recomputed["n_passed"] == stage["metrics"]["n_passed"]
        if recomputed["n_measured"]:
            assert recomputed["max_err_pct"] == pytest.approx(stage["metrics"]["max_err_pct"], abs=1e-6)
    verdict = stage_verdict({k: v["metrics"] for k, v in data["stages"].items()})
    stored = data["verdict"]["verdict"] if isinstance(data["verdict"], dict) else data["verdict"]
    assert verdict["verdict"] == stored