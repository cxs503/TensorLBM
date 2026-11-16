"""Acceptance test for benchmarks/verified/graetz_tube (KPI-LBM-VERIFIED-CAPS).

The test only *reads* the stored machine results: the reference, the two-level
convergence study and the gate evaluation live in
``benchmarks/verified/graetz_tube/{reference_series,study,make_artifacts}.py``.
It asserts the invariants a verification claim must satisfy, not merely that
the code ran.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "verified" / "graetz_tube"
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(ROOT / "src"))

import reference_series as RS  # noqa: E402


@pytest.fixture(scope="module")
def result():
    p = CASE / "result.json"
    if not p.exists():
        pytest.skip("result.json not generated (run make_artifacts.py)")
    return json.loads(p.read_text())


def test_reference_reproduces_published_spectrum():
    """Independent series must reproduce the published eigenvalues / Nu_inf."""
    ref = RS.series(np.array([0.2]), n_modes=10)
    assert abs(RS.LAMBDA[0] - 2.70436) < 1e-5
    assert abs(2 * RS.LAMBDA[0] ** 2 - 14.6275) < 1e-2
    assert abs(ref["Nu"][0] - RS.NU_INF) / RS.NU_INF < 3e-3
    # classic Graetz coefficients (Chandrasekhar 1961)
    assert np.allclose(np.abs(ref["C"][:4]), [1.4764, 0.8061, 0.5888, 0.4759], atol=2e-3)


def test_reference_agrees_with_library_series():
    """Two code paths (independent reference vs tensorlbm.graetz) must agree."""
    from tensorlbm.graetz import graetz_exact

    xs = np.array([0.005, 0.01, 0.02, 0.04])
    ref = RS.series(xs, n_modes=10)["Nu"]
    lib = graetz_exact(x_star=xs, n_modes=10)["Nu"]
    assert np.max(np.abs(lib / ref - 1.0)) < 1e-3


def test_accepted_variant_passes_both_gates(result):
    v = [k for k, d in result["variants"].items() if d["VERIFIED"]]
    assert v, "no geometric convention satisfied G1 (<=3%) + G2 (monotone refinement)"
    for name in v:
        d = result["variants"][name]
        assert d["G1_accuracy"] and d["max_abs_err_gated_56"] <= 0.03
        assert d["G2_monotone"]


def test_error_decreases_under_refinement(result):
    """Every gated station must improve when n_r doubles (no hidden worsening)."""
    for name, d in result["variants"].items():
        e40 = np.abs(np.array(d["levels"]["40"]["rel_err"]))
        e56 = np.abs(np.array(d["levels"]["56"]["rel_err"]))
        idx = [result["x_star"].index(x) for x in result["gated_stations"]]
        assert np.all(e56[idx] < e40[idx]), name


def test_frontier_station_is_disclosed_not_masked(result):
    """The thin-boundary-layer station is reported for both levels and variants."""
    for name, d in result["variants"].items():
        assert "frontier_x0.002_err_40" in d and "frontier_x0.002_err_56" in d
        assert np.isfinite(d["frontier_x0.002_err_56"])


def test_nu_inf_limit_approached(result):
    """The developed asymptote Nu_inf = 3.6568 is approached from above downstream."""
    d = result["variants"][[k for k, x in result["variants"].items() if x["VERIFIED"]][0]]
    nu_last = d["levels"]["56"]["Nu"][-1]
    assert abs(nu_last - RS.NU_INF) / RS.NU_INF < 0.05
