"""CASE-TAYLOR-GREEN-3D round 2 guard: the Chapman-Enskog consistent IC and the rate metric.

The round-1 case initialised ``f = f^eq(u0)``, i.e. with **zero viscous stress**, while the
exact Taylor-Green solution has ``sigma = 2 rho nu S`` at t=0.  That state error is O(k^2)
and it dominated both the reported L2 error and the fitted decay exponent (see
``docs/CASE_TAYLOR_GREEN_3D_round2_IC_attribution.md``).  These tests pin the two claims
that the repair relies on, on grids small enough for CI:

* the CE-consistent IC reproduces the exact stress (energy offset and L2 error drop by
  more than an order of magnitude relative to the equilibrium IC);
* the *asymptotic* decay-rate error (slope of ``ln(E_num/E_exact)``) is what M2 should have
  measured, and it is ~1e-4 relative -- the solver's decay accuracy, not the IC's.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "benchmarks" / "case_taylor_green_3d"
SRC = REPO / "src"


def _load(name: str, filename: str):
    for p in (str(SRC), str(CASE_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(name, CASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ra():
    torch.set_num_threads(2)
    return _load("tg3d_attribution", "run_attribution.py")


def test_ce_ic_lands_on_the_discrete_slow_manifold(ra):
    """f = f^eq + K w Q S (K = -rho tau / c_s^2) removes the O(k^2) state error."""
    eq = ra.run_trace("tg", 16, 0.6, 2e-3, 100, 10, ic="eq")
    ce = ra.run_trace("tg", 16, 0.6, 2e-3, 100, 10, ic="tau")
    # both keep the initial velocity exactly (the extra part carries no mass/momentum)
    assert abs(eq["ic_velocity_rel_err"]) < 1e-10
    assert abs(ce["ic_velocity_rel_err"]) < 1e-10
    # the equilibrium IC leaves a sizeable energy offset, the consistent one does not
    assert abs(eq["offset_final_pct"]) > 1.0
    assert abs(ce["offset_final_pct"]) < 0.2 * abs(eq["offset_final_pct"])
    # and the L2 field error drops by more than a factor 5
    assert ce["l2_final"] < eq["l2_final"] / 5.0
    # mass is conserved in both
    assert eq["mass_drift_rel"] < 1e-12 and ce["mass_drift_rel"] < 1e-12


def test_asymptotic_rate_error_is_the_real_accuracy(ra):
    """The asymptotic rate error is tiny; the whole-history M2 fit is inflated by the startup."""
    r = ra.run_trace("tg", 16, 0.6, 2e-3, 200, 10, ic="eq")
    assert abs(r["asym_rate_err_pct"]) < 0.1, r["asym_rate_err_pct"]
    # round-1 metric (fit window t_start = 0) sees the transient and is much larger
    assert abs(r["M2_round1_pct"]) > 3.0 * abs(r["asym_rate_err_pct"])
    # refitting after the transient recovers the asymptotic value
    late = r["fit_windows"]["100"]["decay_exponent_rel_err_pct"]
    assert abs(late) < abs(r["M2_round1_pct"])


def test_prereg_and_artifacts_are_consistent(ra):
    """The published summary must reproduce from the stored numbers (no hand edits)."""
    prereg = json.loads((CASE_DIR / "prereg_round2.json").read_text())
    assert set(prereg["metrics"]) >= {"R1_M2_inflation", "R2_ic_repair", "R3_order_recovery", "R4_rate_accuracy"}
    summary_path = CASE_DIR / "artifacts" / "attribution_summary.json"
    if not summary_path.exists():
        pytest.skip("attribution artefacts not generated yet")
    s = json.loads(summary_path.read_text())
    r1 = s["R1_M2_inflation"]
    recomputed = abs(r1["M2_pct"]) / abs(r1["asym_pct"])
    assert recomputed == pytest.approx(r1["value"], rel=1e-6)
    assert r1["pass"] == (recomputed >= 5.0)
    r4 = s["R4_rate_accuracy"]
    assert r4["pass"] == (r4["worst_abs_asym_pct"] < 0.1)