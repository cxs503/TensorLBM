"""Tests for CASE-LBM-RB-ONSET-RA1708 (case-pool driver for the Rayleigh-Bénard
onset Rayleigh number).

Two layers:

* **logic** — the bisection/criterion machinery is exercised against a synthetic
  σ(Ra) whose zero is known analytically, so a bug in the bracket validation,
  the marginal band, the turnout gates or the stopping rule is caught without
  running the LBM (seconds).
* **artifacts** — if a production run has been executed
  (``runs/case_rayleigh_benard_onset/summary.json`` or the case ``artifacts/``
  directory), the registered case criteria are re-checked from the JSON.  The
  test is skipped when the artifacts are absent (they are produced by the
  cluster run, not by the unit suite).

The slow end-to-end LBM check is opt-in: ``RB1708_SLOW=1 pytest -k slow``.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "benchmarks" / "case_pool" / "rayleigh_benard_onset"
RUN_DIR = REPO_ROOT / "runs" / "case_rayleigh_benard_onset"


def _load_driver():
    spec = importlib.util.spec_from_file_location("rb_onset_case", CASE_DIR / "run_case.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def drv():
    return _load_driver()


# ---------------------------------------------------------------------------
# logic: criterion + bisection on a synthetic, analytically-known sigma(Ra)
# ---------------------------------------------------------------------------
def test_predeclared_rule_is_frozen(drv):
    """The decision rule must be data on the module, not something recomputed."""
    pre = drv.PREDECLARED
    assert pre["ra_ref"] == pytest.approx(1707.7597)
    assert pre["tol_pct"] == 2.0
    assert pre["bisect_tol_rel"] <= 5e-3
    assert 0.0 < pre["sigma_marginal"] < 0.2
    assert [c["route"] for c in [{"route": "A"}]] == ["A"]
    # the nominal case wavenumber is one roll pair in a Gamma=2 box
    assert abs(3.141592653589793 - pre["a_ref"]) / pre["a_ref"] < 0.02


def test_classify_gates(drv):
    base = {"valid": True, "fit_r2": 0.9999, "sigma_star": 0.5}
    assert drv.classify(base) == "unstable"
    assert drv.classify({**base, "sigma_star": -0.5}) == "stable"
    # near-marginal: amplitude is nearly constant, so R^2 is meaningless — the
    # |sigma*| band decides, not the fit quality
    assert drv.classify({**base, "sigma_star": 0.005, "fit_r2": 0.4}) == "marginal"
    assert drv.classify({**base, "sigma_star": 0.005, "valid": False}) == "marginal"
    # determinate but badly fitted arms do not vote
    assert drv.classify({**base, "fit_r2": 0.95}) == "indeterminate"
    assert drv.classify({**base, "valid": False}) == "indeterminate"


def test_bisection_recovers_a_known_zero(drv, monkeypatch):
    """Synthetic sigma*(Ra) = slope*(Ra - Ra_true): the scan must return Ra_true."""
    ra_true = 1707.7597
    slope = 1.0 / 90.0  # sigma* ~ +-0.19 over the +-10 % bracket

    def fake_arm(ny, nx, ratio, *, steps=None, alpha=None, pr=None):
        s = slope * (ratio * ra_true - ra_true)
        rec = {
            "ny": ny, "nx": nx, "ratio": ratio, "kH": 3.141592653589793,
            "sigma_star": s, "sigma_lattice": s / 1000.0, "fit_r2": 0.99999,
            "fit_points": 100, "fit_window": [0.0, 1.0], "valid": True,
            "steps": 1, "alpha": 1 / 6, "pr": 0.71, "tau_T": 1.0, "tau_f": 0.855,
            "gap_flow": ny - 2, "ma": 1e-3, "amp_initial": 1e-6, "amp_final": 1e-5,
            "route": "A", "k": 0.1, "mode": 1, "ra": ratio * ra_true,
            "sigma_star_over_pr": s / 0.71, "wall_seconds": 0.0,
        }
        rec["verdict"] = drv.classify(rec)
        return rec

    monkeypatch.setattr(drv, "arm", fake_arm)
    res = drv.bisect_onset(32, 64)
    assert res["verdict"] == "ok"
    assert res["stopping_rule"] in ("bracket_width", "sigma_marginal")
    assert res["Ra_c"] == pytest.approx(ra_true, rel=2e-3)
    assert res["abs_err_pct"] < drv.PREDECLARED["tol_pct"]
    assert res["Ra_bracket"][0] < res["Ra_c"] < res["Ra_bracket"][1]
    # bracket endpoints really vote with opposite signs
    assert res["sigma_bracket"][0] < 0 < res["sigma_bracket"][1]
    # the safeguarded interpolation must not cost more arms than plain bisection
    assert res["n_arms"] <= 2 + drv.PREDECLARED["bisect_max_iter"]
    assert res["bracket_expanded"] == 0


def test_bisection_widens_an_invalid_bracket_and_says_so(drv, monkeypatch):
    """A bracket that does not straddle the marginal point must be widened and
    reported, never silently accepted."""
    ra_true = 1707.7597 * 1.15  # onset outside the predeclared bracket

    def fake_arm(ny, nx, ratio, *, steps=None, alpha=None, pr=None):
        s = (ratio - 1.15) * 200.0
        rec = {
            "ny": ny, "nx": nx, "ratio": ratio, "sigma_star": s, "fit_r2": 0.9999,
            "valid": True, "kH": 3.14, "steps": 1, "alpha": 1 / 6, "pr": 0.71,
            "tau_T": 1.0, "tau_f": 0.855, "gap_flow": ny - 2, "ma": 1e-3,
            "amp_initial": 1e-6, "amp_final": 1e-5, "route": "A", "k": 0.1, "mode": 1,
            "ra": ratio * 1707.7597, "sigma_lattice": 0.0, "fit_points": 10,
            "fit_window": [0.0, 1.0], "sigma_star_over_pr": s / 0.71,
            "wall_seconds": 0.0,
        }
        rec["verdict"] = drv.classify(rec)
        return rec

    monkeypatch.setattr(drv, "arm", fake_arm)
    res = drv.bisect_onset(16, 32)
    assert res["verdict"] == "ok"
    assert res["bracket_expanded"] >= 1
    assert res["Ra_c"] == pytest.approx(ra_true, rel=5e-3)


def test_grid_extrapolation_form(drv):
    chain = [{"ny": ny, "Ra_c": 1707.76 + 200.0 / ny} for ny in (32, 48, 64)]
    out = drv.extrapolate_grids(chain)
    assert out["Ra_inf"] == pytest.approx(1707.76, abs=0.05)
    assert out["within_tol_Ra_inf"] is True
    assert out["max_abs_residual"] < 1e-6


# ---------------------------------------------------------------------------
# artifacts: the registered case criteria, re-checked from the run JSON
# ---------------------------------------------------------------------------
def _summary():
    for cand in (RUN_DIR / "summary.json", CASE_DIR / "artifacts" / "summary.json"):
        if cand.exists():
            return json.loads(cand.read_text()), cand
    return None, None


def test_production_artifacts_meet_the_registered_case():
    summary, path = _summary()
    if summary is None:
        pytest.skip("no production artifacts yet (run run_case.py on the cluster)")
    pre = summary["predeclared"]
    assert summary["case_code"] == "CASE-LBM-RB-ONSET-RA1708"
    # control arm first: without it no LBM statement is admissible
    assert summary["route_c"]["control_ok"] is True, summary["route_c"]
    primary = [c for c in summary["route_a"]
               if [c["ny"], c["nx"]] == pre["primary_grid"]][0]
    assert primary["verdict"] == "ok"
    assert primary["abs_err_pct"] < pre["tol_pct"], primary
    assert summary["grid_convergence"]["abs_err_pct_Ra_inf"] < pre["tol_pct"]
    # every voting arm was a real fit and the bracket really straddles zero
    for c in summary["route_a"]:
        assert c["Ra_bracket"][0] < c["Ra_c"] < c["Ra_bracket"][1]
        assert c["sigma_bracket"][0] < 0.0 < c["sigma_bracket"][1]
        for arm in c["history"]:
            if arm["verdict"] in ("stable", "unstable"):
                assert arm["fit_r2"] >= pre["fit_r2_min"]
                assert arm["valid"] is True
    assert all(v for v in summary["checks"].values() if v is not None)
    assert path is not None


@pytest.mark.skipif(os.environ.get("RB1708_SLOW") != "1",
                    reason="set RB1708_SLOW=1 for the end-to-end LBM check (~2 min)")
def test_slow_end_to_end_bisection(drv):
    """One coarse (ny=16) end-to-end bisection through the real LBM solver."""
    saved = list(drv.PREDECLARED["bracket_ratio"])
    try:
        drv.PREDECLARED["bracket_ratio"] = [0.75, 1.60]
        res = drv.bisect_onset(16, 32, tol_rel=2e-2, max_iter=5)
    finally:
        drv.PREDECLARED["bracket_ratio"] = saved
    assert res["verdict"] == "ok"
    assert res["sigma_bracket"][0] < 0.0 < res["sigma_bracket"][1]
    # coarse grid: only require the right ballpark and honest bracketing
    assert abs(res["err_pct"]) < 10.0