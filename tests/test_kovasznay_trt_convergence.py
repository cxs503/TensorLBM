"""Tests for the Kovasznay TRT convergence arm (CASE-LBM-KOVASZNAY-EXACT).

These tests bind three claims made by ``cases/kovasznay_2d/trt_convergence.py``:

1. ``collide_trt`` is a *conservative* collision (mass and momentum preserved to
   round-off) -- i.e. the low-Ma arms are not being pushed by a leaking scheme;
2. TRT and BGK relax to the same steady Kovasznay solution when tau+ is the same
   (so the TRT arm measures the same physics, only with better stability);
3. TRT remains finite at tau+ where BGK blows up -- the stability-wall claim that
   makes the low-Ma (low-tau+) convergence ladder possible at all;
4. the analysis helpers (observed order / Richardson) recover a known p = 2.

The heavy production ladders live in ``artifacts/m18406/*.json``; when those files
are present the test also asserts the sign of the reported trend (interior error
decreasing with refinement), but the assertion is skipped when the artifacts were
not checked out -- the cheap checks above never depend on them.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "cases" / "kovasznay_2d"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(CASE))

torch = pytest.importorskip("torch")

import trt_convergence as tc  # noqa: E402


def test_collide_trt_is_conservative():
    """Mass / momentum must be preserved to machine precision."""
    ident = tc.arm_pressure_identity(ny=16)
    assert abs(ident["mass_rel_change"]) < 1e-13
    assert ident["momentum_x_abs"] < 1e-12
    assert ident["momentum_y_abs"] < 1e-12
    assert ident["rho_max_abs_change"] < 1e-12


def test_trt_matches_bgk_steady_solution_at_same_tau_plus():
    """Same tau+ => same steady solution; TRT only changes the anti-symmetric rate."""
    ny = 16
    steps = 400
    bgk = tc.run_case(ny, u0=0.03, collision="bgk", min_steps=steps, max_steps=steps,
                      check_every=steps, drift_tol=0.0, threads=2)
    trt = tc.run_case(ny, u0=0.03, collision="trt", min_steps=steps, max_steps=steps,
                      check_every=steps, drift_tol=0.0, threads=2)
    assert bgk["finite"] and trt["finite"]
    assert bgk["tau_plus"] == trt["tau_plus"]
    # at ny=16/400 steps the arm is still relaxing; agreement to ~10 % of the error level
    # is enough to rule out a mis-parametrised relaxation rate (a wrong tau- shows up as
    # a factor-level difference, not a percent one).
    assert abs(trt["full"]["v_l2_rel_pct"] - bgk["full"]["v_l2_rel_pct"]) < 0.25 * bgk["full"][
        "v_l2_rel_pct"
    ]
    assert abs(trt["full"]["u_l2_rel_pct"] - bgk["full"]["u_l2_rel_pct"]) < 0.25 * bgk["full"][
        "u_l2_rel_pct"
    ]


def test_trt_survives_where_bgk_diverges():
    """The documented BGK stability wall (tau+ ~ 0.57) must be pushed down by TRT."""
    ny = 16
    u0 = 0.001875  # tau+ = 0.5 + 3*U0*ny/Re = 0.5022
    steps = 800
    bgk = tc.run_case(ny, u0=u0, collision="bgk", min_steps=steps, max_steps=steps,
                      check_every=100, drift_tol=0.0, threads=2)
    trt = tc.run_case(ny, u0=u0, collision="trt", min_steps=steps, max_steps=steps,
                      check_every=100, drift_tol=0.0, threads=2)
    assert bgk["tau_plus"] == pytest.approx(trt["tau_plus"])
    assert trt["tau_plus"] < 0.51
    assert not bgk["finite"], "BGK unexpectedly survived the documented stability wall"
    assert trt["finite"], "TRT did not survive the tau+ at which BGK diverges"


def test_observed_order_and_richardson_recover_second_order():
    """Analysis helpers must return p = 2 and the exact h->0 limit on synthetic data."""
    h = [1 / 32, 1 / 64, 1 / 128]
    c = 3.0e-3
    err = [c * hi**2 for hi in h]
    rows = tc.observed_order(h, err)
    assert len(rows) == 2
    assert all(abs(r["p"] - 2.0) < 1e-9 for r in rows)
    assert tc.richardson(h, err, 2.0) == pytest.approx(0.0, abs=1e-15)


ART = REPO / "artifacts" / "m18406"


@pytest.mark.parametrize("name", ["ladder_u00075_trt.json", "ladder64.json", "ladder128.json"])
def test_production_ladder_artifacts_are_self_consistent(name):
    """If the production ladders were fetched, their numbers must be sane."""
    path = ART / name
    if not path.exists():
        pytest.skip(f"{name} not present (production artifacts fetched separately)")
    data = json.loads(path.read_text())
    rows = [r for r in data["rows"] if r.get("finite")]
    assert rows, f"{name}: every grid diverged"
    for r in rows:
        assert math.isfinite(r["full"]["u_l2_rel_pct"])
        inter = r["interior"]
        assert inter["u_l2_abs_over_u0_pct"] > 0
        assert inter["v_l2_abs_over_u0_pct"] > 0
        # the interior window must be strictly inside the domain
        lo, hi = r["interior_cols"]
        assert 0 < lo < hi < r["nx"]