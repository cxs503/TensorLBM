"""Gate tests for XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK.

Fast subset (<60 s) of the predeclared study: it asserts *invariants*, not just
"the code runs":

* the half-way moving-wall closure reproduces the exact steady Couette profile
  and puts the no-slip surface at y = 0.5;
* the Ladd momentum term conserves mass (no source at the moving wall);
* the reference closure beats the naive ablations by orders of magnitude, i.e.
  the reference case really discriminates between wall models;
* the analytic helpers are the standard closed forms (spot values).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

XV = Path(__file__).resolve().parents[1] / "xval" / "XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK"
sys.path.insert(0, str(XV))

import mwbb  # noqa: E402


@pytest.fixture(scope="module")
def couette_ref():
    return mwbb.run_couette(ny=18, nx=4, tau=0.8, u0=0.05, n_steps=6000)


@pytest.fixture(scope="module")
def stokes_ref():
    return mwbb.run_stokes2(ny=38, nx=4, tau=0.8, u0=0.02, delta_bl=6.0,
                            n_periods=3.0, sample_stride=8)


@pytest.fixture(scope="module")
def stokes_static():
    return mwbb.run_stokes2(ny=38, nx=4, tau=0.8, u0=0.02, delta_bl=6.0,
                            n_periods=3.0, sample_stride=8, variant="static_bb")


@pytest.fixture(scope="module")
def stokes_precollision():
    return mwbb.run_stokes2(ny=38, nx=4, tau=0.8, u0=0.02, delta_bl=6.0,
                            n_periods=3.0, sample_stride=8, variant="ladd_precollision")


@pytest.fixture(scope="module")
def stokes_eqwall():
    return mwbb.run_stokes2(ny=38, nx=4, tau=0.8, u0=0.02, delta_bl=6.0,
                            n_periods=3.0, sample_stride=8, variant="eq_wall_node")


def test_couette_exact_linear_profile(couette_ref):
    """Half-way BB is exact for the steady linear Couette profile (float64)."""
    assert couette_ref["steady"]
    assert couette_ref["max_rel_pct_cf_umax"] < 1e-4


def test_couette_wall_at_half_link(couette_ref):
    """The fitted no-slip surface must sit at the half link, not on the wall node."""
    assert abs(couette_ref["y_wall_fit"]) < 0.01


def test_couette_slope_matches_newtonian_shear(couette_ref):
    """Discrete slope must equal U0/H_eff (i.e. wall shear is right)."""
    assert abs(couette_ref["slope_err_pct"]) < 1e-4


def test_moving_wall_conserves_mass(stokes_ref, couette_ref):
    """The Ladd term injects momentum, never mass."""
    assert abs(stokes_ref["mass_drift_pct"]) < 1e-6
    assert abs(couette_ref["mass_drift_pct"]) < 1e-6


def test_stokes2_within_anchor_tolerance(stokes_ref):
    """Anchor acceptance: max |du|/U0 over the resolved layer < 15%."""
    assert stokes_ref["max_err_pct_U0_resolved"] < 15.0


def test_stokes2_phase_tracks_analytic_solution(stokes_ref):
    p = mwbb.harmonic_fit(stokes_ref, 1.0)
    assert abs(p["phase_err_deg"]) < 5.0
    assert abs(p["amp_err_pct"]) < 5.0


def test_ablation_reference_beats_naive_closures(stokes_ref, stokes_static,
                                                 stokes_precollision, stokes_eqwall):
    """Discriminating power: the reference closure must win by a clear margin."""
    ref = stokes_ref["max_err_pct_U0_resolved"]
    assert ref < stokes_eqwall["max_err_pct_U0_resolved"]
    assert stokes_eqwall["max_err_pct_U0_resolved"] < stokes_static["max_err_pct_U0_resolved"]
    # a term added *before* collision is destroyed by the bounce-back, leaving
    # a wall that does not move at all
    assert stokes_precollision["max_err_pct_U0_resolved"] > 50.0
    assert stokes_static["max_err_pct_U0_resolved"] > 50.0


def test_analytic_helpers_are_standard_closed_forms():
    y = np.array([0.5, 1.5, 8.5])
    u0, nu, omega = 0.02, 0.1, 3.125e-3
    k = math.sqrt(omega / (2 * nu))
    got = mwbb.stokes2_analytic(y, 0.0, u0, omega, nu)
    assert np.allclose(got, u0 * np.exp(-k * (y - 0.5)) * np.cos(-k * (y - 0.5)))
    _, ua = mwbb.couette_analytic(ny=18, u0=0.05)
    # walls at y=0.5 and y=ny-1.5 => first fluid row (y=1) carries U0*0.5/H_eff,
    # top fluid row (y=ny-2) carries U0*(H_eff-0.5)/H_eff, never exactly 0 or U0
    assert math.isclose(ua[1], 0.05 * 0.5 / 16.0, rel_tol=1e-12)
    assert math.isclose(ua[-2], 0.05 * 15.5 / 16.0, rel_tol=1e-12)
    assert ua[0] == 0.0 and math.isclose(ua[-1], 0.05, rel_tol=1e-15)