"""Tests for :mod:`tensorlbm.boundary.slip_wall_calibration` (P106W2-N02).

The four claims the module makes, each pinned by an independent check:

1. **operator algebra** -- the mixed rule's tangential momentum bookkeeping
   recovers an imposed moving-wall velocity exactly (``wall_velocity_from_rule``),
   and ``b = 0`` for ``sigma = 0`` with no offset;
2. **Maxwell condition** -- the wall speed implied by the rule times the
   wall-normal gradient measured by one-sided interior extrapolation equals the
   slip length from the profile inversion, to the level of the discrete
   amplitude defect only;
3. **meshing** -- a mixed wall is a *lattice-scaled* slip model: with ``sigma``
   fixed, ``b_phys = b_lu dx`` shrinks with the mesh, and the retuning recipe
   ``sigma(b_phys, dx)`` holds ``b_phys`` fixed;
4. **drag law** -- ``dCf(b+)`` is monotone on ``b+ in {0,5,10,20}`` on both
   available closure routes, and the literature anchor is explicitly WITHHELD.

All solvers run small enough for CI (``ny <= 25``).
"""

import math

import numpy as np
import pytest

from tensorlbm.boundary import partial_slip_wall as ps
from tensorlbm.boundary import partial_slip_wall3d as p3
from tensorlbm.boundary import slip_wall_calibration as cal

TAU = 0.8
NU = ps.nu_from_tau(TAU)
G = 1.0e-5
NY = 21
STEPS = 30000


# ---------------------------------------------------------------------------
# 1. operator algebra: moving-wall momentum bookkeeping
# ---------------------------------------------------------------------------
def test_rule_recovers_imposed_moving_wall_velocity():
    """`wall_velocity_from_rule` must return the imposed Couette wall speeds."""
    uw = 0.02
    res = cal.plane_channel_3d(0.0, ny=NY, g=0.0, uw_low=-uw, uw_high=uw, steps=STEPS)
    assert res.converged
    lo = cal.wall_velocity_from_rule(res.f_low, 0.0, float(res.rho_profile[0]),
                                     top=False, uw=-uw)
    hi = cal.wall_velocity_from_rule(res.f_high, 0.0, float(res.rho_profile[-1]),
                                     top=True, uw=uw)
    assert np.allclose(lo, -uw, rtol=0, atol=1e-12)
    assert np.allclose(hi, uw, rtol=0, atol=1e-12)


def test_moving_wall_couette_reproduces_analytic_linear_profile():
    uw = 0.02
    res = cal.plane_channel_3d(0.0, ny=NY, g=0.0, uw_low=-uw, uw_high=uw, steps=STEPS)
    ana = uw * (2.0 * (res.coord - 0.5) / (NY - 2) - 1.0)
    assert np.max(np.abs(res.u_profile - ana)) < 1e-9


def test_zero_sigma_has_zero_wall_velocity():
    res = cal.plane_channel_3d(0.0, ny=NY, steps=STEPS)
    lo = cal.wall_velocity_from_rule(res.f_low, 0.0, float(res.rho_profile[0]))
    hi = cal.wall_velocity_from_rule(res.f_high, 0.0, float(res.rho_profile[-1]), top=True)
    assert abs(float(np.mean(lo))) < 1e-15
    assert abs(float(np.mean(hi))) < 1e-15


def test_plane_channel_3d_matches_1d_column():
    """The D3Q19 slab must reduce to the 1D column (same discrete solution)."""
    for sigma in (0.0, 0.5):
        r3 = cal.plane_channel_3d(sigma, ny=NY, steps=STEPS)
        r1 = ps.channel_poiseuille(sigma, ny=NY, steps=STEPS)
        assert r3.converged and r1.converged
        assert np.max(np.abs(r3.u_profile - r1.u)) < 1e-12


# ---------------------------------------------------------------------------
# 2. one-sided wall fit and the Maxwell audit
# ---------------------------------------------------------------------------
def test_one_sided_wall_fit_is_exact_on_a_quadratic():
    b = 0.7
    H = 10.0
    y = np.array([1.0, 2.0, 3.0])
    u = G / (2 * NU) * (H ** 2 - (y - 0.5 - H) ** 2) + G * H * b / NU
    fit = cal.one_sided_wall_fit(u, y, 0.5)
    assert fit["gradient"] == pytest.approx(G * H / NU, rel=1e-12)
    assert fit["residual"] < 1e-18


def test_maxwell_audit_slope_and_gradient():
    """b from the wall rule is exactly ``0.3 sigma/(1-sigma)``; the measured
    wall-normal gradient is the analytic one; the profile-fit route differs only
    by the discrete amplitude defect."""
    out = cal.maxwell_audit(0.5, ny=NY, steps=STEPS)
    assert out["converged"]
    expected_b = cal.SLOPE_DEFAULT * 0.5 / (1.0 - 0.5)
    assert out["b_from_rule_and_gradient"] == pytest.approx(expected_b, rel=1e-6)
    assert out["gradient_rel_err"] < 1e-6
    # the two independent routes agree up to the O(1/H) amplitude defect
    assert out["b_rule_vs_fit_rel_err"] < 0.03


def test_maxwell_defect_is_the_amplitude_defect():
    """``b_fit - b_rule`` must be the same constant at every sigma (it is the
    ``c = -0.065/H`` intercept of the profile-fit law), i.e. it is a property of
    the discrete profile, not of the slip mechanism."""
    defects = []
    for sigma in (0.25, 0.5):
        out = cal.maxwell_audit(sigma, ny=NY, steps=STEPS)
        defects.append(out["b_from_quadratic_inversion"] - out["b_from_rule_and_gradient"])
    assert defects[0] == pytest.approx(defects[1], abs=2e-5)


def test_sigma_from_slip_length_roundtrip():
    for b in (0.1, 0.5, 2.0):
        s = cal.sigma_for_slip_length(b)
        assert cal.SLOPE_DEFAULT * s / (1.0 - s) == pytest.approx(b, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. mesh dependence and the retuning recipe
# ---------------------------------------------------------------------------
def test_fixed_sigma_physical_slip_length_shrinks_with_mesh():
    out = cal.resolution_slip_study(beta=0.10, resolutions=(17, 25), steps=20000,
                                    retune=False)
    rows = out["rows"]
    assert rows[0]["b_phys_fixed_sigma"] > rows[1]["b_phys_fixed_sigma"]
    # b_lu is the *nearly* lattice-invariant quantity: it varies only through the
    # discrete amplitude defect c = -0.065/H (~0.3% between ny=17 and 25)
    assert rows[0]["b_lu_fixed_sigma"] == pytest.approx(rows[1]["b_lu_fixed_sigma"],
                                                        rel=1e-2)
    # ... while the physical slip length changes by ~50% between those two meshes
    assert rows[0]["b_phys_fixed_sigma"] / rows[1]["b_phys_fixed_sigma"] > 1.4


def test_retuning_holds_physical_slip_length():
    out = cal.resolution_slip_study(beta=0.10, resolutions=(17, 25), steps=20000,
                                    retune=True)
    assert out["arm_retuned_rel_err_max"] < 1e-3
    b = [r["b_phys_retuned"] for r in out["rows"]]
    assert abs(b[0] / b[1] - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# 4. free-slip limit, pipe convergence, drag law
# ---------------------------------------------------------------------------
def test_free_slip_asymptotics_is_finite_and_linear_in_one_minus_sigma():
    """sigma -> 1 keeps b finite and on the exact law; nothing blows up."""
    out = cal.free_slip_asymptotics(sigmas=(0.9, 0.95), ny=NY, steps=STEPS)
    assert all(r["converged"] for r in out["rows"])
    for r in out["rows"]:
        expected = cal.SLOPE_DEFAULT * r["sigma"] / (1.0 - r["sigma"])
        assert r["b_rule"] == pytest.approx(expected, rel=1e-5)
        assert r["b"] > 0.0
    assert out["rows"][1]["b"] > 1.5 * out["rows"][0]["b"]
    # exact abscissa: 1/b = (1-sigma)/(a sigma)
    assert out["slope_rel_err_rule_route"] < 1e-4


def test_pipe_inversion_convergence_reduces_bias():
    sigma = cal.sigma_for_slip_length(0.5)      # operator slip length = 0.5
    out = cal.pipe_inversion_convergence(sigma, radii=(17, 25), steps=40000)
    r = out["rows"]
    # the quadratic-fit bias must shrink with radial resolution
    err = [abs(x["b_quadratic"] / 0.5 - 1.0) for x in r]
    assert err[1] < err[0]
    # the model estimator is the accurate one (criterion: 5%)
    assert all(abs(x["b_model"] / 0.5 - 1.0) < 0.03 for x in r)
    assert abs(r[1]["b_model"] / 0.5 - 1.0) < 0.02


def test_drag_law_is_monotone_and_anchor_is_withheld():
    out = cal.drag_law_bplus(b_plus_values=(0.0, 5.0, 10.0, 20.0))
    assert out["rows_monotone"]
    assert out["all_re_theta_monotone"]
    assert out["rows"][0]["dCf_virtual_wall"] == pytest.approx(0.0, abs=1e-12)
    seq = [r["dCf_scaling_k3"] for r in out["rows"]]
    assert all(b > a for a, b in zip(seq, seq[1:]))
    assert "WITHHELD" in out["literature_anchor"]