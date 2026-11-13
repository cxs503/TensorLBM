"""Tests for the XVAL analytic-solution / relative-L2 harness.

Anchor XVAL-LBM-ANALYTIC-SOLUTION-RELATIVE-L2-HARNESS.

Three groups:
1. *convention* -- the norm/energy/coordinate helpers do exactly what the OSS
   reference sources (``lbm-principles-practice/code`` chapter 13 & 5) do;
2. *equivalence* -- the NumPy port of ``vortexdecay.m`` and this repository's
   torch D2Q9 path produce the same diagnostics at identical settings;
3. *gate* -- the harness rejects wrong physics (negative controls) and the
   fluid-node mask behaves as documented.

Runtime is kept small (NX = 32, a fraction of one decay time) so the suite can
run in CI; the full ladder lives in ``benchmarks/verified/tgv2d_analytic_l2_xval``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "benchmarks" / "verified" / "tgv2d_analytic_l2_xval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run as xval  # noqa: E402

from tensorlbm.verification.analytic_l2 import (  # noqa: E402
    cell_centred_coordinates,
    fluid_mask,
    kinetic_energy,
    relative_l2,
    steady_criterion,
    tgv2d_analytic,
    tgv2d_decay_time,
)

NX = 32
TAU = 1.0
U_MAX = 0.04
NU = (2.0 * TAU - 1.0) / 6.0
TD = tgv2d_decay_time(NX, NX, NU)


# --------------------------------------------------------------------------- #
# 1. convention
# --------------------------------------------------------------------------- #
def test_coordinates_are_cell_centred_like_the_oss_driver():
    xx, yy = cell_centred_coordinates(4, 3)
    assert xx[0, 0] == pytest.approx(0.5)
    assert xx[0, 1] == pytest.approx(1.5)
    assert yy[1, 0] == pytest.approx(1.5)
    assert xx.shape == yy.shape == (3, 4)


def test_decay_time_matches_oss_expression():
    k = 2.0 * math.pi / NX
    assert tgv2d_decay_time(NX, NX, NU) == pytest.approx(1.0 / (NU * 2.0 * k * k))


def test_analytic_field_is_exactly_reproduced_and_self_consistent():
    a = tgv2d_analytic(0.0, NX, NX, NU, u_max=U_MAX)
    # zero error against itself, and the density is the lattice EOS rho0 + 3P
    assert relative_l2(a.ux, a.ux).rel_l2 == 0.0
    assert np.allclose(a.rho, 1.0 + 3.0 * a.p)
    # u_max is attained only in the cell-centred sampling; peak <= u_max
    assert np.abs(a.ux).max() <= U_MAX + 1e-12
    # symmetry of the square-box Taylor-Green state: u_y(x,y) = u_x(y,x)
    assert np.allclose(a.uy, -a.ux.T)


def test_analytic_decay_follows_exp_minus_2t_over_td():
    a0 = tgv2d_analytic(0.0, NX, NX, NU, u_max=U_MAX)
    a1 = tgv2d_analytic(0.5 * TD, NX, NX, NU, u_max=U_MAX)
    assert np.abs(a1.ux).max() / np.abs(a0.ux).max() == pytest.approx(math.exp(-0.5), rel=1e-12)


def test_relative_l2_uses_the_reference_fluctuation_as_denominator():
    ref = np.array([[1.0, 2.0], [3.0, 4.0]])
    num = ref + np.array([[0.1, 0.0], [0.0, 0.0]])
    err = relative_l2(num, ref)
    assert err.ssq == pytest.approx(0.01)
    assert err.ref_ssq == pytest.approx(np.sum(ref**2))
    assert err.rel_l2 == pytest.approx(math.sqrt(0.01 / np.sum(ref**2)))
    # density variant: fluctuation about rho0 = 1
    err2 = relative_l2(num, ref, ref_base=1.0)
    assert err2.ref_ssq == pytest.approx(np.sum((ref - 1.0) ** 2))
    assert err2.rel_l2 > err.rel_l2


def test_relative_l2_rejects_undefined_normalisation_and_bad_input():
    with pytest.raises(ValueError):
        relative_l2(np.ones((2, 2)), np.ones((2, 2)), ref_base=1.0)  # zero fluctuation
    with pytest.raises(ValueError):
        relative_l2(np.ones((2, 2)), np.ones((3, 3)))
    with pytest.raises(ValueError):
        relative_l2(np.array([[np.nan]]), np.array([[1.0]]))
    with pytest.raises(ValueError):
        relative_l2(np.ones((2, 2)), np.ones((2, 2)), mask=np.zeros((2, 2), bool))


def test_energy_and_steady_criterion_match_oss_definitions():
    rho = np.full((2, 2), 2.0)
    ux = np.full((2, 2), 3.0)
    uy = np.full((2, 2), 4.0)
    # E = sum rho (u^2 + v^2): no 1/2, no division by N
    assert kinetic_energy(rho, ux, uy) == pytest.approx(4 * 2.0 * 25.0)
    assert steady_criterion([1.0, 1.0 + 1e-13])["converged"] is True
    assert steady_criterion([1.0, 1.0 + 1e-9])["converged"] is False
    with pytest.raises(ValueError):
        steady_criterion([0.0, 1.0])


def test_fluid_mask_excludes_solid_and_boundary_rows():
    solid = np.zeros((5, 4), bool)
    solid[0, :] = True
    m = fluid_mask(solid, extra_excluded_rows=1)
    assert m.shape == (5, 4)
    assert m[0].sum() == 0  # solid row (also the bottom edge)
    assert m[1].sum() == 4  # interior row, countable
    assert m[-1].sum() == 0  # top edge row, excluded by position
    assert m.sum() == 3 * 4
    # the mask is positional, not physical: excluding rows changes the norm value,
    # so a case must declare its mask (see docs, "fluid-node convention")
    with pytest.raises(ValueError):
        fluid_mask(np.zeros((2, 2), bool), extra_excluded_rows=1)


# --------------------------------------------------------------------------- #
# 2. equivalence of the two arms
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def arms():
    kw = dict(nx=NX, tau=TAU, u_max=U_MAX, nsteps=200, nmsg=100)
    return xval.run_oss_port(**kw), xval.run_tensorlbm(**kw)


def test_both_arms_sample_at_the_same_times_and_decay(arms):
    a, b = arms
    assert [s["t"] for s in a.samples] == [100, 200]
    assert [s["t"] for s in b.samples] == [100, 200]
    # energy decays as exp(-2t/td) to within the discretisation error
    for s in b.samples:
        e_scale = 0.5 * U_MAX**2 * NX * NX  # <rho u^2> ; rho ~ 1
        assert s["E"] / (e_scale * math.exp(-2.0 * s["t"] / TD)) == pytest.approx(1.0, abs=0.05)


def test_cross_arm_diagnostics_agree_to_round_off(arms):
    a, b = arms
    rows = xval.compare_arms(a, b)
    worst = max(row[f"d_{k}_rel"] for row in rows for k in ("E", "L2rho", "L2ux", "L2uy"))
    assert worst < 1e-8, f"arms disagree by {worst:.3e}"


def test_velocity_error_vs_analytic_is_below_owner_threshold(arms):
    _, b = arms
    worst_vel = max(max(s["L2ux"], s["L2uy"]) for s in b.samples)
    worst_all = max(max(s["L2rho"], s["L2ux"], s["L2uy"]) for s in b.samples)
    assert worst_vel < 0.03, f"velocity L2 error {worst_vel:.4%} exceeds the 3% target"
    assert worst_all < 0.15, f"L2 error {worst_all:.4%} exceeds the 15% owner standard"


def test_density_plain_normalisation_is_tiny(arms):
    _, b = arms
    assert max(s["L2rho_plain"] for s in b.samples) < 1e-4


# --------------------------------------------------------------------------- #
# 3. the gate can fail
# --------------------------------------------------------------------------- #
def test_wrong_viscosity_is_rejected_by_a_large_margin():
    res = xval.negative_controls(NX, tau_field=TAU, tau_ref=1.2, u_max=U_MAX, tstar=0.5)
    honest = max(res["honest"]["L2ux"], res["honest"]["L2uy"])
    wrong = max(res["wrong_viscosity"]["L2ux"], res["wrong_viscosity"]["L2uy"])
    assert honest < 0.03
    assert wrong > 0.15, f"wrong-viscosity field graded at only {wrong:.3%}"
    assert wrong / honest > 20.0


def test_one_step_time_shift_is_detected():
    res = xval.negative_controls(NX, tau_field=TAU, tau_ref=1.2, u_max=U_MAX, tstar=0.5)
    honest = res["honest"]["L2ux"]
    shifted = res["one_step_shift"]["L2ux"]
    assert shifted > honest, "grading against the wrong time step should inflate the error"


def test_wall_rows_outside_the_mask_pollute_the_norm():
    demo = xval.fluid_node_convention_demo(NX, tau=TAU, u_max=U_MAX, tstar=0.2)
    assert demo["n_nodes_masked"] == demo["n_nodes_full"] - 2 * NX
    # dropping 3% of the nodes shifts the norm by a few percent: the mask is part
    # of the case definition, it is not a cosmetic detail
    assert demo["L2ux_wall_rows_excluded"] == pytest.approx(demo["L2ux_interior"], rel=0.05)
    # the OSS benchmarks sum only over fluid nodes; keeping "wall" rows whose
    # values are unrelated to the interior solution inflates the norm by an order
    # of magnitude (measured 9.2x at NX=32, 33x at NX=64)
    assert demo["L2ux_with_wall_rows"] > 5.0 * demo["L2ux_interior"]


def test_acoustic_transient_period_matches_the_prediction():
    scan = xval.harmonic_scan(NX, tau=TAU, u_max=U_MAX, tstar_max=1.2, every=1)
    predicted = scan["predicted_acoustic_period"]
    measured = scan["measured_deviation_period"]
    assert measured == pytest.approx(predicted, rel=0.12), (measured, predicted)
    # and the amplitude genuinely oscillates around the analytic decay
    ratios = np.array([r["ratio"] for r in scan["samples"]])
    assert ratios.max() > 1.02 and ratios.min() < 0.98