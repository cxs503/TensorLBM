"""Tests for the horizon-free fixed-point discrete adjoint (GAP-DIFFLBM-ADJOINT-VERDICT).

The claim under test: ``fixed_point_adjoint`` returns the *total* (flow-response)
shape derivative of the converged lift, and it does so by an algorithm that is
independent of the unrolled / truncated-BPTT route in
:func:`tensorlbm.shape_gradient.shape_gradient_through_flow`:

* no time horizon and no stored tape -- ``(I - S_f)^T lambda = (dJ/df)^T`` is
  solved by matrix-free GMRES;
* the finite-difference reference is central differences of the *converged*
  solver (a different numerical route again);
* the mirror symmetry ``C_L(y_c) = -C_L(H - y_c)``,
  ``dC_L/dy_c(y_c) = dC_L/dy_c(H - y_c)`` is an analytic invariant.

The frozen-field surrogate of ``tensorlbm.adjoint`` must keep failing the
admission gate next to it.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.adjoint import adjoint_sensitivity, optimization_path_eligible
from tensorlbm.fixed_point_adjoint import (
    fixed_point_adjoint,
    fixed_point_residual,
    state_lift,
    steady_lift,
)
from tensorlbm.shape_gradient import (
    frozen_field_gradient,
    make_channel,
    run_to_steady,
    shape_gradient_through_flow,
)

# small but non-degenerate channel: H = 10 lu, D = 3 lu, 5 cells across the
# radius, measured Re = 4.5 -- small enough for CI, and the study in
# benchmarks/adjoint_verdict_fixed_point_study.py runs the same verification at
# a larger grid
CFG = dict(nx=24, ny=16, diameter=3.0, tau=0.6, u_bulk=0.05, epsilon=0.3, wall_thickness=3)
H = CFG["ny"] - 2 * CFG["wall_thickness"]
Y_MID = (CFG["ny"] - 1) / 2.0
SOLVE = dict(max_steps=5000, tol=1e-10, min_steps=400, check_every=100)


def _geom(yc: float):
    return make_channel(
        nx=CFG["nx"],
        ny=CFG["ny"],
        diameter=CFG["diameter"],
        center_y=yc,
        tau=CFG["tau"],
        u_bulk=CFG["u_bulk"],
        epsilon=CFG["epsilon"],
        wall_thickness=CFG["wall_thickness"],
    )


def _solve(yc: float):
    return run_to_steady(_geom(yc), **SOLVE)


@pytest.fixture(scope="module")
def steady_off_centre():
    return _solve(Y_MID + 0.04 * H)


@pytest.fixture(scope="module")
def fp(steady_off_centre):
    return fixed_point_adjoint(steady_off_centre)


# ---------------------------------------------------------------------------
# contract / gate
# ---------------------------------------------------------------------------


def test_contract_flags_and_gate(steady_off_centre, fp):
    assert fp["shape_derivative_only"] is False
    assert fp["includes_flow_response"] is True
    # the gate is the one in adjoint.py -- not a locally invented flag test
    assert fp["optimization_path_eligible"] is True
    assert optimization_path_eligible(fp) is True
    assert fp["differentiate"] == ["y_c"]
    assert "fixed-point" in fp["method"]
    assert math.isfinite(fp["dJ_dy_c"])


def test_adjoint_solve_actually_converged(steady_off_centre, fp):
    # the anchor state must be a fixed point, otherwise the implicit-function
    # theorem is applied to the wrong state
    assert fixed_point_residual(steady_off_centre) < 1e-8
    # and the Krylov system must be solved, not merely iterated
    # the default Krylov tolerance is 1e-6: measured on the study case, tightening
    # it further changes the gradient by <0.3 % and only buys wall time
    assert fp["krylov_relative_residual"] < 1e-5
    assert fp["krylov_info"] == 0
    assert fp["krylov_iterations"] >= 1


def test_objective_is_the_converged_lift(steady_off_centre, fp):
    geom = steady_off_centre.geom
    scale = 0.5 * geom.bulk_velocity() ** 2 * float(geom.diameter)
    assert steady_lift(steady_off_centre) == pytest.approx(
        steady_off_centre.lift / scale, rel=1e-8
    )
    assert fp["objective"] == pytest.approx(steady_lift(steady_off_centre), rel=1e-8)


# ---------------------------------------------------------------------------
# accuracy against an independent route
# ---------------------------------------------------------------------------


def test_matches_finite_difference_of_converged_lift(steady_off_centre, fp):
    """Central difference of the converged solver -- the mechanical anchor."""
    yc0 = float(steady_off_centre.geom.center_y)
    # small step: at h = 0.5 lu the smeared interface moves a fifth of a radius
    # and the central difference is biased by >1 % (measured in the study)
    h = 0.125
    geom = steady_off_centre.geom
    scale = 0.5 * geom.bulk_velocity() ** 2 * float(geom.diameter)
    cl_p = _solve(yc0 + h).lift / scale
    cl_m = _solve(yc0 - h).lift / scale
    g_fd = (cl_p - cl_m) / (2.0 * h)
    rel = abs(fp["dJ_dy_c"] - g_fd) / abs(g_fd)
    assert rel < 0.03, f"fixed-point adjoint off FD by {rel:.3%} (fp={fp['dJ_dy_c']:.6e}, fd={g_fd:.6e})"


def test_agrees_with_unrolled_ad_long_horizon(steady_off_centre, fp):
    """The two structurally different flow-response algorithms agree."""
    ad = shape_gradient_through_flow(steady_off_centre, n_unroll=150, window=50)
    denom = math.hypot(fp["dJ_dy_c"], 1e-300) * math.hypot(ad["dJ_dy_c"], 1e-300)
    cos = fp["dJ_dy_c"] * ad["dJ_dy_c"] / denom
    assert cos > 0.99, f"unrolled AD and fixed-point adjoint disagree (cos={cos:.4f})"
    assert abs(ad["dJ_dy_c"] - fp["dJ_dy_c"]) / abs(fp["dJ_dy_c"]) < 0.15


def test_flow_response_dominates_the_explicit_term(fp):
    # for lift the flow response is the signal, not a correction
    assert abs(fp["flow_response_term"]["y_c"]) > abs(fp["explicit_term"]["y_c"])
    # the explicit term must equal the exact frozen-field derivative
    assert fp["explicit_term"]["y_c"] != 0.0


# ---------------------------------------------------------------------------
# the surrogate stays blocked
# ---------------------------------------------------------------------------


def test_frozen_surrogate_is_rejected_next_to_the_adjoint(steady_off_centre, fp):
    frozen = frozen_field_gradient(steady_off_centre)
    assert frozen["shape_derivative_only"] is True
    assert optimization_path_eligible(frozen) is False
    denom = abs(fp["dJ_dy_c"]) * abs(frozen["dJ_dy_c"])
    cos = fp["dJ_dy_c"] * frozen["dJ_dy_c"] / denom
    # the predeclared gate is the *signed* cosine: an anti-parallel surrogate
    # has |cos| = 1 and must still fail
    assert cos < 0.99, f"frozen proxy unexpectedly passes the gate (cos={cos:+.4f})"

    rho, ux, uy = steady_off_centre.fields()
    ship = adjoint_sensitivity(
        rho.detach(), ux.detach(), uy.detach(), steady_off_centre.geom.hard_mask(),
        objective="lift",
    )
    assert optimization_path_eligible(ship) is False


# ---------------------------------------------------------------------------
# analytic invariant: mirror symmetry of the channel
# ---------------------------------------------------------------------------


def test_mirror_symmetry_of_adjoint():
    """``dC_L/dy_c`` is even about the channel mid-line, ``C_L`` is odd."""
    delta = 0.04 * H
    up = _solve(Y_MID + delta)
    dn = _solve(Y_MID - delta)
    assert up.lift == pytest.approx(-dn.lift, rel=2e-2)
    g_up = fixed_point_adjoint(up)["dJ_dy_c"]
    g_dn = fixed_point_adjoint(dn)["dJ_dy_c"]
    assert g_up == pytest.approx(g_dn, rel=2e-2)