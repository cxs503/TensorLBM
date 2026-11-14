"""Tests for the flow-response shape gradient (GAP-DIFFLBM-ADJOINT-VERDICT).

The decisive test is :func:`test_unrolled_ad_matches_finite_difference`: the
unrolled discrete adjoint (AD through the LBM step chain, geometry inside the
graph) must reproduce a central finite difference of the *same* finite-time
process to high accuracy.  That is a property of the implementation only -- it
needs no converged flow and runs in seconds.

Plus the contract tests for the frozen-field surrogate: ``adjoint.py`` must
declare ``shape_derivative_only=True`` and must be rejected by the
optimisation-path gate.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.adjoint import (
    SHAPE_DERIVATIVE_ONLY,
    adjoint_sensitivity,
    optimization_path_eligible,
)
from tensorlbm.shape_gradient import (
    ChannelDisk,
    disk_fluid_weight,
    frozen_field_gradient,
    initial_condition,
    lift_from_probe,
    make_channel,
    poiseuille_profile,
    run_to_steady,
    shape_gradient_through_flow,
    step,
    stream_periodic,
)

torch.set_num_threads(2)


# ---------------------------------------------------------------------------
# lattice / geometry unit tests
# ---------------------------------------------------------------------------


def test_stream_periodic_is_a_permutation_and_conserves_mass():
    torch.manual_seed(0)
    f = torch.rand(9, 16, 20, dtype=torch.float64)
    g = stream_periodic(f)
    assert torch.allclose(g.sum(), f.sum(), rtol=0, atol=1e-12)
    # a population with velocity c_q lands at x + c_q
    f = torch.zeros(9, 16, 20, dtype=torch.float64)
    f[1, 3, 4] = 1.0  # c_q = (+1, 0)
    g = stream_periodic(f)
    assert g[1, 3, 5].item() == 1.0
    f = torch.zeros(9, 16, 20, dtype=torch.float64)
    f[2, 3, 4] = 1.0  # c_q = (0, +1)
    g = stream_periodic(f)
    assert g[2, 4, 4].item() == 1.0


def test_disk_weight_saturates_to_hard_mask():
    geom = make_channel(nx=48, ny=40, diameter=10.0, center_y=20.0, epsilon=0.1)
    w = disk_fluid_weight(geom)
    hard = geom.hard_mask()
    # the fluid far from the surface is saturated to exactly 1 (the logistic
    # tail is cut at |phi/eps| > 30), the interior is frozen to < 1e-4
    assert float(w.max()) == 1.0
    assert (w > 1 - 1e-13).sum() > 0.8 * w.numel()
    assert float(w.min()) == 0.0
    # inside/outside classification agrees with the hard mask except in the band
    band = (w > 1e-12) & (w < 1 - 1e-12)
    assert ((w < 0.5) == hard)[~band].all()
    assert band.sum() > 0


def test_disk_weight_is_differentiable_in_centre():
    geom = make_channel(nx=32, ny=32, diameter=8.0, center_y=16.0, epsilon=0.5)
    yc = torch.tensor(16.0, dtype=torch.float64, requires_grad=True)
    w = disk_fluid_weight(geom, center_y=yc)
    w.sum().backward()
    assert yc.grad is not None and math.isfinite(float(yc.grad))
    assert abs(float(yc.grad)) > 0.0


def test_poiseuille_profile_matches_analytic_maximum():
    geom = make_channel(nx=32, ny=64, diameter=4.0, center_y=32.0, tau=0.7, u_bulk=0.03)
    u = poiseuille_profile(geom)
    # discrete maximum is the cell nearest the centre line
    assert abs(float(u.max()) - geom.body_force * geom.channel_height() ** 2
               / (8.0 * geom.kinematic_viscosity())) < 0.02 * float(u.max())


# ---------------------------------------------------------------------------
# the decisive test: unrolled AD == finite difference
# ---------------------------------------------------------------------------


def _finite_time_lift(geom: ChannelDisk, yc: float, n_steps: int) -> float:
    f = initial_condition(geom)
    w = disk_fluid_weight(geom, center_y=yc)
    sw = torch.where(geom.wall_mask(), torch.zeros_like(w), 1.0 - w)
    with torch.no_grad():
        for _ in range(n_steps):
            f, probe = step(f, geom, fluid_weight=w)
        return float(lift_from_probe(probe, sw)[1])


def test_unrolled_ad_matches_finite_difference():
    """d(lift)/d(y_c) over a finite rollout: AD vs central difference."""
    n_steps = 30
    geom = make_channel(nx=40, ny=32, diameter=8.0, center_y=16.0, tau=0.6,
                        u_bulk=0.05, epsilon=0.6)
    f0 = initial_condition(geom)

    # --- AD through the rollout -----------------------------------------
    yc = torch.tensor(16.0, dtype=torch.float64, requires_grad=True)
    f = f0.clone()
    w = disk_fluid_weight(geom, center_y=yc)
    sw = torch.where(geom.wall_mask(), torch.zeros_like(w), 1.0 - w)
    for _ in range(n_steps):
        f, probe = step(f, geom, fluid_weight=w)
    J = lift_from_probe(probe, sw)[1]
    J.backward()
    g_ad = float(yc.grad)

    # --- central difference of the same process -------------------------
    h = 0.05
    g_fd = (_finite_time_lift(geom, 16.0 + h, n_steps)
            - _finite_time_lift(geom, 16.0 - h, n_steps)) / (2 * h)

    assert abs(g_fd) > 1e-12
    # AD is exact for the discrete process; the residual is the O(h^2)
    # truncation of the central difference itself
    assert abs(g_ad - g_fd) / abs(g_fd) < 2e-4, (g_ad, g_fd)


def test_flow_response_is_not_zero():
    """The flow-response term must be present (the whole point of the module)."""
    geom = make_channel(nx=40, ny=32, diameter=8.0, center_y=16.0, tau=0.6,
                        u_bulk=0.05, epsilon=0.6)
    f0 = initial_condition(geom)
    n_steps = 30

    # total derivative (AD through the solver)
    yc = torch.tensor(16.0, dtype=torch.float64, requires_grad=True)
    f = f0.clone()
    for _ in range(n_steps):
        f, probe = step(f, geom, center_y=yc)
    w = disk_fluid_weight(geom, center_y=yc)
    sw = torch.where(geom.wall_mask(), torch.zeros_like(w), 1.0 - w)
    lift_from_probe(probe, sw)[1].backward()
    g_total = float(yc.grad)

    # frozen-field term (geometry only, field frozen at the last state)
    yc2 = torch.tensor(16.0, dtype=torch.float64, requires_grad=True)
    probe_det = probe.detach()
    w2 = disk_fluid_weight(geom, center_y=yc2)
    sw2 = torch.where(geom.wall_mask(), torch.zeros_like(w2), 1.0 - w2)
    lift_from_probe(probe_det, sw2)[1].backward()
    g_direct = float(yc2.grad)

    assert abs(g_total - g_direct) > 1e-9  # response term is real, not round-off


# ---------------------------------------------------------------------------
# frozen-field surrogate contract
# ---------------------------------------------------------------------------


def _toy_state(ny: int = 24, nx: int = 32):
    rho = torch.ones(ny, nx)
    ux = torch.full((ny, nx), 0.05)
    uy = torch.zeros(ny, nx)
    mask = torch.zeros(ny, nx, dtype=torch.bool)
    mask[9:15, 12:18] = True
    return rho, ux, uy, mask


def test_surrogate_declares_shape_derivative_only():
    rho, ux, uy, mask = _toy_state()
    out = adjoint_sensitivity(rho, ux, uy, mask, objective="lift")
    assert SHAPE_DERIVATIVE_ONLY is True
    assert out["shape_derivative_only"] is True
    assert out["includes_flow_response"] is False
    assert out["optimization_path_eligible"] is False
    assert out["frozen_field_verdict"]["anchor"] == "GAP-DIFFLBM-ADJOINT-VERDICT"


def test_gate_rejects_surrogate_and_accepts_flow_response():
    rho, ux, uy, mask = _toy_state()
    assert optimization_path_eligible(adjoint_sensitivity(rho, ux, uy, mask)) is False
    assert optimization_path_eligible(None) is False
    assert optimization_path_eligible({}) is False
    # only a gradient that *claims* both flags is admitted
    assert optimization_path_eligible(
        {"shape_derivative_only": False, "includes_flow_response": True}
    ) is True
    assert optimization_path_eligible(
        {"shape_derivative_only": False, "includes_flow_response": False}
    ) is False


def test_shape_gradient_through_flow_reports_its_contract():
    """Smoke test of the public entry point (tiny grid, tiny unroll)."""
    geom = make_channel(nx=32, ny=24, diameter=6.0, center_y=11.5, tau=0.65,
                        u_bulk=0.04, epsilon=0.5)
    f = initial_condition(geom)
    with torch.no_grad():
        for _ in range(200):
            f, _ = step(f, geom)
    steady = run_to_steady.__wrapped__ if hasattr(run_to_steady, "__wrapped__") else None
    del steady
    from tensorlbm.shape_gradient import SteadySolution

    sol = SteadySolution(f=f, geom=geom, lift=0.0, drag=0.0, steps=200, residual=1.0, history=[])
    out = shape_gradient_through_flow(sol, n_unroll=10, window=5)
    assert out["shape_derivative_only"] is False
    assert out["includes_flow_response"] is True
    assert optimization_path_eligible(out) is True
    assert math.isfinite(out["dJ_dy_c"])
    frozen = frozen_field_gradient(sol)
    assert frozen["shape_derivative_only"] is True
    assert optimization_path_eligible(frozen) is False


@pytest.mark.parametrize("kind", ["lift", "drag", "pressure_loss", "mixing_uniformity"])
def test_surrogate_objectives_still_run(kind):
    rho, ux, uy, mask = _toy_state()
    out = adjoint_sensitivity(rho, ux, uy, mask, objective=kind)
    assert out["n_boundary_nodes"] > 0
    assert len(out["sensitivity_x"]) == out["n_boundary_nodes"]
    assert all(math.isfinite(v) for v in out["sensitivity_x"] + out["sensitivity_y"])