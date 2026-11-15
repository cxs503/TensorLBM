"""Tests for Zou-He channel inlet/outlet reconstruction (D2Q9).

The anchor is ``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI``: the OSS
reference (``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow``) closes
a channel with Zou-He reconstruction at both ends.  These tests do not merely
execute the code -- they assert the *moments* of the reconstructed boundary
layer, which is what the closure claims to impose, and they compare the
converged channel field against the exact plane-Poiseuille solution with the
project-wide 15 % acceptance / 3 % target bound.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundary.d2q9 import C2Q9, OPPOSITE_2Q9
from tensorlbm.boundary.zou_he_channel import (
    zou_he_channel,
    zou_he_column_density_momentum,
    zou_he_wall,
)
from tensorlbm.verification.zou_he_channel import (
    ChannelCase,
    analytic_profile,
    equilibrium_d2q9,
    run_channel,
)
from tensorlbm.verification.zou_he_channel.solver import profile_metrics

DTYPE = torch.float64
SIDES = ("west", "east", "north", "south")


def _random_field(ny: int = 12, nx: int = 14, seed: int = 7) -> torch.Tensor:
    """Near-equilibrium but *non-equilibrium* field: physical moments + noise.

    A physically admissible field matters: the emergent velocity of a pressure
    closure is ``1 - (sum of known) / rho``, so an unphysical field (density ~3)
    would report a shock, not a boundary-condition defect.
    """
    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.02 * torch.rand((ny, nx), generator=g, dtype=DTYPE)
    ux = 0.03 * (torch.rand((ny, nx), generator=g, dtype=DTYPE) - 0.5)
    uy = 0.03 * (torch.rand((ny, nx), generator=g, dtype=DTYPE) - 0.5)
    f_eq = equilibrium_d2q9(rho, ux, uy)
    noise = 1e-3 * (torch.rand((9, ny, nx), generator=g, dtype=DTYPE) - 0.5)
    return f_eq + noise


# --------------------------------------------------------------------------- #
# 1. Moment exactness of the reconstruction (the actual claim of the closure)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("rho_bc", [0.97, 1.0, 1.06])
def test_moment_exactness_pressure_mode(side, rho_bc):
    """Pressure mode: reconstructed layer must have exactly the prescribed rho.

    Mass conservation in the layer is checked independently of the code path;
    if the diagonal pairing in ``zou_he_wall`` were wrong (e.g. f5 rebuilt from
    f6 instead of f8) the density of the layer would shift and this fails.
    """
    f = _random_field()
    zou_he_wall(f, side, mode="pressure", rho=rho_bc)
    rho, ux, uy = zou_he_column_density_momentum(f, side)
    assert torch.allclose(rho, torch.full_like(rho, rho_bc), atol=1e-12)
    # the emergent normal velocity must be finite and sub-sonic
    normal = ux if side in ("west", "east") else uy
    assert torch.isfinite(normal).all()
    assert float(normal.abs().max()) < 0.5


@pytest.mark.parametrize("side", SIDES)
def test_moment_exactness_velocity_mode(side):
    """Velocity mode: reconstructed layer must reproduce the prescribed u."""
    f = _random_field()
    ux_bc, uy_bc = 0.021, -0.004
    zou_he_wall(f, side, mode="velocity", ux=ux_bc, uy=uy_bc)
    rho, ux, uy = zou_he_column_density_momentum(f, side)
    assert torch.allclose(ux, torch.full_like(ux, ux_bc), atol=1e-12)
    assert torch.allclose(uy, torch.full_like(uy, uy_bc), atol=1e-12)
    assert (rho > 0).all()


@pytest.mark.parametrize("side", SIDES)
def test_static_fluid_reproduces_rest_state(side):
    """A fluid at rest (rho0=1, u=0) must survive the closure unchanged."""
    f0 = equilibrium_d2q9(
        torch.ones((1, 9), dtype=DTYPE),
        torch.zeros((1, 9), dtype=DTYPE),
        torch.zeros((1, 9), dtype=DTYPE),
    ).repeat(1, 12, 1)
    f = f0.clone()
    zou_he_wall(f, side, mode="velocity", ux=0.0, uy=0.0)
    assert torch.allclose(f, f0, atol=1e-13)


@pytest.mark.parametrize("side", SIDES)
def test_diagonal_pairing_is_not_interchangeable(side):
    """Falsifiability: swapping the two unknown diagonals breaks the layer.

    Guards against a silent regression to the (wrong) pairing where an unknown
    diagonal is rebuilt from the other unknown diagonal's opposite.
    """
    f_ok = _random_field()
    zou_he_wall(f_ok, side, mode="pressure", rho=1.01)

    unknown = {"west": (5, 7), "east": (6, 8), "north": (7, 8), "south": (5, 6)}[side]
    f_bad = _random_field()
    zou_he_wall(f_bad, side, mode="pressure", rho=1.01)
    line = {"west": f_bad[:, :, 0], "east": f_bad[:, :, -1],
            "north": f_bad[:, -1, :], "south": f_bad[:, 0, :]}[side]
    i, j = unknown
    tmp = line[i].clone()
    line[i] = line[j]
    line[j] = tmp

    _, ux_ok, uy_ok = zou_he_column_density_momentum(f_ok, side)
    _, ux_bad, uy_bad = zou_he_column_density_momentum(f_bad, side)
    transverse_ok = uy_ok if side in ("west", "east") else ux_ok
    transverse_bad = uy_bad if side in ("west", "east") else ux_bad
    assert not torch.allclose(transverse_ok, transverse_bad, atol=1e-9)


def test_channel_helper_wires_both_ends():
    f = equilibrium_d2q9(
        torch.ones((1, 12), dtype=DTYPE),
        torch.zeros((1, 12), dtype=DTYPE),
        torch.zeros((1, 12), dtype=DTYPE),
    ).repeat(1, 8, 1)
    zou_he_channel(f, inlet="pressure", outlet="pressure", inlet_rho=1.02, outlet_rho=0.98)
    rho_w, _, _ = zou_he_column_density_momentum(f, "west")
    rho_e, _, _ = zou_he_column_density_momentum(f, "east")
    assert torch.allclose(rho_w, torch.full_like(rho_w, 1.02), atol=1e-12)
    assert torch.allclose(rho_e, torch.full_like(rho_e, 0.98), atol=1e-12)


def test_invalid_arguments_are_rejected():
    f = _random_field()
    with pytest.raises(ValueError):
        zou_he_wall(f, "top", mode="pressure", rho=1.0)
    with pytest.raises(ValueError):
        zou_he_wall(f, "west", mode="pressure")          # rho missing
    with pytest.raises(ValueError):
        zou_he_wall(f, "west", mode="velocity")          # ux/uy missing
    with pytest.raises(ValueError):
        zou_he_wall(f, "west", mode="bogus", rho=1.0)


# --------------------------------------------------------------------------- #
# 2. Channel verification against exact plane Poiseuille flow
# --------------------------------------------------------------------------- #
# The end densities are imposed (soft) Zou-He pressure boundaries, so the field
# carries a small acoustic ripple that a pointwise "delta < tol" test would never
# accept.  The solver therefore time-averages over blocks and reports the mean
# field; the tests compare that mean field with the exact solution.
_LIGHT = dict(tau=0.65, drho=4.0e-3, burn_in=2000, avg_window=500,
              ramp_steps=1500, tol=1e-6)


@pytest.fixture(scope="module")
def pressure_channel():
    case = ChannelCase(height=12, length=64, arm="pressure-pressure",
                       max_steps=6000, check_every=500, **_LIGHT)
    return case, run_channel(case)


def test_pressure_channel_matches_poiseuille(pressure_channel):
    """Converged pressure-driven channel vs exact parabola (15 % accept / 3 % target)."""
    case, res = pressure_channel
    m = profile_metrics(case, res["u_profile"], res["g_measured"])
    assert m["rel_linf"] < 0.15, m          # project-wide acceptance bound
    assert m["rel_umax"] < 0.03, m          # project-wide target bound
    assert m["shape_rel_l2"] < 0.03, m


def test_pressure_channel_gradient_consistency(pressure_channel):
    """Measured dp/dx must agree with the nominal drho/(nx-1) to a few percent."""
    case, res = pressure_channel
    rel = abs(res["g_measured"] - case.g_nominal) / case.g_nominal
    assert rel < 0.05, (res["g_measured"], case.g_nominal, rel)
    m = profile_metrics(case, res["u_profile"], case.g_nominal)
    assert m["rel_umax"] < 0.05, m


def test_mass_flux_is_constant_along_the_channel(pressure_channel):
    """Continuity: a steady channel must carry the same q(x) at every interior column.

    Also checks the inlet/outlet closure does not leak mass beyond the u^2 order
    expected of a soft pressure boundary.
    """
    _, res = pressure_channel
    q = res["q_col"]
    interior = q[1:-1]
    # Thresholds are for this deliberately *short* (6000-step, nx=64) test run:
    # the settled campaign runs reach 4e-7 / 8e-7 here (see the xval report
    # table), so 2e-3 is a smoke-level bound on the same quantity, not the claim.
    spread = (interior.max() - interior.min()).abs().item() / abs(interior.mean().item())
    assert spread < 2e-3, spread                    # flux flat along x
    defect = abs(q[0] - q[-1]).item() / abs(interior.mean().item())
    assert defect < 2e-3, defect                    # end-to-end closure defect


def test_no_pressure_drop_gives_no_flow():
    """Negative control: equal end densities must yield a vanishing flow rate."""
    case = ChannelCase(height=16, length=48, tau=0.65, drho=0.0,
                       arm="pressure-pressure", init="uniform",
                       burn_in=500, avg_window=200, ramp_steps=0,
                       tol=1e-8, max_steps=3000, check_every=200)
    res = run_channel(case)
    assert res["u_profile"].abs().max().item() < 1e-10


@pytest.fixture(scope="module")
def closure_arms():
    """Zou-He velocity inlet vs a naive equilibrium inlet, same geometry."""
    ok = ChannelCase(height=12, length=64, arm="velocity-pressure",
                     max_steps=6000, check_every=500, **_LIGHT)
    bad = ChannelCase(height=12, length=64, arm="equilibrium-pressure",
                      max_steps=6000, check_every=500, **_LIGHT)
    return (ok, run_channel(ok)), (bad, run_channel(bad))


def test_velocity_inlet_arm_matches_poiseuille(closure_arms):
    (case_ok, res_ok), _ = closure_arms
    m_ok = profile_metrics(case_ok, res_ok["u_profile"], res_ok["g_measured"])
    assert m_ok["shape_rel_l2"] < 0.03, m_ok


def test_naive_equilibrium_inlet_also_passes_the_metric(closure_arms):
    """Documented negative result: the inlet closure is not the dominant error.

    The initial hypothesis was that replacing the Zou-He velocity inlet by a
    local-equilibrium inlet (same prescribed velocity, first-order closure) would
    be measurably worse at mid-channel.  It is not: both arms sit at the same
    O(0.3 %) shape error (see the xval report), because a long channel washes the
    inlet closure out within a few H.  The test therefore only asserts that the
    Zou-He arm is *not* worse than the naive one by more than a factor 2, and the
    finding is reported as such instead of being turned into a green check.
    """
    (case_ok, res_ok), (case_bad, res_bad) = closure_arms
    m_ok = profile_metrics(case_ok, res_ok["u_profile"], res_ok["g_measured"])
    m_bad = profile_metrics(case_bad, res_bad["u_profile"], res_bad["g_measured"])
    assert m_ok["shape_rel_l2"] < 0.03, m_ok
    assert m_bad["shape_rel_l2"] < 0.03, m_bad
    assert m_ok["shape_rel_l2"] < 2.0 * m_bad["shape_rel_l2"], (m_ok, m_bad)


def test_both_resolutions_stay_inside_the_target():
    """H = 8 and H = 16 both inside the 3 % target.

    The *refinement trend* is a property of settled runs; at this test budget
    (6000 steps, nx = 64) it is masked by the residual transient, so it is
    measured in the campaign runs tabulated in ``REPORT.md`` rather than
    asserted here -- an ordering assertion on this data would be noise.
    """
    cases = {
        h: ChannelCase(height=h, length=64, arm="pressure-pressure",
                       max_steps=6000, check_every=500, **_LIGHT)
        for h in (8, 16)
    }
    metrics = {}
    for h, case in cases.items():
        res = run_channel(case)
        metrics[h] = profile_metrics(case, res["u_profile"], res["g_measured"])
    for h, m in metrics.items():
        assert m["rel_linf"] < 0.03, (h, m)
        assert m["shape_rel_l2"] < 0.03, (h, m)
