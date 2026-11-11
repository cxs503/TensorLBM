"""Tests for the two-level (2:1) static refinement of the 2-D D2Q9 solver."""

from __future__ import annotations

import math

import pytest
import torch

# tiny grids: multi-threaded torch is dominated by thread-launch overhead here
torch.set_num_threads(2)

from tensorlbm.two_level_amr2d import (
    TwoLevelConfig,
    TwoLevelD2Q9,
    UniformD2Q9,
    collide_bgk,
    convective_refined_tau,
    cylinder_solid_mask,
    neq_scale_factor,
    obstacle_force,
    stream_mask,
)
from tensorlbm.two_level_gates import (
    _channel_solid,
    _poiseuille_profile,
    gate_conservation,
)


def _periodic_case(shape=(32, 32), steps=40, box=None, **kwargs):
    ny, nx = shape
    yy = (torch.arange(ny, dtype=torch.float64) + 0.5).view(-1, 1)
    xx = (torch.arange(nx, dtype=torch.float64) + 0.5).view(1, -1)
    kx = 2.0 * math.pi / nx
    ky = 2.0 * math.pi / ny
    rho = 1.0 + 1e-3 * torch.sin(kx * xx) * torch.cos(ky * yy)
    ux = 0.02 + 1e-3 * torch.cos(kx * xx) * torch.cos(ky * yy) + torch.zeros_like(rho)
    uy = (1e-3 * torch.sin(ky * yy)).expand(ny, nx).contiguous()
    box = box or (nx // 4, 3 * nx // 4, ny // 4, 3 * ny // 4)
    sim = TwoLevelD2Q9(TwoLevelConfig(shape=shape, box=box, **kwargs))
    sim.fill_uniform(rho, ux, uy)
    return sim, steps


def test_convective_refined_tau_matches_viscosity_scaling():
    tau_c = 0.6
    tau_f = convective_refined_tau(tau_c, 2)
    assert tau_f == pytest.approx(0.5 + 2.0 * (tau_c - 0.5))
    # lattice viscosity doubles on the finer level (same physical viscosity)
    assert (tau_f - 0.5) == pytest.approx(2.0 * (tau_c - 0.5))


def test_neq_scale_is_chapman_enskog_and_invertible():
    tau_c, tau_f, ratio = 0.6, 0.7, 2.0
    forward = neq_scale_factor(tau_c, tau_f, ratio, "chapman_enskog")
    # restriction is the exact inverse: the target spacing swaps with the source
    backward = neq_scale_factor(tau_f, tau_c, 1.0 / ratio, "chapman_enskog")
    assert forward * backward == pytest.approx(1.0)
    assert forward == pytest.approx(tau_f / (ratio * tau_c))
    # the deliberately wrong variants are distinct (negative-control basis)
    assert neq_scale_factor(tau_c, tau_f, ratio, "none") == 1.0
    assert neq_scale_factor(tau_c, tau_f, ratio, "tau_only") != pytest.approx(forward)


def test_prolongate_restrict_round_trip_recovers_the_coarse_state():
    """Spatially uniform state with a non-equilibrium part: exact round trip.

    This is the scaling test of the two interface operators: a factor error in
    ``tau_f / (r tau_c)`` shows up immediately because the non-equilibrium part
    is not recovered.
    """
    sim, _ = _periodic_case()
    ys, xs = sim.config.interior_coarse_slice
    f = sim.coarse.clone()
    f[1] = f[1] * 1.01  # inflate one population -> genuine non-zero neq
    f = f.mean(dim=(1, 2), keepdim=True).expand_as(f).clone()  # spatially uniform
    sim.coarse = f
    restricted = sim.restrict(sim.prolongate(f))
    assert torch.allclose(restricted, f[:, ys, xs], atol=1e-12)


def test_uniform_flow_is_preserved_on_both_levels():
    shape = (24, 32)
    box = (8, 24, 6, 18)
    sim, _ = _periodic_case(shape=shape, box=box)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.03, dtype=torch.float64)
    uy = torch.full(shape, -0.02, dtype=torch.float64)
    sim.fill_uniform(rho, ux, uy)
    m0 = sim.totals()
    for _ in range(10):
        sim.step()
    m1 = sim.totals()
    assert abs(m1[0] - m0[0]) < 1e-12
    assert abs(m1[1] - m0[1]) < 1e-12
    assert abs(m1[2] - m0[2]) < 1e-12
    rho_c, ux_c, uy_c = sim.composite_fields()
    assert float((rho_c - 1.0).abs().max()) < 1e-12
    assert float((ux_c - 0.03).abs().max()) < 1e-9
    assert float((uy_c + 0.02).abs().max()) < 1e-9


def test_mass_and_momentum_conservation():
    result = gate_conservation(shape=(32, 32), steps=40)
    assert result["mass_residual_relative"] <= 1e-12
    assert result["momentum_residual_relative"] <= 1e-8
    assert result["passed"]


def test_reflux_disabled_breaks_mass_conservation():
    """Negative control: without the reflux the mass gate must fail."""
    sim, steps = _periodic_case(reflux=False)
    m0 = sim.totals()[0]
    for _ in range(steps):
        sim.step()
    assert abs(sim.totals()[0] - m0) / m0 > 1e-12


def test_wrong_neq_scaling_inflates_interface_error():
    """Negative control: a wrong non-equilibrium scaling must hurt the interface."""
    from tensorlbm.two_level_gates import _channel_interface_error

    ce = _channel_interface_error("chapman_enskog", 250, tau=1.0, shape=(18, 40))
    wrong = _channel_interface_error("none", 250, tau=1.0, shape=(18, 40))
    assert wrong["interface_layer_error"] > 1.5 * ce["interface_layer_error"]


def test_mixed_periodicity_keeps_diagonal_populations():
    """Regression guard: periodic in x + walls in y must not destroy diagonals.

    Before the per-axis sampling fix the sampler zeroed both axes as soon as
    one of them was non-periodic, so the diagonal populations of a channel were
    silently deleted at the x boundary (measured: channel peak 5.8x too low).
    """
    from tensorlbm.d2q9 import _c_on
    from tensorlbm.two_level_amr2d import _opposite_indices

    shape = (16, 32)
    solid = _channel_solid(*shape)
    c = _c_on(torch.device("cpu")).to(torch.float64)
    opp = _opposite_indices(c)
    f = torch.zeros((9,) + shape, dtype=torch.float64)
    j = int(torch.nonzero((c[:, 0] == 1) & (c[:, 1] == 1))[0, 0])
    f[j, 8, shape[1] - 1] = 1.0  # diagonal population in the rightmost column
    g = stream_mask(f, solid, (True, False), c, opp)
    assert float(g.sum()) == pytest.approx(1.0)
    # wraps in the periodic x direction, steps normally in the wall-bounded y
    assert float(g[j, 9, 0]) == pytest.approx(1.0)


def test_poiseuille_peak_matches_analytic():
    """Mixed periodicity + body force: the discrete channel peak must be right."""
    ny, nx = 26, 40
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    h = ny - 2
    force = 8.0 * nu * 0.05 / (h * h)
    u_exact = _poiseuille_profile(ny, nu, force)
    u_max = float(u_exact.max())
    solid = _channel_solid(ny, nx)
    rho = torch.ones((ny, nx), dtype=torch.float64)
    sim = UniformD2Q9(
        (ny, nx), tau, force=(force, 0.0), periodic=(True, False), solid=solid
    )
    sim.fill_uniform(
        rho, u_exact.view(ny, 1).expand(ny, nx).contiguous(), torch.zeros((ny, nx))
    )
    sim.step(400)
    _, ux, uy = sim.fields()
    err = float((ux[1:-1, nx // 2] - u_exact[1:-1]).abs().max() / u_max)
    assert err < 0.01, f"channel profile error {err:.3%} exceeds 1%"
    assert float(uy[1:-1].abs().max()) < 1e-8


def test_obstacle_force_is_zero_for_a_quiescent_field():
    grid = (16, 16)
    solid = cylinder_solid_mask(grid, (8.0, 8.0), 2.0)
    f = torch.zeros((9,) + grid, dtype=torch.float64)
    c = torch.tensor(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
        dtype=torch.float64,
    )
    assert obstacle_force(f, solid, c) == (0.0, 0.0)


def test_obstacle_force_recovers_a_known_momentum_flux():
    grid = (8, 8)
    solid = torch.zeros(grid, dtype=torch.bool)
    solid[:, 4:] = True  # vertical wall at x = 4
    c = _c_from_repo()
    f = torch.zeros((9,) + grid, dtype=torch.float64)
    f[1, :, 3] = 2.5  # c = (1, 0) hitting the wall
    fx, fy = obstacle_force(f, solid, c)
    assert fx == pytest.approx(2.0 * 1.0 * 2.5 * 8.0)
    assert fy == pytest.approx(0.0)


def _c_from_repo() -> torch.Tensor:
    from tensorlbm.d2q9 import _c_on

    return _c_on(torch.device("cpu")).to(torch.float64)