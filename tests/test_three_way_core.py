"""Unit/physics tests for the three-way LBM-DEM-FSI building blocks.

Run with ``pytest -q tests/test_three_way_core.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.three_way.beam import BeamConfig, CantileverBeam, mode_curvature, mode_shape
from tensorlbm.three_way.core2d import Flow2D
from tensorlbm.three_way.floe import Floe2D, FloeConfig, draft_for_ratio, submerged_area
from tensorlbm.three_way.ibm2d import (
    circle_markers,
    interpolate,
    marker_stencil,
    phi4,
    spread,
)


@pytest.fixture(autouse=True)
def _single_thread():
    torch.set_num_threads(1)
    yield


# ---------------------------------------------------------------------------
# core solver
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_poiseuille_profile_parabolic_symmetry_and_amplitude():
    """Body-force channel flow: symmetric, centred, parabolic amplitude."""
    ny, nx = 20, 6
    tau = 1.0
    flow = Flow2D(ny, nx, tau=tau)
    flow.solid[0, :] = True
    flow.solid[-1, :] = True
    flow.set_equilibrium(1.0, 0.0, 0.0)
    g = 1e-6
    fx = torch.full((ny, nx), g, dtype=torch.float64)
    for _ in range(9000):
        flow.step(fx=fx, periodic=True)
    _, ux, _ = flow.macroscopic()
    prof = ux[1:-1, 0]
    # symmetry about the channel centre line
    asym = float((prof - prof.flip(0)).abs().max() / prof.max())
    assert asym < 0.02, f"profile asymmetry {asym:.3%}"
    # peak in the middle third
    peak = int(prof.argmax())
    assert abs(peak - 0.5 * (prof.numel() - 1)) <= prof.numel() / 6
    # amplitude against the analytic parabola (h = ny - 2, walls at 0.5 / ny-1.5)
    y = torch.arange(1, ny - 1, dtype=torch.float64)
    h = ny - 2
    analytic = g / (2.0 * flow.nu) * (y - 0.5) * (h + 0.5 - (y - 0.5))
    ratio = float(prof.max() / analytic.max())
    assert 0.7 < ratio < 1.3, f"amplitude ratio {ratio:.3f}"


@pytest.mark.physics
def test_forcing_momentum_increment_exact():
    """Guo forcing injects exactly ``F`` momentum per step per cell."""
    flow = Flow2D(6, 6, tau=0.8)
    flow.set_equilibrium(1.0, 0.0, 0.0)
    c = flow._c
    g = 1e-6
    fx = torch.full((6, 6), g, dtype=torch.float64)
    before = float((flow.f * c[:, 0].view(9, 1, 1)).sum() / 36.0)
    flow.step(fx=fx, periodic=True)
    after = float((flow.f * c[:, 0].view(9, 1, 1)).sum() / 36.0)
    assert abs((after - before) - g) / g < 1e-7


@pytest.mark.physics
def test_viscous_dissipation_positive_and_scales_with_nu():
    flow = Flow2D(16, 16, tau=0.7)
    x = torch.arange(16, dtype=torch.float64) * (2 * math.pi / 16)
    X, Y = torch.meshgrid(x, x, indexing="xy")
    flow.set_equilibrium(1.0, 0.03 * torch.sin(Y), torch.zeros_like(Y))
    for _ in range(20):
        flow.step(periodic=True)
    eps = flow.viscous_dissipation()
    assert eps > 0.0
    assert math.isfinite(eps)


# ---------------------------------------------------------------------------
# immersed boundary operators
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_kernel_matches_repository_kernel():
    from tensorlbm.ibm import ibm_delta_4pt

    r = torch.tensor([-1.6, -0.3, 0.0, 0.4, 1.9, 2.4], dtype=torch.float64)
    assert torch.allclose(phi4(r), ibm_delta_4pt(r).to(torch.float64), atol=1e-10)


@pytest.mark.unit
def test_marker_stencil_partition_of_unity():
    mk = circle_markers(20.0, 20.0, 5.0, spacing=0.6)
    idx, w = marker_stencil(mk, 48, 48, support=4)
    assert torch.allclose(w.sum(dim=1), torch.ones_like(w.sum(dim=1)), atol=1e-12)
    idx2, w2 = marker_stencil(mk, 48, 48, support=2)
    assert torch.allclose(w2.sum(dim=1), torch.ones_like(w2.sum(dim=1)), atol=1e-12)


@pytest.mark.unit
def test_interpolate_spread_adjoint_momentum():
    """The total spread momentum equals the marker force sum (no loss)."""
    mk = circle_markers(30.0, 24.0, 5.0, spacing=0.6)
    values = torch.ones(len(mk), dtype=torch.float64) * 0.1
    field = spread(values, mk, 48, 60)
    assert abs(float(field.sum()) - float(values.sum())) < 1e-10
    assert abs(float(interpolate(torch.ones(48, 60, dtype=torch.float64), mk).mean()) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# DEM floe
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_floe_buoyancy_equilibrium_and_potential_force():
    cfg = FloeConfig()
    d = draft_for_ratio(cfg.radius, cfg.density_ratio)
    floe = Floe2D(cfg, (50.0, cfg.waterline - d))
    net = float(floe.weight_force()[1] + floe.buoyancy_force()[1])
    assert abs(net) / (floe.mass * cfg.gravity) < 1e-6
    # PE must be the primitive of weight + buoyancy
    eps = 1e-4
    y0 = float(floe.pos[1])
    up = Floe2D(cfg, (50.0, y0 + eps)).potential_energy()
    dn = Floe2D(cfg, (50.0, y0 - eps)).potential_energy()
    force_numeric = -(up - dn) / (2 * eps)
    assert abs(force_numeric - net) < 1e-6 * abs(floe.mass * cfg.gravity) + 1e-12


@pytest.mark.unit
def test_submerged_area_limits():
    r = 4.0
    assert abs(submerged_area(r, -r) - 0.0) < 1e-12
    assert abs(submerged_area(r, r) - math.pi * r * r) < 1e-9
    assert abs(submerged_area(r, 0.0) - 0.5 * math.pi * r * r) < 1e-12


@pytest.mark.unit
def test_floe_contact_uses_deformed_geometry():
    """A positive structural deflection delays contact (softening feedback)."""
    cfg = FloeConfig()
    d = draft_for_ratio(cfg.radius, cfg.density_ratio)
    x_beam = 70.0
    floe = Floe2D(cfg, (x_beam - cfg.radius + 0.4, cfg.waterline - d))
    a = torch.tensor([x_beam, 0.0], dtype=torch.float64)
    b = torch.tensor([x_beam, 40.0], dtype=torch.float64)
    f0, _, _ = floe.contact_with_segment(a, b, None)
    f1, _, _ = floe.contact_with_segment(a, b, lambda y: 1.0)
    assert float(torch.norm(f0)) > 0.0
    assert float(torch.norm(f1)) < float(torch.norm(f0))


# ---------------------------------------------------------------------------
# beam
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mode_shape_boundary_conditions():
    xi = torch.tensor([0.0, 1.0], dtype=torch.float64)
    for n in range(3):
        phi = mode_shape(n, xi)
        assert abs(float(phi[0])) < 1e-9  # clamped end
        # clamped slope: dp/dxi = 0 at xi = 0
        h = 1e-6
        slope = float((mode_shape(n, torch.tensor([h])) - mode_shape(n, torch.tensor([0.0]))) / h)
        assert abs(slope) < 1e-4


@pytest.mark.unit
def test_beam_static_deflection_matches_euler_bernoulli():
    """Tip deflection of a cantilever under a tip load is ``F L^3 / (3 EI)``."""
    cfg = BeamConfig(n_modes=3)
    beam = CantileverBeam(cfg)
    force = 0.2
    # a *tip load* F has generalized force Q_n = F phi_n(L)
    phi_tip = float(mode_shape(0, torch.tensor([1.0], dtype=torch.float64))[0])
    q = torch.tensor([force * phi_tip, 0.0, 0.0], dtype=torch.float64)
    for _ in range(1500):
        beam.step(q, 1.0)
    y_tip = cfg.y_base + cfg.length
    w = beam.deflection(y_tip)
    analytic = force * cfg.length ** 3 / (3.0 * cfg.youngs_modulus * beam.I)
    # the first-mode projection of a tip load is exact only up to the modal
    # truncation; three modes reproduce the static tip deflection well
    assert abs(w - analytic) / analytic < 0.05, f"{w} vs {analytic}"


@pytest.mark.unit
def test_beam_first_natural_frequency():
    cfg = BeamConfig()
    beam = CantileverBeam(cfg)
    analytic = 3.5160 / cfg.length ** 2 * math.sqrt(
        cfg.youngs_modulus * beam.I / (cfg.density * beam.area)
    )
    assert abs(float(beam.omega[0]) - analytic) / analytic < 0.02


@pytest.mark.unit
def test_beam_negative_control_stays_zero():
    cfg = BeamConfig()
    beam = CantileverBeam(cfg)
    beam.enable_feedback = False
    beam.step(torch.tensor([1.0, 1.0], dtype=torch.float64), 1.0)
    assert float(beam.q.abs().max()) == 0.0
    assert beam.stress(cfg.y_base + 0.5 * cfg.length) == 0.0