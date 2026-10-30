"""Unit + short end-to-end tests for ``tensorlbm.cac_moving_body`` (W11-WE).

Covers the pieces that the water-entry benchmark leans on and that can be
checked analytically:

* geometry masks (wedge footprint, disc symmetry),
* link bookkeeping (``links_into`` is the transpose of ``links_from``),
* moving-wall bounce-back reduces to the plain swap at ``u_w = 0`` and is
  antisymmetric in ``u_w`` about that point,
* momentum exchange is zero for a symmetric configuration and flips sign with
  the wall velocity,
* the Neumann wall-ghost policy is a no-op on a uniform field and produces a
  zero-normal-gradient ghost on a layered field,
* a short end-to-end run with a prescribed-motion body stays finite.

All CPU, all fast (no water-entry physics is asserted here -- see
``benchmarks/pending/phasefield_water_entry/`` for that).
"""

import math

import pytest
import torch

from tensorlbm import cac_lbm as m
from tensorlbm import cac_moving_body as mb


def _tank(shape, *, z_periodic=True):
    return mb.wall_box(shape, z_periodic=z_periodic)


def test_wall_box_faces():
    shape = (4, 10, 12)
    w = _tank(shape)
    assert w[:, 0, :].all() and w[:, -1, :].all()
    assert w[:, :, 0].all() and w[:, :, -1].all()
    assert not w[0, 3:-3, 3:-3].any(), "z interior must be open for a periodic slab"
    w6 = _tank(shape, z_periodic=False)
    assert w6[0, :, :].all() and w6[-1, :, :].all()


def test_wedge_mask_is_symmetric_and_monotone():
    shape = (4, 80, 120)
    body = mb.wedge_mask(
        shape,
        apex_xy=(60.0, 40.0),
        deadrise_rad=math.radians(10.0),
        half_length=40.0,
        thickness=6.0,
    )
    row = body[0, :, :]
    # symmetric about the apex column (column-by-column, |x - apex| pairing)
    for d in range(1, 45):
        assert torch.equal(row[:, 60 - d], row[:, 60 + d]), d
    # the tip is the lowest solid cell and nothing exists below it
    assert bool(body[0, 40, 60])
    assert not bool(body[0, 39, 60])
    # solid lies ABOVE the wetted surface, so it thins out with |x - apex|
    assert int(row[:, 60].sum()) == 7  # thickness 6 + the tip cell
    assert int(row[:, 55].sum()) < int(row[:, 60].sum())
    # top face is flat at apex + thickness
    assert bool(body[0, 46, 60]) and not bool(body[0, 47, 60])
    # footprint bounded by the requested half length
    cols = torch.nonzero(row.any(dim=0)).flatten()
    assert int(cols.max() - cols.min()) <= 2 * 40 + 1


def test_disc_mask_area_and_centroid():
    shape = (4, 64, 64)
    body = mb.disc_mask(shape, centre_xy=(32.0, 32.0), radius=8.0)
    per_layer = int(body[0].sum())
    assert abs(per_layer - math.pi * 64) < 0.08 * math.pi * 64
    assert int(body.sum()) == per_layer * shape[0]
    yy, xx = torch.nonzero(body[0], as_tuple=True)
    assert float(yy.float().mean()) == pytest.approx(32.0, abs=0.5)
    assert float(xx.float().mean()) == pytest.approx(32.0, abs=0.5)


def test_link_tables_are_transposes():
    shape = (3, 9, 11)
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[1, 4, 5] = True
    into = mb.links_into(mask)  # x + c_q in mask
    frm = mb.links_from(mask)  # x - c_q in mask
    # one interior cell: 19 true entries for the "into" table (18 links + rest)
    assert int(into.sum()) == 19 and int(frm.sum()) == 19
    # for every non-rest direction, "into" and "from" see the same number of
    # links (each is the other one's transpose), and q=0 is the mask itself
    assert torch.equal(into[0], mask) and torch.equal(frm[0], mask)
    for q in range(1, 19):
        assert int(into[q].sum()) == int(frm[q].sum()), q


def test_moving_wall_bounce_reduces_to_static_swap_and_is_antisymmetric():
    shape = (4, 12, 12)
    body = mb.disc_mask(shape, centre_xy=(6.0, 6.0), radius=3.0)
    fluid = ~body
    torch.manual_seed(0)
    dist = torch.rand(19, *shape, dtype=torch.float64)
    dist_pre = torch.rand(19, *shape, dtype=torch.float64)
    density = torch.ones(shape, dtype=torch.float64)

    static = mb.moving_wall_bounce(
        dist, dist_pre, body=body, fluid=fluid, u_wall=(0.0, 0.0, 0.0), density=density
    )
    plus = mb.moving_wall_bounce(
        dist, dist_pre, body=body, fluid=fluid, u_wall=(0.01, 0.0, 0.0), density=density
    )
    minus = mb.moving_wall_bounce(
        dist, dist_pre, body=body, fluid=fluid, u_wall=(-0.01, 0.0, 0.0), density=density
    )
    # static path is the plain opposite swap on the link cells
    link = fluid.unsqueeze(0) & mb.links_from(body)
    plain = torch.where(
        link,
        dist_pre[torch.tensor([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])],
        dist,
    )
    assert torch.allclose(static, plain)
    # antisymmetry about u_w = 0 in the wall contribution
    wall_cells = link.any(dim=0)
    assert torch.allclose(plus[:, wall_cells] + minus[:, wall_cells], 2.0 * static[:, wall_cells])


def test_momentum_exchange_symmetric_and_antisymmetric():
    shape = (4, 16, 16)
    body = mb.disc_mask(shape, centre_xy=(8.0, 8.0), radius=3.0)
    fluid = ~body
    from tensorlbm.d3q19 import C

    y_dir = next(q for q, (cx, cy, cz) in enumerate(C.tolist()) if (cx, cy, cz) == (0, 1, 0))
    dist_pre = torch.zeros(19, *shape, dtype=torch.float64)
    dist_pre[y_dir] = 0.1  # one direction only -> asymmetric on purpose
    density = torch.ones(shape, dtype=torch.float64)
    zero = mb.momentum_exchange(
        dist_pre, body=body, fluid=fluid, u_wall=(0.0, 0.0, 0.0), density=density
    )
    # symmetric body + uniform +y-fluxes: x and z must cancel
    assert zero[0] == pytest.approx(0.0, abs=1e-12)
    assert zero[2] == pytest.approx(0.0, abs=1e-12)
    assert zero[1] != 0.0
    # the u_w correction must shift the force linearly and only through rho*w*cuw
    shifted = mb.momentum_exchange(
        dist_pre, body=body, fluid=fluid, u_wall=(0.0, 0.02, 0.0), density=density
    )
    assert shifted[1] != pytest.approx(zero[1])
    # doubling the wall velocity from +v to -v mirrors the correction about v=0
    back = mb.momentum_exchange(
        dist_pre, body=body, fluid=fluid, u_wall=(0.0, -0.02, 0.0), density=density
    )
    assert shifted[1] + back[1] == pytest.approx(2.0 * zero[1], rel=1e-12)


def test_neumann_wall_ghost_is_noop_on_uniform_field():
    shape = (4, 8, 8)
    wall = _tank(shape)
    sim = m.CACSim(
        shape,
        rho_l=1.0,
        rho_g=0.1,
        mu_l=0.1,
        mu_g=0.01,
        sigma=0.01,
        width=4.0,
        axes="pwp",
        wall_mask=wall,
        wall_phi_policy="neumann",
    )
    sim.initialize(torch.full(shape, 0.7))
    before = sim.f.clone()
    sim.refresh_solid_ghosts()
    assert torch.allclose(sim.f, before, atol=1e-12)


def test_neumann_wall_ghost_matches_neighbour_mean_on_layered_field():
    shape = (4, 8, 8)
    wall = _tank(shape)
    sim = m.CACSim(
        shape,
        rho_l=1.0,
        rho_g=0.1,
        mu_l=0.1,
        mu_g=0.01,
        sigma=0.01,
        width=4.0,
        axes="pwp",
        wall_mask=wall,
        wall_phi_policy="neumann",
    )
    phi = torch.linspace(0.0, 1.0, shape[1]).view(1, -1, 1).expand(shape).contiguous()
    sim.initialize(phi)
    sim.refresh_solid_ghosts()
    got = sim.f.sum(dim=0)
    # bottom wall cell must inherit the first fluid row, not its seeded value
    assert float(got[:, 0, 4].mean()) == pytest.approx(float(phi[:, 1, 4].mean()), abs=1e-6)
    assert float(got[:, -1, 4].mean()) == pytest.approx(float(phi[:, -2, 4].mean()), abs=1e-6)


def test_set_wall_mask_rebuilds_derived_tensors():
    shape = (4, 8, 8)
    sim = m.CACSim(
        shape, rho_l=1.0, rho_g=0.1, mu_l=0.1, mu_g=0.01, sigma=0.01, width=4.0, axes="pwp"
    )
    assert sim.wall_mask is None and sim.inv_tau_f.abs().sum() > 0
    wall = _tank(shape)
    sim.set_wall_mask(wall)
    assert sim.fluid_mask.sum() == (~wall).sum()
    assert float(sim.inv_tau_f[0][wall].abs().sum()) == 0.0
    assert float(sim.fm_f[0][wall].abs().sum()) == 0.0


def test_moving_body_end_to_end_stays_finite():
    shape = (4, 24, 24)
    wall = _tank(shape)
    sim = m.CACSim(
        shape,
        rho_l=1.0,
        rho_g=0.01,
        mu_l=0.1,
        mu_g=0.001,
        sigma=0.01,
        width=4.0,
        mobility=0.05,
        p0=0.01,
        gravity=(0.0, -1e-4, 0.0),
        gravity_mode="rho",
        axes="pwp",
        wall_mask=wall,
    )
    sim.initialize(m.init_phi_layers(shape, width=4.0, y0=14.0))
    driver = mb.MovingBodyCAC(sim, static_wall=wall)
    body = mb.disc_mask(shape, centre_xy=(12.0, 18.0), radius=3.0)
    driver.set_body(body)
    for step in range(1, 61):
        moved = mb.disc_mask(shape, centre_xy=(12.0, 18.0 - 0.01 * step), radius=3.0)
        driver.step(moved, (0.0, -0.01, 0.0))
    d = m.diagnostics(sim)
    assert d["nan_f"] == 0 and d["nan_g"] == 0
    assert d["u_max_fluid"] < 0.2
    assert all(abs(f) < 10.0 for f in driver.force)


def _two_phase_tank(shape, *, y0, policy="freeze", g=1e-4):
    wall = _tank(shape)
    sim = m.CACSim(
        shape,
        rho_l=1.0,
        rho_g=0.01,
        mu_l=0.1,
        mu_g=0.001,
        sigma=0.01,
        width=4.0,
        mobility=0.05,
        p0=0.01,
        gravity=(0.0, -g, 0.0),
        gravity_mode="rho",
        axes="pwp",
        wall_mask=wall,
        wall_phi_policy=policy,
    )
    sim.initialize(m.init_phi_layers(shape, width=4.0, y0=y0))
    return sim, wall


def test_static_body_conserves_phase_mass():
    """Full-way bounce inside the body must not destroy phase mass.

    Regression for the first implementation of this module, which bounced on
    the fluid side only: the body's own ghosts were never refreshed, so every
    streamed-in population vanished.  A 5x5x4 block lost 50 % of the phase
    integral in 100 steps.
    """
    shape = (4, 40, 48)
    sim, wall = _two_phase_tank(shape, y0=20.0)
    driver = mb.MovingBodyCAC(sim, static_wall=wall)
    body = mb.disc_mask(shape, centre_xy=(24.0, 24.0), radius=5.0)
    driver.step(body, (0.0, 0.0, 0.0))
    start = float(sim.f.sum(dim=0)[sim.fluid_mask].sum())
    for _ in range(120):
        driver.step(body, (0.0, 0.0, 0.0))
    end = float(sim.f.sum(dim=0)[sim.fluid_mask].sum())
    # the base solver itself drifts at the 1e-4 level here (interface creep,
    # blocker B2); the bug this guards against was a 50 % loss.
    assert abs(end - start) / start < 1e-3, (start, end)


def test_static_body_driver_matches_plain_step():
    """With u = 0 the driver must reproduce a plain ``CACSim.step()`` state."""
    shape = (4, 32, 40)
    sim_a, wall = _two_phase_tank(shape, y0=16.0)
    body = mb.disc_mask(shape, centre_xy=(20.0, 20.0), radius=4.0)
    driver = mb.MovingBodyCAC(sim_a, static_wall=wall)
    for _ in range(3):
        driver.step(body, (0.0, 0.0, 0.0))
    sim_b, _ = _two_phase_tank(shape, y0=16.0)
    sim_b.set_wall_mask(wall | body)
    for _ in range(3):
        sim_b.step()
    assert torch.allclose(sim_a.f, sim_b.f, atol=1e-12)
    assert torch.allclose(sim_a.g, sim_b.g, atol=1e-12)


def test_momentum_exchange_delta_is_antisymmetric_under_swap():
    """The force reader is odd in the boundary operation (pins the factor)."""
    shape = (2, 24, 32)
    sim, wall = _two_phase_tank(shape, y0=12.0)
    body = mb.disc_mask(shape, centre_xy=(16.0, 16.0), radius=4.0) & ~wall
    fluid = ~(wall | body)
    from tensorlbm.boundaries3d import bounce_back_cells_3d

    sim.collide()
    sim.stream()
    g_before = sim.g
    g_after = bounce_back_cells_3d(sim.g, body)
    forward = mb.momentum_exchange_delta(g_before, g_after, body=body, fluid=fluid)
    backward = mb.momentum_exchange_delta(g_after, g_before, body=body, fluid=fluid)
    for a in range(3):
        assert forward[a] == pytest.approx(-backward[a], rel=1e-12, abs=1e-12), a
    # no boundary change -> no force
    zero = mb.momentum_exchange_delta(g_before, g_before, body=body, fluid=fluid)
    assert all(abs(v) == 0.0 for v in zero)


def test_static_body_in_zero_gravity_feels_no_force():
    """Force-reader offset: with g = 0 a static body must read ~0."""
    shape = (4, 40, 32)
    sim, wall = _two_phase_tank(shape, y0=30.0, policy="neumann", g=0.0)
    body = mb.disc_mask(shape, centre_xy=(16.0, 16.0), radius=5.0)
    driver = mb.MovingBodyCAC(sim, static_wall=wall)
    tail = []
    for step in range(1, 201):
        driver.step(body, (0.0, 0.0, 0.0))
        if step > 100:
            tail.append(driver.force)
    n = len(tail)
    # 1e-3 in lattice force units; the longer probe arm (2000 steps) reaches
    # 1e-5, the residual here is the still-relaxing interface (sigma = 0.01).
    for a in range(3):
        assert abs(sum(f[a] for f in tail) / n) < 1e-3, a


def test_archimedes_buoyancy_gate():
    """A static submerged body must feel ``rho_l g V`` (force-reader gate)."""
    shape = (4, 48, 32)
    g = 1e-4
    sim, wall = _two_phase_tank(shape, y0=40.0, policy="neumann", g=g)
    body = mb.disc_mask(shape, centre_xy=(16.0, 16.0), radius=5.0)
    driver = mb.MovingBodyCAC(sim, static_wall=wall)
    expected = 1.0 * g * float(body.sum().item())
    fy = []
    for step in range(1, 601):
        driver.step(body, (0.0, 0.0, 0.0))
        if step > 400:
            fy.append(driver.force[1])
    mean = sum(fy) / len(fy)
    assert mean == pytest.approx(expected, rel=0.15), (mean, expected)
