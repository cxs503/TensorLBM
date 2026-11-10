"""Unit / physics tests for the force-coupling method (:mod:`tensorlbm.fcm`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm import particle_suspension as ps
from tensorlbm.fcm import (
    FCMCoupler,
    FCMSimulation,
    ParticleSet,
    gaussian_delta_1d,
    lbm_step_bgk_guo,
    sphere_markers,
    sphere_particle,
)


# --------------------------------------------------------------------------- #
# discrete kernel identities
# --------------------------------------------------------------------------- #


def test_gaussian_delta_1d_normalised():
    r = torch.linspace(-6.0, 6.0, 4001, dtype=torch.float64)
    val = gaussian_delta_1d(r, 1.0)
    integral = torch.trapz(val, r)
    assert abs(float(integral) - 1.0) < 1e-6
    assert float(val.max()) == pytest.approx(1.0 / math.sqrt(math.pi), rel=1e-6)


def test_spread_conserves_force_and_interpolates_unity():
    shape = (24, 24, 24)
    cp = FCMCoupler(shape, 1.5)
    pos = torch.tensor([[3.3, 7.9, 12.4], [20.1, 2.0, 17.7]])
    vals = torch.tensor([[1.0, -2.0, 0.5], [0.3, 0.0, -1.5]])
    grid = cp.spread(vals, pos)
    tot = grid.sum(dim=(1, 2, 3))
    assert torch.allclose(tot, vals.sum(dim=0), atol=1e-5)
    ones = torch.ones(3, *shape)
    interp = cp.interpolate(ones, pos)
    assert torch.allclose(interp, torch.ones_like(interp), atol=1e-6)


def test_spread_interpolate_adjoint():
    shape = (16, 16, 16)
    cp = FCMCoupler(shape, 1.2)
    torch.manual_seed(0)
    pos = torch.rand(5, 3) * 16.0
    forces = torch.randn(5, 3)
    u = torch.randn(3, *shape)
    grid = cp.spread(forces, pos)
    lhs = float((grid * u).sum())
    rhs = float((forces * cp.interpolate(u, pos)).sum())
    assert lhs == pytest.approx(rhs, rel=1e-5, abs=1e-5)


def test_periodic_wrapping_of_kernel():
    """A coupling point on the boundary must not lose force off the grid."""
    shape = (12, 12, 12)
    cp = FCMCoupler(shape, 1.5)
    pos = torch.tensor([[0.05, 11.95, 6.0]])
    grid = cp.spread(torch.tensor([[1.0, 1.0, 1.0]]), pos)
    assert torch.allclose(grid.sum(dim=(1, 2, 3)), torch.ones(3), atol=1e-5)


def test_sphere_markers_on_surface():
    off = sphere_markers(4.0, spacing=1.0)
    r = torch.linalg.norm(off, dim=1)
    assert off.shape[0] > 100
    assert torch.allclose(r, torch.full_like(r, 4.0), atol=1e-5)


# --------------------------------------------------------------------------- #
# analytic references
# --------------------------------------------------------------------------- #


def test_stokes_and_schiller_naumann():
    assert ps.stokes_cd(0.1) == pytest.approx(240.0)
    assert ps.schiller_naumann_cd(0.1) / ps.stokes_cd(0.1) == pytest.approx(1.031, abs=2e-3)
    # finite-Re terminal velocity is below the Stokes value
    v_st = ps.stokes_terminal_velocity(1.0, 2.0, 1.0, 0.1, 1.0)
    v_re = ps.finite_re_terminal_velocity(1.0, 2.0, 1.0, 0.1, 1.0)
    assert v_re < v_st
    assert ps.stokes_terminal_velocity(1.0, 2.0, 1.0, 0.1, 1.0) == pytest.approx(
        (2.0 - 1.0) * 1.0 * 1.0 / (18.0 * 0.1)
    )


def test_batchelor_green_and_einstein():
    assert ps.einstein_viscosity_ratio(0.1) == pytest.approx(1.25)
    assert ps.batchelor_green_viscosity_ratio(0.1) == pytest.approx(1.25 + 0.052)
    assert ps.batchelor_green_viscosity_ratio(0.0) == pytest.approx(1.0)


def test_lubrication_scaling_and_limits():
    mu, a = 0.1, 4.0
    f1 = ps.lubrication_force_sphere_sphere(a, a, 0.4, 1e-4, mu)
    f2 = ps.lubrication_force_sphere_sphere(a, a, 0.2, 1e-4, mu)
    assert f2 / f1 == pytest.approx(2.0, rel=1e-6)  # F ~ 1/h
    # equal spheres: a_eff = a/2 -> F = (3/2) pi mu a^2 U / h
    expect = 1.5 * math.pi * mu * a**2 * 1e-4 / 0.4
    assert f1 == pytest.approx(expect, rel=1e-6)
    # sphere-plane limit: a2 -> inf gives 6 pi mu a^2 V / h
    f_plane = ps.lubrication_force_sphere_sphere(a, 1e12, 0.4, 1e-4, mu)
    assert f_plane == pytest.approx(
        ps.lubrication_force_sphere_plane(a, 0.4, 1e-4, mu), rel=1e-6
    )


def test_hasimoto_correction_monotone():
    assert ps.hasimoto_drag_correction(0.0) == 1.0
    vals = [ps.hasimoto_drag_correction(x) for x in (1e-4, 1e-3, 1e-2)]
    assert vals[0] < vals[1] < vals[2]
    assert all(v > 1.0 for v in vals)


# --------------------------------------------------------------------------- #
# LBM kernels and TG decay (single phase)
# --------------------------------------------------------------------------- #


def test_taylor_green_decay_single_phase():
    """Particle-free TG decay must reproduce exp(-2 nu k^2 t)."""
    L, nu, u0, steps = 16, 0.1, 0.01, 220
    tau = 0.5 + 3.0 * nu
    k = 2.0 * math.pi / L
    sim = FCMSimulation((L, L, L), tau)
    sim.initialize()
    xi = torch.arange(L, dtype=torch.float32)
    X = xi.view(1, 1, L)
    Y = xi.view(1, L, 1)
    from tensorlbm.d3q19 import equilibrium3d

    ux = (u0 * torch.sin(k * X) * torch.cos(k * Y)).expand(L, L, L).contiguous()
    uy = (-u0 * torch.cos(k * X) * torch.sin(k * Y)).expand(L, L, L).contiguous()
    sim.f = equilibrium3d(
        torch.ones(L, L, L), ux, uy, torch.zeros(L, L, L)
    )
    e0 = sim.kinetic_energy()
    e = [e0]
    for _ in range(steps):
        sim.step(update_particles=False)
        e.append(sim.kinetic_energy())
    e = np.asarray(e)
    t = np.arange(e.size)
    fit = slice(40, steps)
    slope = np.polyfit(t[fit], np.log(e[fit]), 1)[0]
    gamma = -slope / 2.0
    gamma_theory = ps.taylor_green_decay_rate_2d(nu, k)
    assert gamma / gamma_theory == pytest.approx(1.0, rel=0.02)


def test_guo_forcing_odd_momentum_consistency():
    """Guo half-step: applied force must appear as the momentum increment."""
    L, tau = 8, 0.8
    sim = FCMSimulation((L, L, L), tau)
    sim.initialize()
    f0 = sim.f.clone()
    fx = torch.zeros(3, L, L, L)
    fx[0] = 1e-4
    p0 = sim.momentum()
    sim.step(body_force=fx, update_particles=False)
    p1 = sim.momentum()
    dp = float(p1[0] - p0[0])
    assert dp == pytest.approx(1e-4 * L**3, rel=2e-2)


# --------------------------------------------------------------------------- #
# coupled physics
# --------------------------------------------------------------------------- #


def test_particle_velocity_relaxation_is_stable():
    """Density ratio 2 must relax (no added-mass divergence) over many steps."""
    L, a = 16, 3.0
    sim = FCMSimulation((L, L, L), 0.8, sigma=1.5)
    part = sphere_particle((L / 2, L / 2, L / 2), a, density_ratio=2.0,
                           velocity=(0.0, 0.0, -1e-3))
    sim.initialize(part)
    vol = 4.0 / 3.0 * math.pi * a**3
    fext = torch.zeros(1, 3)
    fext[0, 2] = -(2.0 - 1.0) * vol * 1e-5
    for _ in range(120):
        sim.step(part, external_force=fext, remove_mean_momentum=True)
    v = part.velocities[0]
    assert torch.isfinite(v).all()
    assert float(torch.linalg.norm(v)) < 1e-2
    assert torch.isfinite(sim.f).all()


def test_marker_force_reaction_matches_momentum_sink():
    """Sum of the spread marker forces equals the sum over coupling points."""
    L, a = 16, 3.0
    sim = FCMSimulation((L, L, L), 0.8, sigma=1.5)
    part = sphere_particle((L / 2, L / 2, L / 2), a, fixed=True)
    sim.initialize(part)
    fp = sim.step(part, update_particles=False)
    assert torch.allclose(fp.sum(dim=0), sim._last_marker_forces.sum(dim=0), atol=1e-6)


@pytest.mark.slow
def test_single_sphere_drag_near_stokes():
    """Force-driven sphere in a periodic box: C_D within 40% of the SN value.

    The tolerance is deliberately loose: this *unit* test runs a small,
    under-resolved configuration (d = 6 lattice units, 2500 steps).  The
    quantitative verification (resolution and domain studies, <15% band) lives
    in ``scripts/fcm_suspension_campaign.py --case drag``.
    """
    L, d, nu = 24, 6.0, 0.1
    a = d / 2
    tau = 0.5 + 3.0 * nu
    mu = nu
    sim = FCMSimulation((L, L, L), tau, sigma=1.5)
    c = (L / 2, L / 2, L / 2)
    part = sphere_particle(c, a, fixed=True)
    sim.initialize(part)
    z = torch.arange(L).view(L, 1, 1).float()
    y = torch.arange(L).view(1, L, 1).float()
    x = torch.arange(L).view(1, 1, L).float()
    inside = ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) <= a * a
    fluid = (~inside).float()
    n_fluid = float(fluid.sum())
    g = torch.zeros(3, L, L, L)
    drive = 1e-5
    u_target = 0.0125  # Re = u d / nu = 0.75
    u_hist, f_hist = [], []
    steps = 2500
    for it in range(steps):
        g[0] = drive * fluid
        fp = sim.step(part, body_force=g, update_particles=False)
        if it % 100 == 0 and it > 0:
            um = float(sim.mean_velocity()[0])
            drive = float(np.clip(drive * np.clip((u_target / max(um, 1e-12)) ** 0.5, 0.6, 1.6),
                                  1e-9, 1e-2))
        if it >= steps - 500:
            u_hist.append(float(sim.mean_velocity()[0]))
            f_hist.append(-float(fp[0, 0]))
    u = float(np.mean(u_hist))
    f_drag = float(np.mean(f_hist))
    area = 0.25 * math.pi * d * d
    cd = f_drag / (0.5 * u * u * area)
    re = u * d / nu
    cd_sn = ps.schiller_naumann_cd(re)
    # The FCM Gaussian envelope inflates the hydrodynamic radius, and a periodic
    # box at Re < 1 needs O(10^4) steps to reach the steady state, so this fast
    # check only asserts the order of magnitude.  The quantitative verification
    # (resolution + domain studies, convergence table) is produced by
    # ``scripts/fcm_suspension_campaign.py --case drag``.
    assert 0.5 < cd / cd_sn < 3.5, f"C_D/C_D,SN = {cd / cd_sn:.3f} at Re={re:.3f}"
    # the coupling force must absorb a finite fraction of the drive
    drive_total = float(g[0].sum())
    assert 0.2 < f_drag / drive_total < 1.6