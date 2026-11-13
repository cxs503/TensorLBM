"""Unit tests for the pressure-evolution + Allen-Cahn port (XVAL m18194)."""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.phasefield import pf_pressure_evolution as M
from tensorlbm.phasefield.cac_h_population import _neighbours, get_stencil

ST = get_stencil("d2q9")
DT = torch.float64


def _smooth(shape=(16, 16), seed=3):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(shape, generator=g, dtype=DT)
    return torch.nn.functional.avg_pool2d(a.view(1, 1, *shape), 3, 1, 1).view(shape) * 0 + a


def test_moment_matrix_and_viscous_modes():
    m = torch.tensor(M.D2Q9_MOMENTS, dtype=DT)
    assert m.shape == (9, 9)
    # rows 7 and 8 are p_xx = c_x^2 - c_y^2 and p_xy = c_x c_y
    c = ST.C.to(DT)
    pxx = c[:, 0] ** 2 - c[:, 1] ** 2
    pxy = c[:, 0] * c[:, 1]
    assert torch.allclose(m[7], pxx)
    assert torch.allclose(m[8], pxy)
    assert M.VISCOUS_MOMENTS == (7, 8)
    inv = torch.linalg.inv(m)
    r = torch.randn((9, 4, 4), dtype=DT)
    lam0 = torch.zeros(9, dtype=DT)
    # lambda = 0 projects every mode out
    assert float(M.mrt_relax(r, lam0, m, inv).abs().max()) < 1e-12
    # lambda = 1 is the identity
    assert float((M.mrt_relax(r, torch.ones(9, dtype=DT), m, inv) - r).abs().max()) < 1e-12


def test_gbar_equilibrium_gives_pressure():
    u = (torch.zeros(8, 8, dtype=DT), torch.zeros(8, 8, dtype=DT))
    p = 0.31 * torch.ones(8, 8, dtype=DT)
    rho = 7.0 * torch.ones(8, 8, dtype=DT)
    gamma = M.maxwellian(ST, u, 2, dtype=DT, device="cpu")
    c = ST.C.to(DT)
    g = M.gbar_equilibrium(gamma, p, rho, ST)
    assert float((g.sum(0) - p).abs().max()) < 1e-14
    assert float((g * c[:, 0].view(-1, 1, 1)).sum().abs().max()) < 1e-14  # at rest


def test_interface_term_has_no_pressure_source_when_static():
    """Regression: the reference's centred/mixed stencils must not pump pressure.

    ``sum_i I_i`` is the mass (=pressure) source of the f-population.  For a
    static field (u = 0) both directional stencils must sum to zero, otherwise the
    pressure drifts linearly in time (the bug found while porting).
    """
    sol = M.PFPressureEvolution(shape=(24, 24), rho_h=10.0, rho_l=1.0)
    phi = sol.tanh_drop((24, 24), 7.0, 4.0)
    nb = _neighbours(phi, ST)
    grad = M.__dict__["isotropic_gradient"](phi, ST, nb) if "isotropic_gradient" in M.__dict__ else None
    from tensorlbm.phasefield.cac_h_population import isotropic_gradient
    grad = isotropic_gradient(phi, ST, nb)
    mu = M.chem_potential(phi, ST, sigma=0.05, width=4.0, nb=nb)
    u = (torch.zeros_like(phi), torch.zeros_like(phi))
    gamma = M.maxwellian(ST, u, 2, dtype=DT, device="cpu")
    # The centred stencil is exactly antisymmetric -> the source is round-off.
    term_c = M.interface_term(gamma, M.directional_centred(phi, ST, nb), mu, u, grad,
                              10.0, 1.0, ST)
    assert float(term_c.sum(0).abs().max()) < 1e-14 * max(
        1.0, float(term_c.abs().max()))
    # The reference's *mixed* (upwind) stencil is not exactly conservative: its
    # weight sum is zero only to second order, so a small O(truncation) pressure
    # source survives (measured ~10% of the term scale at W = 4).  This is an
    # inherited property of the OSS scheme, quantified here so that a future
    # change cannot silently make it worse.
    term_m = M.interface_term(gamma, M.directional_mixed(phi, ST, nb), mu, u, grad,
                              10.0, 1.0, ST)
    src = float(term_m.sum(0).abs().max())
    scale = float(term_m.abs().max())
    assert 0.0 < src < 0.2 * scale, (src, scale)


def test_mass_conservation_is_exact():
    sol = M.PFPressureEvolution(shape=(32, 32), rho_h=100.0, rho_l=1.0,
                                sigma=0.05, width=4.0, mobility=0.05,
                                viscosity_h=1 / 6, viscosity_l=1 / 6)
    sol.initialize(sol.tanh_drop((32, 32), 8.0, 4.0))
    sol.run(200)
    assert sol.mass_error() < 1e-11
    assert float(sol.phi.min()) > -1e-9 and float(sol.phi.max()) < 1.0 + 1e-9


def test_single_phase_control_is_quiescent():
    """phi_h = phi_l -> mu = 0 identically: no interface force, no pressure drift."""
    sol = M.PFPressureEvolution(shape=(24, 24), rho_h=1.0, rho_l=1.0,
                                phi_h=0.0, phi_l=0.0, sigma=0.05, width=4.0,
                                viscosity_h=1 / 6, viscosity_l=1 / 6)
    sol.initialize(torch.zeros(24, 24, dtype=DT))
    _, _, grad, mu = sol.kinetics()
    assert float(mu.abs().max()) == 0.0
    p0 = sol.pressure()
    sol.run(120)
    p1 = sol.pressure()
    u = sol.velocity()
    assert float((p1 - p0).abs().max()) < 1e-12
    assert float((u[0] ** 2 + u[1] ** 2).max()) < 1e-14


def test_taylor_green_recovers_viscosity():
    n, nu = 32, 0.1
    sol = M.PFPressureEvolution(shape=(n, n), rho_h=1.0, rho_l=1.0,
                                phi_h=0.0, phi_l=0.0, viscosity_h=nu,
                                viscosity_l=nu)
    phi = torch.zeros(n, n, dtype=DT)
    y = torch.arange(n, dtype=DT).view(n, 1).expand(n, n)
    k = 2 * math.pi / n
    u0 = 0.01
    ux, uy = u0 * torch.sin(k * y), torch.zeros_like(y)
    sol.initialize(phi, u0=(ux, uy))
    sol.f = M.gbar_equilibrium(M.maxwellian(ST, (ux, uy), 2, dtype=DT, device="cpu"),
                               torch.zeros_like(phi), torch.ones_like(phi), ST)
    rec = []
    for step in range(1201):
        if step % 100 == 0:
            u = sol.velocity()
            rec.append((step, float(2 / n * (u[0] * torch.sin(k * y)).sum(1).mean())))
        sol.step()
    t = torch.tensor([r[0] for r in rec], dtype=DT)
    a = torch.tensor([abs(r[1]) for r in rec], dtype=DT)
    slope = float((torch.log(a[1:]) - torch.log(a[:-1])).mean() / 100.0)
    nu_fit = -slope / k ** 2
    assert abs(nu_fit - nu) / nu < 0.05


def test_laplace_law_coarse():
    """Small static drop must already show dp = sigma/R within 25%."""
    n, R, W, sigma, nu = 40, 9.0, 4.0, 0.05, 1 / 6
    sol = M.PFPressureEvolution(shape=(n, n), rho_h=100.0, rho_l=1.0, sigma=sigma,
                                width=W, mobility=0.05, viscosity_h=nu, viscosity_l=nu)
    sol.initialize(sol.tanh_drop((n, n), R, W))
    c = 0.5 * (n - 1)
    y, x = torch.meshgrid(torch.arange(n, dtype=DT), torch.arange(n, dtype=DT),
                          indexing="ij")
    r = torch.sqrt((x - c) ** 2 + (y - c) ** 2)
    r_eff = math.sqrt(float(sol.phi.sum()) / math.pi)
    sol.run(1500)
    p = sol.pressure()
    dp = float(p[r < r_eff - W].mean() - p[r > r_eff + W].mean())
    ref = M.laplace_pressure_2d(sigma, r_eff)
    assert abs(dp - ref) / ref < 0.25, (dp, ref)
