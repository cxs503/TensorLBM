"""Unit tests for the two-PDF partial-density colour-gradient model.

Invariants checked (not just "code executes"):

* ``viscosity_from_phi`` reproduces ``nu_a`` / ``nu_b`` at ``phi = +-1`` and the
  harmonic mean at ``phi = 0``; ``tau = 3 nu + 1/2``;
* ``two_pdf_macros`` recovers the partial densities / colour / velocity that
  were assembled by hand from the two PDFs;
* one full step conserves **each** partial density to round-off (the defining
  property of the partial-density bookkeeping);
* the recolouring stencil is mass-free, ``sum_i w_i c_i = 0``, so the split of
  the total PDF cannot change ``rho_A`` / ``rho_B``;
* the surface-tension perturbation is mass- and momentum-free;
* the ``phi = +1`` single-phase limit reproduces the half-way-bounce-back
  Poiseuille parabola;
* the literal MF-LBM recolouring stencil (``4/9`` axes) is *unstable* at this
  amplitude scaling (documented negative-partial-density failure mode).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundaries import bounce_back_cells
from tensorlbm.color_gradient_two_pdf import (
    colorgradient_two_pdf_step,
    pressure_jump,
    recolor_weights,
    relative_l2,
    tau_from_phi,
    two_pdf_macros,
    viscosity_from_phi,
)
from tensorlbm.d2q9 import C, equilibrium
from tensorlbm.solver import stream

DT = torch.float64


def _random_state(ny=12, nx=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=DT).view(9, 1, 1)
    rho_a = 0.5 + 0.4 * torch.rand((ny, nx), dtype=DT, generator=g)
    rho_b = 0.5 + 0.4 * torch.rand((ny, nx), dtype=DT, generator=g)
    ux = 1e-3 * torch.randn((ny, nx), dtype=DT, generator=g)
    uy = 1e-3 * torch.randn((ny, nx), dtype=DT, generator=g)
    f = w * rho_a.unsqueeze(0) * (1.0 + 3.0 * ux.unsqueeze(0))
    g_ = w * rho_b.unsqueeze(0) * (1.0 + 3.0 * uy.unsqueeze(0))
    return f, g_


def test_viscosity_and_tau_from_phi():
    phi = torch.tensor([-1.0, 0.0, 1.0], dtype=DT)
    nu = viscosity_from_phi(phi, 2.0 / 3.0, 1.0 / 15.0)
    assert abs(float(nu[2]) - 2.0 / 3.0) < 1e-14
    assert abs(float(nu[0]) - 1.0 / 15.0) < 1e-14
    harmonic = 2.0 / (1.0 / (2.0 / 3.0) + 1.0 / (1.0 / 15.0))
    assert abs(float(nu[1]) - harmonic) < 1e-14
    tau = tau_from_phi(phi, 2.0 / 3.0, 1.0 / 15.0)
    assert torch.allclose(tau, 3.0 * nu + 0.5)
    with pytest.raises(ValueError):
        viscosity_from_phi(phi, 0.0, 1.0)


def test_macros_roundtrip():
    f, g = _random_state()
    rho_a, rho_b, rho, phi, ux, uy = two_pdf_macros(f, g)
    assert torch.allclose(rho_a, f.sum(dim=0))
    assert torch.allclose(rho_b, g.sum(dim=0))
    assert torch.allclose(rho, rho_a + rho_b)
    assert torch.allclose(phi, (rho_a - rho_b) / rho)
    c = C.to(DT)
    jx = ((f + g) * c[:, 0].view(9, 1, 1)).sum(dim=0)
    assert torch.allclose(ux * rho, jx)
    assert float(phi.abs().max()) <= 1.0 + 1e-12


def test_partial_density_mass_conserved_per_step():
    f, g = _random_state(ny=16, nx=8, seed=3)
    m_a0, m_b0 = float(f.sum()), float(g.sum())
    m_a, m_b = m_a0, m_b0
    for _ in range(5):
        fo, go = colorgradient_two_pdf_step(f, g, nu_a=0.2, nu_b=1.0 / 30.0, A=0.04,
                                            beta=0.9, gx=1e-6)
        m_a, m_b = float(fo.sum()), float(go.sum())
        f, g = fo, go
    assert abs(m_a - m_a0) / abs(m_a0) < 1e-13
    assert abs(m_b - m_b0) / abs(m_b0) < 1e-13


def test_recolor_stencil_is_mass_free():
    f, _ = _random_state()
    w_rec = recolor_weights(f)
    assert w_rec.shape == (9,)
    assert float(w_rec[0]) == 0.0
    c = C.to(DT)
    for nvec in [(1.0, 0.0), (0.0, 1.0), (0.6, 0.8), (-0.28, 0.96)]:
        n = torch.tensor(nvec, dtype=DT)
        proj = (c * n).sum(dim=1)
        assert abs(float((w_rec * proj).sum())) < 1e-15
    # isotropic weights too
    w_iso = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=DT)
    n = torch.tensor([0.6, 0.8], dtype=DT)
    assert abs(float((w_iso * (c * n).sum(dim=1)).sum())) < 1e-15


def test_surface_tension_perturbation_mass_and_momentum_free():
    ny, nx = 32, 8
    j = torch.arange(ny, dtype=DT).view(ny, 1)
    fa = 0.5 * (1.0 - torch.tanh((j - ny // 2) / 2.0)).expand(ny, nx).contiguous()
    z = torch.zeros((ny, nx), dtype=DT)
    f, g = equilibrium(fa, z, z), equilibrium(1.0 - fa, z, z)
    # with A=0 vs A=0.04 and beta=0 the *only* difference is the perturbation
    f0, g0 = colorgradient_two_pdf_step(f, g, A=0.0, beta=0.0)
    f1, g1 = colorgradient_two_pdf_step(f, g, A=0.04, beta=0.0)
    d_f, d_g = f1 - f0, g1 - g0
    assert abs(float(d_f.sum()) + float(d_g.sum())) < 1e-15
    c = C.to(DT)
    for d in (d_f, d_g):
        mom = (d * c[:, 0].view(9, 1, 1)).sum(dim=0)
        assert float(mom.abs().max()) < 1e-15


def test_single_phase_limit_matches_poiseuille():
    ny, nx, steps = 32, 8, 8000
    gx, nu = 5e-6, 1.0 / 6.0
    z = torch.zeros((ny, nx), dtype=DT)
    f = equilibrium(torch.ones((ny, nx), dtype=DT), z, z)
    g = equilibrium(torch.zeros((ny, nx), dtype=DT), z, z)
    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = wall[-1, :] = True
    for _ in range(steps):
        f, g = colorgradient_two_pdf_step(f, g, nu_a=nu, nu_b=nu, A=0.04, beta=0.9,
                                          gx=gx, solid_mask=wall)
        f, g = bounce_back_cells(stream(f), wall), bounce_back_cells(stream(g), wall)
    _, _, _, _, ux, _ = two_pdf_macros(f, g)
    W = float(ny - 2)
    Y = torch.arange(1, ny - 1, dtype=DT) - 0.5
    ana = gx * Y * (W - Y) / (2.0 * nu)
    rel = relative_l2(ux[1:-1, nx // 2], ana)
    assert rel < 0.01, f"single-phase Poiseuille relL2={rel:.3%} exceeds 1%"


def test_recolor_keeps_partial_densities_non_negative():
    """The production stencil (``recolor="iso"``) keeps both partial densities
    non-negative over an advecting interface.  The literal MF-LBM ``4/9``
    stencil is rejected on this evidence: with the same amplitude scaling it
    drives the minority partial density negative (measured min rho_sigma
    -0.11 at ny=32 after 1000 steps; recorded in
    ``xval/XVAL-LBM-TWO-PDF-PARTIAL-DENSITY-COLORGRADIENT/artifacts``)."""
    ny, nx = 32, 4
    j = torch.arange(ny, dtype=DT).view(ny, 1)
    fa = 0.5 * (1.0 - torch.tanh((j - ny // 2) / 2.0)).expand(ny, nx).contiguous()
    z = torch.zeros((ny, nx), dtype=DT)
    f, g = equilibrium(fa, z, z), equilibrium(1.0 - fa, z, z)
    for _ in range(1200):
        f, g = colorgradient_two_pdf_step(f, g, A=0.04, beta=0.9, recolor="iso")
    rho_a, rho_b, _rho, _phi, _ux, _uy = two_pdf_macros(f, g)
    assert min(float(rho_a.min()), float(rho_b.min())) > -1e-12
    assert float((rho_a + rho_b).sum()) > 0.0
