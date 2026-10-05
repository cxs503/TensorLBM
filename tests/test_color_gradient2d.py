"""Smoke tests for the two-tau color-gradient engine.

Covers the invariants the verified two-layer Poiseuille case
(benchmarks/verified/poiseuille_two_phase_cg) relies on: per-color mass
conservation at machine precision across the collision/recolor step,
NaN-free evolution for the equal-viscosity and 10:1 viscosity-contrast
geometries, and driven-flow sanity.
"""

import torch

from tensorlbm.boundaries import bounce_back_cells
from tensorlbm.color_gradient2d import color_gradient_two_tau_step
from tensorlbm.d2q9 import C, equilibrium
from tensorlbm.solver import stream


def _run_channel(ny=32, nx=8, tau_r=1.0, tau_b=1.0, gx=5e-6, steps=400):
    half = ny // 2
    j = torch.arange(ny, dtype=torch.float64).view(ny, 1)
    frac_r = 0.5 * (1.0 - torch.tanh((j - half) / 2.0)).expand(ny, nx).contiguous()
    z = torch.zeros((ny, nx), dtype=torch.float64)
    f_r = equilibrium(1.0 * frac_r, z, z)
    f_b = equilibrium(1.0 * (1.0 - frac_r), z, z)
    m_r0, m_b0 = float(f_r.sum()), float(f_b.sum())
    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True
    for _ in range(steps):
        f_r, f_b = color_gradient_two_tau_step(
            f_r,
            f_b,
            tau_r=tau_r,
            tau_b=tau_b,
            A=0.04,
            beta=0.9,
            gx=gx,
            solid_mask=wall,
        )
        f_r = bounce_back_cells(stream(f_r), wall)
        f_b = bounce_back_cells(stream(f_b), wall)
        assert torch.isfinite(f_r).all() and torch.isfinite(f_b).all()
    return f_r, f_b, m_r0, m_b0


def _drift_per_step(f, m0, steps):
    return abs(float(f.sum()) - m0) / steps


def test_mass_conservation_equal_viscosity():
    f_r, f_b, m_r0, m_b0 = _run_channel()
    assert _drift_per_step(f_r, m_r0, 400) < 1e-12
    assert _drift_per_step(f_b, m_b0, 400) < 1e-12


def test_mass_conservation_viscosity_contrast():
    f_r, f_b, m_r0, m_b0 = _run_channel(tau_r=2.5, tau_b=0.7, gx=1.5e-7)
    assert _drift_per_step(f_r, m_r0, 400) < 1e-12
    assert _drift_per_step(f_b, m_b0, 400) < 1e-12


def test_driven_flow_sanity():
    f_r, f_b, _, _ = _run_channel()
    f = f_r + f_b
    ux = (f * C.to(torch.float64)[:, 0].view(9, 1, 1)).sum(dim=0) / f.sum(dim=0)
    assert float(ux[1:-1].mean()) > 0.0
