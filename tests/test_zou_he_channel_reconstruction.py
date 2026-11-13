"""Zou/He channel inlet/outlet reconstruction — closure + physics regression.

Anchor: ``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION``.

Two classes of assertions:

* **Closure (exact)** — after applying the boundary condition to an arbitrary
  non-equilibrium population field, the boundary column must reproduce the
  prescribed macroscopic quantities to round-off.  This is an algebraic
  invariant of the reconstruction (Zou & He 1997, sec. II), not a tolerance
  choice, so the bound is ``1e-12``.
* **Physics** — a channel driven by a Zou/He pressure inlet + pressure outlet
  must develop the analytic plane Poiseuille profile
  ``u(y) = G y (H - y) / (2 nu)``, ``G = dp/dx = (rho_in - rho_out)/(3 L)``,
  with the amplitude predicted *a priori* (no fitting), to better than the
  owner's 15 % standard.
"""

from __future__ import annotations

import torch

from tensorlbm.boundary.boundaries import (
    apply_zou_he_pressure_channel_boundaries,
    zou_he_inlet_pressure,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

DTYPE = torch.float64


def _random_field(ny: int = 18, nx: int = 12, seed: int = 19970317) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand((9, ny, nx), generator=gen, dtype=DTYPE) + 0.1


def test_inlet_velocity_reconstruction_is_exact() -> None:
    """``sum f``, ``sum f cx``, ``sum f cy`` match (rho, ux, uy) to round-off."""
    f = _random_field()
    ny = f.shape[1]
    u_in = torch.linspace(0.02, 0.09, ny, dtype=DTYPE)
    uy_in = torch.linspace(-0.02, 0.02, ny, dtype=DTYPE)

    g = zou_he_inlet_velocity(f, u_in, uy_in)
    rho, ux, uy = macroscopic(g)

    f0, f2, f4 = f[0, :, 0], f[2, :, 0], f[4, :, 0]
    f3, f6, f7 = f[3, :, 0], f[6, :, 0], f[7, :, 0]
    rho_target = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_in)

    assert torch.allclose(rho[:, 0], rho_target, rtol=1e-12, atol=1e-13)
    assert torch.allclose(ux[:, 0], u_in, rtol=1e-12, atol=1e-13)
    assert torch.allclose(uy[:, 0], uy_in, rtol=0, atol=1e-13)


def test_inlet_pressure_reconstruction_is_exact() -> None:
    """West pressure inlet: rho prescribed, ux implied by the mass constraint."""
    f = _random_field()
    ny = f.shape[1]
    rho_in = 1.004
    uy_in = torch.linspace(-0.01, 0.01, ny, dtype=DTYPE)

    g = zou_he_inlet_pressure(f, rho_in, uy_in)
    rho, ux, uy = macroscopic(g)

    f0, f2, f4 = f[0, :, 0], f[2, :, 0], f[4, :, 0]
    f3, f6, f7 = f[3, :, 0], f[6, :, 0], f[7, :, 0]
    ux_expected = 1.0 - (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / rho_in

    assert torch.allclose(rho[:, 0], torch.full_like(rho[:, 0], rho_in), rtol=1e-12, atol=1e-13)
    assert torch.allclose(ux[:, 0], ux_expected, rtol=0, atol=1e-13)
    assert torch.allclose(uy[:, 0], uy_in, rtol=0, atol=1e-13)


def test_outlet_pressure_reconstruction_is_exact() -> None:
    """East pressure outlet: rho prescribed, uy=0 by default and free if given."""
    f = _random_field()
    ny = f.shape[1]
    rho_out = 0.996

    for uy_out in (0.0, 0.013):
        g = zou_he_outlet_pressure(f, rho_out, uy_out)
        rho, _ux, uy = macroscopic(g)
        assert torch.allclose(
            rho[:, -1], torch.full_like(rho[:, -1], rho_out), rtol=1e-12, atol=1e-13
        )
        target = torch.full_like(uy[:, -1], float(uy_out))
        assert torch.allclose(uy[:, -1], target, rtol=0, atol=1e-13)


def test_outlet_pressure_default_matches_historical_formula() -> None:
    """``uy_out=0`` stays bit-compatible with the pre-extension reconstruction."""
    f = _random_field()
    rho_out = 1.0
    got = zou_he_outlet_pressure(f, rho_out)

    f1, f2, f4, f5, f8 = f[1, :, -1], f[2, :, -1], f[4, :, -1], f[5, :, -1], f[8, :, -1]
    ux = -1.0 + (f[0, :, -1] + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_out
    ref = f.clone()
    ref[3, :, -1] = f1 - (2.0 / 3.0) * rho_out * ux
    ref[7, :, -1] = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux
    ref[6, :, -1] = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux

    assert torch.allclose(got, ref, rtol=0, atol=1e-15)


def test_known_populations_are_not_touched() -> None:
    """Only the unknown (in-flowing / out-flowing) populations are rewritten."""
    f = _random_field()
    g_in = zou_he_inlet_velocity(f, 0.05, 0.0)
    for q in (0, 2, 3, 4, 6, 7):
        assert torch.equal(g_in[q, :, 0], f[q, :, 0])
    g_out = zou_he_outlet_pressure(f, 1.0)
    for q in (0, 1, 2, 4, 5, 8):
        assert torch.equal(g_out[q, :, -1], f[q, :, -1])


def _poiseuille(y: torch.Tensor, h: float, u_max: float) -> torch.Tensor:
    s = y / h
    return u_max * 4.0 * s * (1.0 - s)


def test_pressure_driven_channel_matches_analytic_poiseuille() -> None:
    """Pressure inlet + pressure outlet reproduce the Poiseuille parabola."""
    nx, ny, tau = 60, 12, 0.8
    nu = (tau - 0.5) / 3.0
    h = ny - 1  # full-way bounce-back: no-slip plane at the wall node centre
    d_rho = 0.004
    rho_in, rho_out = 1.0 + d_rho / 2, 1.0 - d_rho / 2
    g_force = d_rho / 3.0 / (nx - 1)
    u_max = g_force * h * h / (8.0 * nu)

    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True
    obstacle = torch.zeros_like(wall)

    rho = torch.ones((ny, nx), dtype=DTYPE)
    zeros = torch.zeros_like(rho)
    f = equilibrium(rho, zeros, zeros)
    for _ in range(6000):
        f = collide_bgk(f, tau)
        f = stream(f)
        f = apply_zou_he_pressure_channel_boundaries(f, rho_in, rho_out, wall, obstacle)

    _, ux, _ = macroscopic(f)
    col = nx // 2
    y = torch.arange(1, ny - 1, dtype=DTYPE)
    ana = _poiseuille(y, h, u_max)
    got = ux[1:-1, col]
    dp_max = float((got - ana).abs().max() / ana.abs().max())

    assert torch.isfinite(got).all()
    assert dp_max < 0.15, f"dp_max={dp_max:.4f} exceeds the 15 % standard"
    # mass conservation across the channel cross-section
    q_in = float((macroscopic(f)[0][1:-1, 0] * ux[1:-1, 0]).sum())
    q_out = float((macroscopic(f)[0][1:-1, -1] * ux[1:-1, -1]).sum())
    assert abs(q_out - q_in) / abs(q_in) < 0.01