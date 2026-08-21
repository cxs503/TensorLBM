"""Tests for the power-law (generalised Newtonian) D2Q9 model.

Covers:
1. central-difference velocity gradients (exact on linear fields),
2. shear rate γ̇ = sqrt(2·S:S) for a 1-D channel profile,
3. power-law viscosity ν(γ̇) = K·γ̇^(n-1) (n=1 Newtonian limit, monotonicity,
   clamping),
4. collision: mass conservation, shape, and the n=1 ⇒ BGK limit,
5. end-to-end channel flow:
   - n=1 reproduces the Newtonian Poiseuille profile
     u(y) = fₓ/(2ν)·(y−0.5)·(ny−1.5−(y−0.5))  (half-way BB walls at 0.5/ny−1.5),
   - n=0.5 (shear-thinning) and n=1.5 (shear-thickening) match the analytic
     power-law profile u(y) = U_max·(1 − (|y−y_c|/Hh)^((n+1)/n)).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
from tensorlbm.powerlaw import (
    central_difference,
    collide_powerlaw_bgk,
    collide_powerlaw_bgk_forced,
    powerlaw_viscosity,
    strain_rate_shear_rate_2d,
    tau_from_viscosity,
)
from tensorlbm.solver import collide_bgk, stream

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_powerlaw_channel(
    H: int,
    n: float,
    K: float,
    fx: float,
    n_steps: int,
    nu_min: float = 1e-5,
    nu_max: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a body-force-driven power-law channel flow and return (ux, tau_eff).

    ny = H + 2, walls at rows 0 and ny-1 (pre-streaming half-way bounce-back),
    x periodic.  Returns the final ux field (ny, nx) and the last τ_eff field.
    """
    torch.manual_seed(0)
    ny, nx = H + 2, H
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall[0, :] = True
    wall[-1, :] = True
    opp = OPPOSITE.to(DEVICE)

    f = equilibrium(
        torch.ones((ny, nx), device=DEVICE),
        torch.zeros((ny, nx), device=DEVICE),
        torch.zeros((ny, nx), device=DEVICE),
    )

    tau_eff = torch.full((ny, nx), 3.0 * K + 0.5, device=DEVICE)
    f_pre = f.clone()
    col = H // 2
    hist = []
    steady = False
    for step in range(1, n_steps + 1):
        f = collide_powerlaw_bgk_forced(
            f, fx, K, n, nu_min=nu_min, nu_max=nu_max, tau_field=tau_eff
        )
        f = torch.where(wall.unsqueeze(0), f_pre[opp], f)  # pre-streaming BB
        f = stream(f)
        rho, ux, uy = macroscopic(f)
        gamma = strain_rate_shear_rate_2d(ux, uy)
        nu = powerlaw_viscosity(gamma, K, n, nu_min=nu_min, nu_max=nu_max)
        tau_eff = tau_from_viscosity(nu)
        f_pre = f.clone()
        if step % 200 == 0:
            _, ux2, _ = macroscopic(f)
            hist.append(float(ux2[ny // 2, col].item()))
            if step >= min(n_steps // 2, 4000) and len(hist) >= 10:
                recent = hist[-10:]
                mean = sum(recent) / len(recent)
                drift = (max(recent) - min(recent)) / max(abs(mean), 1e-12)
                if drift < 1e-5:
                    steady = True
                    break

    # Time-average the centre-column profile over the last 100 steps
    _, ux_f, _ = macroscopic(f)
    if steady:
        profs = []
        for _ in range(100):
            f = collide_powerlaw_bgk_forced(
                f, fx, K, n, nu_min=nu_min, nu_max=nu_max, tau_field=tau_eff
            )
            f = torch.where(wall.unsqueeze(0), f_pre[opp], f)
            f = stream(f)
            _, ux_a, _ = macroscopic(f)
            profs.append(ux_a[:, col].clone())
            rho, ux, uy = macroscopic(f)
            gamma = strain_rate_shear_rate_2d(ux, uy)
            nu = powerlaw_viscosity(gamma, K, n, nu_min=nu_min, nu_max=nu_max)
            tau_eff = tau_from_viscosity(nu)
            f_pre = f.clone()
        ux_avg = torch.stack(profs).mean(0)
        ux_f = ux_avg.unsqueeze(1).expand(ny, nx)

    return ux_f, tau_eff


def _powerlaw_analytic_profile(H: int, n: float, U_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Fluid-cell analytic profile and the corresponding y coordinates.

    Walls at y = 0.5 and y = ny-1.5 (half-way BB), centreline at
    y_c = (ny-1)/2, half-height Hh = (ny-2)/2.  Fluid cell centres are
    y = 1 .. ny-2 (i.e. y_phys = y - 0.5 in [0.5, H+0.5]).
    """
    ny = H + 2
    Hh = H / 2.0  # effective half-height: walls at y_phys=0.5 and H+0.5
    y_c = (ny - 1) / 2.0
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = U_max * (1.0 - (np.abs(y_phys - y_c) / Hh) ** ((n + 1.0) / n))
    return u_ana, y_phys


# ---------------------------------------------------------------------------
# 1. central differences
# ---------------------------------------------------------------------------


def test_central_difference_linear_exact():
    ny, nx = 10, 12
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32), torch.arange(nx, dtype=torch.float32), indexing="ij"
    )
    a, b, c = 0.3, -0.2, 1.5
    field = a * yy + b * xx + c
    d_dy = central_difference(field, 0)
    d_dx = central_difference(field, 1)
    assert torch.allclose(d_dy, torch.full_like(field, a), atol=1e-6)
    assert torch.allclose(d_dx, torch.full_like(field, b), atol=1e-6)


def test_central_difference_quadratic_second_order():
    ny = 20
    yy = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, 3)
    field = yy**2
    d = central_difference(field, 0)
    # interior: exact 2y
    interior = d[1:-1, 0]
    exact = 2.0 * yy[1:-1, 0]
    assert torch.allclose(interior, exact, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. shear rate
# ---------------------------------------------------------------------------


def test_shear_rate_channel_profile():
    """For a 1-D profile uₓ = c·y², γ̇ = |∂uₓ/∂y| = 2c|y| in the interior."""
    ny = 20
    yy = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, 4)
    ux = 0.5 * yy**2
    uy = torch.zeros_like(ux)
    gamma = strain_rate_shear_rate_2d(ux, uy)
    exact = torch.abs(yy)
    assert torch.allclose(gamma[1:-1], exact[1:-1], atol=1e-5)


def test_shear_rate_linear_plug_zero():
    """Uniform velocity → zero shear rate everywhere."""
    ux = torch.full((8, 8), 0.05)
    uy = torch.zeros((8, 8))
    gamma = strain_rate_shear_rate_2d(ux, uy)
    assert float(gamma.max().item()) < 1e-12


# ---------------------------------------------------------------------------
# 3. power-law viscosity
# ---------------------------------------------------------------------------


def test_powerlaw_viscosity_newtonian_limit():
    gamma = torch.tensor([0.0, 1e-6, 0.01, 1.0])
    nu = powerlaw_viscosity(gamma, consistency_index=0.05, flow_index=1.0)
    assert torch.allclose(nu, torch.full_like(gamma, 0.05), atol=1e-8)


def test_powerlaw_viscosity_monotonicity():
    gamma = torch.tensor([1e-3, 1e-2, 1e-1])
    nu_thin = powerlaw_viscosity(gamma, consistency_index=0.01, flow_index=0.5)
    nu_thick = powerlaw_viscosity(gamma, consistency_index=1.0, flow_index=1.5)
    # shear-thinning: ν decreases with γ̇
    assert float(nu_thin[0]) > float(nu_thin[-1])
    # shear-thickening: ν increases with γ̇
    assert float(nu_thick[0]) < float(nu_thick[-1])
    # exact values ν = K·γ̇^(n-1)
    assert float(nu_thin[1].item()) == pytest.approx(0.01 * 1e-2**-0.5, rel=1e-6)
    assert float(nu_thick[1].item()) == pytest.approx(1.0 * 1e-2**0.5, rel=1e-6)


def test_powerlaw_viscosity_clamping():
    gamma = torch.tensor([1e-12, 1.0])
    nu = powerlaw_viscosity(gamma, consistency_index=1.0, flow_index=0.5, nu_min=0.01, nu_max=0.3)
    assert float(nu.min().item()) >= 0.01 - 1e-6
    assert float(nu.max().item()) <= 0.3 + 1e-6


def test_powerlaw_viscosity_validation():
    with pytest.raises(ValueError):
        powerlaw_viscosity(torch.ones(4), consistency_index=0.0, flow_index=1.0)
    with pytest.raises(ValueError):
        powerlaw_viscosity(torch.ones(4), consistency_index=1.0, flow_index=0.0)
    with pytest.raises(ValueError):
        tau_from_viscosity(torch.ones(4), tau_min=0.5)


# ---------------------------------------------------------------------------
# 4. collision
# ---------------------------------------------------------------------------


def test_collide_powerlaw_bgk_newtonian_limit_matches_bgk():
    torch.manual_seed(1)
    ny, nx = 16, 12
    rho = torch.ones((ny, nx))
    ux = 0.02 * torch.rand(ny, nx)
    uy = 0.01 * torch.rand(ny, nx)
    f = equilibrium(rho, ux, uy)
    # perturb slightly so the non-equilibrium part is exercised
    f = f + 0.001 * torch.randn_like(f)

    K = 0.05
    f_pl = collide_powerlaw_bgk(f, consistency_index=K, flow_index=1.0)
    f_bgk = collide_bgk(f, tau=3.0 * K + 0.5)
    assert torch.allclose(f_pl, f_bgk, atol=1e-6)


def test_collide_powerlaw_bgk_mass_and_shape():
    torch.manual_seed(2)
    ny, nx = 12, 10
    rho = torch.ones((ny, nx))
    ux = 0.02 * torch.rand(ny, nx)
    uy = 0.01 * torch.rand(ny, nx)
    f = equilibrium(rho, ux, uy) + 0.001 * torch.randn(9, ny, nx)
    m0 = float(f.sum().item())
    f_out = collide_powerlaw_bgk(f, consistency_index=0.003, flow_index=0.5)
    assert f_out.shape == f.shape
    assert torch.allclose(f_out.sum(), torch.tensor(m0), atol=1e-5)
    assert torch.isfinite(f_out).all()
    # per-cell tau must be > 0.5 everywhere
    _, ux2, uy2 = macroscopic(f)
    gamma = strain_rate_shear_rate_2d(ux2, uy2)
    tau = tau_from_viscosity(powerlaw_viscosity(gamma, 0.003, 0.5))
    assert float(tau.min().item()) > 0.5


def test_collide_powerlaw_bgk_tau_field_equivalent():
    torch.manual_seed(3)
    ny, nx = 12, 10
    f = equilibrium(
        torch.ones(ny, nx), 0.02 * torch.rand(ny, nx), 0.01 * torch.rand(ny, nx)
    ) + 0.001 * torch.randn(9, ny, nx)
    K, n = 0.004, 0.6
    rho, ux, uy = macroscopic(f)
    tau_eff = tau_from_viscosity(powerlaw_viscosity(strain_rate_shear_rate_2d(ux, uy), K, n))
    f_internal = collide_powerlaw_bgk(f, K, n)
    f_external = collide_powerlaw_bgk(f, K, n, tau_field=tau_eff)
    assert torch.allclose(f_internal, f_external, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. end-to-end channel flow
# ---------------------------------------------------------------------------


def test_powerlaw_channel_newtonian_regression():
    """n = 1 must reproduce the Newtonian Poiseuille profile exactly."""
    H, n = 40, 1.0
    U_max, nu_w = 0.03, 0.05
    Hh = H / 2.0
    gamma_w = (n + 1.0) * U_max / (n * Hh)
    K = nu_w * gamma_w ** (1.0 - n)
    fx = nu_w * gamma_w / Hh
    ux, _ = _run_powerlaw_channel(H, n, K, fx, n_steps=40000)
    u_ana, _ = _powerlaw_analytic_profile(H, 1.0, U_max)
    u_num = ux[1:-1, H // 2].cpu().numpy().astype(np.float64)
    l2 = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    assert l2 < 0.05, f"Newtonian regression L2 error too large: {l2:.2e}"


def test_powerlaw_channel_shear_thinning_profile():
    """n = 0.5 (shear-thinning) matches u(y) = U_max·(1 − (|y−y_c|/Hh)³)."""
    H, n = 40, 0.5
    U_max, nu_w = 0.03, 0.05
    Hh = H / 2.0
    gamma_w = (n + 1.0) * U_max / (n * Hh)
    K = nu_w * gamma_w ** (1.0 - n)
    fx = nu_w * gamma_w / Hh
    ux, _ = _run_powerlaw_channel(H, n, K, fx, n_steps=40000)
    u_ana, _ = _powerlaw_analytic_profile(H, n, U_max)
    u_num = ux[1:-1, H // 2].cpu().numpy().astype(np.float64)
    l2 = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    assert l2 < 0.05, f"n=0.5 profile L2 error too large: {l2:.4f}"


def test_powerlaw_channel_shear_thickening_profile():
    """n = 1.5 (shear-thickening) matches u(y) = U_max·(1 − (|y−y_c|/Hh)^(5/3))."""
    H, n = 40, 1.5
    U_max, nu_w = 0.03, 0.05
    Hh = H / 2.0
    gamma_w = (n + 1.0) * U_max / (n * Hh)
    K = nu_w * gamma_w ** (1.0 - n)
    fx = nu_w * gamma_w / Hh
    ux, _ = _run_powerlaw_channel(H, n, K, fx, n_steps=40000)
    u_ana, _ = _powerlaw_analytic_profile(H, n, U_max)
    u_num = ux[1:-1, H // 2].cpu().numpy().astype(np.float64)
    l2 = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    assert l2 < 0.05, f"n=1.5 profile L2 error too large: {l2:.4f}"


def test_powerlaw_channel_profile_shapes_differ_from_newtonian():
    """Sanity: shear-thinning profile is flatter, thickening is rounder."""
    H, fx = 40, 1e-4
    ux_newt, _ = _run_powerlaw_channel(H, 1.0, 0.05, fx, n_steps=40000)
    Hh = H / 2.0
    # n=0.5, same wall viscosity scale
    gamma_w = 1.5 * 0.03 / (0.5 * Hh)
    K = 0.05 * gamma_w**0.5
    fx2 = 0.05 * gamma_w / Hh
    ux_thin, _ = _run_powerlaw_channel(H, 0.5, K, fx2, n_steps=40000)
    ux_thick, _ = _run_powerlaw_channel(H, 1.5, 1.0, 6.25e-6, n_steps=6000)

    def _shape(u):
        col = u[1:-1, H // 2].cpu().numpy()
        return (col - col.min()) / (col.max() - col.min())

    shape_newt, shape_thin, shape_thick = _shape(ux_newt), _shape(ux_thin), _shape(ux_thick)
    y = np.linspace(0.0, 1.0, H)
    # plug-likeness: deviation from linear ramp between wall and centre
    dev_newt = float(np.max(np.abs(shape_newt - y)))
    dev_thin = float(np.max(np.abs(shape_thin - y)))
    dev_thick = float(np.max(np.abs(shape_thick - y)))
    assert dev_thin > dev_newt, "shear-thinning profile should be flatter than Newtonian"
    assert dev_thick < dev_newt, "shear-thickening profile should be rounder than Newtonian"
