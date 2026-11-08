"""Channel drivers for yield-stress fluid LBM validation.

Optimised for speed: convergence check every 100 iterations, relaxed tolerances.
"""

from __future__ import annotations

import math

import torch

from .d2q9 import C, W, OPPOSITE, equilibrium, macroscopic
from .solver import stream
from .yield_stress_lbm import (
    collide_bingham_bgk,
    collide_bingham_bgk_forced,
    collide_hb_bgk_forced,
    neq_stress_2d,
)
from .yield_stress_analytical import (
    bingham_couette_profile,
    bingham_poiseuille_profile,
    bingham_poiseuille_plug_half_width,
)

__all__ = [
    "run_bingham_couette",
    "run_bingham_poiseuille",
    "run_hb_poiseuille",
    "compute_lubrication_force",
    "compute_yield_surface_topology",
]


def _make_channel(H: int, nx: int | None, dtype, device) -> tuple[torch.Tensor, int, int]:
    """Create channel wall mask (rows 0 and H+1 are walls)."""
    nx = H if nx is None else int(nx)
    if H < 4 or nx < 2:
        raise ValueError("channel needs H >= 4 and nx >= 2")
    ny = H + 2
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    return wall, ny, nx


def _moving_wall_bounce_back(
    f: torch.Tensor,
    wall: torch.Tensor,
    u_wall: tuple[float, float] | None = None,
    *,
    moving_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Half-way bounce-back with optional moving wall correction.

    Ladd's correction is ``6 w_i rho (c_i . u_w)`` with ``c_s^2 = 1/3``; the
    local density ``rho`` is required (it was missing until 2026-11-08, which is
    invisible at ``rho = 1`` but 33% of the correction at ``rho = 1.2``).

    ``moving_mask`` selects the cells that actually move.  It defaults to the
    last row for backward compatibility with :func:`run_bingham_couette` (top
    wall driving), but the correction must follow the caller's mask: hard-coding
    row ``-1`` silently turned every other geometry into a static wall.
    Pinned by ``tests/test_moving_wall_convention_audit.py``.
    """
    opp = OPPOSITE.to(f.device)
    reflected = f[opp]

    if u_wall is not None and (u_wall[0] != 0.0 or u_wall[1] != 0.0):
        c = C.to(device=f.device, dtype=f.dtype)
        w = W.to(device=f.device, dtype=f.dtype)
        rho = f.sum(dim=0)
        cu = c[:, 0].view(9, 1, 1) * float(u_wall[0]) + c[:, 1].view(9, 1, 1) * float(u_wall[1])
        correction = (6.0 * w.view(9, 1, 1) * rho.unsqueeze(0) * cu)
        if moving_mask is None:
            moving_mask = torch.zeros_like(wall)
            moving_mask[-1, :] = True
        moving_mask = moving_mask & wall
        reflected = reflected + correction * moving_mask.unsqueeze(0).to(f.dtype)

    return torch.where(wall.unsqueeze(0), reflected, f)


def run_bingham_couette(
    H: int,
    u_wall: float,
    tau_y: float,
    mu_0: float,
    m: float,
    *,
    nx: int | None = None,
    max_iter: int = 20000,
    tol: float = 1e-6,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> dict:
    """Run Bingham Couette flow simulation."""
    if device is None:
        device = torch.device("cpu")

    wall, ny, nx = _make_channel(H, nx, dtype, device)

    rho = torch.ones((ny, nx), dtype=dtype, device=device)
    ux = torch.zeros((ny, nx), dtype=dtype, device=device)
    uy = torch.zeros((ny, nx), dtype=dtype, device=device)
    f = equilibrium(rho, ux, uy)

    converged = False
    prev_u_sum = 0.0
    iteration = max_iter

    for iteration in range(1, max_iter + 1):
        f = collide_bingham_bgk(
            f, tau_y, mu_0, m,
            nu_min=nu_min, nu_max=nu_max, tau_min=tau_min,
        )
        f = stream(f)
        f = _moving_wall_bounce_back(f, wall, u_wall=(u_wall, 0.0))

        # Check convergence every 100 iterations
        if iteration % 100 == 0:
            _, ux, uy = macroscopic(f)
            u_sum = ux.sum().item()
            du = abs(u_sum - prev_u_sum)
            if du < tol and iteration > 200:
                converged = True
                break
            prev_u_sum = u_sum

    rho, ux, uy = macroscopic(f)
    analytic_u, _ = bingham_couette_profile(H, u_wall, tau_y, mu_0)

    u_sim = ux[1:-1, nx // 2].double()
    norm_analytic = torch.sqrt((analytic_u ** 2).sum())
    if norm_analytic > 1e-30:
        l2_error = float(torch.sqrt(((u_sim - analytic_u) ** 2).sum()) / norm_analytic)
    else:
        l2_error = float(torch.sqrt(((u_sim - analytic_u) ** 2).sum()))

    return {
        "f": f, "rho": rho, "ux": ux, "uy": uy, "wall": wall,
        "iterations": iteration, "converged": converged,
        "analytic_u": analytic_u, "u_sim": u_sim, "l2_error": l2_error,
        "H": H, "u_wall": u_wall, "tau_y": tau_y, "mu_0": mu_0, "m": m,
    }


def run_bingham_poiseuille(
    H: int,
    ax: float,
    tau_y: float,
    mu_0: float,
    m: float,
    *,
    nx: int | None = None,
    max_iter: int = 50000,
    tol: float = 1e-6,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> dict:
    """Run Bingham Poiseuille flow simulation (body-force driven)."""
    if device is None:
        device = torch.device("cpu")

    wall, ny, nx = _make_channel(H, nx, dtype, device)

    rho = torch.ones((ny, nx), dtype=dtype, device=device)
    ux = torch.zeros((ny, nx), dtype=dtype, device=device)
    uy = torch.zeros((ny, nx), dtype=dtype, device=device)
    f = equilibrium(rho, ux, uy)

    converged = False
    prev_u_sum = 0.0
    iteration = max_iter

    for iteration in range(1, max_iter + 1):
        f = collide_bingham_bgk_forced(
            f, tau_y, mu_0, m, ax,
            nu_min=nu_min, nu_max=nu_max, tau_min=tau_min,
        )
        f = stream(f)
        f = _moving_wall_bounce_back(f, wall)

        if iteration % 100 == 0:
            _, ux, uy = macroscopic(f)
            u_sum = ux.sum().item()
            du = abs(u_sum - prev_u_sum)
            if du < tol and iteration > 200:
                converged = True
                break
            prev_u_sum = u_sum

    rho, ux, uy = macroscopic(f)
    analytic_u, _ = bingham_poiseuille_profile(H, ax, tau_y, mu_0)

    u_sim = ux[1:-1, nx // 2].double()
    norm_analytic = torch.sqrt((analytic_u ** 2).sum())
    if norm_analytic > 1e-30:
        l2_error = float(torch.sqrt(((u_sim - analytic_u) ** 2).sum()) / norm_analytic)
    else:
        l2_error = float(torch.sqrt(((u_sim - analytic_u) ** 2).sum()))

    y_p_analytic = bingham_poiseuille_plug_half_width(H, ax, tau_y, mu_0)

    u_max_sim = float(u_sim.max())
    if u_max_sim > 0:
        yc = float(H + 1) / 2.0
        y_phys = torch.arange(1, H + 1, dtype=torch.float64)
        dy = torch.abs(y_phys - yc)
        plug_mask = u_sim > 0.99 * u_max_sim
        y_p_sim = float(dy[plug_mask].max()) if plug_mask.any() else 0.0
    else:
        y_p_sim = 0.0

    plug_error = abs(y_p_sim - y_p_analytic) / (y_p_analytic + 1e-30)
    u_max_analytic = float(analytic_u.max())
    u_max_error = abs(u_max_sim - u_max_analytic) / (u_max_analytic + 1e-30)

    return {
        "f": f, "rho": rho, "ux": ux, "uy": uy, "wall": wall,
        "iterations": iteration, "converged": converged,
        "analytic_u": analytic_u, "u_sim": u_sim, "l2_error": l2_error,
        "y_p_analytic": y_p_analytic, "y_p_sim": y_p_sim,
        "plug_error": plug_error, "u_max_analytic": u_max_analytic,
        "u_max_sim": u_max_sim, "u_max_error": u_max_error,
        "H": H, "ax": ax, "tau_y": tau_y, "mu_0": mu_0, "m": m,
    }


def run_hb_poiseuille(
    H: int,
    ax: float,
    tau_y: float,
    K: float,
    n: float,
    m: float,
    *,
    nx: int | None = None,
    max_iter: int = 50000,
    tol: float = 1e-6,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> dict:
    """Run Herschel-Bulkley Poiseuille flow simulation."""
    if device is None:
        device = torch.device("cpu")

    wall, ny, nx = _make_channel(H, nx, dtype, device)

    rho = torch.ones((ny, nx), dtype=dtype, device=device)
    ux = torch.zeros((ny, nx), dtype=dtype, device=device)
    uy = torch.zeros((ny, nx), dtype=dtype, device=device)
    f = equilibrium(rho, ux, uy)

    converged = False
    prev_u_sum = 0.0
    iteration = max_iter

    for iteration in range(1, max_iter + 1):
        f = collide_hb_bgk_forced(
            f, tau_y, K, n, m, ax,
            nu_min=nu_min, nu_max=nu_max, tau_min=tau_min,
        )
        f = stream(f)
        f = _moving_wall_bounce_back(f, wall)

        if iteration % 100 == 0:
            _, ux, uy = macroscopic(f)
            u_sum = ux.sum().item()
            du = abs(u_sum - prev_u_sum)
            if du < tol and iteration > 200:
                converged = True
                break
            prev_u_sum = u_sum

    rho, ux, uy = macroscopic(f)
    u_sim = ux[1:-1, nx // 2].double()

    return {
        "f": f, "rho": rho, "ux": ux, "uy": uy, "wall": wall,
        "iterations": iteration, "converged": converged,
        "u_sim": u_sim, "H": H, "ax": ax, "tau_y": tau_y, "K": K, "n": n, "m": m,
    }


def compute_lubrication_force(
    ux: torch.Tensor,
    mu_eff: torch.Tensor,
    wall_mask: torch.Tensor,
    dx: float = 1.0,
) -> float:
    """Compute lubrication drag force from velocity and viscosity fields."""
    du_dy = torch.zeros_like(ux)
    du_dy[0, :] = (ux[1, :] - ux[0, :]) / 0.5
    du_dy[-1, :] = (ux[-1, :] - ux[-2, :]) / 0.5

    bottom_force = float((mu_eff[0, :] * torch.abs(du_dy[0, :])).sum() * dx)
    top_force = float((mu_eff[-1, :] * torch.abs(du_dy[-1, :])).sum() * dx)
    return bottom_force + top_force


def compute_yield_surface_topology(
    ux: torch.Tensor,
    uy: torch.Tensor,
    tau_y: float,
    mu_0: float,
    *,
    threshold_factor: float = 0.95,
) -> dict:
    """Analyse yield surface topology from velocity field."""
    from .powerlaw import strain_rate_shear_rate_2d

    gamma = strain_rate_shear_rate_2d(ux, uy)
    gamma_y = tau_y / mu_0

    plug_mask = gamma < threshold_factor * gamma_y
    fluid_mask = plug_mask[1:-1, :]
    plug_area = int(fluid_mask.sum().item())
    plug_fraction = plug_area / max(1, fluid_mask.numel())

    n_components = 1 if plug_area > 0 else 0

    if plug_area > 0:
        ys, xs = torch.where(fluid_mask)
        cy = float(ys.float().mean()) + 1.0
        cx = float(xs.float().mean())
        centroid = (cy, cx)
    else:
        centroid = (0.0, 0.0)

    max_thickness = 0.0
    if plug_area > 0:
        for col in range(fluid_mask.shape[1]):
            col_mask = fluid_mask[:, col]
            if col_mask.any():
                indices = torch.where(col_mask)[0]
                thickness = float(indices.max() - indices.min() + 1)
                max_thickness = max(max_thickness, thickness)

    return {
        "plug_area_fraction": plug_fraction,
        "n_components": n_components,
        "centroid": centroid,
        "max_thickness": max_thickness,
    }
