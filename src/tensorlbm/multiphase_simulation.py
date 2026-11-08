"""SCMP simulation utilities for thermodynamic verification.

Provides:
- Static droplet initialization and simulation
- Phase separation (bulk) simulation
- Pressure tensor computation for surface tension
- Spurious velocity measurement

All operations are vectorized using torch for performance.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch

from .d2q9 import C, W, equilibrium, macroscopic


# ============================================================================
# Streaming (periodic D2Q9) — vectorized
# ============================================================================

def stream_d2q9(f: torch.Tensor) -> torch.Tensor:
    """Stream D2Q9 distributions with periodic boundaries (vectorized).

    Args:
        f: Post-collision distributions, shape (9, ny, nx).

    Returns:
        Streamed distributions, same shape.
    """
    device = f.device
    c = C.to(device)
    streamed = torch.empty_like(f)
    for i in range(9):
        cy, cx = int(c[i, 1]), int(c[i, 0])
        if cy == 0 and cx == 0:
            streamed[i] = f[i]
        else:
            streamed[i] = torch.roll(f[i], shifts=(cy, cx), dims=(-2, -1))
    return streamed


# ============================================================================
# SCMP neighbor sum — vectorized
# ============================================================================

def sc_neighbor_sum(psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sum_i w_i psi(x + c_i) c_i for SC force (vectorized).

    Uses the backward gather convention: samples psi at x - c_i.

    Args:
        psi: Pseudopotential field, shape (ny, nx).

    Returns:
        (sum_x, sum_y) each of shape (ny, nx).
    """
    device = psi.device
    c = C.to(device)  # (9, 2) int64
    w = W.to(device)  # (9,) float32

    ny, nx = psi.shape

    # Build gather indices for all 9 directions at once
    cy = c[:, 1]  # (9,)
    cx = c[:, 0]  # (9,)

    # Source indices: x - c_i
    y_src = (torch.arange(ny, device=device).unsqueeze(0) - cy.unsqueeze(1)) % ny  # (9, ny)
    x_src = (torch.arange(nx, device=device).unsqueeze(0) - cx.unsqueeze(1)) % nx  # (9, nx)

    # Gather: psi_shifts[i, j, k] = psi[y_src[i,j], x_src[i,k]]
    # Use advanced indexing
    y_idx = y_src.unsqueeze(2)  # (9, ny, 1)
    x_idx = x_src.unsqueeze(1)  # (9, 1, nx)
    psi_shifts = psi[y_idx, x_idx]  # (9, ny, nx)

    # Weighted sum
    cx_f = c[:, 0].float().view(9, 1, 1)  # (9, 1, 1)
    cy_f = c[:, 1].float().view(9, 1, 1)
    w_3d = w.view(9, 1, 1)

    sum_x = (w_3d * cx_f * psi_shifts).sum(0)  # (ny, nx)
    sum_y = (w_3d * cy_f * psi_shifts).sum(0)

    return sum_x, sum_y


def scmp_force(
    rho: torch.Tensor,
    G: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SC single-component interaction force.

    F = -G * psi(rho) * sum_i w_i c_i psi(rho(x + c_i))

    Args:
        rho: Density field, shape (ny, nx).
        G: Coupling constant (G > 0 for attraction).
        psi_fn: Pseudopotential function.

    Returns:
        (Fx, Fy) force components, each shape (ny, nx).
    """
    psi = psi_fn(rho)
    sum_x, sum_y = sc_neighbor_sum(psi)
    Fx = -G * psi * sum_x
    Fy = -G * psi * sum_y
    return Fx, Fy


def collide_scmp(
    f: torch.Tensor,
    G: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor],
    tau: float = 1.0,
) -> torch.Tensor:
    """BGK collision with SC single-component force (velocity-shift forcing).

    Args:
        f: Distributions, shape (9, ny, nx).
        G: SC coupling constant.
        psi_fn: Pseudopotential function.
        tau: Relaxation time.

    Returns:
        Post-collision distributions, same shape.
    """
    rho, ux, uy = macroscopic(f)
    Fx, Fy = scmp_force(rho, G, psi_fn)

    rho_s = torch.clamp(rho, min=1e-12)
    ux_eq = ux + tau * Fx / rho_s
    uy_eq = uy + tau * Fy / rho_s

    feq = equilibrium(rho, ux_eq, uy_eq)
    f_out = f - (f - feq) / tau
    return f_out


def simulate_scmp(
    f: torch.Tensor,
    G: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor],
    tau: float = 1.0,
    n_steps: int = 1000,
    measure_interval: int = 100,
) -> dict:
    """Run SCMP simulation and collect statistics.

    Args:
        f: Initial distributions, shape (9, ny, nx).
        G: SC coupling constant.
        psi_fn: Pseudopotential function.
        tau: Relaxation time.
        n_steps: Number of time steps.
        measure_interval: Interval for collecting statistics.

    Returns:
        Dictionary with simulation results.
    """
    max_u_history = []

    for step in range(n_steps):
        f = collide_scmp(f, G, psi_fn, tau)
        f = stream_d2q9(f)

        if (step + 1) % measure_interval == 0:
            rho, ux, uy = macroscopic(f)
            max_u = float(torch.sqrt(ux**2 + uy**2).max())
            max_u_history.append((step + 1, max_u))

    rho_final, ux_final, uy_final = macroscopic(f)
    return {
        'f_final': f,
        'rho_final': rho_final,
        'ux_final': ux_final,
        'uy_final': uy_final,
        'max_u_history': max_u_history,
    }


# ============================================================================
# Initialization utilities
# ============================================================================

def init_uniform(rho0: float, ny: int, nx: int, device: str = 'cpu') -> torch.Tensor:
    """Initialize uniform density field in equilibrium."""
    rho = torch.full((ny, nx), rho0, dtype=torch.float64, device=device)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    return equilibrium(rho, ux, uy)


def init_droplet(
    rho_v: float,
    rho_l: float,
    radius: float,
    ny: int,
    nx: int,
    device: str = 'cpu',
) -> torch.Tensor:
    """Initialize a circular liquid droplet in vapor."""
    y = torch.arange(ny, dtype=torch.float64, device=device)
    x = torch.arange(nx, dtype=torch.float64, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    cy, cx = ny / 2.0, nx / 2.0
    r = torch.sqrt((yy - cy)**2 + (xx - cx)**2)

    rho = torch.where(r < radius,
                      torch.full_like(r, rho_l, device=device),
                      torch.full_like(r, rho_v, device=device))

    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    return equilibrium(rho, ux, uy)


def init_phase_separation(
    rho_avg: float,
    perturbation: float,
    ny: int,
    nx: int,
    device: str = 'cpu',
) -> torch.Tensor:
    """Initialize for bulk phase separation with small perturbation."""
    rho = torch.full((ny, nx), rho_avg, dtype=torch.float64, device=device)
    rho = rho + perturbation * torch.randn(ny, nx, dtype=torch.float64, device=device)
    rho = torch.clamp(rho, min=1e-6)

    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    return equilibrium(rho, ux, uy)


# ============================================================================
# Measurement utilities
# ============================================================================

def measure_droplet_pressure(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    radius: float,
    G: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[float, float]:
    """Measure pressure inside and outside a static droplet.

    Uses the mechanical EOS: p = rho*cs^2 - (G*cs^2/2)*psi^2
    """
    ny, nx = rho.shape
    cy, cx = ny / 2.0, nx / 2.0

    y = torch.arange(ny, dtype=torch.float64, device=rho.device)
    x = torch.arange(nx, dtype=torch.float64, device=rho.device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt((yy - cy)**2 + (xx - cx)**2)

    cs2 = 1.0 / 3.0
    psi = psi_fn(rho)
    p = rho * cs2 - 0.5 * G * cs2 * psi**2

    inside_mask = r < radius * 0.7
    outside_mask = r > radius * 1.3

    p_inside = float(p[inside_mask].mean()) if inside_mask.any() else 0.0
    p_outside = float(p[outside_mask].mean()) if outside_mask.any() else 0.0

    return p_inside, p_outside


def measure_max_velocity(ux: torch.Tensor, uy: torch.Tensor) -> float:
    """Measure maximum velocity magnitude (spurious velocity)."""
    return float(torch.sqrt(ux**2 + uy**2).max())


def measure_density_std(rho: torch.Tensor) -> float:
    """Measure density field standard deviation."""
    return float(rho.std())
