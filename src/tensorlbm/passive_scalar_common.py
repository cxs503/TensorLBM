"""Common passive-scalar LBM module — composable advection-diffusion step.

Extracts the passive-scalar transport LBM into a reusable module that can
be composed with *any* collision / turbulence / multiphase solver.  The
scalar field evolves on a D3Q7 lattice, while the momentum field can be
D3Q19 or D3Q27.

Physical model
--------------
The scalar field φ (concentration, pollutant, tracer) obeys::

    ∂φ/∂t + u · ∇φ = D ∇²φ + S(x, t)

The LBM implementation uses the D3Q7 lattice with BGK collision::

    g_i^eq = w_i · φ · (1 + 4 · (c_i · u))
    g_i*   = g_i − (g_i − g_i^eq) / τ_D
    τ_D    = D / c_s² + 0.5 = 4D + 0.5

where ``c_s² = 1/4`` for D3Q7.

Design principles
-----------------
* **No solver hot-path changes** — standalone composable step function.
* **Supports D3Q19 and D3Q27** — velocity is extracted from the momentum
  distribution; the scalar lattice is always D3Q7.
* **Composable** — call after the momentum step::

      f = collide_any(f, tau)
      f = stream(f)
      g, phi = passive_scalar_step(f, g, lattice="D3Q19", tau_d=0.8)

References
----------
Shi, B., & Guo, Z. (2009).
    Lattice Boltzmann model for nonlinear convection-diffusion equations.
    *Phys. Rev. E* 79, 016701.
"""
from __future__ import annotations

import functools
from typing import Any

import torch

__all__ = [
    "scalar_equilibrium_3d",
    "scalar_collide_bgk_3d",
    "scalar_stream_3d",
    "scalar_macroscopic_3d",
    "passive_scalar_step",
]

# Reuse D3Q7 constants from thermal_common
from .thermal_common import C_D3Q7, W_D3Q7

_stream_scalar_cache: dict[
    tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


@functools.cache
def _c_scalar(device: torch.device) -> torch.Tensor:
    return C_D3Q7.to(device)


@functools.cache
def _w_scalar(device: torch.device) -> torch.Tensor:
    return W_D3Q7.to(device)


# ---------------------------------------------------------------------------
# D3Q7 scalar lattice operators
# ---------------------------------------------------------------------------


def scalar_equilibrium_3d(
    phi: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Compute the D3Q7 scalar equilibrium distribution.

    g_i^eq = w_i · φ · (1 + 4 · (cx_i·ux + cy_i·uy + cz_i·uz))

    Args:
        phi: Scalar concentration field, shape ``(nz, ny, nx)``.
        ux:  x-velocity, shape ``(nz, ny, nx)``.
        uy:  y-velocity, shape ``(nz, ny, nx)``.
        uz:  z-velocity, shape ``(nz, ny, nx)``.

    Returns:
        Equilibrium distribution, shape ``(7, nz, ny, nx)``.
    """
    device = phi.device
    c = _c_scalar(device)
    w = _w_scalar(device).view(7, 1, 1, 1)
    cx = c[:, 0].view(7, 1, 1, 1).float()
    cy = c[:, 1].view(7, 1, 1, 1).float()
    cz = c[:, 2].view(7, 1, 1, 1).float()
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    return w * phi.unsqueeze(0) * (1.0 + 4.0 * cu)


def scalar_collide_bgk_3d(
    g: torch.Tensor,
    phi: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    tau_d: float,
    source: torch.Tensor | None = None,
) -> torch.Tensor:
    """BGK collision for the D3Q7 scalar distribution.

    Args:
        g:      Scalar distribution, shape ``(7, nz, ny, nx)``.
        phi:    Scalar concentration, shape ``(nz, ny, nx)``.
        ux:     x-velocity, shape ``(nz, ny, nx)``.
        uy:     y-velocity, shape ``(nz, ny, nx)``.
        uz:     z-velocity, shape ``(nz, ny, nx)``.
        tau_d:  Scalar relaxation time (τ_D = 4D + 0.5).
        source: Optional scalar source term, shape ``(nz, ny, nx)``.

    Returns:
        Post-collision scalar distribution, shape ``(7, nz, ny, nx)``.
    """
    g_eq = scalar_equilibrium_3d(phi, ux, uy, uz)
    g_out = g - (g - g_eq) / tau_d
    if source is not None:
        device = g.device
        w = _w_scalar(device).view(7, 1, 1, 1)
        g_out = g_out + w * source.unsqueeze(0)
    return g_out


def scalar_stream_3d(g: torch.Tensor) -> torch.Tensor:
    """Periodic streaming for the D3Q7 scalar distribution.

    Args:
        g: Scalar distribution, shape ``(7, nz, ny, nx)``.

    Returns:
        Streamed distribution of the same shape.
    """
    nz, ny, nx = g.shape[1], g.shape[2], g.shape[3]
    device = g.device
    c = _c_scalar(device)

    cache_key = (nz, ny, nx, device.type, device.index)
    if cache_key not in _stream_scalar_cache:
        z_src = (torch.arange(nz, device=device).unsqueeze(0) - c[:, 2].unsqueeze(1)) % nz
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx
        q_idx = torch.arange(7, device=device).view(7, 1, 1, 1).expand(7, nz, ny, nx)
        z_idx = z_src.view(7, nz, 1, 1).expand(7, nz, ny, nx)
        y_idx = y_src.view(7, 1, ny, 1).expand(7, nz, ny, nx)
        x_idx = x_src.view(7, 1, 1, nx).expand(7, nz, ny, nx)
        _stream_scalar_cache[cache_key] = (q_idx, z_idx, y_idx, x_idx)

    q_idx, z_idx, y_idx, x_idx = _stream_scalar_cache[cache_key]
    return g[q_idx, z_idx, y_idx, x_idx]


def scalar_macroscopic_3d(g: torch.Tensor) -> torch.Tensor:
    """Recover the scalar concentration from D3Q7 distributions.

    φ = Σ_i g_i

    Args:
        g: Scalar distribution, shape ``(7, nz, ny, nx)``.

    Returns:
        Scalar concentration field, shape ``(nz, ny, nx)``.
    """
    return g.sum(dim=0)


# ---------------------------------------------------------------------------
# Combined passive scalar step
# ---------------------------------------------------------------------------


def passive_scalar_step(
    f: torch.Tensor,
    g: torch.Tensor,
    *,
    tau_d: float = 0.8,
    lattice: str = "D3Q19",
    source: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One composable passive-scalar LBM step (collision + streaming).

    Extracts velocity from the momentum distribution *f*, then evolves the
    scalar distribution *g* on D3Q7 via BGK collision + periodic streaming.

    Designed to be inserted into *any* time loop::

        for step in range(n_steps):
            f = collide_any(f, tau)      # any collision
            f = stream(f)                 # any streaming
            g, phi = passive_scalar_step(f, g, tau_d=0.8, lattice="D3Q19")

    Args:
        f:      Momentum distribution, shape ``(Q, nz, ny, nx)``.
        g:      Scalar distribution (D3Q7), shape ``(7, nz, ny, nx)``.
        tau_d:  Scalar relaxation time (τ_D = 4D + 0.5).
        lattice: Momentum lattice — ``"D3Q19"`` or ``"D3Q27"``.
        source: Optional scalar source term, shape ``(nz, ny, nx)``.
        mask:   Optional solid mask for velocity zeroing.

    Returns:
        ``(g_updated, phi_updated)`` — updated scalar distribution and
        concentration field.
    """
    lattice_u = lattice.upper()
    if lattice_u == "D3Q19":
        from .d3q19 import macroscopic3d

        _, ux, uy, uz = macroscopic3d(f)
    elif lattice_u == "D3Q27":
        from .d3q27 import macroscopic27

        _, ux, uy, uz = macroscopic27(f)
    else:
        raise ValueError(f"lattice must be 'D3Q19' or 'D3Q27', got {lattice!r}")

    # Zero velocity in solid cells
    if mask is not None:
        ux = ux.masked_fill(mask, 0.0)
        uy = uy.masked_fill(mask, 0.0)
        uz = uz.masked_fill(mask, 0.0)

    # Recover scalar
    phi = scalar_macroscopic_3d(g)

    # Collision + streaming
    g = scalar_collide_bgk_3d(g, phi, ux, uy, uz, tau_d, source=source)
    g = scalar_stream_3d(g)

    # Recover updated scalar
    phi = scalar_macroscopic_3d(g)

    return g, phi
