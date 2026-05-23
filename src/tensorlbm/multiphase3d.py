"""D3Q19 multiphase lattice Boltzmann model — Shan-Chen two-component (SCMC).

Extends the D2Q9 :mod:`multiphase` module to three dimensions using the D3Q19
velocity set.  Currently implements the Shan-Chen two-component model, which is
the most practical choice for large-scale 3D simulations such as sphere water
entry.

The single-component (SCMP) and Color-Gradient (CG) models follow the same
pattern and can be added by replacing the pseudopotential / recoloring kernels
in the 2D implementations with 3D analogs.

References
----------
Shan & Chen (1993) Phys. Rev. E 47 1815
Shan & Chen (1994) Phys. Rev. E 49 2941
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

from .d3q19 import C, W, equilibrium3d, macroscopic3d
from .multiphase import psi_exp, psi_linear, psi_power  # re-export for convenience

_CS2 = 1.0 / 3.0

# Cache for SC neighbour-sum gather indices keyed by (nz, ny, nx, device_type, device_index)
_sc3d_cache: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _c_on_3d(device: torch.device) -> torch.Tensor:
    return C.to(device)


def _w_on_3d(device: torch.device) -> torch.Tensor:
    return W.to(device)


# ---------------------------------------------------------------------------
# Shan-Chen neighborhood sum for D3Q19
# ---------------------------------------------------------------------------

def _sc_neighbor_weighted_sum_3d(
    psi: torch.Tensor,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Σᵢ wᵢ ψ(x+cᵢ) cᵢ for the 3D SC interaction force.

    Uses a vectorised gather (same strategy as :func:`~tensorlbm.solver3d.stream3d`)
    instead of a Python for-loop, eliminating all GPU→CPU synchronisations and
    reducing kernel launches to a small constant.  Index tensors are cached per
    (shape, device) to avoid re-allocation on every call.

    Args:
        psi:         Scalar field of shape ``(nz, ny, nx)``.
        solid_mask:  Optional boolean mask ``(nz, ny, nx)``.  Solid/wall cells
                     are zeroed in ψ before the neighbour sum.

    Returns:
        ``(Fx, Fy, Fz)`` each of shape ``(nz, ny, nx)``.
    """
    if solid_mask is not None:
        psi = psi.masked_fill(solid_mask, 0.0)

    device = psi.device
    nz, ny, nx = psi.shape[-3], psi.shape[-2], psi.shape[-1]
    c = _c_on_3d(device)   # (19, 3)  int64
    w = _w_on_3d(device)   # (19,)    float32

    # Build and cache gather index tensors (one-time cost per unique shape/device)
    cache_key = (nz, ny, nx, device.type, device.index)
    if cache_key not in _sc3d_cache:
        cz = c[:, 2]  # (19,)
        cy = c[:, 1]  # (19,)
        cx = c[:, 0]  # (19,)
        z_src = (torch.arange(nz, device=device).unsqueeze(0) - cz.unsqueeze(1)) % nz
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - cy.unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - cx.unsqueeze(1)) % nx
        # shape: (19, nz/ny/nx, 1, 1) or similar for broadcasting to (19, nz, ny, nx)
        _sc3d_cache[cache_key] = (
            z_src.view(19, nz, 1, 1),  # (19, nz, 1, 1)
            y_src.view(19, 1, ny, 1),  # (19, 1, ny, 1)
            x_src.view(19, 1, 1, nx),  # (19, 1, 1, nx)
        )

    z_idx, y_idx, x_idx = _sc3d_cache[cache_key]
    # psi_shifts: (19, nz, ny, nx) – all shifted copies gathered in one operation
    psi_shifts = psi[z_idx, y_idx, x_idx]   # advanced-index gather, no Python loop

    # w * c components: (19, 1, 1, 1) for broadcasting over (nz, ny, nx)
    cx_float = c[:, 0].float().view(19, 1, 1, 1)
    cy_float = c[:, 1].float().view(19, 1, 1, 1)
    cz_float = c[:, 2].float().view(19, 1, 1, 1)
    w_4d = w.view(19, 1, 1, 1)

    Fx = (w_4d * cx_float * psi_shifts).sum(0)   # (nz, ny, nx)
    Fy = (w_4d * cy_float * psi_shifts).sum(0)   # (nz, ny, nx)
    Fz = (w_4d * cz_float * psi_shifts).sum(0)   # (nz, ny, nx)
    return Fx, Fy, Fz


# ---------------------------------------------------------------------------
# Shan-Chen two-component (3D)
# ---------------------------------------------------------------------------

def sc_two_component_force_3d(
    rho1: torch.Tensor,
    rho2: torch.Tensor,
    G_12: float,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shan-Chen interaction + body forces for two 3D components.

    Args:
        rho1:        Density of component 1, shape ``(nz, ny, nx)``.
        rho2:        Density of component 2, shape ``(nz, ny, nx)``.
        G_12:        Coupling constant (> 0 → repulsive → phase separation).
        gx:          x body-force acceleration.
        gy:          y body-force acceleration.
        gz:          z body-force acceleration (negative = downward if z is up).
        solid_mask:  Optional boolean mask ``(nz, ny, nx)`` of solid/wall cells.

    Returns:
        ``(Fx1, Fy1, Fz1, Fx2, Fy2, Fz2)`` each of shape ``(nz, ny, nx)``.
    """
    sx2, sy2, sz2 = _sc_neighbor_weighted_sum_3d(rho2, solid_mask)
    Fx1 = -G_12 * rho1 * sx2 + rho1 * gx
    Fy1 = -G_12 * rho1 * sy2 + rho1 * gy
    Fz1 = -G_12 * rho1 * sz2 + rho1 * gz

    sx1, sy1, sz1 = _sc_neighbor_weighted_sum_3d(rho1, solid_mask)
    Fx2 = -G_12 * rho2 * sx1 + rho2 * gx
    Fy2 = -G_12 * rho2 * sy1 + rho2 * gy
    Fz2 = -G_12 * rho2 * sz1 + rho2 * gz

    return Fx1, Fy1, Fz1, Fx2, Fy2, Fz2


def collide_sc_two_component_3d(
    f1: torch.Tensor,
    f2: torch.Tensor,
    G_12: float = 0.9,
    tau1: float = 1.0,
    tau2: float = 1.0,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shan-Chen two-component BGK collision step for D3Q19.

    Args:
        f1:          Component-1 distribution, shape ``(19, nz, ny, nx)``.
        f2:          Component-2 distribution, shape ``(19, nz, ny, nx)``.
        G_12:        SC coupling constant (> 0 for phase separation).
        tau1:        Relaxation time for component 1.
        tau2:        Relaxation time for component 2.
        gx:          x body-force acceleration.
        gy:          y body-force acceleration.
        gz:          z body-force acceleration.
        solid_mask:  Optional boolean mask ``(nz, ny, nx)`` of solid/wall cells.

    Returns:
        Updated ``(f1, f2)`` after BGK collision.
    """
    rho1, ux1, uy1, uz1 = macroscopic3d(f1)
    rho2, ux2, uy2, uz2 = macroscopic3d(f2)

    Fx1, Fy1, Fz1, Fx2, Fy2, Fz2 = sc_two_component_force_3d(
        rho1, rho2, G_12, gx, gy, gz, solid_mask,
    )

    rho1_s = torch.clamp(rho1, min=1e-12)
    rho2_s = torch.clamp(rho2, min=1e-12)

    feq1 = equilibrium3d(
        rho1,
        ux1 + tau1 * Fx1 / rho1_s,
        uy1 + tau1 * Fy1 / rho1_s,
        uz1 + tau1 * Fz1 / rho1_s,
    )
    feq2 = equilibrium3d(
        rho2,
        ux2 + tau2 * Fx2 / rho2_s,
        uy2 + tau2 * Fy2 / rho2_s,
        uz2 + tau2 * Fz2 / rho2_s,
    )

    f1_out = f1 - (f1 - feq1) / tau1
    f2_out = f2 - (f2 - feq2) / tau2

    # Solid cells skip collision.
    if solid_mask is not None:
        mask_4d = solid_mask.unsqueeze(0)  # (1, nz, ny, nx)
        f1_out = torch.where(mask_4d, f1, f1_out)
        f2_out = torch.where(mask_4d, f2, f2_out)

    return f1_out, f2_out


# ---------------------------------------------------------------------------
# Shan-Chen single-component (3D)
# ---------------------------------------------------------------------------

def collide_sc_single_component_3d(
    f: torch.Tensor,
    G: float = -4.0,
    tau: float = 1.0,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] = psi_exp,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    solid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shan-Chen single-component multiphase BGK collision for D3Q19.

    Args:
        f:           Distribution tensor, shape ``(19, nz, ny, nx)``.
        G:           SC self-coupling constant (< 0 → attractive → phase sep.).
        tau:         Relaxation time.
        psi_fn:      Pseudopotential callable.
        gx:          x body-force acceleration.
        gy:          y body-force acceleration.
        gz:          z body-force acceleration.
        solid_mask:  Optional boolean mask ``(nz, ny, nx)`` of solid/wall cells.

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    psi = psi_fn(rho)
    sx, sy, sz = _sc_neighbor_weighted_sum_3d(psi, solid_mask)
    rho_s = torch.clamp(rho, min=1e-12)
    Fx = -G * psi * sx + rho * gx
    Fy = -G * psi * sy + rho * gy
    Fz = -G * psi * sz + rho * gz
    feq = equilibrium3d(
        rho,
        ux + tau * Fx / rho_s,
        uy + tau * Fy / rho_s,
        uz + tau * Fz / rho_s,
    )
    f_out = f - (f - feq) / tau
    if solid_mask is not None:
        f_out = torch.where(solid_mask.unsqueeze(0), f, f_out)
    return f_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # 3D SC two-component
    "sc_two_component_force_3d",
    "collide_sc_two_component_3d",
    # 3D SC single-component
    "collide_sc_single_component_3d",
    # Re-exported pseudopotential helpers (same as 2D)
    "psi_linear",
    "psi_exp",
    "psi_power",
]
