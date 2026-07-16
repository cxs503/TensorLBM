"""D3Q27 Shan-Chen multiphase lattice Boltzmann model.

Extends the D3Q19 :mod:`multiphase3d` module to the D3Q27 velocity set.
The D3Q27 lattice has 27 directions covering all combinations of
``(cx, cy, cz) ∈ {−1, 0, 1}³``.  Compared to D3Q19 it includes the 8 corner
directions (|c| = √3) and therefore achieves 4th-order isotropy, which can
reduce numerical artefacts in flows with strong corner-region gradients.

This module implements the Shan-Chen single-component (SCMP) and
two-component (SCMC) multiphase collision operators for D3Q27, following
the same algorithmic pattern as the D3Q19 versions in
:mod:`tensorlbm.multiphase3d` but using the 27-direction neighbour gather.

References
----------
Shan & Chen (1993) Phys. Rev. E 47 1815
Shan & Chen (1994) Phys. Rev. E 49 2941
Qian (1992) for D3Q27 lattice weights
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

from .d3q27 import C, W, equilibrium27, macroscopic27
from .multiphase import (
    psi_carnahan_starling,
    psi_exp,
    psi_linear,
    psi_peng_robinson,
    psi_power,
)  # re-export for convenience

_CS2 = 1.0 / 3.0

# Cache for SC neighbour-sum gather indices keyed by (nz, ny, nx, device_type, device_index)
_sc27_cache: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _c_on_27(device: torch.device) -> torch.Tensor:
    return C.to(device)


def _w_on_27(device: torch.device) -> torch.Tensor:
    return W.to(device)


# ---------------------------------------------------------------------------
# Shan-Chen neighborhood sum for D3Q27
# ---------------------------------------------------------------------------

def _sc_neighbor_weighted_sum_27(
    psi: torch.Tensor,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Σᵢ wᵢ ψ(x+cᵢ) cᵢ for the 3D SC interaction force (D3Q27).

    Uses a vectorised gather (same strategy as
    :func:`~tensorlbm.d3q27.stream27`) instead of a Python for-loop,
    eliminating all GPU→CPU synchronisations and reducing kernel launches
    to a small constant.  Index tensors are cached per (shape, device) to
    avoid re-allocation on every call.

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
    c = _c_on_27(device)   # (27, 3)  int64
    w = _w_on_27(device)   # (27,)    float32

    # Build and cache gather index tensors (one-time cost per unique shape/device)
    cache_key = (nz, ny, nx, device.type, device.index)
    if cache_key not in _sc27_cache:
        cz = c[:, 2]  # (27,)
        cy = c[:, 1]  # (27,)
        cx = c[:, 0]  # (27,)
        z_src = (torch.arange(nz, device=device).unsqueeze(0) - cz.unsqueeze(1)) % nz
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - cy.unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - cx.unsqueeze(1)) % nx
        _sc27_cache[cache_key] = (
            z_src.view(27, nz, 1, 1),  # (27, nz, 1, 1)
            y_src.view(27, 1, ny, 1),  # (27, 1, ny, 1)
            x_src.view(27, 1, 1, nx),  # (27, 1, 1, nx)
        )

    z_idx, y_idx, x_idx = _sc27_cache[cache_key]
    # psi_shifts: (27, nz, ny, nx) – all shifted copies gathered in one operation
    psi_shifts = psi[z_idx, y_idx, x_idx]   # advanced-index gather, no Python loop

    # w * c components: (27, 1, 1, 1) for broadcasting over (nz, ny, nx)
    cx_float = c[:, 0].float().view(27, 1, 1, 1)
    cy_float = c[:, 1].float().view(27, 1, 1, 1)
    cz_float = c[:, 2].float().view(27, 1, 1, 1)
    w_4d = w.view(27, 1, 1, 1)

    Fx = (w_4d * cx_float * psi_shifts).sum(0)   # (nz, ny, nx)
    Fy = (w_4d * cy_float * psi_shifts).sum(0)   # (nz, ny, nx)
    Fz = (w_4d * cz_float * psi_shifts).sum(0)   # (nz, ny, nx)
    return Fx, Fy, Fz


# ---------------------------------------------------------------------------
# Shan-Chen two-component (D3Q27)
# ---------------------------------------------------------------------------

def sc_two_component_force_27(
    rho1: torch.Tensor,
    rho2: torch.Tensor,
    G_12: float,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shan-Chen interaction + body forces for two D3Q27 components.

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
    sx2, sy2, sz2 = _sc_neighbor_weighted_sum_27(rho2, solid_mask)
    Fx1 = -G_12 * rho1 * sx2 + rho1 * gx
    Fy1 = -G_12 * rho1 * sy2 + rho1 * gy
    Fz1 = -G_12 * rho1 * sz2 + rho1 * gz

    sx1, sy1, sz1 = _sc_neighbor_weighted_sum_27(rho1, solid_mask)
    Fx2 = -G_12 * rho2 * sx1 + rho2 * gx
    Fy2 = -G_12 * rho2 * sy1 + rho2 * gy
    Fz2 = -G_12 * rho2 * sz1 + rho2 * gz

    return Fx1, Fy1, Fz1, Fx2, Fy2, Fz2


def collide_sc_two_component_27(
    f1: torch.Tensor,
    f2: torch.Tensor,
    G_12: float = 0.9,
    tau1: float = 1.0,
    tau2: float = 1.0,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    solid_mask: torch.Tensor | None = None,
    use_guo: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shan-Chen two-component BGK collision step for D3Q27.

    Args:
        f1:          Component-1 distribution, shape ``(27, nz, ny, nx)``.
        f2:          Component-2 distribution, shape ``(27, nz, ny, nx)``.
        G_12:        SC coupling constant (> 0 for phase separation).
        tau1:        Relaxation time for component 1.
        tau2:        Relaxation time for component 2.
        gx:          x body-force acceleration.
        gy:          y body-force acceleration.
        gz:          z body-force acceleration.
        solid_mask:  Optional boolean mask ``(nz, ny, nx)`` of solid/wall cells.
        use_guo:     If True, use Guo (2002) second-order forcing instead of
                     the velocity-shift.  Guo forcing adds a correction term
                     Δfᵢ = (1 − 1/(2τ))·wᵢ·[(cᵢ−u)/cs² + (cᵢ·u)·cᵢ/cs⁴]·F
                     which improves stability at high-density gradients and
                     is the standard in waLBerla (``lbm::force_model::GuoField``).

    Returns:
        Updated ``(f1, f2)`` after BGK collision.
    """
    device = f1.device
    rho1, ux1, uy1, uz1 = macroscopic27(f1)
    rho2, ux2, uy2, uz2 = macroscopic27(f2)

    Fx1, Fy1, Fz1, Fx2, Fy2, Fz2 = sc_two_component_force_27(
        rho1, rho2, G_12, gx, gy, gz, solid_mask,
    )

    rho1_s = torch.clamp(rho1, min=1e-12)
    rho2_s = torch.clamp(rho2, min=1e-12)

    if use_guo:
        # --- Guo forcing (second-order, waLBerla pattern) ---
        f1_out, f2_out = _bgk_collision_guo_27(
            f1, f2, rho1, rho2, ux1, uy1, uz1, ux2, uy2, uz2,
            Fx1, Fy1, Fz1, Fx2, Fy2, Fz2,
            tau1, tau2, device,
        )
    else:
        # --- Velocity-shift (first-order, original TensorLBM) ---
        feq1 = equilibrium27(
            rho1,
            ux1 + tau1 * Fx1 / rho1_s,
            uy1 + tau1 * Fy1 / rho1_s,
            uz1 + tau1 * Fz1 / rho1_s,
        )
        feq2 = equilibrium27(
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


def _bgk_collision_guo_27(
    f1: torch.Tensor,
    f2: torch.Tensor,
    rho1: torch.Tensor,
    rho2: torch.Tensor,
    ux1: torch.Tensor,
    uy1: torch.Tensor,
    uz1: torch.Tensor,
    ux2: torch.Tensor,
    uy2: torch.Tensor,
    uz2: torch.Tensor,
    Fx1: torch.Tensor,
    Fy1: torch.Tensor,
    Fz1: torch.Tensor,
    Fx2: torch.Tensor,
    Fy2: torch.Tensor,
    Fz2: torch.Tensor,
    tau1: float,
    tau2: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BGK collision with Guo (2002) second-order forcing for SC two-component (D3Q27).

    The Guo forcing formula:
        Δfᵢ = (1 − 1/(2τ)) · wᵢ · [(cᵢ − u)/cs² + (cᵢ·u)·cᵢ/cs⁴] · F

    This is applied as a post-collision correction to reduce spurious
    currents in multiphase flows — the standard approach used by
    waLBerla (``lbm::force_model::GuoField``).

    References
    ----------
    Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308
    """
    cs2 = 1.0 / 3.0
    cs4 = cs2 * cs2
    w = _w_on_27(device).view(27, 1, 1, 1)  # (27, 1, 1, 1)
    c = _c_on_27(device)
    cx = c[:, 0].float().view(27, 1, 1, 1)
    cy = c[:, 1].float().view(27, 1, 1, 1)
    cz = c[:, 2].float().view(27, 1, 1, 1)

    # Velocity-shift equilibrium for BGK step
    feq1 = equilibrium27(
        rho1,
        ux1 + tau1 * Fx1 / torch.clamp(rho1, min=1e-12),
        uy1 + tau1 * Fy1 / torch.clamp(rho1, min=1e-12),
        uz1 + tau1 * Fz1 / torch.clamp(rho1, min=1e-12),
    )
    feq2 = equilibrium27(
        rho2,
        ux2 + tau2 * Fx2 / torch.clamp(rho2, min=1e-12),
        uy2 + tau2 * Fy2 / torch.clamp(rho2, min=1e-12),
        uz2 + tau2 * Fz2 / torch.clamp(rho2, min=1e-12),
    )

    f1_post = f1 - (f1 - feq1) / tau1
    f2_post = f2 - (f2 - feq2) / tau2

    # Guo correction term for component 1
    cu1 = cx * ux1.unsqueeze(0) + cy * uy1.unsqueeze(0) + cz * uz1.unsqueeze(0)
    term_a1 = (cx - ux1.unsqueeze(0)) * Fx1.unsqueeze(0) + (cy - uy1.unsqueeze(0)) * Fy1.unsqueeze(0) + (cz - uz1.unsqueeze(0)) * Fz1.unsqueeze(0)
    term_b1 = cu1 * (cx * Fx1.unsqueeze(0) + cy * Fy1.unsqueeze(0) + cz * Fz1.unsqueeze(0))
    delta_f1 = (1.0 - 1.0 / (2.0 * tau1)) * w * (term_a1 / cs2 + term_b1 / cs4)

    # Guo correction term for component 2
    cu2 = cx * ux2.unsqueeze(0) + cy * uy2.unsqueeze(0) + cz * uz2.unsqueeze(0)
    term_a2 = (cx - ux2.unsqueeze(0)) * Fx2.unsqueeze(0) + (cy - uy2.unsqueeze(0)) * Fy2.unsqueeze(0) + (cz - uz2.unsqueeze(0)) * Fz2.unsqueeze(0)
    term_b2 = cu2 * (cx * Fx2.unsqueeze(0) + cy * Fy2.unsqueeze(0) + cz * Fz2.unsqueeze(0))
    delta_f2 = (1.0 - 1.0 / (2.0 * tau2)) * w * (term_a2 / cs2 + term_b2 / cs4)

    return f1_post + delta_f1, f2_post + delta_f2


# ---------------------------------------------------------------------------
# Shan-Chen single-component (D3Q27)
# ---------------------------------------------------------------------------

def collide_sc_single_component_27(
    f: torch.Tensor,
    G: float = -4.0,
    tau: float = 1.0,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] = psi_exp,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    solid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shan-Chen single-component multiphase BGK collision for D3Q27.

    Args:
        f:           Distribution tensor, shape ``(27, nz, ny, nx)``.
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
    rho, ux, uy, uz = macroscopic27(f)
    psi = psi_fn(rho)
    sx, sy, sz = _sc_neighbor_weighted_sum_27(psi, solid_mask)
    rho_s = torch.clamp(rho, min=1e-12)
    Fx = -G * psi * sx + rho * gx
    Fy = -G * psi * sy + rho * gy
    Fz = -G * psi * sz + rho * gz
    feq = equilibrium27(
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
    # D3Q27 SC two-component
    "sc_two_component_force_27",
    "collide_sc_two_component_27",
    # D3Q27 SC single-component
    "collide_sc_single_component_27",
    # Re-exported pseudopotential helpers (same as 2D / D3Q19)
    "psi_linear",
    "psi_exp",
    "psi_power",
    "psi_carnahan_starling",
    "psi_peng_robinson",
]
