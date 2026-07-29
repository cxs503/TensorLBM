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
from .multiphase import psi_exp, psi_linear, psi_power, psi_carnahan_starling, psi_peng_robinson  # re-export for convenience
from .solver3d import _get_d3q19_mrt_matrices
from .turbulence import _neq_stress_norm_3d, _smagorinsky_tau

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
    use_guo: bool = False,
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
        use_guo:     If True, use Guo (2002) second-order forcing instead of
                     the velocity-shift.  Guo forcing adds a correction term
                     Δfᵢ = (1 − 1/(2τ))·wᵢ·[(cᵢ−u)/cs² + (cᵢ·u)·cᵢ/cs⁴]·F
                     which improves stability at high-density gradients and
                     is the standard in waLBerla (``lbm::force_model::GuoField``).

    Returns:
        Updated ``(f1, f2)`` after BGK collision.
    """
    device = f1.device
    rho1, ux1, uy1, uz1 = macroscopic3d(f1)
    rho2, ux2, uy2, uz2 = macroscopic3d(f2)

    Fx1, Fy1, Fz1, Fx2, Fy2, Fz2 = sc_two_component_force_3d(
        rho1, rho2, G_12, gx, gy, gz, solid_mask,
    )

    rho1_s = torch.clamp(rho1, min=1e-12)
    rho2_s = torch.clamp(rho2, min=1e-12)

    if use_guo:
        # --- Guo forcing (second-order, waLBerla pattern) ---
        f1_out, f2_out = _bgk_collision_guo_3d(
            f1, f2, rho1, rho2, ux1, uy1, uz1, ux2, uy2, uz2,
            Fx1, Fy1, Fz1, Fx2, Fy2, Fz2,
            tau1, tau2, device,
        )
    else:
        # --- Velocity-shift (first-order, original TensorLBM) ---
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


def _bgk_collision_guo_3d(
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
    """BGK collision with Guo (2002) second-order forcing for SC two-component.

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
    w = _w_on_3d(device).view(19, 1, 1, 1)  # (19, 1, 1, 1)
    c = _c_on_3d(device)
    cx = c[:, 0].float().view(19, 1, 1, 1)
    cy = c[:, 1].float().view(19, 1, 1, 1)
    cz = c[:, 2].float().view(19, 1, 1, 1)

    # Velocity-shift equilibrium for BGK step
    feq1 = equilibrium3d(
        rho1,
        ux1 + tau1 * Fx1 / torch.clamp(rho1, min=1e-12),
        uy1 + tau1 * Fy1 / torch.clamp(rho1, min=1e-12),
        uz1 + tau1 * Fz1 / torch.clamp(rho1, min=1e-12),
    )
    feq2 = equilibrium3d(
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
    # 3D Color-Gradient
    "color_gradient_step_3d",
    # 3D Free-Energy / Phase-Field
    "init_free_energy_g_3d",
    "free_energy_step_3d",
    # Re-exported pseudopotential helpers (same as 2D)
    "psi_linear",
    "psi_exp",
    "psi_power",
    "psi_carnahan_starling",
    "psi_peng_robinson",
    # MRT + Smagorinsky multiphase collision
    "collide_cg_mrt_3d",
    "collide_sc_mrt_3d",
    "init_hydrostatic_pressure_3d",
]


# ---------------------------------------------------------------------------
# Model 3 – Color-Gradient (3D, D3Q19)
# ---------------------------------------------------------------------------
#
# Algorithm mirrors the 2-D Latva-Kokko & Rothman (2005) CG model:
#   1. BGK collision on the total distribution.
#   2. Surface-tension perturbation: Δfᵢ = (A/2)|∇φ| wᵢ [(cᵢ·n̂)² − 1/3]
#   3. Recoloring to maintain phase separation.
#
# The 3-D phase-field gradient is approximated by central differences in all
# three coordinate directions.


def _grad_phase_field_3d(
    rho_r: torch.Tensor,
    rho_b: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """3-D phase-field gradient and unit normal for the CG model.

    For a field of shape ``(nz, ny, nx)``:
        dim 0 ↔ z,  dim 1 ↔ y,  dim 2 ↔ x.

    Returns:
        ``(phi, mag, nhat_x, nhat_y, nhat_z)`` all of shape ``(nz, ny, nx)``.
    """
    n = rho_r + rho_b
    n_safe = torch.clamp(n, min=1e-12)
    phi = (rho_r - rho_b) / n_safe

    dphi_dx = 0.5 * (torch.roll(phi, -1, dims=2) - torch.roll(phi, 1, dims=2))
    dphi_dy = 0.5 * (torch.roll(phi, -1, dims=1) - torch.roll(phi, 1, dims=1))
    dphi_dz = 0.5 * (torch.roll(phi, -1, dims=0) - torch.roll(phi, 1, dims=0))

    mag = torch.sqrt(dphi_dx ** 2 + dphi_dy ** 2 + dphi_dz ** 2)
    mag_safe = torch.clamp(mag, min=1e-12)
    nhat_x = dphi_dx / mag_safe
    nhat_y = dphi_dy / mag_safe
    nhat_z = dphi_dz / mag_safe

    return phi, mag, nhat_x, nhat_y, nhat_z


def color_gradient_step_3d(
    f_r: torch.Tensor,
    f_b: torch.Tensor,
    tau: float = 1.0,
    A: float = 0.04,
    beta: float = 0.7,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Color-Gradient two-phase step for D3Q19.

    Performs one full 3-D CG iteration:
      (a) BGK collision on the total distribution;
      (b) Surface-tension perturbation proportional to |∇φ|;
      (c) Recoloring to restore the two phases.

    Args:
        f_r:         Red (heavy) component distribution, shape ``(19, nz, ny, nx)``.
        f_b:         Blue (light) component distribution, shape ``(19, nz, ny, nx)``.
        tau:         Shared relaxation time (same for both components).
        A:           Surface-tension coefficient (larger → stronger tension).
        beta:        Recoloring parameter ∈ (0, 1].  β→1 gives sharp interfaces.
        gx:          x body-force acceleration.
        gy:          y body-force acceleration.
        gz:          z body-force acceleration.
        solid_mask:  Optional boolean mask ``(nz, ny, nx)`` of solid/wall cells.

    Returns:
        Updated ``(f_r, f_b)`` after collision + recoloring.
    """
    device = f_r.device
    c = _c_on_3d(device)
    w = _w_on_3d(device)

    cx = c[:, 0].float().view(19, 1, 1, 1)
    cy = c[:, 1].float().view(19, 1, 1, 1)
    cz = c[:, 2].float().view(19, 1, 1, 1)
    w_v = w.view(19, 1, 1, 1)

    # --- 1. Total distribution and macroscopic quantities ---
    f_total = f_r + f_b
    rho_r_s = f_r.sum(dim=0)
    rho_b_s = f_b.sum(dim=0)
    rho = rho_r_s + rho_b_s
    rho_safe = torch.clamp(rho, min=1e-12)

    ux = (f_total * cx).sum(dim=0) / rho_safe
    uy = (f_total * cy).sum(dim=0) / rho_safe
    uz = (f_total * cz).sum(dim=0) / rho_safe

    feq = equilibrium3d(rho, ux + tau * gx, uy + tau * gy, uz + tau * gz)
    f_post = f_total - (f_total - feq) / tau

    if solid_mask is not None:
        f_post = torch.where(solid_mask.unsqueeze(0), f_total, f_post)

    # --- 2. Surface-tension perturbation ---
    # Keep original densities at solid cells.  The solid_mask already prevents
    # collision at solid cells (line 383), so solid-cell distributions stay at
    # their initialized values.  Letting the gradient see these values keeps
    # the phase-field smooth across the boundary.
    rho_r_m = rho_r_s
    rho_b_m = rho_b_s
    _phi, mag, nhat_x, nhat_y, nhat_z = _grad_phase_field_3d(rho_r_m, rho_b_m)

    ci_dot_n = (
        cx * nhat_x.unsqueeze(0)
        + cy * nhat_y.unsqueeze(0)
        + cz * nhat_z.unsqueeze(0)
    )  # (19, nz, ny, nx)
    B_iso = 1.0 / 3.0
    perturbation = (A / 2.0) * mag.unsqueeze(0) * w_v * (ci_dot_n ** 2 - B_iso)
    f_post = f_post + perturbation

    # --- 3. Recoloring step (Latva-Kokko & Rothman 2005) ---
    # feq at zero velocity with unit density equals wᵢ, so reuse w_v directly
    # instead of allocating new tensors via equilibrium3d(ones, zeros, zeros, zeros).
    recolor_amp = (
        beta * (rho_r_s * rho_b_s / rho_safe).unsqueeze(0) * ci_dot_n * w_v
    )

    f_r_out = (rho_r_s / rho_safe).unsqueeze(0) * f_post + recolor_amp
    f_b_out = (rho_b_s / rho_safe).unsqueeze(0) * f_post - recolor_amp

    if solid_mask is not None:
        mask_4d = solid_mask.unsqueeze(0)
        f_r_out = torch.where(mask_4d, f_r, f_r_out)
        f_b_out = torch.where(mask_4d, f_b, f_b_out)

    return f_r_out, f_b_out


# ---------------------------------------------------------------------------
# Model 4 – Free-Energy / Phase-Field (3D, D3Q19)
# ---------------------------------------------------------------------------
#
# Extends the 2-D Swift et al. formulation to three dimensions.
# Two coupled LBM equations:
#   f  – momentum distribution for total density ρ and velocity u.
#   g  – order-parameter distribution that advects the phase field φ = Σᵢ gᵢ.
#
# Chemical potential: μ = −Aφ + Bφ³ − κ∇²φ
# Driving force:      Fᵢ = −φ ∇μ + ρ_eff g_body   (Korteweg force)


def _laplacian_3d(field: torch.Tensor) -> torch.Tensor:
    """3-D Laplacian via second-order central differences (periodic).

    For a field of shape ``(nz, ny, nx)``:
        dim 0 ↔ z,  dim 1 ↔ y,  dim 2 ↔ x.
    """
    return (
        torch.roll(field, 1, dims=2) + torch.roll(field, -1, dims=2)
        + torch.roll(field, 1, dims=1) + torch.roll(field, -1, dims=1)
        + torch.roll(field, 1, dims=0) + torch.roll(field, -1, dims=0)
        - 6.0 * field
    )


def init_free_energy_g_3d(
    phi: torch.Tensor,
    ux: torch.Tensor | None = None,
    uy: torch.Tensor | None = None,
    uz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Initialise the 3-D FE order-parameter distribution in equilibrium.

    Args:
        phi: Initial phase field, shape ``(nz, ny, nx)``.
        ux:  Initial x-velocity (optional, defaults to zero).
        uy:  Initial y-velocity (optional, defaults to zero).
        uz:  Initial z-velocity (optional, defaults to zero).

    Returns:
        Equilibrium distribution g, shape ``(19, nz, ny, nx)``.
    """
    device = phi.device
    c = _c_on_3d(device)
    w = _w_on_3d(device).view(19, 1, 1, 1)

    if ux is None:
        ux = torch.zeros_like(phi)
    if uy is None:
        uy = torch.zeros_like(phi)
    if uz is None:
        uz = torch.zeros_like(phi)

    cx = c[:, 0].float().view(19, 1, 1, 1)
    cy = c[:, 1].float().view(19, 1, 1, 1)
    cz = c[:, 2].float().view(19, 1, 1, 1)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    u_sq = (ux ** 2 + uy ** 2 + uz ** 2).unsqueeze(0)
    return w * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u_sq)


def free_energy_step_3d(
    f: torch.Tensor,
    g: torch.Tensor,
    tau_f: float = 1.0,
    tau_g: float = 0.7,
    A: float = 0.1,
    B: float = 0.1,
    kappa: float = 0.02,
    Gamma: float = 0.5,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    rho_heavy: float | None = None,
    rho_light: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Free-Energy two-phase step for D3Q19.

    Two coupled LBM equations:
      • **f**: momentum distribution for total density ρ and velocity u.
      • **g**: order-parameter distribution that advects the phase field φ.

    The Korteweg stress drives interface dynamics.  When *rho_heavy* and
    *rho_light* are given the gravitational body force is scaled by the local
    effective density (Boussinesq buoyancy).

    Args:
        f:          Momentum distribution, shape ``(19, nz, ny, nx)``.
        g:          Order-parameter distribution, shape ``(19, nz, ny, nx)``;
                    its zeroth moment is the phase field φ = Σᵢ gᵢ.
        tau_f:      Relaxation time for momentum (ν = cs²(τ_f − ½)).
        tau_g:      Relaxation time for phase field (M = cs²(τ_g − ½)).
        A:          Double-well coefficient.
        B:          Quartic coefficient.
        kappa:      Interfacial-tension parameter (gradient penalty).
        Gamma:      Phase-field mobility coupling.
        gx:         x body-force acceleration.
        gy:         y body-force acceleration.
        gz:         z body-force acceleration.
        rho_heavy:  Effective density for the φ=+1 phase (Boussinesq buoyancy).
        rho_light:  Effective density for the φ=−1 phase.

    Returns:
        Updated ``(f, g)`` after one collision step.
    """
    device = f.device
    c = _c_on_3d(device)
    w = _w_on_3d(device)
    cx = c[:, 0].float().view(19, 1, 1, 1)
    cy = c[:, 1].float().view(19, 1, 1, 1)
    cz = c[:, 2].float().view(19, 1, 1, 1)
    w_v = w.view(19, 1, 1, 1)

    # Macroscopic quantities
    rho, ux, uy, uz = macroscopic3d(f)
    phi = g.sum(dim=0)  # order parameter

    # Effective density for buoyancy (Boussinesq approximation)
    if rho_heavy is not None and rho_light is not None:
        phi_c = phi.clamp(-1.0, 1.0)
        rho_eff = 0.5 * ((1.0 + phi_c) * rho_heavy + (1.0 - phi_c) * rho_light)
    else:
        rho_eff = rho

    # Chemical potential: μ = −Aφ + Bφ³ − κ∇²φ
    mu = -A * phi + B * phi ** 3 - kappa * _laplacian_3d(phi)

    # Korteweg (capillary) body force: F_cap = −φ ∇μ + ρ_eff g_body
    grad_mu_x = 0.5 * (torch.roll(mu, -1, dims=2) - torch.roll(mu, 1, dims=2))
    grad_mu_y = 0.5 * (torch.roll(mu, -1, dims=1) - torch.roll(mu, 1, dims=1))
    grad_mu_z = 0.5 * (torch.roll(mu, -1, dims=0) - torch.roll(mu, 1, dims=0))
    Fx = -phi * grad_mu_x + rho_eff * gx
    Fy = -phi * grad_mu_y + rho_eff * gy
    Fz = -phi * grad_mu_z + rho_eff * gz

    rho_s = torch.clamp(rho, min=1e-12)
    ux_eq = ux + tau_f * Fx / rho_s
    uy_eq = uy + tau_f * Fy / rho_s
    uz_eq = uz + tau_f * Fz / rho_s

    # Collision for f (BGK with Korteweg + buoyancy force)
    feq = equilibrium3d(rho, ux_eq, uy_eq, uz_eq)
    f_out = f - (f - feq) / tau_f

    # Equilibrium for g  (D=3, cs²=1/3 → diff_factor = 3|c|² − 3)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    u_sq = (ux ** 2 + uy ** 2 + uz ** 2).unsqueeze(0)
    geq_adv = w_v * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u_sq)
    c_sq = cx ** 2 + cy ** 2 + cz ** 2
    diff_factor = c_sq / _CS2 - 3.0  # = 3|c|² − 3;  Σ_i wᵢ diff_factor = 0
    geq_diff = w_v * Gamma * diff_factor * mu.unsqueeze(0)
    geq = geq_adv + geq_diff
    g_out = g - (g - geq) / tau_g

    return f_out, g_out

# ---------------------------------------------------------------------------
# MRT + Smagorinsky multiphase collision operators
# ---------------------------------------------------------------------------


def _mrt_collision_field(
    f, feq, matrix, matrix_inv,
    s_e, s_eps, s_q, s_pi, s_nu_field,
    nz, ny, nx, device,
):
    """MRT collision with spatially-varying stress relaxation (Smagorinsky)."""
    f_flat = f.reshape(19, -1)
    feq_flat = feq.reshape(19, -1)
    s_nu_flat = s_nu_field.reshape(-1)
    s_fixed = torch.tensor(
        [0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, 0.0, s_q,
         1.0, 1.0, 1.0, 1.0, 1.0, s_pi, s_pi, 1.0, 1.0, 1.0],
        dtype=f.dtype, device=device,
    )
    m = matrix @ f_flat
    m_eq = matrix @ feq_flat
    dm = m - m_eq
    m_star = m - s_fixed.unsqueeze(1) * dm
    for k in range(9, 14):
        m_star[k] = m[k] - s_nu_flat * dm[k]
    return (matrix_inv @ m_star).reshape(19, nz, ny, nx)


def _mrt_collision_uniform(
    f, feq, matrix, matrix_inv,
    s_e, s_eps, s_q, s_pi, s_nu,
    nz, ny, nx, device,
):
    """MRT collision with uniform relaxation rates (no Smagorinsky)."""
    f_flat = f.reshape(19, -1)
    feq_flat = feq.reshape(19, -1)
    s_vec = torch.tensor(
        [0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, 0.0, s_q,
         s_nu, s_nu, s_nu, s_nu, s_nu, s_pi, s_pi, 1.0, 1.0, 1.0],
        dtype=f.dtype, device=device,
    )
    m = matrix @ f_flat
    m_eq = matrix @ feq_flat
    m_star = m - s_vec.unsqueeze(1) * (m - m_eq)
    return (matrix_inv @ m_star).reshape(19, nz, ny, nx)


def _guo_correction_3d(ux, uy, uz, Fx, Fy, Fz, s_nu_field, device):
    """Guo forcing correction for a single component (3D)."""
    cs2 = 1.0 / 3.0
    cs4 = cs2 * cs2
    w_v = _w_on_3d(device).view(19, 1, 1, 1)
    c = _c_on_3d(device)
    cx = c[:, 0].float().view(19, 1, 1, 1)
    cy = c[:, 1].float().view(19, 1, 1, 1)
    cz = c[:, 2].float().view(19, 1, 1, 1)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    term_a = ((cx - ux.unsqueeze(0)) * Fx.unsqueeze(0)
              + (cy - uy.unsqueeze(0)) * Fy.unsqueeze(0)
              + (cz - uz.unsqueeze(0)) * Fz.unsqueeze(0))
    term_b = cu * (cx * Fx.unsqueeze(0) + cy * Fy.unsqueeze(0) + cz * Fz.unsqueeze(0))
    return (1.0 - 0.5 * s_nu_field.unsqueeze(0)) * w_v * (term_a / cs2 + term_b / cs4)


def collide_cg_mrt_3d(
    f_r, f_b,
    tau=1.0, A=0.04, beta=0.7,
    gx=0.0, gy=0.0, gz=0.0,
    solid_mask=None,
    s_e=1.19, s_eps=1.4, s_q=1.2,
    s_pi=None, C_s=0.0, use_guo=False,
):
    """Color-Gradient step with MRT collision + optional Smagorinsky LES.

    When ``C_s > 0``: Smagorinsky sub-grid model (waLBerla default 0.1).
    When ``C_s = 0``: pure MRT (OpenLB pattern, more stable than BGK).
    When ``use_guo=True``: Guo second-order forcing (waLBerla GuoField).

    References
    ----------
    waLBerla ``DamBreakRectangular.prm`` (SmagorinskyConstant=0.1)
    OpenLB ``rayleighTaylor3d.cpp`` (MRT for multiphase)
    """
    device = f_r.device
    w_v = _w_on_3d(device).view(19, 1, 1, 1)
    c = _c_on_3d(device)
    cx = c[:, 0].float().view(19, 1, 1, 1)
    cy = c[:, 1].float().view(19, 1, 1, 1)
    cz = c[:, 2].float().view(19, 1, 1, 1)

    f_total = f_r + f_b
    rho_r_s = f_r.sum(dim=0)
    rho_b_s = f_b.sum(dim=0)
    rho = rho_r_s + rho_b_s
    rho_safe = rho.clamp(min=1e-12)
    nz, ny, nx = f_total.shape[1], f_total.shape[2], f_total.shape[3]

    ux = (f_total * cx).sum(dim=0) / rho_safe
    uy = (f_total * cy).sum(dim=0) / rho_safe
    uz = (f_total * cz).sum(dim=0) / rho_safe
    if s_pi is None:
        s_pi = s_e

    matrix, matrix_inv = _get_d3q19_mrt_matrices(device)

    if C_s > 0.0:
        feq = equilibrium3d(rho, ux, uy, uz)
        tau_eff = _smagorinsky_tau(tau, _neq_stress_norm_3d(f_total - feq), rho, C_s)
        s_nu_f = 1.0 / tau_eff
        f_post = _mrt_collision_field(
            f_total, feq, matrix, matrix_inv,
            s_e, s_eps, s_q, s_pi, s_nu_f, nz, ny, nx, device,
        )
        if use_guo:
            f_post += _guo_correction_3d(
                ux, uy, uz, rho * gx, rho * gy, rho * gz, s_nu_f, device,
            )
    else:
        feq = equilibrium3d(rho, ux + tau * gx, uy + tau * gy, uz + tau * gz)
        f_post = _mrt_collision_uniform(
            f_total, feq, matrix, matrix_inv,
            s_e, s_eps, s_q, s_pi, 1.0 / tau, nz, ny, nx, device,
        )

    if solid_mask is not None:
        f_post = torch.where(solid_mask.unsqueeze(0), f_total, f_post)

    # Surface-tension perturbation
    _phi, mag, nhat_x, nhat_y, nhat_z = _grad_phase_field_3d(rho_r_s, rho_b_s)
    ci_dot_n = cx * nhat_x.unsqueeze(0) + cy * nhat_y.unsqueeze(0) + cz * nhat_z.unsqueeze(0)
    f_post += (A / 2.0) * mag.unsqueeze(0) * w_v * (ci_dot_n ** 2 - 1.0 / 3.0)

    # Recoloring
    amp = beta * (rho_r_s * rho_b_s / rho_safe).unsqueeze(0) * ci_dot_n * w_v
    f_r_out = (rho_r_s / rho_safe).unsqueeze(0) * f_post + amp
    f_b_out = (rho_b_s / rho_safe).unsqueeze(0) * f_post - amp
    if solid_mask is not None:
        m4 = solid_mask.unsqueeze(0)
        f_r_out = torch.where(m4, f_r, f_r_out)
        f_b_out = torch.where(m4, f_b, f_b_out)
    return f_r_out, f_b_out


def collide_sc_mrt_3d(
    f1, f2,
    G_12=0.9, tau=1.0,
    gx=0.0, gy=0.0, gz=0.0,
    solid_mask=None,
    s_e=1.19, s_eps=1.4, s_q=1.2,
    s_pi=None, C_s=0.0, use_guo=False,
):
    """Shan-Chen two-component MRT collision + optional Smagorinsky LES.

    MRT independently relaxes non-hydrodynamic moments, improving stability
    for SC multiphase at density ratios > 3:1 (OpenLB pattern).
    """
    device = f1.device
    rho1, ux1, uy1, uz1 = macroscopic3d(f1)
    rho2, ux2, uy2, uz2 = macroscopic3d(f2)

    Fx1, Fy1, Fz1, Fx2, Fy2, Fz2 = sc_two_component_force_3d(
        rho1, rho2, G_12, gx, gy, gz, solid_mask,
    )

    rho1_s = rho1.clamp(min=1e-12)
    rho2_s = rho2.clamp(min=1e-12)
    nz, ny, nx = f1.shape[1], f1.shape[2], f1.shape[3]
    if s_pi is None:
        s_pi = s_e

    matrix, matrix_inv = _get_d3q19_mrt_matrices(device)

    if C_s > 0.0:
        feq1 = equilibrium3d(rho1, ux1, uy1, uz1)
        tau_eff1 = _smagorinsky_tau(tau, _neq_stress_norm_3d(f1 - feq1), rho1, C_s)
        feq2 = equilibrium3d(rho2, ux2, uy2, uz2)
        tau_eff2 = _smagorinsky_tau(tau, _neq_stress_norm_3d(f2 - feq2), rho2, C_s)
        s_nu1 = 1.0 / tau_eff1
        s_nu2 = 1.0 / tau_eff2
        f1_out = _mrt_collision_field(
            f1, feq1, matrix, matrix_inv,
            s_e, s_eps, s_q, s_pi, s_nu1, nz, ny, nx, device,
        )
        f2_out = _mrt_collision_field(
            f2, feq2, matrix, matrix_inv,
            s_e, s_eps, s_q, s_pi, s_nu2, nz, ny, nx, device,
        )
        if use_guo:
            f1_out += _guo_correction_3d(ux1, uy1, uz1, Fx1, Fy1, Fz1, s_nu1, device)
            f2_out += _guo_correction_3d(ux2, uy2, uz2, Fx2, Fy2, Fz2, s_nu2, device)
    else:
        s_nu = 1.0 / tau
        feq1 = equilibrium3d(
            rho1, ux1 + tau * Fx1 / rho1_s,
            uy1 + tau * Fy1 / rho1_s,
            uz1 + tau * Fz1 / rho1_s,
        )
        feq2 = equilibrium3d(
            rho2, ux2 + tau * Fx2 / rho2_s,
            uy2 + tau * Fy2 / rho2_s,
            uz2 + tau * Fz2 / rho2_s,
        )
        f1_out = _mrt_collision_uniform(
            f1, feq1, matrix, matrix_inv,
            s_e, s_eps, s_q, s_pi, s_nu, nz, ny, nx, device,
        )
        f2_out = _mrt_collision_uniform(
            f2, feq2, matrix, matrix_inv,
            s_e, s_eps, s_q, s_pi, s_nu, nz, ny, nx, device,
        )

    if solid_mask is not None:
        m4 = solid_mask.unsqueeze(0)
        f1_out = torch.where(m4, f1, f1_out)
        f2_out = torch.where(m4, f2, f2_out)
    return f1_out, f2_out


# ---------------------------------------------------------------------------
# Hydrostatic pressure initialisation (waLBerla dam break pattern)
# ---------------------------------------------------------------------------


def init_hydrostatic_pressure_3d(f, solid_mask, gy, water_height):
    """Initialise hydrostatic pressure gradient for dam-break IC.

    Mirrors waLBerla's ``initHydrostaticPressure`` in DamBreakRectangular.
    Sets rho(y) = rho0*(1+|gy|*(h-y)/cs²) in fluid, then recomputes f_eq.
    """
    device = f.device
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    ys = torch.arange(ny, dtype=torch.float32, device=device).view(1, ny, 1)
    rho0 = macroscopic3d(f)[0].mean()
    cs2 = 1.0 / 3.0
    hydro = rho0 * (1.0 + abs(gy) * (water_height - ys) / cs2)
    hydro = hydro.clamp(min=rho0 * 0.5, max=rho0 * 2.0)
    fluid = ~solid_mask
    rho_hydro = torch.where(fluid, hydro.expand(nz, ny, nx), macroscopic3d(f)[0])
    zero = torch.zeros((nz, ny, nx), device=device)
    return torch.where(solid_mask.unsqueeze(0), f, equilibrium3d(rho_hydro, zero, zero, zero))
