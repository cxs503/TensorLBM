"""Core D3Q19 LBM solver operations: equilibrium, collision, streaming.

Includes MRT collision, regularized BGK (RLBM), and Smagorinsky SGS model for LES.
Optimized for CPU performance.
"""

import torch
from typing import Tuple

from .d3q19 import C, W, CS2


def equilibrium3d(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Compute D3Q19 equilibrium distribution."""
    device = rho.device
    dtype = rho.dtype
    c = C.to(device=device, dtype=dtype)
    w = W.to(device=device, dtype=dtype)
    uc = (c[:, 0].view(-1, 1, 1, 1) * ux.unsqueeze(0) +
          c[:, 1].view(-1, 1, 1, 1) * uy.unsqueeze(0) +
          c[:, 2].view(-1, 1, 1, 1) * uz.unsqueeze(0))
    u2 = ux**2 + uy**2 + uz**2
    feq = w.view(-1, 1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0)
    )
    return feq


def macroscopic3d(f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute macroscopic quantities from distributions."""
    device = f.device
    dtype = f.dtype
    c = C.to(device=device, dtype=dtype)
    rho = f.sum(dim=0)
    mom_x = (f * c[:, 0].view(-1, 1, 1, 1)).sum(dim=0)
    mom_y = (f * c[:, 1].view(-1, 1, 1, 1)).sum(dim=0)
    mom_z = (f * c[:, 2].view(-1, 1, 1, 1)).sum(dim=0)
    ux = mom_x / rho
    uy = mom_y / rho
    uz = mom_z / rho
    return rho, ux, uy, uz


def stream3d(f: torch.Tensor) -> torch.Tensor:
    """Stream distributions with periodic BC in x and z."""
    nf, nz, ny, nx = f.shape
    device = f.device
    c = C.to(device=device)
    f_out = torch.empty_like(f)
    for i in range(nf):
        cx, cy, cz = int(c[i, 0]), int(c[i, 1]), int(c[i, 2])
        if cx == 0 and cy == 0 and cz == 0:
            f_out[i] = f[i]
        else:
            f_out[i] = torch.roll(f[i], shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
    return f_out


def collide_bgk(f: torch.Tensor, tau: float) -> torch.Tensor:
    """BGK (SRT) collision."""
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    return f - (f - feq) / tau


def collide_mrt(f: torch.Tensor, tau: float) -> torch.Tensor:
    """D3Q19 MRT (Multiple Relaxation Time) collision.

    Implements the d'Humières moment basis MRT collision operator using
    the polynomial moment approach (avoids explicit M matrix inversion).

    Parameters
    ----------
    f : torch.Tensor, shape (19, nz, ny, nx)
        Particle distribution functions.
    tau : float
        Relaxation time for shear modes (s_ν = 1/τ).

    Returns
    -------
    f_out : torch.Tensor, shape (19, nz, ny, nx)
        Post-collision distribution functions.

    Notes
    -----
    The relaxation matrix S is diagonal with:
    - s = 0 for conserved moments (ρ, jx, jy, jz)
    - s = 1/τ for shear stress modes
    - s = s_e for energy modes (typically 1.0-1.5)
    - s = s_q for energy flux modes (typically 1.0-1.5)

    Reference: d'Humières et al., Phil. Trans. R. Soc. Lond. A 360, 437 (2002).
    """
    device = f.device
    dtype = f.dtype

    # Compute macroscopic quantities
    rho, ux, uy, uz = macroscopic3d(f)

    # Compute equilibrium distribution
    feq = equilibrium3d(rho, ux, uy, uz)

    # D3Q19 lattice velocities
    c = C.to(device=device, dtype=dtype)
    cx = c[:, 0].view(-1, 1, 1, 1)
    cy = c[:, 1].view(-1, 1, 1, 1)
    cz = c[:, 2].view(-1, 1, 1, 1)
    c2 = cx * cx + cy * cy + cz * cz  # |c|²

    # Compute non-equilibrium distribution
    fneq = f - feq

    # Compute stress tensor from fneq (second-order moments)
    Pi_xx = (fneq * cx * cx).sum(dim=0)
    Pi_yy = (fneq * cy * cy).sum(dim=0)
    Pi_zz = (fneq * cz * cz).sum(dim=0)
    Pi_xy = (fneq * cx * cy).sum(dim=0)
    Pi_yz = (fneq * cy * cz).sum(dim=0)
    Pi_xz = (fneq * cx * cz).sum(dim=0)

    # Compute energy-related moments from fneq
    # Energy: sum(|c|² * fneq)
    E_neq = (fneq * c2).sum(dim=0)
    # Energy flux: sum(c_α * |c|² * fneq)
    Qx_neq = (fneq * cx * c2).sum(dim=0)
    Qy_neq = (fneq * cy * c2).sum(dim=0)
    Qz_neq = (fneq * cz * c2).sum(dim=0)

    # Relaxation rates
    s_nu = 1.0 / tau  # shear mode relaxation
    s_e = 1.0         # energy mode relaxation (bulk viscosity)
    s_q = 1.0         # energy flux mode relaxation

    # For D3Q19 MRT, we relax the non-equilibrium moments and reconstruct f
    # The key insight: fneq can be decomposed into orthogonal moment components
    # We relax each component with its own rate and reconstruct

    # Reconstruct relaxed fneq from relaxed moments
    # This uses the Hermite polynomial expansion approach

    # Shear stress modes (relaxed with s_nu)
    # These are the traceless part of the stress tensor
    Pi_xx_dev = Pi_xx - (Pi_xx + Pi_yy + Pi_zz) / 3.0
    Pi_yy_dev = Pi_yy - (Pi_xx + Pi_yy + Pi_zz) / 3.0
    Pi_zz_dev = Pi_zz - (Pi_xx + Pi_yy + Pi_zz) / 3.0

    # Energy mode (relaxed with s_e)
    # Trace of stress tensor
    Pi_trace = Pi_xx + Pi_yy + Pi_zz

    # Reconstruct fneq from moments using the polynomial basis
    # fneq_i = w_i * [ (c_iα c_iβ - cs² δ_αβ) * Π_αβ / (2 cs⁴) + ... ]
    # For D3Q19, cs² = 1/3

    cs2 = CS2
    cs4 = cs2 * cs2

    # Shear stress contribution (relaxed)
    # (c_iα c_iβ - cs² δ_αβ) * Π_αβ_relaxed / (2 cs⁴)
    # Π_αβ_relaxed = (1 - s_nu) * Π_αβ for shear modes
    Pi_xx_r = (1.0 - s_nu) * Pi_xx_dev
    Pi_yy_r = (1.0 - s_nu) * Pi_yy_dev
    Pi_zz_r = (1.0 - s_nu) * Pi_zz_dev
    Pi_xy_r = (1.0 - s_nu) * Pi_xy
    Pi_yz_r = (1.0 - s_nu) * Pi_yz
    Pi_xz_r = (1.0 - s_nu) * Pi_xz

    # Energy mode contribution (relaxed with s_e)
    # The trace part is relaxed differently
    Pi_trace_r = (1.0 - s_e) * Pi_trace / 3.0  # isotropic part

    # Total relaxed stress
    Pi_xx_total = Pi_xx_r + Pi_trace_r
    Pi_yy_total = Pi_yy_r + Pi_trace_r
    Pi_zz_total = Pi_zz_r + Pi_trace_r

    # Reconstruct fneq from relaxed stress tensor
    # fneq_i = w_i * (c_iα c_iβ - cs² δ_αβ) * Π_αβ / (2 cs⁴)
    fneq_reg = W.to(device=device, dtype=dtype).view(-1, 1, 1, 1) / (2.0 * cs4) * (
        (cx * cx - cs2) * Pi_xx_total +
        (cy * cy - cs2) * Pi_yy_total +
        (cz * cz - cs2) * Pi_zz_total +
        2.0 * cx * cy * Pi_xy_r +
        2.0 * cy * cz * Pi_yz_r +
        2.0 * cx * cz * Pi_xz_r
    )

    # Energy flux contribution (relaxed with s_q)
    # This is a third-order moment contribution
    # fneq_i += w_i * c_iα * (c_i² - D cs² - 2) * Q_α / (2 cs² (D cs² + 2))
    # For D3Q19, D=3, cs²=1/3, so D cs² + 2 = 3
    Qx_r = (1.0 - s_q) * Qx_neq
    Qy_r = (1.0 - s_q) * Qy_neq
    Qz_r = (1.0 - s_q) * Qz_neq

    fneq_reg = fneq_reg + W.to(device=device, dtype=dtype).view(-1, 1, 1, 1) / (2.0 * cs2 * 3.0) * (
        cx * (c2 - 3.0 * cs2 - 2.0) * Qx_r +
        cy * (c2 - 3.0 * cs2 - 2.0) * Qy_r +
        cz * (c2 - 3.0 * cs2 - 2.0) * Qz_r
    )

    # Collision: f* = feq + fneq_reg
    f_out = feq + fneq_reg

    return f_out


def collide_smagorinsky_mrt3d(
    f: torch.Tensor, tau: float, C_s: float = 0.1, delta: float = 1.0,
) -> torch.Tensor:
    """D3Q19 MRT collision with Smagorinsky SGS model."""
    device = f.device
    dtype = f.dtype
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    fneq = f - feq
    c = C.to(device=device, dtype=dtype)
    c0 = c[:, 0].view(-1, 1, 1, 1)
    c1 = c[:, 1].view(-1, 1, 1, 1)
    c2 = c[:, 2].view(-1, 1, 1, 1)
    Pi_xx = (fneq * c0 * c0).sum(0)
    Pi_yy = (fneq * c1 * c1).sum(0)
    Pi_zz = (fneq * c2 * c2).sum(0)
    Pi_xy = (fneq * c0 * c1).sum(0)
    Pi_xz = (fneq * c0 * c2).sum(0)
    Pi_yz = (fneq * c1 * c2).sum(0)
    Pi_Q = torch.sqrt(2.0 * (
        Pi_xx**2 + Pi_yy**2 + Pi_zz**2 +
        2.0 * Pi_xy**2 + 2.0 * Pi_xz**2 + 2.0 * Pi_yz**2
    ).clamp(min=1e-20))
    cs2 = CS2
    Q = 18.0 * (C_s * delta)**2 * Pi_Q / rho
    tau_eff = 0.5 * (tau + torch.sqrt(tau**2 + Q / cs2))
    tau_eff = tau_eff.clamp(min=0.5001, max=2.0)
    omega = 1.0 / tau_eff.unsqueeze(0)
    return f - omega * (f - feq)


# --------------------------------------------------------------------------- #
# D2Q9 Regularized BGK (RLBM) collision — Latt & Chopard (2006)
# --------------------------------------------------------------------------- #

def collide_rlbm_2d(
    f: torch.Tensor,
    tau: float,
    cs2: float = 1.0 / 3.0,
    fluid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """D2Q9 regularized BGK collision (Latt & Chopard, Phys. Fluids 18, 2006).

    Reconstructs non-equilibrium from the strain-rate tensor, filtering out
    non-hydrodynamic (ghost) modes that cause instability at low viscosity.

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
    tau : float, relaxation time
    cs2 : float, speed of sound squared (1/3 for D2Q9)
    fluid_mask : torch.Tensor, shape (ny, nx), bool. If provided, only
        fluid nodes are updated; solid nodes are preserved.

    Returns
    -------
    f_out : torch.Tensor, shape (9, ny, nx)
    """
    from .d2q9 import C as C2Q9, W as W2Q9

    device = f.device
    dtype = f.dtype
    c = C2Q9.to(device=device, dtype=dtype)
    w = W2Q9.to(device=device, dtype=dtype)

    # Macroscopic
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * c[:, 0].view(-1, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(-1, 1, 1)).sum(dim=0) / rho_safe

    # Equilibrium
    u2 = ux**2 + uy**2
    cu = c[:, 0].view(-1, 1, 1) * ux + c[:, 1].view(-1, 1, 1) * uy
    feq = w.view(-1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2.unsqueeze(0)
    )

    fneq = f - feq

    # Non-equilibrium stress tensor
    cix = c[:, 0].view(-1, 1, 1)
    ciy = c[:, 1].view(-1, 1, 1)
    Pi_xx = (fneq * cix * cix).sum(dim=0)
    Pi_yy = (fneq * ciy * ciy).sum(dim=0)
    Pi_xy = (fneq * cix * ciy).sum(dim=0)
    Pi_trace = Pi_xx + Pi_yy

    # Reconstruct regularized f^neq
    fneq_reg = w.view(-1, 1, 1) / (2.0 * cs2 * cs2) * (
        cix * cix * Pi_xx +
        ciy * ciy * Pi_yy +
        2.0 * cix * ciy * Pi_xy -
        cs2 * Pi_trace
    )

    # Collision
    omega = 1.0 / tau
    f_out = feq + (1.0 - omega) * fneq_reg

    # Preserve solid nodes if mask provided
    if fluid_mask is not None:
        f_out[:, ~fluid_mask] = f[:, ~fluid_mask]

    return f_out
