"""Momentum-flux pressure measurement for SC/D2Q9 two-phase LBM.

In the Shan-Chen model, the mechanical pressure has two contributions:
1. Kinetic pressure: p_kin = ρcs² (from the momentum flux tensor)
2. Interaction pressure: p_int = (G/2) cs² ψ² (from the SC force)

The total thermodynamic pressure is:
    p = p_kin + p_int = ρcs² + (G/2) cs² ψ²

The momentum flux tensor Π_αβ = Σᵢ fᵢ cᵢα cᵢβ only captures the kinetic
part.  To get the full pressure, we must add the interaction contribution.

This module provides functions to compute the full thermodynamic pressure
and measure the pressure jump across a curved interface.

References
----------
Wagner & Pohl (2004) Europhys. Lett. 65, 145.
Shan & Chen (1994) Phys. Rev. E 49, 2941.
"""

from __future__ import annotations

import torch

from .sc_core import SCState, psi_exp


def momentum_flux_tensor(state: SCState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the momentum flux tensor Π_αβ = Σᵢ fᵢ cᵢα cᵢβ.

    This gives the KINETIC pressure contribution only.

    Args:
        state: SC simulation state with populations f.

    Returns:
        (Pi_xx, Pi_xy, Pi_yy) each of shape (ny, nx).
    """
    f = state.f
    device = f.device
    dtype = f.dtype

    # D2Q9 velocities
    c = torch.tensor([
        [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, 1], [-1, -1], [1, -1],
    ], device=device, dtype=dtype)

    cx = c[:, 0].view(9, 1, 1)  # (9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    # Π_αβ = Σᵢ fᵢ cᵢα cᵢβ
    Pi_xx = (f * cx * cx).sum(dim=0)  # (ny, nx)
    Pi_xy = (f * cx * cy).sum(dim=0)
    Pi_yy = (f * cy * cy).sum(dim=0)

    return Pi_xx, Pi_xy, Pi_yy


def momentum_flux_pressure(state: SCState) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract thermodynamic pressure from momentum flux + interaction.

    The full thermodynamic pressure is:
        p = ρcs² + (G/2) cs² ψ²

    The momentum flux tensor gives only ρcs², so we add the interaction
    contribution (G/2) cs² ψ².

    Args:
        state: SC simulation state.

    Returns:
        (p, p_aniso) where p is the isotropic pressure and p_aniso is
        the anisotropic part (Π_xx - Π_yy)/2.
    """
    Pi_xx, Pi_xy, Pi_yy = momentum_flux_tensor(state)

    # Kinetic pressure (trace / 2 for 2D)
    p_kin = (Pi_xx + Pi_yy) / 2.0

    # Interaction pressure: p_int = (G/2) cs² ψ²
    cs2 = 1.0 / 3.0
    G = state.config.G
    psi = state.config.psi_fn(state.rho)
    p_int = (G / 2.0) * cs2 * psi**2

    # Total thermodynamic pressure
    p = p_kin + p_int

    # Anisotropic part (deviatoric stress) - only from kinetic part
    p_aniso = (Pi_xx - Pi_yy) / 2.0

    return p, p_aniso


def pressure_jump_droplet(
    state: SCState,
    center: tuple[float, float],
    R_target: float,
    interface_width: float = 3.0,
) -> dict[str, float]:
    """Measure the pressure jump across a droplet interface.

    Samples the pressure at two radial positions:
    - Inside: r = R_target - 2*interface_width (well inside the droplet)
    - Outside: r = R_target + 2*interface_width (well outside the droplet)

    The pressure jump Δp = p_inside - p_outside should equal σ/R for a
    2D cylindrical droplet.

    Args:
        state: SC simulation state with a droplet.
        center: (cx, cy) center of the droplet in lattice units.
        R_target: Expected droplet radius.
        interface_width: Width of the diffuse interface (for sampling positions).

    Returns:
        Dictionary with keys:
        - 'p_inside': average pressure inside the droplet
        - 'p_outside': average pressure outside the droplet
        - 'dp': pressure jump p_inside - p_outside
        - 'dp_relative': dp / p_outside
        - 'R_inner': inner sampling radius
        - 'R_outer': outer sampling radius
    """
    p, _ = momentum_flux_pressure(state)

    ny, nx = p.shape
    device = p.device
    dtype = p.dtype

    # Create radial distance field from droplet center
    y = torch.arange(ny, device=device, dtype=dtype)
    x = torch.arange(nx, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    cx, cy = center
    # Periodic distance
    dx = X - cx
    dy = Y - cy
    # Handle periodic wrapping
    dx = dx - nx * torch.round(dx / nx)
    dy = dy - ny * torch.round(dy / ny)
    r = torch.sqrt(dx**2 + dy**2)

    # Sampling radii
    R_inner = R_target - 2.0 * interface_width
    R_outer = R_target + 2.0 * interface_width

    # Masks for sampling regions (annular regions to avoid interface)
    mask_inner = (r >= R_inner - 1.0) & (r <= R_inner + 1.0)
    mask_outer = (r >= R_outer - 1.0) & (r <= R_outer + 1.0)

    # Average pressure in each region
    if mask_inner.sum() > 0:
        p_inside = p[mask_inner].mean().item()
    else:
        p_inside = float('nan')

    if mask_outer.sum() > 0:
        p_outside = p[mask_outer].mean().item()
    else:
        p_outside = float('nan')

    dp = p_inside - p_outside
    dp_relative = dp / p_outside if p_outside > 0 else float('nan')

    return {
        'p_inside': p_inside,
        'p_outside': p_outside,
        'dp': dp,
        'dp_relative': dp_relative,
        'R_inner': R_inner,
        'R_outer': R_outer,
    }
