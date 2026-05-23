from __future__ import annotations

import torch

from .d2q9 import C as C2D, equilibrium, macroscopic
from .d3q19 import C as C3D, equilibrium3d, macroscopic3d


# ---------------------------------------------------------------------------
# 2-D helpers (D2Q9)
# ---------------------------------------------------------------------------

def _pi_neq_norm_2d(f: torch.Tensor, feq: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of the non-equilibrium stress tensor for D2Q9.

    Π^neq_{αβ} = Σ_i c_{i,α} c_{i,β} (f_i − f_i^eq)

    Returns a tensor of shape ``(ny, nx)``.
    """
    device = f.device
    c = C2D.to(device).float()
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    fneq = f - feq

    pi_xx = (cx * cx * fneq).sum(dim=0)
    pi_yy = (cy * cy * fneq).sum(dim=0)
    pi_xy = (cx * cy * fneq).sum(dim=0)

    # |Π|² = π_xx² + π_yy² + 2·π_xy²
    return torch.sqrt(torch.clamp(pi_xx**2 + pi_yy**2 + 2.0 * pi_xy**2, min=0.0))


def collide_smagorinsky_bgk(
    f: torch.Tensor,
    tau_0: float,
    C_s: float = 0.1,
) -> torch.Tensor:
    """Smagorinsky LES BGK collision step for D2Q9.

    Computes a cell-local effective relaxation time from the magnitude of the
    non-equilibrium stress tensor (Hou et al. 1994, *J. Stat. Phys.* **68**):

        τ_eff(x) = τ_0/2 + √(τ_0²/4 + 18 · C_s² · |Π^neq(x)|)

    The sub-grid viscosity is ν_sgs = C_s² · |S̄| where the strain-rate
    magnitude |S̄| is proportional to |Π^neq|.  Conservation of mass and
    momentum is guaranteed because the BGK form is used.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        tau_0: Molecular (minimum) relaxation time; ν_mol = (τ_0 − ½) / 3.
        C_s: Smagorinsky constant (default 0.1; typical range 0.1–0.18).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    pi_norm = _pi_neq_norm_2d(f, feq)

    tau_eff = 0.5 * (tau_0 + torch.sqrt(tau_0**2 + 18.0 * C_s**2 * pi_norm))
    return f - (f - feq) / tau_eff.unsqueeze(0)


# ---------------------------------------------------------------------------
# 3-D helpers (D3Q19)
# ---------------------------------------------------------------------------

def _pi_neq_norm_3d(f: torch.Tensor, feq: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of the non-equilibrium stress tensor for D3Q19.

    Returns a tensor of shape ``(nz, ny, nx)``.
    """
    device = f.device
    c = C3D.to(device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
    fneq = f - feq

    pi_xx = (cx * cx * fneq).sum(dim=0)
    pi_yy = (cy * cy * fneq).sum(dim=0)
    pi_zz = (cz * cz * fneq).sum(dim=0)
    pi_xy = (cx * cy * fneq).sum(dim=0)
    pi_xz = (cx * cz * fneq).sum(dim=0)
    pi_yz = (cy * cz * fneq).sum(dim=0)

    # |Π|² = π_xx² + π_yy² + π_zz² + 2(π_xy² + π_xz² + π_yz²)
    return torch.sqrt(
        torch.clamp(
            pi_xx**2 + pi_yy**2 + pi_zz**2
            + 2.0 * (pi_xy**2 + pi_xz**2 + pi_yz**2),
            min=0.0,
        )
    )


def collide_smagorinsky_bgk3d(
    f: torch.Tensor,
    tau_0: float,
    C_s: float = 0.1,
) -> torch.Tensor:
    """Smagorinsky LES BGK collision step for D3Q19.

    Computes a cell-local effective relaxation time from the magnitude of the
    non-equilibrium stress tensor (Hou et al. 1994):

        τ_eff(x) = τ_0/2 + √(τ_0²/4 + 18 · C_s² · |Π^neq(x)|)

    Conservation of mass and momentum is guaranteed by the BGK form.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau_0: Molecular (minimum) relaxation time; ν_mol = (τ_0 − ½) / 3.
        C_s: Smagorinsky constant (default 0.1; typical range 0.1–0.18).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    pi_norm = _pi_neq_norm_3d(f, feq)

    tau_eff = 0.5 * (tau_0 + torch.sqrt(tau_0**2 + 18.0 * C_s**2 * pi_norm))
    return f - (f - feq) / tau_eff.unsqueeze(0)


__all__ = [
    "collide_smagorinsky_bgk",
    "collide_smagorinsky_bgk3d",
]
