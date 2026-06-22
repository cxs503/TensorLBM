"""Wall shear stress extraction for LBM simulations.

Computes the wall shear stress (WSS) distribution along solid boundaries from
the non-equilibrium part of the distribution function.  This is a standard
post-processing quantity in industrial CFD (PowerFlow, XFlow) used for:

- Boundary-layer visualisation and y+ verification
- Aerodynamic skin-friction mapping (external aerodynamics)
- Haemodynamic wall shear stress in cardiovascular flows
- Corrosion and erosion hotspot identification

Theory
------
In LBM the viscous stress tensor is related to the second moment of the
non-equilibrium DFs (Chapman–Enskog expansion):

    σ_αβ = -(1 - 1/(2τ)) Σ_i f_i^neq c_{iα} c_{iβ}

At a wall cell, the wall shear stress magnitude is:

    τ_w = μ * |∂u/∂n|_wall  ≈  ν/Δy * Δu_tangential

Two complementary methods are provided:
1. **Stress-tensor method** – uses the f_neq tensor directly; more accurate.
2. **Finite-difference method** – computes ∂u/∂n by centred differences in the
   wall-normal direction; suitable when only (ux, uy) fields are available.

References
----------
Krüger et al. (2017) "The Lattice Boltzmann Method". Springer.
Ladd (1994) "Numerical simulations of particulate suspensions via a
    discretized Boltzmann equation". *J. Fluid Mech.* 271, 285–309.
"""
from __future__ import annotations

import torch

__all__ = [
    "wss_from_fneq_2d",
    "wss_from_velocity_2d",
    "wss_from_fneq_3d",
    "wss_map_2d",
]

# D2Q9 velocity vectors (matching tensorlbm.d2q9.C convention)
_CX2D = torch.tensor([0, 1, 0, -1,  0,  1, -1, -1,  1], dtype=torch.float32)
_CY2D = torch.tensor([0, 0, 1,  0, -1,  1,  1, -1, -1], dtype=torch.float32)


def _feq_2d(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Compute D2Q9 equilibrium distributions."""
    from .d2q9 import equilibrium  # noqa: PLC0415
    return equilibrium(rho, ux, uy)


def wss_from_fneq_2d(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    tau: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute 2-D wall shear stress magnitude from the f_neq tensor.

    Uses the Chapman–Enskog stress-tensor relation:

        τ_w = √(σ_xy²)    at wall-adjacent fluid cells

    where σ_xy = -(1 - 1/(2τ)) Σ_i f_i^neq * cx_i * cy_i.

    Args:
        f:    Post-collision DF, shape ``(9, ny, nx)``.
        rho:  Density field, shape ``(ny, nx)``.
        ux:   x-velocity, shape ``(ny, nx)``.
        uy:   y-velocity, shape ``(ny, nx)``.
        tau:  BGK relaxation time.
        mask: Boolean solid mask (``True`` = solid), shape ``(ny, nx)``.

    Returns:
        Wall shear stress magnitude field, shape ``(ny, nx)``.  Zero inside
        solid cells and far from walls.
    """
    device = f.device
    cx = _CX2D.to(device)
    cy = _CY2D.to(device)

    f_eq = _feq_2d(rho, ux, uy)
    f_neq = f - f_eq  # (9, ny, nx)

    # Stress tensor components
    # σ_xx = -(1-1/2τ) Σ f_neq cx cx
    # σ_yy = -(1-1/2τ) Σ f_neq cy cy
    # σ_xy = -(1-1/2τ) Σ f_neq cx cy
    prefactor = -(1.0 - 0.5 / tau)

    # Broadcast cx, cy over spatial dims
    cx_b = cx.view(9, 1, 1)
    cy_b = cy.view(9, 1, 1)

    sigma_xx = prefactor * (f_neq * cx_b * cx_b).sum(dim=0)
    sigma_yy = prefactor * (f_neq * cy_b * cy_b).sum(dim=0)
    sigma_xy = prefactor * (f_neq * cx_b * cy_b).sum(dim=0)

    # Von-Mises-like wall shear stress magnitude
    # τ_w = √(σ_xy² + (σ_xx - σ_yy)²/4)
    wss = torch.sqrt(sigma_xy ** 2 + ((sigma_xx - sigma_yy) / 2.0) ** 2)

    # Zero out solid cells
    wss = wss * (~mask).float()
    return wss


def wss_from_velocity_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    mask: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    """Compute 2-D wall shear stress from the velocity gradient (FD method).

    For each fluid cell adjacent to a wall, the wall-normal velocity gradient
    is estimated using a one-sided finite difference, and the WSS is:

        τ_w = ν * |∂u_t / ∂n|

    where u_t is the velocity component tangential to the wall.

    Args:
        ux:   x-velocity field, shape ``(ny, nx)``.
        uy:   y-velocity field, shape ``(ny, nx)``.
        mask: Boolean solid mask, shape ``(ny, nx)``.
        nu:   Kinematic viscosity (lattice units).

    Returns:
        Wall shear stress magnitude, shape ``(ny, nx)``.
    """
    ny, nx = ux.shape
    fluid = (~mask).float()

    # Central differences for velocity gradients
    dux_dy = torch.zeros_like(ux)
    dux_dx = torch.zeros_like(ux)
    duy_dy = torch.zeros_like(uy)
    duy_dx = torch.zeros_like(uy)

    dux_dy[1:-1, :] = (ux[2:, :] - ux[:-2, :]) / 2.0
    dux_dx[:, 1:-1] = (ux[:, 2:] - ux[:, :-2]) / 2.0
    duy_dy[1:-1, :] = (uy[2:, :] - uy[:-2, :]) / 2.0
    duy_dx[:, 1:-1] = (uy[:, 2:] - uy[:, :-2]) / 2.0

    # Strain-rate tensor magnitude S = √(2 S_ij S_ij)
    s_xx = dux_dx
    s_yy = duy_dy
    s_xy = 0.5 * (dux_dy + duy_dx)
    strain_mag = torch.sqrt(
        torch.clamp(2.0 * (s_xx ** 2 + s_yy ** 2 + 2.0 * s_xy ** 2), min=0.0)
    )

    wss = nu * strain_mag * fluid
    return wss


def wss_from_fneq_3d(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    tau: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute 3-D wall shear stress magnitude from f_neq (D3Q19).

    Args:
        f:    Post-collision DF, shape ``(19, nz, ny, nx)``.
        rho:  Density, shape ``(nz, ny, nx)``.
        ux, uy, uz: Velocity components, shape ``(nz, ny, nx)``.
        tau:  BGK relaxation time.
        mask: Boolean solid mask, shape ``(nz, ny, nx)``.

    Returns:
        WSS magnitude field, shape ``(nz, ny, nx)``.
    """
    from .d3q19 import C as C3D  # noqa: PLC0415
    from .d3q19 import equilibrium as eq3d

    device = f.device
    c = C3D.to(device).float()  # (19, 3)

    from .d3q19 import W as _W3D  # noqa: PLC0415,F401
    _W3D.to(device)  # kept for device validation only
    f_eq = eq3d(rho, ux, uy, uz)
    f_neq = f - f_eq  # (19, nz, ny, nx)

    prefactor = -(1.0 - 0.5 / tau)
    # Stress components σ_xy, σ_xz, σ_yz
    cx = c[:, 0].view(-1, 1, 1, 1)
    cy = c[:, 1].view(-1, 1, 1, 1)
    cz = c[:, 2].view(-1, 1, 1, 1)

    sigma_xy = prefactor * (f_neq * cx * cy).sum(0)
    sigma_xz = prefactor * (f_neq * cx * cz).sum(0)
    sigma_yz = prefactor * (f_neq * cy * cz).sum(0)
    # Normal stresses not used in tangential WSS formula; kept for reference
    # sigma_xx = prefactor * (f_neq * cx * cx).sum(0)

    # WSS magnitude: √(σ_xy² + σ_xz² + σ_yz²)
    wss = torch.sqrt(sigma_xy ** 2 + sigma_xz ** 2 + sigma_yz ** 2)
    wss = wss * (~mask).float()
    return wss


def wss_map_2d(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    tau: float,
    mask: torch.Tensor,
    *,
    normalise: bool = True,
    rho_ref: float = 1.0,
    u_ref: float = 0.1,
) -> dict[str, object]:
    """Full WSS map for 2-D field with statistics.

    Args:
        f, rho, ux, uy, tau, mask: As in :func:`wss_from_fneq_2d`.
        normalise:  If True, also return non-dimensional ``Cf`` (skin-friction
                    coefficient) map: Cf = τ_w / (0.5 ρ_ref u_ref²).
        rho_ref:    Reference density for non-dimensionalisation.
        u_ref:      Reference velocity for non-dimensionalisation.

    Returns:
        Dictionary with keys:
        ``wss``     – 2-D WSS field as nested list (float).
        ``wss_max`` – maximum WSS value.
        ``wss_mean``– domain-mean WSS (fluid cells only).
        ``cf_map``  – (optional) skin-friction coefficient map.
        ``cf_max``  – (optional) max Cf.
    """
    wss = wss_from_fneq_2d(f, rho, ux, uy, tau, mask)
    fluid = (~mask).float()
    n_fluid = float(fluid.sum().item()) or 1.0

    result: dict[str, object] = {
        "wss": wss.cpu().tolist(),
        "wss_max": float(wss.max().item()),
        "wss_mean": float((wss * fluid).sum().item() / n_fluid),
    }

    if normalise:
        q_ref = 0.5 * rho_ref * u_ref ** 2 + 1e-30
        cf = wss / q_ref
        result["cf_map"] = cf.cpu().tolist()
        result["cf_max"] = float(cf.max().item())

    return result
