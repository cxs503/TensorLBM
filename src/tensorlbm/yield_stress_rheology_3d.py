"""3D yield-stress fluid rheology: Papanastasiou & Herschel-Bulkley regularisation for D3Q19.

Extends the 2D yield-stress rheology to 3D (D3Q19) with proper tensor
contraction for the shear-rate recovery from non-equilibrium stress.

The key difference from 2D is the full 3D stress tensor:

    Pi_ab^neq = -2 * c_s^2 * tau * rho * S_ab

with c_s^2 = 1/3, and the shear rate:

    gamma_dot = sqrt(2 * S:S) = 3 * |Pi^neq| / (sqrt(2) * rho * tau)

where |Pi^neq| = sqrt(Pi_xx^2 + Pi_yy^2 + Pi_zz^2 + 2*Pi_xy^2 + 2*Pi_xz^2 + 2*Pi_yz^2).

References
----------
* Papanastasiou, T.C. (1987) *J. Rheol.* 31, 385.
* Bingham, E.C. (1922) *Fluidity and Plasticity*, McGraw-Hill.
* Herschel, W.H. & Bulkley, R. (1926) *Proc. Am. Soc. Test. Mater.* 26, 621.
"""

from __future__ import annotations

import math

import torch

from .yield_stress_rheology import (
    papanastasiou_viscosity,
    herschel_bulkley_viscosity,
    bingham_viscosity,
    bi_viscosity_model,
    effective_relaxation_time,
    YieldStressRheologyError,
    BinghamParams,
    HerschelBulkleyParams,
)

__all__ = [
    "neq_stress_3d",
    "shear_rate_from_stress_3d",
    "compute_tau_field_bingham_3d",
    "compute_tau_field_hb_3d",
    "strain_rate_shear_rate_3d",
]


def neq_stress_3d(
    f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Macroscopic state and full 3D non-equilibrium stress tensor.

    Args:
        f: Distribution tensor of shape (19, nz, ny, nx).

    Returns:
        (rho, ux, uy, uz, Pi_xx, Pi_yy, Pi_zz, Pi_xy, Pi_xz, Pi_yz)
        where Pi_ab = sum_i c_ia c_ib (f_i - f_i^eq).
    """
    from .d3q19 import C, equilibrium3d, macroscopic3d

    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    fneq = f - feq

    c = C.to(device=f.device, dtype=f.dtype)  # (19, 3)
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)

    pi_xx = (cx * cx * fneq).sum(0)
    pi_yy = (cy * cy * fneq).sum(0)
    pi_zz = (cz * cz * fneq).sum(0)
    pi_xy = (cx * cy * fneq).sum(0)
    pi_xz = (cx * cz * fneq).sum(0)
    pi_yz = (cy * cz * fneq).sum(0)

    return rho, ux, uy, uz, pi_xx, pi_yy, pi_zz, pi_xy, pi_xz, pi_yz


def shear_rate_from_stress_3d(
    pi_xx: torch.Tensor,
    pi_yy: torch.Tensor,
    pi_zz: torch.Tensor,
    pi_xy: torch.Tensor,
    pi_xz: torch.Tensor,
    pi_yz: torch.Tensor,
    rho: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Recover shear rate from 3D non-equilibrium stress (consistent with LBM).

    From Pi_ab^neq = -2 * c_s^2 * tau * rho * S_ab with c_s^2 = 1/3:

        gamma_dot = sqrt(2 * S:S) = 3 * |Pi^neq| / (sqrt(2) * rho * tau)

    where |Pi^neq| = sqrt(Pi_xx^2 + Pi_yy^2 + Pi_zz^2 + 2*Pi_xy^2 + 2*Pi_xz^2 + 2*Pi_yz^2).

    Args:
        pi_xx, pi_yy, pi_zz: Normal stress components.
        pi_xy, pi_xz, pi_yz: Shear stress components.
        rho: Density field.
        tau: Relaxation time field.

    Returns:
        Shear rate field.
    """
    pi_norm = torch.sqrt(
        torch.clamp(
            pi_xx * pi_xx + pi_yy * pi_yy + pi_zz * pi_zz
            + 2.0 * pi_xy * pi_xy + 2.0 * pi_xz * pi_xz + 2.0 * pi_yz * pi_yz,
            min=0.0,
        )
    )
    rho_safe = torch.clamp(rho, min=1e-12)
    tau_safe = torch.clamp(tau, min=1e-12)
    gamma = 3.0 * pi_norm / (math.sqrt(2.0) * rho_safe * tau_safe)
    return gamma


def strain_rate_shear_rate_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Compute shear rate from velocity gradients via central differences (3D).

    gamma_dot = sqrt(2 * S:S) where S_ab = 0.5 * (du_a/dx_b + du_b/dx_a).

    Uses central differences in the interior and one-sided at boundaries.

    Args:
        ux, uy, uz: Velocity component fields of shape (nz, ny, nx).

    Returns:
        Shear rate field of shape (nz, ny, nx).
    """
    # Central differences (interior), one-sided at boundaries
    def grad_x(field):
        g = torch.zeros_like(field)
        g[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / 2.0
        g[:, :, 0] = field[:, :, 1] - field[:, :, 0]
        g[:, :, -1] = field[:, :, -1] - field[:, :, -2]
        return g

    def grad_y(field):
        g = torch.zeros_like(field)
        g[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / 2.0
        g[:, 0, :] = field[:, 1, :] - field[:, 0, :]
        g[:, -1, :] = field[:, -1, :] - field[:, -2, :]
        return g

    def grad_z(field):
        g = torch.zeros_like(field)
        g[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / 2.0
        g[0, :, :] = field[1, :, :] - field[0, :, :]
        g[-1, :, :] = field[-1, :, :] - field[-2, :, :]
        return g

    dudx = grad_x(ux)
    dudy = grad_y(ux)
    dudz = grad_z(ux)
    dvdx = grad_x(uy)
    dvdy = grad_y(uy)
    dvdz = grad_z(uy)
    dwdx = grad_x(uz)
    dwdy = grad_y(uz)
    dwdz = grad_z(uz)

    # S:S = S_xx^2 + S_yy^2 + S_zz^2 + 2*S_xy^2 + 2*S_xz^2 + 2*S_yz^2
    s_xx = dudx
    s_yy = dvdy
    s_zz = dwdz
    s_xy = 0.5 * (dudy + dvdx)
    s_xz = 0.5 * (dudz + dwdx)
    s_yz = 0.5 * (dwdy + dvdz)

    gamma = torch.sqrt(
        torch.clamp(
            2.0 * (s_xx * s_xx + s_yy * s_yy + s_zz * s_zz)
            + 4.0 * (s_xy * s_xy + s_xz * s_xz + s_yz * s_yz),
            min=0.0,
        )
    )
    return gamma


def compute_tau_field_bingham_3d(
    f: torch.Tensor,
    tau_y: float,
    mu_0: float,
    m: float,
    *,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    use_neq: bool = True,
) -> torch.Tensor:
    """Compute per-cell relaxation time for Papanastasiou-regularised Bingham (3D).

    Args:
        f: Distribution tensor (19, nz, ny, nx).
        tau_y: Yield stress.
        mu_0: Plastic viscosity.
        m: Papanastasiou regularisation parameter.
        nu_min: Minimum kinematic viscosity.
        nu_max: Maximum kinematic viscosity.
        tau_min: Minimum relaxation time.
        tau_max: Optional maximum relaxation time.
        use_neq: If True, use consistent shear rate from non-equilibrium stress.

    Returns:
        Relaxation time field (nz, ny, nx).
    """
    rho, _, _, _, pi_xx, pi_yy, pi_zz, pi_xy, pi_xz, pi_yz = neq_stress_3d(f)

    if use_neq:
        tau_guess = torch.full_like(rho, 0.5 + 3.0 * mu_0)
        gamma = shear_rate_from_stress_3d(pi_xx, pi_yy, pi_zz, pi_xy, pi_xz, pi_yz, rho, tau_guess)
    else:
        _, ux, uy, uz = macroscopic3d_from_f(f)
        gamma = strain_rate_shear_rate_3d(ux, uy, uz)

    mu_eff = papanastasiou_viscosity(gamma, tau_y, mu_0, m, mu_plast=None)
    nu_eff = mu_eff  # rho = 1 in lattice units
    nu_eff = torch.clamp(nu_eff, min=nu_min, max=nu_max)
    tau = effective_relaxation_time(nu_eff, rho=1.0, tau_min=tau_min, tau_max=tau_max)
    return tau


def compute_tau_field_hb_3d(
    f: torch.Tensor,
    tau_y: float,
    K: float,
    n: float,
    m: float,
    *,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    use_neq: bool = True,
) -> torch.Tensor:
    """Compute per-cell relaxation time for Papanastasiou-regularised Herschel-Bulkley (3D).

    Args:
        f: Distribution tensor (19, nz, ny, nx).
        tau_y: Yield stress.
        K: Consistency index.
        n: Flow index.
        m: Papanastasiou regularisation parameter.
        nu_min: Minimum kinematic viscosity.
        nu_max: Maximum kinematic viscosity.
        tau_min: Minimum relaxation time.
        tau_max: Optional maximum relaxation time.
        use_neq: If True, use consistent shear rate from non-equilibrium stress.

    Returns:
        Relaxation time field (nz, ny, nx).
    """
    rho, _, _, _, pi_xx, pi_yy, pi_zz, pi_xy, pi_xz, pi_yz = neq_stress_3d(f)

    if use_neq:
        tau_guess = torch.full_like(rho, 0.5 + 3.0 * K)
        gamma = shear_rate_from_stress_3d(pi_xx, pi_yy, pi_zz, pi_xy, pi_xz, pi_yz, rho, tau_guess)
    else:
        _, ux, uy, uz = macroscopic3d_from_f(f)
        gamma = strain_rate_shear_rate_3d(ux, uy, uz)

    mu_eff = herschel_bulkley_viscosity(gamma, tau_y, K, n, m, mu_plast=None)
    nu_eff = mu_eff
    nu_eff = torch.clamp(nu_eff, min=nu_min, max=nu_max)
    tau = effective_relaxation_time(nu_eff, rho=1.0, tau_min=tau_min, tau_max=tau_max)
    return tau


def macroscopic3d_from_f(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover macroscopic variables from distribution (helper)."""
    from .d3q19 import macroscopic3d
    return macroscopic3d(f)
