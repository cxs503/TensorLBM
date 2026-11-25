"""Reusable D3Q19 second-order stress collision kernels.

These kernels are deliberately limited to a verifiable second-order Hermite
stress projection.  They are *not* complete cascaded/central-moment, cumulant,
or entropic KBC implementations: they do not transform or relax all higher
central/cumulant modes, and they do not solve an entropy/positivity condition.
"""

from __future__ import annotations

import torch

from .d3q19 import C, W, equilibrium3d, macroscopic3d

_CS2 = 1.0 / 3.0


def _lattice_views(f: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return D3Q19 views with the historical float32 lattice constants.

    Public stress kernels deliberately use the package's float32 D3Q19
    constants regardless of population dtype.  This preserves the established
    CG contract: float64 populations promote operations, but do not silently
    upgrade the lattice weights or ``cs2``.
    """
    c = C.to(device=f.device, dtype=torch.float32)
    w = W.to(device=f.device, dtype=torch.float32).view(19, 1, 1, 1)
    return (c[:, 0].view(19, 1, 1, 1), c[:, 1].view(19, 1, 1, 1), c[:, 2].view(19, 1, 1, 1), w)


def second_order_stress_d3q19(f_neq: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return ``(xx, yy, zz, xy, xz, yz)`` of the second-order Hermite moments.

    This computes the Hermite moments ``M_alpha = sum_i w_i h_alpha(c_i) f_neq,i``,
    which is required for the correct inverse Hermite reconstruction.
    """
    cx, cy, cz, w = _lattice_views(f_neq)
    h_xx, h_yy, h_zz = cx * cx - _CS2, cy * cy - _CS2, cz * cz - _CS2
    return (
        (w * h_xx * f_neq).sum(0),
        (w * h_yy * f_neq).sum(0),
        (w * h_zz * f_neq).sum(0),
        (w * cx * cy * f_neq).sum(0),
        (w * cx * cz * f_neq).sum(0),
        (w * cy * cz * f_neq).sum(0),
    )


def reconstruct_second_order_stress_d3q19(
    pi_xx: torch.Tensor,
    pi_yy: torch.Tensor,
    pi_zz: torch.Tensor,
    pi_xy: torch.Tensor,
    pi_xz: torch.Tensor,
    pi_yz: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the D3Q19 second-order Hermite non-equilibrium projection.

    The reconstruction uses the inverse Hermite normalization:
    f_neq = sum_alpha (h_alpha(c) / theta_alpha) * pi_alpha
    where theta_xx = theta_yy = theta_zz = 2/9, theta_xy = theta_xz = theta_yz = 1/9.

    The result is projected to have zero unweighted mass and momentum, ensuring
    that adding it to the equilibrium distribution preserves those quantities.
    """
    # Construct a lightweight shape carrier; all six components must be fields.
    shape = pi_xx.shape
    if not all(p.shape == shape for p in (pi_yy, pi_zz, pi_xy, pi_xz, pi_yz)):
        raise ValueError("all D3Q19 stress components must have identical shapes")
    carrier = pi_xx.new_empty((19, *shape))
    cx, cy, cz, w = _lattice_views(carrier)
    h_xx, h_yy, h_zz = cx * cx - _CS2, cy * cy - _CS2, cz * cz - _CS2
    # theta_xx = 2/9, so 1/theta_xx = 9/2 = 4.5
    # theta_xy = 1/9, so 1/theta_xy = 9
    f_neq = (
        4.5 * h_xx * pi_xx.unsqueeze(0)
        + 4.5 * h_yy * pi_yy.unsqueeze(0)
        + 4.5 * h_zz * pi_zz.unsqueeze(0)
        + 9.0 * h_xy(cx, cy) * pi_xy.unsqueeze(0)
        + 9.0 * h_xy(cx, cz) * pi_xz.unsqueeze(0)
        + 9.0 * h_xy(cy, cz) * pi_yz.unsqueeze(0)
    )
    # Project out unweighted mass and momentum.
    # The Hermite polynomials h_xx, h_yy, h_zz, h_xy, h_xz, h_yz are orthogonal
    # to the mass (1) and momentum (cx, cy, cz) modes with respect to the weight w,
    # but their unweighted sums are not zero. We subtract the unweighted mass
    # and momentum to ensure f_neq has zero unweighted mass and momentum.
    mass = f_neq.sum(0)
    momentum_x = (cx * f_neq).sum(0)
    momentum_y = (cy * f_neq).sum(0)
    momentum_z = (cz * f_neq).sum(0)
    # Subtract unweighted mass: f_neq -= mass / 19
    f_neq = f_neq - mass.unsqueeze(0) / 19.0
    # Subtract unweighted momentum: f_neq -= sum_j (c_ij * momentum_j) / (sum_i c_ij^2)
    # For D3Q19, sum_i c_ix^2 = sum_i c_iy^2 = sum_i c_iz^2 = 10
    # (not 19 * cs2 = 6.33, because the lattice is not isotropic in the unweighted sense)
    f_neq = f_neq - (cx * momentum_x + cy * momentum_y + cz * momentum_z) / 10.0
    return f_neq


def h_xy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Small helper that keeps the reconstruction expression readable."""
    return a * b


def collide_bgk_d3q19(f: torch.Tensor, tau: float) -> torch.Tensor:
    """D3Q19 BGK collision (named here for a common collision-kernel surface)."""
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    return f - (f - feq) / tau


def collide_mrt_d3q19(f: torch.Tensor, tau: float, **rates: float) -> torch.Tensor:
    """D3Q19 MRT kernel, delegating to the established production baseline.

    This wrapper deliberately does not replace or alter :mod:`solver3d`'s MRT
    implementation; it provides an explicit common-kernel name only.
    """
    from .solver3d import collide_mrt3d

    return collide_mrt3d(f, tau, **rates)


def collide_regularized_stress_d3q19(
    f_total: torch.Tensor,
    feq: torch.Tensor,
    tau: float | torch.Tensor,
) -> torch.Tensor:
    """Relax only the D3Q19 second-order non-equilibrium stress.

    ``feq`` is deliberately supplied by the caller.  This lets force-aware
    users retain their own equilibrium construction instead of silently
    recomputing unshifted hydrodynamics in this common kernel.

    *tau* may be a scalar or a per-cell tensor ``(nz, ny, nx)``; the
    relaxation rate ``omega = 1/tau`` broadcasts over the stress components.
    """
    pi_xx, pi_yy, pi_zz, pi_xy, pi_xz, pi_yz = second_order_stress_d3q19(f_total - feq)
    omega = 1.0 / tau
    return feq + reconstruct_second_order_stress_d3q19(
        (1.0 - omega) * pi_xx,
        (1.0 - omega) * pi_yy,
        (1.0 - omega) * pi_zz,
        (1.0 - omega) * pi_xy,
        (1.0 - omega) * pi_xz,
        (1.0 - omega) * pi_yz,
    )


def collide_central_stress_d3q19(
    f_total: torch.Tensor,
    feq: torch.Tensor,
    tau: float | torch.Tensor,
    s_bulk: float | None = None,
) -> torch.Tensor:
    """Second-order central-stress approximation with independent trace rate.

    This is not a full cascaded collision: it only relaxes the second-order
    trace/deviatoric stresses and projects away every higher-order mode.

    *tau* may be a scalar or a per-cell tensor ``(nz, ny, nx)``.
    """
    pi_xx, pi_yy, pi_zz, pi_xy, pi_xz, pi_yz = second_order_stress_d3q19(f_total - feq)
    omega_shear = 1.0 / tau
    omega_bulk = omega_shear if s_bulk is None else s_bulk
    trace = pi_xx + pi_yy + pi_zz
    dev_xx = pi_xx - trace / 3.0
    dev_yy = pi_yy - trace / 3.0
    dev_zz = pi_zz - trace / 3.0
    trace = (1.0 - omega_bulk) * trace
    dev_xx = (1.0 - omega_shear) * dev_xx
    dev_yy = (1.0 - omega_shear) * dev_yy
    dev_zz = (1.0 - omega_shear) * dev_zz
    pi_xx = dev_xx + trace / 3.0
    pi_yy = dev_yy + trace / 3.0
    pi_zz = dev_zz + trace / 3.0
    return feq + reconstruct_second_order_stress_d3q19(
        pi_xx,
        pi_yy,
        pi_zz,
        (1.0 - omega_shear) * pi_xy,
        (1.0 - omega_shear) * pi_xz,
        (1.0 - omega_shear) * pi_yz,
    )


__all__ = [
    "second_order_stress_d3q19",
    "reconstruct_second_order_stress_d3q19",
    "collide_bgk_d3q19",
    "collide_mrt_d3q19",
    "collide_regularized_stress_d3q19",
    "collide_central_stress_d3q19",
    "collide_cumulant_d3q19",
]


def collide_cumulant_d3q19(
    f: torch.Tensor,
    tau: float,
    C_s: float = 0.1,
) -> torch.Tensor:
    """D3Q19 cumulant collision (2nd-order stress relaxation).

    This is a minimal cumulant kernel: it relaxes the second-order central
    moments (stress) and leaves higher-order modes untouched.  It is
    deliberately *not* a full cascaded/cumulant implementation.

    *C_s* is a dummy parameter kept for API compatibility with the
    ``advanced_collision_d3q19`` import path used by ``free_surface_lbm``.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq
    pi_xx, pi_yy, pi_zz, pi_xy, pi_xz, pi_yz = second_order_stress_d3q19(f_neq)
    omega = 1.0 / tau
    return feq + reconstruct_second_order_stress_d3q19(
        (1.0 - omega) * pi_xx,
        (1.0 - omega) * pi_yy,
        (1.0 - omega) * pi_zz,
        (1.0 - omega) * pi_xy,
        (1.0 - omega) * pi_xz,
        (1.0 - omega) * pi_yz,
    )
