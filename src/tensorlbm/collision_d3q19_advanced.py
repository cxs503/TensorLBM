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
    return (c[:, 0].view(19, 1, 1, 1), c[:, 1].view(19, 1, 1, 1),
            c[:, 2].view(19, 1, 1, 1), w)


def second_order_stress_d3q19(f_neq: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return ``(xx, yy, zz, xy, xz, yz)`` of ``sum_i c_ia c_ib f_neq,i``."""
    cx, cy, cz, _ = _lattice_views(f_neq)
    return (
        (cx * cx * f_neq).sum(0), (cy * cy * f_neq).sum(0),
        (cz * cz * f_neq).sum(0), (cx * cy * f_neq).sum(0),
        (cx * cz * f_neq).sum(0), (cy * cz * f_neq).sum(0),
    )


def reconstruct_second_order_stress_d3q19(
    pi_xx: torch.Tensor, pi_yy: torch.Tensor, pi_zz: torch.Tensor,
    pi_xy: torch.Tensor, pi_xz: torch.Tensor, pi_yz: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the D3Q19 second-order Hermite non-equilibrium projection."""
    # Construct a lightweight shape carrier; all six components must be fields.
    shape = pi_xx.shape
    if not all(p.shape == shape for p in (pi_yy, pi_zz, pi_xy, pi_xz, pi_yz)):
        raise ValueError("all D3Q19 stress components must have identical shapes")
    carrier = pi_xx.new_empty((19, *shape))
    cx, cy, cz, w = _lattice_views(carrier)
    h_xx, h_yy, h_zz = cx * cx - _CS2, cy * cy - _CS2, cz * cz - _CS2
    return 4.5 * w * (
        h_xx * pi_xx.unsqueeze(0) + h_yy * pi_yy.unsqueeze(0) + h_zz * pi_zz.unsqueeze(0)
        + 2.0 * h_xy(cx, cy) * pi_xy.unsqueeze(0)
        + 2.0 * h_xy(cx, cz) * pi_xz.unsqueeze(0)
        + 2.0 * h_xy(cy, cz) * pi_yz.unsqueeze(0)
    )


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
    f_total: torch.Tensor, feq: torch.Tensor, tau: float,
) -> torch.Tensor:
    """Relax only the D3Q19 second-order non-equilibrium stress.

    ``feq`` is deliberately supplied by the caller.  This lets force-aware
    users retain their own equilibrium construction instead of silently
    recomputing unshifted hydrodynamics in this common kernel.
    """
    pi_xx, pi_yy, pi_zz, pi_xy, pi_xz, pi_yz = second_order_stress_d3q19(f_total - feq)
    omega = 1.0 / tau
    return feq + reconstruct_second_order_stress_d3q19(
        (1.0 - omega) * pi_xx, (1.0 - omega) * pi_yy, (1.0 - omega) * pi_zz,
        (1.0 - omega) * pi_xy, (1.0 - omega) * pi_xz, (1.0 - omega) * pi_yz,
    )


def collide_central_stress_d3q19(
    f_total: torch.Tensor, feq: torch.Tensor, tau: float, s_bulk: float | None = None,
) -> torch.Tensor:
    """Second-order central-stress approximation with independent trace rate.

    This is not a full cascaded collision: it only relaxes the second-order
    trace/deviatoric stresses and projects away every higher-order mode.
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
        pi_xx, pi_yy, pi_zz,
        (1.0 - omega_shear) * pi_xy, (1.0 - omega_shear) * pi_xz,
        (1.0 - omega_shear) * pi_yz,
    )


__all__ = [
    "second_order_stress_d3q19", "reconstruct_second_order_stress_d3q19",
    "collide_bgk_d3q19", "collide_mrt_d3q19", "collide_regularized_stress_d3q19",
    "collide_central_stress_d3q19",
]
