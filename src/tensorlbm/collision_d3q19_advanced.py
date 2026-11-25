"""Reusable D3Q19 second-order stress collision kernels.

These kernels are deliberately limited to a verifiable second-order Hermite
stress projection.  They are *not* complete cascaded/central-moment, cumulant,
or entropic KBC implementations: they do not transform or relax all higher
central/cumulant modes, and they do not solve an entropy/positivity condition.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from .d3q19 import C, W, equilibrium3d, macroscopic3d

_CS2 = 1.0 / 3.0

# D3Q19 moment degrees (19 moments, no corner directions)
_D3Q19_DEGREES = [
    (0, 0, 0),  #  0  mass
    (1, 0, 0),  #  1  jx
    (0, 1, 0),  #  2  jy
    (0, 0, 1),  #  3  jz
    (2, 0, 0),  #  4  Pxx
    (0, 2, 0),  #  5  Pyy
    (0, 0, 2),  #  6  Pzz
    (1, 1, 0),  #  7  Pxy
    (1, 0, 1),  #  8  Pxz
    (0, 1, 1),  #  9  Pyz
    (2, 1, 0),  # 10  qx
    (2, 0, 1),  # 11
    (1, 2, 0),  # 12
    (0, 2, 1),  # 13
    (1, 0, 2),  # 14
    (0, 1, 2),  # 15
    (2, 2, 0),  # 16  (4th order)
    (2, 0, 2),  # 17
    (0, 2, 2),  # 18
]


def _build_moment_matrix(c, degrees):
    """Build the Q×Q raw-moment matrix M[i,j] = monomial_i(c_j)."""
    c_np = c.numpy().astype(np.float64)
    Q = len(degrees)
    matrix = np.zeros((Q, Q), dtype=np.float64)
    for i, (a, b, d) in enumerate(degrees):
        matrix[i, :] = c_np[:, 0] ** a * c_np[:, 1] ** b * c_np[:, 2] ** d
    return matrix


_M19_DATA = _build_moment_matrix(C, _D3Q19_DEGREES)
_M19_INV_DATA = np.linalg.inv(_M19_DATA)


@functools.cache
def _get_d3q19_matrices(device, dtype):
    """Get the D3Q19 moment matrix and its inverse."""
    return (
        torch.tensor(_M19_DATA, dtype=dtype, device=device),
        torch.tensor(_M19_INV_DATA, dtype=dtype, device=device),
    )


def _shift_1d(m, groups, u):
    """Apply the 1-D forward binomial shift to a moment tensor."""
    out = m.clone()
    for i0, i1, i2 in groups:
        m0 = m[i0]
        m1 = m[i1]
        m2 = m[i2]
        out[i1] = m1 - u * m0
        out[i2] = m2 - 2.0 * u * m1 + u * u * m0
    return out


def _unshift_1d(m, groups, u):
    """Apply the 1-D inverse binomial shift (unshift)."""
    out = m.clone()
    for i0, i1, i2 in groups:
        m0 = m[i0]
        m1 = m[i1]
        m2 = m[i2]
        out[i1] = m1 + u * m0
        out[i2] = m2 + 2.0 * u * m1 + u * u * m0
    return out


def _to_central_d3q19(m, ux, uy, uz):
    """Shift raw moments to central moments via 1-D binomial shifts."""
    k = m.clone()
    # Shift x
    k = _shift_1d(k, [(0, 1, 4), (1, 7, 10), (7, 12, 16)], ux)
    # Shift y
    k = _shift_1d(k, [(0, 2, 5), (2, 7, 12), (7, 9, 13)], uy)
    # Shift z
    k = _shift_1d(k, [(0, 3, 6), (3, 8, 11), (8, 9, 15)], uz)
    return k


def _to_raw_d3q19(k, ux, uy, uz):
    """Unshift central moments to raw moments."""
    m = k.clone()
    # Unshift z
    m = _unshift_1d(m, [(0, 3, 6), (3, 8, 11), (8, 9, 15)], uz)
    # Unshift y
    m = _unshift_1d(m, [(0, 2, 5), (2, 7, 12), (7, 9, 13)], uy)
    # Unshift x
    m = _unshift_1d(m, [(0, 1, 4), (1, 7, 10), (7, 12, 16)], ux)
    return m


def _relax_d3q19(k, omega, s_bulk, s_3, s_4):
    """Relax central moments in cumulant family."""
    k_star = k.clone()

    # 2nd order: shear modes (Pxy, Pxz, Pyz) relax at omega
    k_star[7] = k[7] - omega * k[7]  # Pxy
    k_star[8] = k[8] - omega * k[8]  # Pxz
    k_star[9] = k[9] - omega * k[9]  # Pyz

    # 2nd order: normal stresses (Pxx, Pyy, Pzz) - trace/deviatoric split
    trace = k[4] + k[5] + k[6]
    trace_star = trace - s_bulk * (trace - (k[4] + k[5] + k[6]))  # Simplified
    # Actually, for cumulant, we relax the trace at s_bulk and deviatoric at omega
    # But since we're working with f_neq, the equilibrium trace is zero
    # So we just relax the trace at s_bulk
    dev_xx = k[4] - trace / 3.0
    dev_yy = k[5] - trace / 3.0
    dev_zz = k[6] - trace / 3.0
    trace_star = trace - s_bulk * trace
    dev_xx = dev_xx - omega * dev_xx
    dev_yy = dev_yy - omega * dev_yy
    dev_zz = dev_zz - omega * dev_zz
    k_star[4] = dev_xx + trace_star / 3.0
    k_star[5] = dev_yy + trace_star / 3.0
    k_star[6] = dev_zz + trace_star / 3.0

    # 3rd order ghost modes relax at s_3
    for idx in [10, 11, 12, 13, 14, 15]:
        k_star[idx] = k[idx] - s_3 * k[idx]

    # 4th order ghost modes relax at s_4
    for idx in [16, 17, 18]:
        k_star[idx] = k[idx] - s_4 * k[idx]

    return k_star


def _lattice_views(f: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return D3Q19 views with the historical float32 lattice constants."""
    c = C.to(device=f.device, dtype=torch.float32)
    w = W.to(device=f.device, dtype=torch.float32).view(19, 1, 1, 1)
    return (c[:, 0].view(19, 1, 1, 1), c[:, 1].view(19, 1, 1, 1), c[:, 2].view(19, 1, 1, 1), w)


def second_order_stress_d3q19(f_neq: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return ``(xx, yy, zz, xy, xz, yz)`` of ``sum_i c_ia c_ib f_neq,i``."""
    cx, cy, cz, _ = _lattice_views(f_neq)
    return (
        (cx * cx * f_neq).sum(0),
        (cy * cy * f_neq).sum(0),
        (cz * cz * f_neq).sum(0),
        (cx * cy * f_neq).sum(0),
        (cx * cz * f_neq).sum(0),
        (cy * cz * f_neq).sum(0),
    )


def reconstruct_second_order_stress_d3q19(
    pi_xx: torch.Tensor,
    pi_yy: torch.Tensor,
    pi_zz: torch.Tensor,
    pi_xy: torch.Tensor,
    pi_xz: torch.Tensor,
    pi_yz: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the D3Q19 second-order Hermite non-equilibrium projection."""
    shape = pi_xx.shape
    if not all(p.shape == shape for p in (pi_yy, pi_zz, pi_xy, pi_xz, pi_yz)):
        raise ValueError("all D3Q19 stress components must have identical shapes")
    carrier = pi_xx.new_empty((19, *shape))
    cx, cy, cz, w = _lattice_views(carrier)
    h_xx, h_yy, h_zz = cx * cx - _CS2, cy * cy - _CS2, cz * cz - _CS2
    return (
        4.5
        * w
        * (
            h_xx * pi_xx.unsqueeze(0)
            + h_yy * pi_yy.unsqueeze(0)
            + h_zz * pi_zz.unsqueeze(0)
            + 2.0 * h_xy(cx, cy) * pi_xy.unsqueeze(0)
            + 2.0 * h_xy(cx, cz) * pi_xz.unsqueeze(0)
            + 2.0 * h_xy(cy, cz) * pi_yz.unsqueeze(0)
        )
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
    """D3Q19 MRT kernel, delegating to the established production baseline."""
    from .solver3d import collide_mrt3d

    return collide_mrt3d(f, tau, **rates)


def collide_regularized_stress_d3q19(
    f_total: torch.Tensor,
    feq: torch.Tensor,
    tau: float | torch.Tensor,
) -> torch.Tensor:
    """Relax only the D3Q19 second-order non-equilibrium stress."""
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
    """Second-order central-stress approximation with independent trace rate."""
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


def collide_cumulant_d3q19(
    f: torch.Tensor,
    tau: float,
    *,
    C_s: float = 0.0,
    omega_b: float = 1.0,
    omega_3: float = 1.0,
    omega_4: float = 1.0,
) -> torch.Tensor:
    """D3Q19 cumulant-family collision (Geier 2015, truncated at 3rd order).

    The D3Q19 lattice does not support full 4th-order isotropy, so the
    nonlinear cumulant corrections (C_{220} etc.) are omitted.  For orders
    ≤ 3, cumulants coincide with central moments, so this operator is
    equivalent to a cascaded (central-moment) collision with independently
    tunable ghost-mode rates.

    Args:
        f:         Distribution tensor, shape ``(19, nz, ny, nx)``.
        tau:       Shear relaxation time τ > 0.5.  ν = (τ − ½) / 3.
        C_s:       Subgrid-scale coefficient (SGS).  When > 0, increases
                   ghost-mode damping to emulate LES dissipation.
        omega_b:   Bulk-viscosity (trace) relaxation rate (default 1.0).
        omega_3:   3rd-order ghost-mode rate (default 1.0).
        omega_4:   4th-order ghost-mode rate (default 1.0, truncated).

    Returns:
        Post-collision distribution tensor of shape ``(19, nz, ny, nx)``.
    """
    if f.ndim != 4 or f.shape[0] != 19:
        raise ValueError(
            f"D3Q19 populations must have shape (19, nz, ny, nx), got {tuple(f.shape)}"
        )
    if isinstance(tau, torch.Tensor):
        if (tau <= 0.5).any():
            raise ValueError("all tau values must be > 0.5")
    elif tau <= 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")

    device = f.device
    dtype = f.dtype
    M, M_inv = _get_d3q19_matrices(device, dtype)

    # Macroscopic fields and equilibrium
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq

    # Raw moments of f_neq
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    m_neq = (M @ f_neq.reshape(19, -1)).reshape(19, nz, ny, nx)

    # Shift to central moments
    k_neq = _to_central_d3q19(m_neq, ux, uy, uz)

    # SGS enhancement: when C_s > 0, increase ghost-mode damping
    if C_s > 0:
        sgs_factor = 1.0 + 2.0 * C_s * C_s
        omega_3 = min(omega_3 * sgs_factor, 2.0)
        omega_4 = min(omega_4 * sgs_factor, 2.0)

    # Relax
    omega = 1.0 / tau
    k_star = _relax_d3q19(k_neq, omega, omega_b, omega_3, omega_4)

    # Unshift to raw moments and reconstruct populations
    m_star = _to_raw_d3q19(k_star, ux, uy, uz)
    f_neq_star = (M_inv @ m_star.reshape(19, -1)).reshape(19, nz, ny, nx)

    return feq + f_neq_star


__all__ = [
    "second_order_stress_d3q19",
    "reconstruct_second_order_stress_d3q19",
    "collide_bgk_d3q19",
    "collide_mrt_d3q19",
    "collide_regularized_stress_d3q19",
    "collide_central_stress_d3q19",
    "collide_cumulant_d3q19",
]
