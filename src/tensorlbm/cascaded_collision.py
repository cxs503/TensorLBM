"""Full cascaded central-moment collision operators for D3Q19 and D3Q27.

This module implements the *complete* central-moment (CM) hierarchy, upgrading
the second-order stress approximation in :mod:`collision_d3q19_advanced` and the
regularized reconstruction in :mod:`advanced_collision` to a full cascaded
collision that transforms, relaxes, and back-transforms **every** lattice mode.

Theory
------
Central moments are raw moments shifted to the co-moving frame:

    κ_{abc} = Σ_i f_i (c_{ix} - u_x)^a (c_{iy} - u_y)^b (c_{iz} - u_z)^c

The shift is applied order-by-order via the binomial expansion, decomposed into
three sequential 1-D shifts along x, y, z.  Each 1-D shift is a triangular
operation on moments grouped by the other two dimensions' polynomial degrees.

Relaxation uses independent rates per order, with a trace/deviatoric split at
2nd order (bulk vs. shear).  Higher orders are relaxed at uniform per-order
rates (s_3, s_4, and for D3Q27 also s_5, s_6).

The collision operates on the non-equilibrium part ``f_neq = f - f_eq``.  Since
``f_neq`` has zero mass and momentum, the 0th/1st central moments vanish and
are trivially conserved.  The equilibrium central moments of ``f_neq`` are zero,
so relaxation reduces to ``κ* = (1 - s) κ``.

References
----------
Premnath, K. N. & Banerjee, S. (2009). Cascaded lattice Boltzmann automata for
    two-dimensional fluid dynamics. *Phys. Rev. E* 80, 036702.
Geier, M., Schönherr, M., Pasquali, A. & Krafczyk, M. (2015). The cumulant
    lattice Boltzmann equation in three dimensions. *Comput. Math. Appl.* 70.
"""
from __future__ import annotations

import functools

import numpy as np
import torch

from .d3q19 import C as _C19, equilibrium3d, macroscopic3d
from .d3q27 import C as _C27, equilibrium27, macroscopic27

# ---------------------------------------------------------------------------
# Moment degree orderings
# ---------------------------------------------------------------------------
#
# On both lattices cx ∈ {−1, 0, 1}, so cx^3 = cx and cx^4 = cx^2.  The
# independent monomials are products of {1, cx, cx^2} across the three
# dimensions, giving exactly Q independent moments per lattice.
#
# D3Q19 has no corner directions (±1,±1,±1), so every monomial involving
# cx·cy·cz is identically zero and omitted.  This leaves 19 moments.
#
# D3Q27 includes the 8 corners, so all 27 monomials are present.

_D3Q19_DEGREES: list[tuple[int, int, int]] = [
    (0, 0, 0),   #  0  mass
    (1, 0, 0),   #  1  jx
    (0, 1, 0),   #  2  jy
    (0, 0, 1),   #  3  jz
    (2, 0, 0),   #  4  Pxx
    (0, 2, 0),   #  5  Pyy
    (0, 0, 2),   #  6  Pzz
    (1, 1, 0),   #  7  Pxy
    (1, 0, 1),   #  8  Pxz
    (0, 1, 1),   #  9  Pyz
    (2, 1, 0),   # 10  qx (3rd order)
    (2, 0, 1),   # 11
    (1, 2, 0),   # 12
    (0, 2, 1),   # 13
    (1, 0, 2),   # 14
    (0, 1, 2),   # 15
    (2, 2, 0),   # 16  (4th order)
    (2, 0, 2),   # 17
    (0, 2, 2),   # 18
]

_D3Q27_DEGREES: list[tuple[int, int, int]] = [
    (0, 0, 0),   #  0  mass
    (1, 0, 0),   #  1  jx
    (0, 1, 0),   #  2  jy
    (0, 0, 1),   #  3  jz
    (2, 0, 0),   #  4  Pxx
    (0, 2, 0),   #  5  Pyy
    (0, 0, 2),   #  6  Pzz
    (1, 1, 0),   #  7  Pxy
    (1, 0, 1),   #  8  Pxz
    (0, 1, 1),   #  9  Pyz
    (2, 1, 0),   # 10  (3rd order)
    (2, 0, 1),   # 11
    (1, 2, 0),   # 12
    (0, 2, 1),   # 13
    (1, 0, 2),   # 14
    (0, 1, 2),   # 15
    (1, 1, 1),   # 16
    (2, 2, 0),   # 17  (4th order)
    (2, 0, 2),   # 18
    (0, 2, 2),   # 19
    (2, 1, 1),   # 20  (4th order, mixed)
    (1, 2, 1),   # 21
    (1, 1, 2),   # 22
    (2, 2, 1),   # 23  (5th order)
    (2, 1, 2),   # 24
    (1, 2, 2),   # 25
    (2, 2, 2),   # 26  (6th order)
]

# Order boundaries for relaxation grouping
_D3Q19_ORDER_BOUNDS = {
    "conserved": (0, 4),    # indices 0-3
    "second": (4, 10),      # indices 4-9
    "third": (10, 16),      # indices 10-15
    "fourth": (16, 19),     # indices 16-18
}

_D3Q27_ORDER_BOUNDS = {
    "conserved": (0, 4),    # indices 0-3
    "second": (4, 10),      # indices 4-9
    "third": (10, 17),      # indices 10-16
    "fourth": (17, 23),     # indices 17-22
    "fifth": (23, 26),      # indices 23-25
    "sixth": (26, 27),      # index 26
}


# ---------------------------------------------------------------------------
# Shift-group construction
# ---------------------------------------------------------------------------

def _build_shift_groups(
    degrees: list[tuple[int, int, int]],
) -> tuple[list[tuple[int, int, int]], ...]:
    """Return ``(x_groups, y_groups, z_groups)`` for the 1-D shift decomposition.

    Each group is a ``(i0, i1, i2)`` tuple of moment indices whose polynomial
    degree in the shifted dimension is 0, 1, 2 respectively (with the other two
    dimensions' degrees held fixed).  Groups that lack all three degrees are
    omitted — those moments are unaffected by the shift in that dimension.
    """
    groups: list[list[tuple[int, int, int]]] = []
    for dim in range(3):
        other = [d for d in range(3) if d != dim]
        bucket: dict[tuple[int, int], dict[int, int]] = {}
        for idx, deg in enumerate(degrees):
            key = (deg[other[0]], deg[other[1]])
            bucket.setdefault(key, {})[deg[dim]] = idx
        dim_groups: list[tuple[int, int, int]] = []
        for key in sorted(bucket):
            m = bucket[key]
            if 0 in m and 1 in m and 2 in m:
                dim_groups.append((m[0], m[1], m[2]))
        groups.append(dim_groups)
    return tuple(groups)


_D3Q19_SHIFT_GROUPS = _build_shift_groups(_D3Q19_DEGREES)
_D3Q27_SHIFT_GROUPS = _build_shift_groups(_D3Q27_DEGREES)


# ---------------------------------------------------------------------------
# Moment matrix construction (cached per device/dtype)
# ---------------------------------------------------------------------------

def _build_moment_matrix(
    c: torch.Tensor,
    degrees: list[tuple[int, int, int]],
) -> np.ndarray:
    """Build the Q×Q raw-moment matrix ``M[i,j] = monomial_i(c_j)``."""
    c_np = c.numpy().astype(np.float64)
    Q = len(degrees)
    matrix = np.zeros((Q, Q), dtype=np.float64)
    for i, (a, b, d) in enumerate(degrees):
        matrix[i, :] = c_np[:, 0] ** a * c_np[:, 1] ** b * c_np[:, 2] ** d
    assert np.linalg.matrix_rank(matrix) == Q, "moment matrix is rank-deficient"
    return matrix


_M19_DATA = _build_moment_matrix(_C19, _D3Q19_DEGREES)
_M19_INV_DATA = np.linalg.inv(_M19_DATA)
_M27_DATA = _build_moment_matrix(_C27, _D3Q27_DEGREES)
_M27_INV_DATA = np.linalg.inv(_M27_DATA)


@functools.cache
def _get_d3q19_matrices(
    device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(_M19_DATA, dtype=dtype, device=device),
        torch.tensor(_M19_INV_DATA, dtype=dtype, device=device),
    )


@functools.cache
def _get_d3q27_matrices(
    device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(_M27_DATA, dtype=dtype, device=device),
        torch.tensor(_M27_INV_DATA, dtype=dtype, device=device),
    )


# ---------------------------------------------------------------------------
# 1-D shift / unshift (binomial velocity shift)
# ---------------------------------------------------------------------------

def _shift_1d(
    m: torch.Tensor,
    groups: list[tuple[int, int, int]],
    u: torch.Tensor,
) -> torch.Tensor:
    """Apply the 1-D forward binomial shift to a moment tensor.

    For each triplet ``(i0, i1, i2)`` of moments with degree 0/1/2 in the
    shifted dimension::

        m'[i0] = m[i0]
        m'[i1] = m[i1] - u * m[i0]
        m'[i2] = m[i2] - 2u * m[i1] + u² * m[i0]
    """
    out = m.clone()
    for i0, i1, i2 in groups:
        m0 = m[i0]
        m1 = m[i1]
        m2 = m[i2]
        out[i1] = m1 - u * m0
        out[i2] = m2 - 2.0 * u * m1 + u * u * m0
    return out


def _unshift_1d(
    m: torch.Tensor,
    groups: list[tuple[int, int, int]],
    u: torch.Tensor,
) -> torch.Tensor:
    """Apply the 1-D inverse binomial shift (unshift).

    For each triplet ``(i0, i1, i2)``::

        m[i0] = m'[i0]
        m[i1] = m'[i1] + u * m'[i0]
        m[i2] = m'[i2] + 2u * m'[i1] + u² * m'[i0]
    """
    out = m.clone()
    for i0, i1, i2 in groups:
        m0 = m[i0]
        m1 = m[i1]
        m2 = m[i2]
        out[i1] = m1 + u * m0
        out[i2] = m2 + 2.0 * u * m1 + u * u * m0
    return out


# ---------------------------------------------------------------------------
# Full 3-D shift / unshift
# ---------------------------------------------------------------------------

def _to_central_d3q19(
    m: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
) -> torch.Tensor:
    """Shift D3Q19 raw moments → central moments (x, then y, then z)."""
    x_g, y_g, z_g = _D3Q19_SHIFT_GROUPS
    m = _shift_1d(m, x_g, ux)
    m = _shift_1d(m, y_g, uy)
    m = _shift_1d(m, z_g, uz)
    return m


def _to_raw_d3q19(
    k: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
) -> torch.Tensor:
    """Unshift D3Q19 central moments → raw moments (z, then y, then x)."""
    x_g, y_g, z_g = _D3Q19_SHIFT_GROUPS
    k = _unshift_1d(k, z_g, uz)
    k = _unshift_1d(k, y_g, uy)
    k = _unshift_1d(k, x_g, ux)
    return k


def _to_central_d3q27(
    m: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
) -> torch.Tensor:
    """Shift D3Q27 raw moments → central moments."""
    x_g, y_g, z_g = _D3Q27_SHIFT_GROUPS
    m = _shift_1d(m, x_g, ux)
    m = _shift_1d(m, y_g, uy)
    m = _shift_1d(m, z_g, uz)
    return m


def _to_raw_d3q27(
    k: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
) -> torch.Tensor:
    """Unshift D3Q27 central moments → raw moments."""
    x_g, y_g, z_g = _D3Q27_SHIFT_GROUPS
    k = _unshift_1d(k, z_g, uz)
    k = _unshift_1d(k, y_g, uy)
    k = _unshift_1d(k, x_g, ux)
    return k


# ---------------------------------------------------------------------------
# Cascaded relaxation
# ---------------------------------------------------------------------------

def _relax_d3q19(
    k: torch.Tensor,
    omega_shear: float | torch.Tensor,
    omega_bulk: float | torch.Tensor,
    omega_3: float | torch.Tensor,
    omega_4: float | torch.Tensor,
) -> torch.Tensor:
    """Relax D3Q19 central moments with trace/deviatoric split at 2nd order.

    Parameters
    ----------
    k
        Central moments of ``f_neq``, shape ``(19, *spatial)``.
    omega_shear
        Shear relaxation rate ``1/τ`` (controls kinematic viscosity).
    omega_bulk
        Bulk (trace) relaxation rate.
    omega_3
        3rd-order ghost-mode rate.
    omega_4
        4th-order ghost-mode rate.
    """
    out = k.clone()

    # 0th / 1st order (indices 0-3): conserved, no relaxation.

    # 2nd order (indices 4-9): trace/deviatoric split.
    kxx, kyy, kzz = k[4], k[5], k[6]
    kxy, kxz, kyz = k[7], k[8], k[9]
    trace = kxx + kyy + kzz
    dev_xx = kxx - trace / 3.0
    dev_yy = kyy - trace / 3.0
    dev_zz = kzz - trace / 3.0

    trace_s = (1.0 - omega_bulk) * trace
    dev_xx_s = (1.0 - omega_shear) * dev_xx
    dev_yy_s = (1.0 - omega_shear) * dev_yy
    dev_zz_s = (1.0 - omega_shear) * dev_zz

    out[4] = dev_xx_s + trace_s / 3.0
    out[5] = dev_yy_s + trace_s / 3.0
    out[6] = dev_zz_s + trace_s / 3.0
    out[7] = (1.0 - omega_shear) * kxy
    out[8] = (1.0 - omega_shear) * kxz
    out[9] = (1.0 - omega_shear) * kyz

    # 3rd order (indices 10-15)
    lo, hi = _D3Q19_ORDER_BOUNDS["third"]
    for i in range(lo, hi):
        out[i] = (1.0 - omega_3) * k[i]

    # 4th order (indices 16-18)
    lo, hi = _D3Q19_ORDER_BOUNDS["fourth"]
    for i in range(lo, hi):
        out[i] = (1.0 - omega_4) * k[i]

    return out


def _relax_d3q27(
    k: torch.Tensor,
    omega_shear: float | torch.Tensor,
    omega_bulk: float | torch.Tensor,
    omega_3: float | torch.Tensor,
    omega_4: float | torch.Tensor,
    omega_5: float | torch.Tensor,
    omega_6: float | torch.Tensor,
) -> torch.Tensor:
    """Relax D3Q27 central moments with trace/deviatoric split at 2nd order."""
    out = k.clone()

    # 0th / 1st order: conserved.

    # 2nd order: trace/deviatoric split.
    kxx, kyy, kzz = k[4], k[5], k[6]
    kxy, kxz, kyz = k[7], k[8], k[9]
    trace = kxx + kyy + kzz
    dev_xx = kxx - trace / 3.0
    dev_yy = kyy - trace / 3.0
    dev_zz = kzz - trace / 3.0

    trace_s = (1.0 - omega_bulk) * trace
    dev_xx_s = (1.0 - omega_shear) * dev_xx
    dev_yy_s = (1.0 - omega_shear) * dev_yy
    dev_zz_s = (1.0 - omega_shear) * dev_zz

    out[4] = dev_xx_s + trace_s / 3.0
    out[5] = dev_yy_s + trace_s / 3.0
    out[6] = dev_zz_s + trace_s / 3.0
    out[7] = (1.0 - omega_shear) * kxy
    out[8] = (1.0 - omega_shear) * kxz
    out[9] = (1.0 - omega_shear) * kyz

    # 3rd order (indices 10-16)
    lo, hi = _D3Q27_ORDER_BOUNDS["third"]
    for i in range(lo, hi):
        out[i] = (1.0 - omega_3) * k[i]

    # 4th order (indices 17-22)
    lo, hi = _D3Q27_ORDER_BOUNDS["fourth"]
    for i in range(lo, hi):
        out[i] = (1.0 - omega_4) * k[i]

    # 5th order (indices 23-25)
    lo, hi = _D3Q27_ORDER_BOUNDS["fifth"]
    for i in range(lo, hi):
        out[i] = (1.0 - omega_5) * k[i]

    # 6th order (index 26)
    lo, hi = _D3Q27_ORDER_BOUNDS["sixth"]
    for i in range(lo, hi):
        out[i] = (1.0 - omega_6) * k[i]

    return out


# ---------------------------------------------------------------------------
# Collision operators
# ---------------------------------------------------------------------------

def collide_cascaded_d3q19(
    f: torch.Tensor,
    tau: float,
    s_bulk: float = 1.0,
    s_3: float = 1.0,
    s_4: float = 1.0,
) -> torch.Tensor:
    """Full cascaded central-moment collision for D3Q19.

    Transforms populations → raw moments → central moments, relaxes each
    order independently (trace/deviatoric split at 2nd order), and
    back-transforms to populations.

    Parameters
    ----------
    f
        Distribution tensor of shape ``(19, nz, ny, nx)``.
    tau
        Shear relaxation time (τ > ½).  Kinematic viscosity ν = (τ − ½)/3.
    s_bulk
        Bulk (trace) relaxation rate (default 1.0).
    s_3
        3rd-order ghost-mode rate (default 1.0).
    s_4
        4th-order ghost-mode rate (default 1.0).

    Returns
    -------
    Post-collision distribution of the same shape.
    """
    if f.ndim != 4 or f.shape[0] != 19:
        raise ValueError(f"D3Q19 populations must have shape (19, nz, ny, nx), got {tuple(f.shape)}")
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

    # Relax
    omega = 1.0 / tau
    k_star = _relax_d3q19(k_neq, omega, s_bulk, s_3, s_4)

    # Unshift to raw moments and reconstruct populations
    m_star = _to_raw_d3q19(k_star, ux, uy, uz)
    f_neq_star = (M_inv @ m_star.reshape(19, -1)).reshape(19, nz, ny, nx)

    return feq + f_neq_star


def collide_cascaded_d3q27(
    f: torch.Tensor,
    tau: float,
    s_bulk: float = 1.0,
    s_3: float = 1.0,
    s_4: float = 1.0,
    s_5: float | None = None,
    s_6: float | None = None,
) -> torch.Tensor:
    """Full cascaded central-moment collision for D3Q27.

    Transforms populations → raw moments → central moments, relaxes each
    order independently (trace/deviatoric split at 2nd order), and
    back-transforms to populations.

    Parameters
    ----------
    f
        Distribution tensor of shape ``(27, nz, ny, nx)``.
    tau
        Shear relaxation time (τ > ½).  Kinematic viscosity ν = (τ − ½)/3.
    s_bulk
        Bulk (trace) relaxation rate (default 1.0).
    s_3
        3rd-order ghost-mode rate (default 1.0).
    s_4
        4th-order ghost-mode rate (default 1.0).
    s_5
        5th-order ghost-mode rate (defaults to *s_4*).
    s_6
        6th-order ghost-mode rate (defaults to *s_4*).

    Returns
    -------
    Post-collision distribution of the same shape.
    """
    if f.ndim != 4 or f.shape[0] != 27:
        raise ValueError(f"D3Q27 populations must have shape (27, nz, ny, nx), got {tuple(f.shape)}")
    if isinstance(tau, torch.Tensor):
        if (tau <= 0.5).any():
            raise ValueError("all tau values must be > 0.5")
    elif tau <= 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")

    if s_5 is None:
        s_5 = s_4
    if s_6 is None:
        s_6 = s_4

    device = f.device
    dtype = f.dtype
    M, M_inv = _get_d3q27_matrices(device, dtype)

    # Macroscopic fields and equilibrium
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq

    # Raw moments of f_neq
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    m_neq = (M @ f_neq.reshape(27, -1)).reshape(27, nz, ny, nx)

    # Shift to central moments
    k_neq = _to_central_d3q27(m_neq, ux, uy, uz)

    # Relax
    omega = 1.0 / tau
    k_star = _relax_d3q27(k_neq, omega, s_bulk, s_3, s_4, s_5, s_6)

    # Unshift to raw moments and reconstruct populations
    m_star = _to_raw_d3q27(k_star, ux, uy, uz)
    f_neq_star = (M_inv @ m_star.reshape(27, -1)).reshape(27, nz, ny, nx)

    return feq + f_neq_star


__all__ = [
    "collide_cascaded_d3q19",
    "collide_cascaded_d3q27",
    "_to_central_d3q19",
    "_to_raw_d3q19",
    "_to_central_d3q27",
    "_to_raw_d3q27",
    "_get_d3q19_matrices",
    "_get_d3q27_matrices",
    "_D3Q19_DEGREES",
    "_D3Q27_DEGREES",
]
