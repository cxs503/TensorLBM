"""D3Q27 collision operators with full path activation.

Implements BGK, MRT (cascaded/central-moment), and Cumulant collision operators
for the D3Q27 lattice, ensuring all 27 velocity directions are properly
activated (no dead enumeration).

The D3Q27 lattice has 27 velocity directions covering all combinations of
(cx, cy, cz) ∈ {−1, 0, 1}³. Compared to D3Q19 it includes the 8 corner
directions (|c| = √3) and therefore achieves 4th-order isotropy.

Lattice weights (Qian, 1992):
- Rest (0,0,0):           w = 8/27
- Face-centre (|c|=1):    w = 2/27  (×6)
- Edge-centre (|c|=√2):   w = 1/54  (×12)
- Corner     (|c|=√3):    w = 1/216 (×8)

References:
    Qian, Y. H., d'Humières, D., & Lallemand, P. (1992).
    "Lattice BGK models for Navier-Stokes equation."
    Europhysics Letters, 17(6), 479.

    Geier, M., Schönherr, M., Pasquali, A., & Krafczyk, M. (2015).
    "The cumulant lattice Boltzmann equation in three dimensions."
    Computers & Mathematics with Applications, 70(4), 507-547.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

# ── D3Q27 lattice descriptor ──────────────────────────────────────────────

_C_DATA = [
    [0, 0, 0],
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    [1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1],
    [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1],
]

C_D3Q27 = torch.tensor(_C_DATA, dtype=torch.int64)

_w_rest = 8.0 / 27.0
_w_face = 2.0 / 27.0
_w_edge = 1.0 / 54.0
_w_corner = 1.0 / 216.0

_W_DATA = [_w_rest] + [_w_face] * 6 + [_w_edge] * 12 + [_w_corner] * 8
W_D3Q27 = torch.tensor(_W_DATA, dtype=torch.float32)
W_D3Q27_EXACT64 = torch.tensor(_W_DATA, dtype=torch.float64)


def _build_opposite() -> torch.Tensor:
    c_list = [tuple(row) for row in _C_DATA]
    opp = []
    for cx, cy, cz in c_list:
        target = (-cx, -cy, -cz)
        opp.append(c_list.index(target))
    return torch.tensor(opp, dtype=torch.int64)


OPPOSITE_D3Q27 = _build_opposite()


# ── Moment degree ordering (proper basis for D3Q27) ───────────────────────
# Independent monomials: products of {1, cx, cx²} across 3 dimensions = 27

_D3Q27_DEGREES = [
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
    (2, 1, 0),  # 10  (3rd order)
    (2, 0, 1),  # 11
    (1, 2, 0),  # 12
    (0, 2, 1),  # 13
    (1, 0, 2),  # 14
    (0, 1, 2),  # 15
    (1, 1, 1),  # 16
    (2, 2, 0),  # 17  (4th order)
    (2, 0, 2),  # 18
    (0, 2, 2),  # 19
    (2, 1, 1),  # 20  (4th order, mixed)
    (1, 2, 1),  # 21
    (1, 1, 2),  # 22
    (2, 2, 1),  # 23  (5th order)
    (2, 1, 2),  # 24
    (1, 2, 2),  # 25
    (2, 2, 2),  # 26  (6th order)
]

# Order boundaries for relaxation grouping
_D3Q27_ORDER_BOUNDS = {
    "conserved": (0, 4),
    "second": (4, 10),
    "third": (10, 17),
    "fourth": (17, 23),
    "fifth": (23, 26),
    "sixth": (26, 27),
}


def _build_shift_groups():
    """Build 1-D shift groups for D3Q27."""
    groups = []
    for dim in range(3):
        other = [d for d in range(3) if d != dim]
        bucket = {}
        for idx, deg in enumerate(_D3Q27_DEGREES):
            key = (deg[other[0]], deg[other[1]])
            bucket.setdefault(key, {})[deg[dim]] = idx
        dim_groups = []
        for key in sorted(bucket):
            m = bucket[key]
            if 0 in m and 1 in m and 2 in m:
                dim_groups.append((m[0], m[1], m[2]))
        groups.append(dim_groups)
    return tuple(groups)


_D3Q27_SHIFT_GROUPS = _build_shift_groups()


def _build_moment_matrix():
    """Build the 27×27 raw-moment matrix M[i,j] = monomial_i(c_j)."""
    c_np = np.array(_C_DATA, dtype=np.float64)
    Q = 27
    matrix = np.zeros((Q, Q), dtype=np.float64)
    for i, (a, b, d) in enumerate(_D3Q27_DEGREES):
        matrix[i, :] = c_np[:, 0] ** a * c_np[:, 1] ** b * c_np[:, 2] ** d
    assert np.linalg.matrix_rank(matrix) == Q, "moment matrix is rank-deficient"
    return matrix


_M27_DATA = _build_moment_matrix()
_M27_INV_DATA = np.linalg.inv(_M27_DATA)


@functools.cache
def _get_d3q27_matrices(device: torch.device, dtype: torch.dtype):
    return (
        torch.tensor(_M27_DATA, dtype=dtype, device=device),
        torch.tensor(_M27_INV_DATA, dtype=dtype, device=device),
    )


# ── 1-D shift / unshift ──────────────────────────────────────────────────

def _shift_1d(m, groups, u):
    """Apply 1-D forward binomial shift: raw → central moments."""
    out = m.clone()
    for i0, i1, i2 in groups:
        m0 = m[i0]
        m1 = m[i1]
        m2 = m[i2]
        out[i1] = m1 - u * m0
        out[i2] = m2 - 2.0 * u * m1 + u * u * m0
    return out


def _unshift_1d(m, groups, u):
    """Apply 1-D inverse binomial shift: central → raw moments."""
    out = m.clone()
    for i0, i1, i2 in groups:
        m0 = m[i0]
        m1 = m[i1]
        m2 = m[i2]
        out[i1] = m1 + u * m0
        out[i2] = m2 + 2.0 * u * m1 + u * u * m0
    return out


def _to_central(m, ux, uy, uz):
    """Shift D3Q27 raw moments → central moments."""
    x_g, y_g, z_g = _D3Q27_SHIFT_GROUPS
    m = _shift_1d(m, x_g, ux)
    m = _shift_1d(m, y_g, uy)
    m = _shift_1d(m, z_g, uz)
    return m


def _to_raw(k, ux, uy, uz):
    """Unshift D3Q27 central moments → raw moments."""
    x_g, y_g, z_g = _D3Q27_SHIFT_GROUPS
    k = _unshift_1d(k, z_g, uz)
    k = _unshift_1d(k, y_g, uy)
    k = _unshift_1d(k, x_g, ux)
    return k


# ── Equilibrium distribution ──────────────────────────────────────────────

def equilibrium_d3q27(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Compute D3Q27 equilibrium distribution.

    Args:
        rho: density field (nz, ny, nx)
        ux, uy, uz: velocity components (nz, ny, nx)

    Returns:
        f_eq: (27, nz, ny, nx)
    """
    device = rho.device
    dtype = rho.dtype
    c = C_D3Q27.to(device=device, dtype=dtype)  # (27, 3)
    w = W_D3Q27.to(device=device, dtype=dtype).view(27, 1, 1, 1)  # (27, 1, 1, 1)

    u_sq = ux * ux + uy * uy + uz * uz  # (nz, ny, nx)
    cu = (c[:, 0].view(27, 1, 1, 1) * ux
          + c[:, 1].view(27, 1, 1, 1) * uy
          + c[:, 2].view(27, 1, 1, 1) * uz)  # (27, nz, ny, nx)

    return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))


def macroscopic_d3q27(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover macroscopic quantities from D3Q27 distributions.

    Args:
        f: (27, nz, ny, nx)

    Returns:
        (rho, ux, uy, uz) each (nz, ny, nx)
    """
    device = f.device
    c = C_D3Q27.to(device=device, dtype=f.dtype)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * c[:, 0].view(27, 1, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(27, 1, 1, 1)).sum(dim=0) / rho_safe
    uz = (f * c[:, 2].view(27, 1, 1, 1)).sum(dim=0) / rho_safe
    return rho, ux, uy, uz


# ── BGK collision ─────────────────────────────────────────────────────────

def collide_bgk_d3q27(f: torch.Tensor, tau: float) -> torch.Tensor:
    """BGK collision for D3Q27.

    f_new = f - (f - f_eq) / tau

    Args:
        f: (27, nz, ny, nx)
        tau: relaxation time (> 0.5)

    Returns:
        post-collision f
    """
    rho, ux, uy, uz = macroscopic_d3q27(f)
    feq = equilibrium_d3q27(rho, ux, uy, uz)
    return f - (f - feq) / tau


# ── MRT / Cascaded collision ──────────────────────────────────────────────

def collide_mrt_d3q27(
    f: torch.Tensor,
    tau: float,
    omega_b: float = 1.0,
    omega_3: float = 1.0,
    omega_4: float = 1.0,
) -> torch.Tensor:
    """Cascaded (central-moment) collision for D3Q27.

    Uses the proper moment matrix and 1-D shift decomposition.

    Args:
        f: (27, nz, ny, nx)
        tau: shear relaxation time
        omega_b: bulk viscosity relaxation rate
        omega_3: 3rd-order ghost relaxation rate
        omega_4: 4th+ order ghost relaxation rate

    Returns:
        post-collision f
    """
    device = f.device
    dtype = f.dtype
    nz, ny, nx = f.shape[1:]
    N = nz * ny * nx

    rho, ux, uy, uz = macroscopic_d3q27(f)
    rho_f = rho.view(N)
    ux_f = ux.view(N)
    uy_f = uy.view(N)
    uz_f = uz.view(N)

    M, Minv = _get_d3q27_matrices(device, dtype)

    # Transform to raw moment space
    f_flat = f.view(27, N)
    m = M @ f_flat  # (27, N)

    # Shift to central moments
    k = _to_central(m, ux_f, uy_f, uz_f)

    # Relax (working on f_neq central moments, so equilibrium = 0)
    omega = 1.0 / tau

    # 2nd order: trace/deviatoric split
    kxx, kyy, kzz = k[4], k[5], k[6]
    kxy, kxz, kyz = k[7], k[8], k[9]
    trace = kxx + kyy + kzz
    dev_xx = kxx - trace / 3.0
    dev_yy = kyy - trace / 3.0
    dev_zz = kzz - trace / 3.0

    trace_s = (1.0 - omega_b) * trace
    dev_xx_s = (1.0 - omega) * dev_xx
    dev_yy_s = (1.0 - omega) * dev_yy
    dev_zz_s = (1.0 - omega) * dev_zz

    k[4] = dev_xx_s + trace_s / 3.0
    k[5] = dev_yy_s + trace_s / 3.0
    k[6] = dev_zz_s + trace_s / 3.0
    k[7] = (1.0 - omega) * kxy
    k[8] = (1.0 - omega) * kxz
    k[9] = (1.0 - omega) * kyz

    # 3rd order
    lo, hi = _D3Q27_ORDER_BOUNDS["third"]
    k[lo:hi] = (1.0 - omega_3) * k[lo:hi]

    # 4th+ order
    for order in ["fourth", "fifth", "sixth"]:
        lo, hi = _D3Q27_ORDER_BOUNDS[order]
        k[lo:hi] = (1.0 - omega_4) * k[lo:hi]

    # Unshift back to raw moments
    m_s = _to_raw(k, ux_f, uy_f, uz_f)

    # Transform back to populations
    f_new_flat = Minv @ m_s
    return f_new_flat.view(27, nz, ny, nx)


# ── Cumulant collision (Geier 2015) ───────────────────────────────────────

def collide_cumulant_d3q27(
    f: torch.Tensor,
    tau: float,
    omega_b: float = 1.0,
    omega_3: float = 1.0,
    omega_4: float = 1.0,
) -> torch.Tensor:
    """Cumulant collision for D3Q27 (Geier 2015).

    Full cumulant hierarchy with 1/ρ factors for 4th-order cumulants.

    Args:
        f: (27, nz, ny, nx)
        tau: shear relaxation time
        omega_b: bulk viscosity relaxation rate
        omega_3: 3rd-order ghost relaxation rate
        omega_4: 4th-order ghost relaxation rate

    Returns:
        post-collision f
    """
    device = f.device
    dtype = f.dtype
    nz, ny, nx = f.shape[1:]
    N = nz * ny * nx

    rho, ux, uy, uz = macroscopic_d3q27(f)
    rho_f = rho.view(N)
    ux_f = ux.view(N)
    uy_f = uy.view(N)
    uz_f = uz.view(N)

    M, Minv = _get_d3q27_matrices(device, dtype)
    f_flat = f.view(27, N)

    # Raw moments
    m = M @ f_flat

    # Central moments
    k = _to_central(m, ux_f, uy_f, uz_f)

    # Cumulants: for order ≤ 3, cumulants = central moments
    # For 4th order, cumulants have 1/ρ correction
    rho_safe = torch.clamp(rho_f, min=1e-12)

    c = k.clone()

    # 4th order cumulant corrections (Geier 2015)
    # C_{220} = κ_{220} - (κ_{200}·κ_{020} + 2·κ_{110}²) / ρ
    c[17] = k[17] - (k[4] * k[5] + 2 * k[7]**2) / rho_safe  # C220
    c[18] = k[18] - (k[4] * k[6] + 2 * k[8]**2) / rho_safe  # C202
    c[19] = k[19] - (k[5] * k[6] + 2 * k[9]**2) / rho_safe  # C022
    c[20] = k[20] - (k[4] * k[9] + 2 * k[8] * k[7]) / rho_safe  # C211
    c[21] = k[21] - (k[5] * k[8] + 2 * k[7] * k[9]) / rho_safe  # C121
    c[22] = k[22] - (k[6] * k[7] + 2 * k[8] * k[9]) / rho_safe  # C112

    # Relaxation
    omega = 1.0 / tau

    # 2nd order: trace/deviatoric split
    kxx, kyy, kzz = c[4], c[5], c[6]
    kxy, kxz, kyz = c[7], c[8], c[9]
    trace = kxx + kyy + kzz
    dev_xx = kxx - trace / 3.0
    dev_yy = kyy - trace / 3.0
    dev_zz = kzz - trace / 3.0

    trace_s = (1.0 - omega_b) * trace
    dev_xx_s = (1.0 - omega) * dev_xx
    dev_yy_s = (1.0 - omega) * dev_yy
    dev_zz_s = (1.0 - omega) * dev_zz

    c[4] = dev_xx_s + trace_s / 3.0
    c[5] = dev_yy_s + trace_s / 3.0
    c[6] = dev_zz_s + trace_s / 3.0
    c[7] = (1.0 - omega) * kxy
    c[8] = (1.0 - omega) * kxz
    c[9] = (1.0 - omega) * kyz

    # 3rd order
    lo, hi = _D3Q27_ORDER_BOUNDS["third"]
    c[lo:hi] = (1.0 - omega_3) * c[lo:hi]

    # 4th+ order cumulants (relax to zero)
    for order in ["fourth", "fifth", "sixth"]:
        lo, hi = _D3Q27_ORDER_BOUNDS[order]
        c[lo:hi] = (1.0 - omega_4) * c[lo:hi]

    # Back-transform: cumulants → central moments (4th order)
    k_back = c.clone()
    k_back[17] = c[17] + (c[4] * c[5] + 2 * c[7]**2) / rho_safe
    k_back[18] = c[18] + (c[4] * c[6] + 2 * c[8]**2) / rho_safe
    k_back[19] = c[19] + (c[5] * c[6] + 2 * c[9]**2) / rho_safe
    k_back[20] = c[20] + (c[4] * c[9] + 2 * c[8] * c[7]) / rho_safe
    k_back[21] = c[21] + (c[5] * c[8] + 2 * c[7] * c[9]) / rho_safe
    k_back[22] = c[22] + (c[6] * c[7] + 2 * c[8] * c[9]) / rho_safe

    # Unshift back to raw moments
    m_s = _to_raw(k_back, ux_f, uy_f, uz_f)

    # Transform back to populations
    f_new_flat = Minv @ m_s
    return f_new_flat.view(27, nz, ny, nx)
