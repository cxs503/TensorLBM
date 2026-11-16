"""D2Q9 lattice Boltzmann kernel: velocities, weights, equilibrium, macroscopic.

This is the 2-D counterpart of the D3Q19 kernel used by the multiphase and
wetting modules.  All quantities are in lattice units.
"""

from __future__ import annotations

import functools

import torch

# D2Q9 lattice velocities (cx, cy), weights, and opposite-direction mapping
C = torch.tensor(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=torch.int64,
)
W = torch.tensor(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=torch.float32,
)
OPPOSITE = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)
CS2 = 1.0 / 3.0  # lattice speed of sound squared

# Exact double-precision D2Q9 weights
W_EXACT: tuple[float, ...] = (
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
)

# ---------------------------------------------------------------------------
# D2Q9 MRT transformation matrix (Guo/Shyy moment basis)
# ---------------------------------------------------------------------------
# Reference: Guo, Zheng & Shi, Phys. Rev. E 78, 036704 (2008)
#            d'Humières, Prog. Astronaut. Aeronaut. 159, 450 (1992)
#
# Moments (row order):
#   m0 = ρ          (density, conserved)
#   m1 = e          (energy, non-conserved)
#   m2 = ε          (energy square, non-conserved)
#   m3 = jx         (x-momentum, conserved)
#   m4 = qx         (x-energy flux, non-conserved)
#   m5 = jy         (y-momentum, conserved)
#   m6 = qy         (y-energy flux, non-conserved)
#   m7 = pxx        (stress xx, non-conserved)
#   m8 = pxy        (stress xy, non-conserved)
# ---------------------------------------------------------------------------

M_D2Q9 = torch.tensor(
    [
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1],  # m0 = ρ
        [-4, -1, -1, -1, -1,  2,  2,  2,  2],  # m1 = e
        [ 4, -2, -2, -2, -2,  1,  1,  1,  1],  # m2 = ε
        [ 0,  1,  0, -1,  0,  1, -1, -1,  1],  # m3 = jx
        [ 0, -2,  0,  2,  0,  1, -1, -1,  1],  # m4 = qx
        [ 0,  0,  1,  0, -1,  1,  1, -1, -1],  # m5 = jy
        [ 0,  0, -2,  0,  2,  1,  1, -1, -1],  # m6 = qy
        [ 0,  1, -1,  1, -1,  0,  0,  0,  0],  # m7 = pxx
        [ 0,  0,  0,  0,  0,  1, -1,  1, -1],  # m8 = pxy
    ],
    dtype=torch.float64,
)

# Pre-compute inverse of M_D2Q9
MINV_D2Q9 = torch.linalg.inv(M_D2Q9)


@functools.cache
def _c_on(device: torch.device) -> torch.Tensor:
    return C.to(device)


@functools.cache
def _w_for(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """D2Q9 weights built from exact double literals for ``dtype``."""
    return torch.tensor(W_EXACT, dtype=dtype, device=device)


@functools.cache
def _m_on(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """D2Q9 MRT transformation matrix on device."""
    return M_D2Q9.to(device=device, dtype=dtype)


@functools.cache
def _minv_on(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """D2Q9 MRT inverse transformation matrix on device."""
    return MINV_D2Q9.to(device=device, dtype=dtype)


def equilibrium(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute D2Q9 equilibrium distribution f_eq.

    Parameters
    ----------
    rho, ux, uy : torch.Tensor
        Macroscopic fields, shape ``(ny, nx)``.

    Returns
    -------
    feq : torch.Tensor, shape ``(9, ny, nx)``
    """
    if not (rho.shape == ux.shape == uy.shape):
        raise ValueError(
            f"shape mismatch: rho={tuple(rho.shape)}, ux={tuple(ux.shape)}, uy={tuple(uy.shape)}"
        )
    if device is None:
        device = rho.device
    c = _c_on(device)
    w = _w_for(rho.dtype, device).view(9, 1, 1)

    u_sq = ux * ux + uy * uy
    cu = c[:, 0].view(9, 1, 1) * ux + c[:, 1].view(9, 1, 1) * uy
    return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))


def macroscopic(
    f: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover rho, ux, uy from particle distributions.

    Parameters
    ----------
    f : torch.Tensor, shape ``(9, ny, nx)``

    Returns
    -------
    rho, ux, uy : torch.Tensor, shape ``(ny, nx)``
    """
    if device is None:
        device = f.device
    c = _c_on(device)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * c[:, 0].view(9, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0) / rho_safe
    return rho, ux, uy


def stream(f: torch.Tensor) -> torch.Tensor:
    """Stream distributions with periodic boundary conditions.

    Parameters
    ----------
    f : torch.Tensor, shape ``(9, ny, nx)``

    Returns
    -------
    f_out : torch.Tensor, shape ``(9, ny, nx)``
    """
    nf, ny, nx = f.shape
    device = f.device
    c = _c_on(device)
    f_out = torch.empty_like(f)

    for i in range(nf):
        cx, cy = int(c[i, 0]), int(c[i, 1])
        if cx == 0 and cy == 0:
            f_out[i] = f[i]
        else:
            f_out[i] = torch.roll(f[i], shifts=(-cy, -cx), dims=(0, 1))
    return f_out


def collide_bgk(f: torch.Tensor, tau: float) -> torch.Tensor:
    """BGK (SRT) collision."""
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau


def _equilibrium_moments(
    rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor
) -> torch.Tensor:
    """Compute equilibrium moments for D2Q9 MRT.

    Parameters
    ----------
    rho, ux, uy : torch.Tensor, shape (ny, nx)

    Returns
    -------
    m_eq : torch.Tensor, shape (9, ny, nx)
        Equilibrium moments in the Guo/Shyy basis.
    """
    u2 = ux * ux + uy * uy
    # m0_eq = ρ
    # m1_eq = ρ(-2 + 3·u²)
    # m2_eq = ρ(1 - 3·u²)
    # m3_eq = ρ·ux
    # m4_eq = -ρ·ux
    # m5_eq = ρ·uy
    # m6_eq = -ρ·uy
    # m7_eq = ρ(ux² - uy²)
    # m8_eq = ρ·ux·uy
    m_eq = torch.stack(
        [
            rho,
            rho * (-2.0 + 3.0 * u2),
            rho * (1.0 - 3.0 * u2),
            rho * ux,
            -rho * ux,
            rho * uy,
            -rho * uy,
            rho * (ux * ux - uy * uy),
            rho * ux * uy,
        ],
        dim=0,
    )
    return m_eq


def collide_mrt(
    f: torch.Tensor,
    tau: float,
    s_e: float = 1.0,
    s_q: float = 1.0,
) -> torch.Tensor:
    """D2Q9 MRT (Multiple Relaxation Time) collision.

    Implements the Guo/Shyy moment basis MRT collision operator:
    1. Transform f → moment space: m = M·f
    2. Relax moments: m* = m - S·(m - m_eq)
    3. Transform back: f* = M⁻¹·m*

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
        Particle distribution functions.
    tau : float
        Relaxation time for shear modes (s_ν = 1/τ).
    s_e : float, optional
        Relaxation rate for energy modes (m1, m2). Default 1.0.
    s_q : float, optional
        Relaxation rate for energy flux modes (m4, m6). Default 1.0.

    Returns
    -------
    f_out : torch.Tensor, shape (9, ny, nx)
        Post-collision distribution functions.

    Notes
    -----
    The relaxation matrix S is diagonal:
        S = diag(0, s_e, s_e, 0, s_q, 0, s_q, s_ν, s_ν)
    where s_ν = 1/τ, and conserved moments (m0, m3, m5) have s=0.

    Reference: Guo, Zheng & Shi, Phys. Rev. E 78, 036704 (2008).
    """
    device = f.device
    dtype = f.dtype

    # Get transformation matrices
    M = _m_on(device, dtype)
    Minv = _minv_on(device, dtype)

    # Compute macroscopic quantities
    rho, ux, uy = macroscopic(f, device=device)

    # Transform to moment space: m = M @ f
    # f shape: (9, ny, nx) → reshape to (9, N) for matmul
    ny, nx = f.shape[1], f.shape[2]
    f_flat = f.view(9, -1)  # (9, N)
    m = M @ f_flat  # (9, N)

    # Compute equilibrium moments
    m_eq = _equilibrium_moments(rho, ux, uy)
    m_eq_flat = m_eq.view(9, -1)  # (9, N)

    # Build relaxation matrix S (diagonal)
    s_nu = 1.0 / tau
    S = torch.zeros(9, dtype=dtype, device=device)
    S[0] = 0.0    # ρ (conserved)
    S[1] = s_e    # e
    S[2] = s_e    # ε
    S[3] = 0.0    # jx (conserved)
    S[4] = s_q    # qx
    S[5] = 0.0    # jy (conserved)
    S[6] = s_q    # qy
    S[7] = s_nu   # pxx
    S[8] = s_nu   # pxy

    # Relax: m* = m - S·(m - m_eq)
    m_diff = m - m_eq_flat
    m_star = m - S.view(9, 1) * m_diff

    # Transform back: f* = M⁻¹·m*
    f_star_flat = Minv @ m_star
    f_out = f_star_flat.view(9, ny, nx)

    return f_out
