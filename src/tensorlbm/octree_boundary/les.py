"""LES sub-grid model on octree-shell leaves via neighbor-table gathers.

The WALE/Smagorinsky operators in ``tensorlbm.turbulence`` are written for
regular grid tensors (torch.roll neighbours).  Octree-shell leaves are a
sparse SoA (``f_leaf (Q, n_leaf)``) whose spatial neighbours are only known
through ``neighbor_table`` (Q, n_leaf) with sentinels
``SHELL_OUTSIDE / SOLID / DOMAIN_OUT / FANOUT``.

Here we compute the velocity-gradient tensor with neighbour *gathers*
(``f[..., neighbor_table[d, :]]``), so the sub-grid eddy viscosity is
physically meaningful on the shell.

Gradients are one-sided when the opposite neighbour is missing (sentinel),
which is the correct fallback for a shell whose inner neighbour is SOLID
(wall) and outer neighbour is SHELL_OUTSIDE (ghost-filled).
"""
from __future__ import annotations

import torch

from tensorlbm.d3q27 import macroscopic27
from tensorlbm.d3q19 import macroscopic3d

# sentinels (must match geometry.py)
SHELL_OUTSIDE = -1
SOLID = -2
DOMAIN_OUT = -3
FANOUT = -4


def _leaf_macros(f: torch.Tensor, Q: int):
    """(rho, ux, uy, uz) per leaf from (Q, n_leaf) populations."""
    if Q == 27:
        rho, ux, uy, uz = macroscopic27(f.view(Q, 1, 1, -1))
    else:
        rho, ux, uy, uz = macroscopic3d(f.view(Q, 1, 1, -1))
    return (rho.reshape(-1), ux.reshape(-1), uy.reshape(-1), uz.reshape(-1))


def _gather_velocity(
    f: torch.Tensor, d: int, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
    neighbor_table: torch.Tensor, n_leaf: int,
):
    """Velocity of the neighbour in direction d; NaN where missing."""
    nb = neighbor_table[d]
    valid = nb >= 0
    idx = nb.clamp(min=0)
    # zeros for invalid entries (masked later)
    gx = torch.where(valid, ux[idx], torch.zeros_like(ux))
    gy = torch.where(valid, uy[idx], torch.zeros_like(uy))
    gz = torch.where(valid, uz[idx], torch.zeros_like(uz))
    return gx, gy, gz, valid

def _gradient(f, d: int, u, neighbor_table, n_leaf, dx):
    """One-sided central difference ∂u/∂x_d via neighbour gather."""
    g, _, _, valid = _gather_velocity(
        f, d, u, torch.zeros_like(u), torch.zeros_like(u), neighbor_table, n_leaf,
    )
    return torch.where(valid, (g - u) / dx, torch.zeros_like(u))


def leaf_wale_nu_t(
    f: torch.Tensor,
    neighbor_table: torch.Tensor,
    C_w: float,
    dx: float,
) -> torch.Tensor:
    """WALE eddy viscosity on octree leaves (neighbour-gather gradients).

    Args:
        f: (Q, n_leaf) populations.
        neighbor_table: (Q, n_leaf) int64 with sentinels.
        C_w: WALE constant.
        dx: leaf spacing (lattice units).

    Returns:
        nu_t (n_leaf,) >= 0.
    """
    Q = f.shape[0]
    rho, ux, uy, uz = _leaf_macros(f, Q)
    n_leaf = f.shape[1]
    nbt = neighbor_table

    # velocity gradient tensor g_ij = ∂u_i / ∂x_j  (x=2, y=1, z=0 dims)
    g11 = _gradient(f, 2, ux, nbt, n_leaf, dx)   # ∂ux/∂x
    g12 = _gradient(f, 1, ux, nbt, n_leaf, dx)   # ∂ux/∂y
    g13 = _gradient(f, 0, ux, nbt, n_leaf, dx)   # ∂ux/∂z
    g21 = _gradient(f, 2, uy, nbt, n_leaf, dx)
    g22 = _gradient(f, 1, uy, nbt, n_leaf, dx)
    g23 = _gradient(f, 0, uy, nbt, n_leaf, dx)
    g31 = _gradient(f, 2, uz, nbt, n_leaf, dx)
    g32 = _gradient(f, 1, uz, nbt, n_leaf, dx)
    g33 = _gradient(f, 0, uz, nbt, n_leaf, dx)

    # S_ij = (g_ij + g_ji)/2
    S11 = 0.5 * (g11 + g11); S12 = 0.5 * (g12 + g21); S13 = 0.5 * (g13 + g31)
    S21 = S12;              S22 = 0.5 * (g22 + g22); S23 = 0.5 * (g23 + g32)
    S31 = S13;              S32 = S23;              S33 = 0.5 * (g33 + g33)

    S2 = (S11**2 + S22**2 + S33**2
          + 2 * (S12**2 + S13**2 + S23**2))          # |S|²
    S3 = (S11 * (S22 * S33 - S23 * S32)
          - S12 * (S21 * S33 - S23 * S31)
          + S13 * (S21 * S32 - S22 * S31))           # det(S)

    # WALE: nu_t = (C_w dx)² * S3^(2/3) / (S2^(5/2) + S3^(5/3))
    # Use sign-preserving real power for possibly-negative S3.
    S3_abs = S3.abs().clamp_min(1e-30)
    num = S3_abs ** (2.0 / 3.0)
    den = S2 ** (2.5) + S3_abs ** (5.0 / 3.0) + 1e-12
    nu_t = (C_w * dx) ** 2 * num / den
    return torch.clamp(nu_t, min=0.0)


def leaf_smagorinsky_nu_t(
    f: torch.Tensor,
    neighbor_table: torch.Tensor,
    C_s: float,
    dx: float,
) -> torch.Tensor:
    """Smagorinsky eddy viscosity on octree leaves (neighbour-gather gradients)."""
    Q = f.shape[0]
    rho, ux, uy, uz = _leaf_macros(f, Q)
    n_leaf = f.shape[1]
    nbt = neighbor_table

    g11 = _gradient(f, 2, ux, nbt, n_leaf, dx)
    g12 = _gradient(f, 1, ux, nbt, n_leaf, dx)
    g13 = _gradient(f, 0, ux, nbt, n_leaf, dx)
    g21 = _gradient(f, 2, uy, nbt, n_leaf, dx)
    g22 = _gradient(f, 1, uy, nbt, n_leaf, dx)
    g23 = _gradient(f, 0, uy, nbt, n_leaf, dx)
    g31 = _gradient(f, 2, uz, nbt, n_leaf, dx)
    g32 = _gradient(f, 1, uz, nbt, n_leaf, dx)
    g33 = _gradient(f, 0, uz, nbt, n_leaf, dx)

    S2 = (g11**2 + g22**2 + g33**2
          + 0.5 * ((g12 + g21)**2 + (g13 + g31)**2 + (g23 + g32)**2))
    nu_t = (C_s * dx) ** 2 * torch.sqrt(S2.clamp(min=0.0))
    return torch.clamp(nu_t, min=0.0)


def leaf_les_collide(
    f: torch.Tensor,
    tau: float,
    neighbor_table: torch.Tensor,
    *,
    model: str = "wale",
    C_w: float = 0.5,
    C_s: float = 0.05,
    dx: float = 0.25,
) -> torch.Tensor:
    """MRT collision with LES on octree leaves (stable at tau->0.5).

    The sub-grid eddy viscosity comes from neighbour-table gathers (spatially
    correct on the SoA shell); the collision is D3Q27 MRT so the scheme stays
    stable at high Re (BGK is not).
    """
    Q = f.shape[0]
    if Q != 27:
        raise NotImplementedError("leaf LES requires D3Q27")
    if model == "wale":
        nu_t = leaf_wale_nu_t(f, neighbor_table, C_w, dx)
    else:
        nu_t = leaf_smagorinsky_nu_t(f, neighbor_table, C_s, dx)

    from tensorlbm.turbulence import _get_d3q27_mrt_matrices, _nu_t_to_tau_eff
    from tensorlbm.d3q27 import macroscopic27, equilibrium27

    device = f.device
    M, M_inv = _get_d3q27_mrt_matrices(device)

    rho, ux, uy, uz = macroscopic27(f.view(Q, 1, 1, -1))
    feq = equilibrium27(rho, ux, uy, uz)
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)
    s_nu_flat = (1.0 / tau_eff).reshape(-1)

    # moment space: m = M f ; feq_m = M feq ; relax non-conserved rows
    f4 = f.view(Q, 1, 1, -1)
    n = f4.shape[-1]
    m = M @ f4.view(Q, -1)
    meq = M @ feq.view(Q, -1)
    s = torch.ones(Q, n, device=device, dtype=f.dtype)
    # row 0..3 conserved (rho, jx, jy, jz): s=0 keeps them unchanged
    s[0:4] = 0.0
    # row 4 energy, row 19 energy^2, rows 10-18 heat flux, rows 20-26 4th order
    s[4] = 1.19
    s[10:19] = 1.2
    s[19] = 1.4
    s[20:27] = 1.19
    # shear rows 5..9 relaxed with 1/tau_eff (per-leaf eddy viscosity)
    s[5:10] = s_nu_flat.view(1, -1).expand(5, n)
    m_post = m + s * (meq - m)
    f_out = (M_inv @ m_post).view(Q, 1, 1, n)
    return f_out
