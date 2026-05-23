from __future__ import annotations

import functools

import torch

from .d3q19 import C, equilibrium3d, macroscopic3d


def _build_d3q19_mrt_matrices() -> tuple[list[list[float]], list[list[float]]]:
    """Compute and return (M, M_inv) as nested Python lists (float64 precision)."""
    import numpy as np

    c_np = C.numpy().astype(np.float64)
    cx, cy, cz = c_np[:, 0], c_np[:, 1], c_np[:, 2]
    e2 = cx**2 + cy**2 + cz**2
    e4 = e2**2

    matrix = np.array(
        [
            np.ones(19),
            19.0 * e2 - 30.0,
            (21.0 * e4 - 53.0 * e2 + 24.0) / 2.0,
            cx,
            cx * e2 * (5.0 * e2 - 9.0) / 2.0,
            cy,
            cy * e2 * (5.0 * e2 - 9.0) / 2.0,
            cz,
            cz * e2 * (5.0 * e2 - 9.0) / 2.0,
            3.0 * cx**2 - e2,
            cx**2 - cy**2,
            cx * cy,
            cx * cz,
            cy * cz,
            (3.0 * e2 - 5.0) * (3.0 * cx**2 - e2) / 2.0,
            (3.0 * e2 - 5.0) * (cx**2 - cy**2) / 2.0,
            cx**2 * cy,
            cx**2 * cz,
            cy**2 * cx,
        ]
    )
    assert np.linalg.matrix_rank(matrix) == 19, "D3Q19 MRT matrix is rank-deficient"
    matrix_inv = np.linalg.inv(matrix)
    return matrix.tolist(), matrix_inv.tolist()


_M_D3Q19_DATA, _M_D3Q19_INV_DATA = _build_d3q19_mrt_matrices()


@functools.cache
def _get_d3q19_mrt_matrices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.tensor(_M_D3Q19_DATA, dtype=torch.float32, device=device)
    matrix_inv = torch.tensor(_M_D3Q19_INV_DATA, dtype=torch.float32, device=device)
    return matrix, matrix_inv


def collide_bgk3d(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Single-relaxation-time BGK collision step for D3Q19."""
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    return f - (f - feq) / tau


def collide_mrt3d(
    f: torch.Tensor,
    tau: float,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """Multi-relaxation-time (MRT) collision step for D3Q19.

    The shear viscosity is determined by *tau* (same as BGK). Independent
    relaxation rates for non-hydrodynamic moments improve stability at high
    Reynolds numbers.

    Moment relaxation rates (vector s, length 19):
        0: rho      – 0 (conserved)
        1: e        – s_e
        2: eps      – s_eps
        3,5,7: jx,jy,jz – 0 (conserved)
        4,6,8: qx,qy,qz – s_q
        9–13: stress    – 1/tau
        14–15: Txx,Tww  – s_pi (defaults to s_e)
        16–18: cubic    – 1 (fully relax non-physical modes)

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau: Relaxation time for shear stress.
        s_e: Relaxation rate for energy moment.
        s_eps: Relaxation rate for energy-square moment.
        s_q: Relaxation rate for heat-flux moments.
        s_pi: Relaxation rate for higher-order stress moments
              (defaults to *s_e* when *None*).

    Returns:
        Updated distribution tensor of the same shape.
    """
    if s_pi is None:
        s_pi = s_e
    device = f.device
    matrix, matrix_inv = _get_d3q19_mrt_matrices(device)

    s_nu = 1.0 / tau
    s_vec = torch.tensor(
        [
            0.0,
            s_e,
            s_eps,
            0.0,
            s_q,
            0.0,
            s_q,
            0.0,
            s_q,
            s_nu,
            s_nu,
            s_nu,
            s_nu,
            s_nu,
            s_pi,
            s_pi,
            1.0,
            1.0,
            1.0,
        ],
        dtype=f.dtype,
        device=device,
    )

    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(19, -1)
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    feq_flat = feq.reshape(19, -1)

    moments = matrix @ f_flat
    moments_eq = matrix @ feq_flat
    moments_star = moments - s_vec.unsqueeze(1) * (moments - moments_eq)
    return (matrix_inv @ moments_star).reshape(19, nz, ny, nx)


def stream3d(f: torch.Tensor) -> torch.Tensor:
    """Vectorised streaming step for D3Q19 (periodic boundaries).

    Replaces the per-direction ``torch.roll`` loop with a single advanced-index
    gather over all 19 directions simultaneously.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.

    Returns:
        Streamed tensor of the same shape.
    """
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    device = f.device
    c = C.to(device)

    z_src = (torch.arange(nz, device=device).unsqueeze(0) - c[:, 2].unsqueeze(1)) % nz
    y_src = (torch.arange(ny, device=device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
    x_src = (torch.arange(nx, device=device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx

    q_idx = torch.arange(19, device=device).view(19, 1, 1, 1).expand(19, nz, ny, nx)
    z_idx = z_src.view(19, nz, 1, 1).expand(19, nz, ny, nx)
    y_idx = y_src.view(19, 1, ny, 1).expand(19, nz, ny, nx)
    x_idx = x_src.view(19, 1, 1, nx).expand(19, nz, ny, nx)

    return f[q_idx, z_idx, y_idx, x_idx]


def correct_mass3d(f: torch.Tensor, target_mass: float) -> torch.Tensor:
    """Redistribute mass uniformly to correct global mass drift (3-D).

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        target_mass: Desired total mass.

    Returns:
        Rescaled distribution tensor.
    """
    current = f.sum()
    if current.abs() < 1e-30:
        return f
    return f * (target_mass / current)


__all__ = [
    "collide_bgk3d",
    "collide_mrt3d",
    "stream3d",
    "correct_mass3d",
]
