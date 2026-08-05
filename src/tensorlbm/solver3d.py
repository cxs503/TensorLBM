from __future__ import annotations

import functools  # retained for compatibility; internal cache functions use explicit dicts
from typing import Any

import torch

from .d3q19 import OPPOSITE as _OPPOSITE_3D
from .d3q19 import C, equilibrium3d, macroscopic3d

# Cache for streaming index tensors keyed by (nz, ny, nx, device_type, device_index)
_stream3d_cache: dict[
    tuple[Any, ...],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}

# Manual dict cache keyed by (device_str, dtype) — avoids the multi-GPU
# collision that @functools.cache causes for torch.device arguments.
_mrt3d_matrix_cache: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


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


def _get_d3q19_mrt_matrices(
    device: torch.device, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached (M, M_inv) for D3Q19 MRT, keyed by (str(device), dtype).

    Uses an explicit string key so that cuda:0, cuda:1, … are cached
    independently — ``@functools.cache`` on a ``torch.device`` argument
    is unsafe in multi-GPU settings because devices with the same type
    but different indices can collide in the hash.
    """
    key = (str(device), dtype)
    if key not in _mrt3d_matrix_cache:
        matrix = torch.tensor(_M_D3Q19_DATA, dtype=dtype, device=device)
        matrix_inv = torch.tensor(_M_D3Q19_INV_DATA, dtype=dtype, device=device)
        _mrt3d_matrix_cache[key] = (matrix, matrix_inv)
    return _mrt3d_matrix_cache[key]


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
    """Streaming step for D3Q19 using torch.roll (memory-optimized).

    Uses torch.roll per direction instead of cached index tensors.
    Eliminates 4×[19,N] int64 index tensors (~6GB for 10M cells),
    trading a small speed cost for massive memory savings.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.

    Returns:
        Streamed tensor of the same shape.
    """
    # D3Q19 velocity vectors: (cx, cy, cz) per direction
    # Pull scheme: out[q](x) = f[q](x - c_q) → shift by +c_q
    shifts = [
        (0, 0, 0),  # 0: rest
        (1, 0, 0),  # 1: +x
        (-1, 0, 0),  # 2: -x
        (0, 1, 0),  # 3: +y
        (0, -1, 0),  # 4: -y
        (0, 0, 1),  # 5: +z
        (0, 0, -1),  # 6: -z
        (1, 1, 0),  # 7: +x+y
        (-1, -1, 0),  # 8: -x-y  (was -x+y — bug: must match C)
        (1, -1, 0),  # 9: +x-y
        (-1, 1, 0),  # 10: -x+y (was -x-y — bug: must match C)
        (1, 0, 1),  # 11: +x+z
        (-1, 0, -1),  # 12: -x-z (was -x+z — bug: must match C)
        (1, 0, -1),  # 13: +x-z
        (-1, 0, 1),  # 14: -x+z (was -x-z — bug: must match C)
        (0, 1, 1),  # 15: +y+z
        (0, -1, -1),  # 16: -y-z (was -y+z — bug: must match C)
        (0, 1, -1),  # 17: +y-z
        (0, -1, 1),  # 18: -y+z (was -y-z — bug: must match C)
    ]
    # dims: (z, y, x) → roll dims = (3, 2, 1) for (x, y, z)
    out = torch.empty_like(f)
    # q=0: rest direction — no shift needed, just copy (skip torch.roll)
    out[0] = f[0]
    for q in range(1, 19):
        sx, sy, sz = shifts[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def stream3d_index_select(f: torch.Tensor) -> torch.Tensor:
    """Streaming step for D3Q19 using a cached advanced-index gather.

    Compared to :func:`stream3d` (which uses per-direction ``torch.roll``),
    this variant pre-computes and caches four integer index tensors of shape
    ``(19, nz, ny, nx)``.  The single advanced-index gather is more
    GPU-friendly and avoids 18 separate roll dispatches, at the cost of
    higher memory for the index tensors (~6 × 19 × N int64 bytes).  For
    large grids (≥10 M cells) prefer :func:`stream3d` to save memory.

    Index tensors are cached per ``(nz, ny, nx, device_type, device_index)``
    so they are only allocated once per unique grid shape and device.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.

    Returns:
        Streamed tensor of the same shape.
    """
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    device = f.device
    cache_key = (nz, ny, nx, device.type, device.index)

    if cache_key not in _stream3d_cache:
        c_np = C.numpy()  # (19, 3) — cx, cy, cz
        # Pull scheme: out[q](r) = f[q](r - c_q)
        # For each direction q: src_z = (z - cz_q) % nz, etc.
        z_base = torch.arange(nz, device=device)  # (nz,)
        y_base = torch.arange(ny, device=device)  # (ny,)
        x_base = torch.arange(nx, device=device)  # (nx,)

        q_idx_list, z_idx_list, y_idx_list, x_idx_list = [], [], [], []
        for q in range(19):
            cxq, cyq, czq = int(c_np[q, 0]), int(c_np[q, 1]), int(c_np[q, 2])
            z_src = (z_base - czq) % nz  # (nz,)
            y_src = (y_base - cyq) % ny  # (ny,)
            x_src = (x_base - cxq) % nx  # (nx,)
            # broadcast to (nz, ny, nx) then prepend q-dim
            z_idx = z_src.view(nz, 1, 1).expand(nz, ny, nx)
            y_idx = y_src.view(1, ny, 1).expand(nz, ny, nx)
            x_idx = x_src.view(1, 1, nx).expand(nz, ny, nx)
            q_idx_list.append(torch.full((nz, ny, nx), q, dtype=torch.int64, device=device))
            z_idx_list.append(z_idx)
            y_idx_list.append(y_idx)
            x_idx_list.append(x_idx)

        q_idx = torch.stack(q_idx_list, dim=0)  # (19, nz, ny, nx)
        z_idx = torch.stack(z_idx_list, dim=0)  # (19, nz, ny, nx)
        y_idx = torch.stack(y_idx_list, dim=0)  # (19, nz, ny, nx)
        x_idx = torch.stack(x_idx_list, dim=0)  # (19, nz, ny, nx)
        _stream3d_cache[cache_key] = (q_idx, z_idx, y_idx, x_idx)

    q_idx, z_idx, y_idx, x_idx = _stream3d_cache[cache_key]
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


def collide_trt3d(
    f: torch.Tensor,
    tau_plus: float,
    lambda_trt: float = 3.0 / 16.0,
) -> torch.Tensor:
    """Two-relaxation-time (TRT) collision step for D3Q19.

    Uses two independent relaxation rates: *τ₊* controls the symmetric part
    (sets viscosity ν = (τ₊ − ½) / 3) and *τ₋* controls the anti-symmetric
    part (derived from the magic parameter Λ). Setting Λ = 3/16 eliminates
    wall-placement errors in Poiseuille flow (Ginzburg 2008).

    Reference
    ---------
    Ginzburg, I. (2008). Two-relaxation-time lattice Boltzmann scheme.
    *Commun. Comput. Phys.* 3(2), 427–478.

    Args:
        f:           Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau_plus:    Symmetric relaxation time (τ₊ > 0.5).
        lambda_trt:  Magic parameter Λ (default 3/16).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)

    tau_minus = 0.5 + lambda_trt / (tau_plus - 0.5)

    opp = _OPPOSITE_3D.to(f.device)
    f_plus = 0.5 * (f + f[opp])
    f_minus = 0.5 * (f - f[opp])
    feq_plus = 0.5 * (feq + feq[opp])
    feq_minus = 0.5 * (feq - feq[opp])

    return f - (f_plus - feq_plus) / tau_plus - (f_minus - feq_minus) / tau_minus


def collide_rlbm3d(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Regularized BGK (RLBM) collision step for D3Q19.

    Projects the non-equilibrium distribution onto the second-order Hermite
    polynomial subspace before BGK relaxation, filtering out higher-order
    ghost modes for improved stability at low viscosity (τ → 0.5).
    See Latt & Chopard, *Math. Comput. Simul.* (2006).

    Args:
        f:   Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau: Relaxation time (τ > 0.5). Kinematic viscosity ν = (τ − ½)/3.

    Returns:
        Updated distribution tensor of the same shape.
    """
    from .d3q19 import _c_on, _w_on  # noqa: PLC0415

    device = f.device
    c = _c_on(device).to(f.dtype)
    w = _w_on(device).to(f.dtype)

    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    fneq = f - feq

    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)

    # Second-order non-equilibrium moments Π_αβ
    pi_xx = (cx * cx * fneq).sum(dim=0)
    pi_yy = (cy * cy * fneq).sum(dim=0)
    pi_zz = (cz * cz * fneq).sum(dim=0)
    pi_xy = (cx * cy * fneq).sum(dim=0)
    pi_xz = (cx * cz * fneq).sum(dim=0)
    pi_yz = (cy * cz * fneq).sum(dim=0)

    cs2 = 1.0 / 3.0
    h_xx = cx * cx - cs2
    h_yy = cy * cy - cs2
    h_zz = cz * cz - cs2
    h_xy = cx * cy
    h_xz = cx * cz
    h_yz = cy * cz
    w_view = w.view(19, 1, 1, 1)
    fneq_reg = (
        (9.0 / 2.0)
        * w_view
        * (
            h_xx * pi_xx
            + h_yy * pi_yy
            + h_zz * pi_zz
            + 2.0 * h_xy * pi_xy
            + 2.0 * h_xz * pi_xz
            + 2.0 * h_yz * pi_yz
        )
    )

    return feq + (1.0 - 1.0 / tau) * fneq_reg


__all__ = [
    "collide_bgk3d",
    "collide_mrt3d",
    "collide_rlbm3d",
    "collide_trt3d",
    "stream3d",
    "stream3d_index_select",
    "correct_mass3d",
]
