"""D3Q27 lattice constants and equilibrium distribution.

The D3Q27 lattice has 27 velocity directions covering all combinations of
(cx, cy, cz) ∈ {−1, 0, 1}³. Compared to D3Q19 it includes the 8 corner
directions (|c| = √3) and therefore achieves 4th-order isotropy, which can
reduce numerical artefacts in flows with strong corner-region gradients
(e.g. flows past bluff bodies or in confined geometries).

Lattice weights (Qian, 1992):

- Rest (0,0,0):           w = 8/27
- Face-centre (|c|=1):    w = 2/27  (×6)
- Edge-centre (|c|=√2):   w = 1/54  (×12)
- Corner     (|c|=√3):    w = 1/216 (×8)
"""
from __future__ import annotations

import functools
from typing import Any

import torch

# Cache for streaming index tensors keyed by (nz, ny, nx, device_type, device_index)
_stream27_cache: dict[
    tuple[Any, ...],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}

_C_DATA = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [-1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, -1, 1],
    [0, 1, -1],
    [0, -1, -1],
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [1, 1, -1],
    [-1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1],
]

C = torch.tensor(_C_DATA, dtype=torch.int64)

_w_rest = 8.0 / 27.0
_w_face = 2.0 / 27.0
_w_edge = 1.0 / 54.0
_w_corner = 1.0 / 216.0

_W_DATA = [_w_rest] + [_w_face] * 6 + [_w_edge] * 12 + [_w_corner] * 8
W = torch.tensor(_W_DATA, dtype=torch.float32)


def _build_opposite() -> torch.Tensor:
    c_list = [tuple(row) for row in _C_DATA]
    opp = []
    for cx, cy, cz in c_list:
        target = (-cx, -cy, -cz)
        opp.append(c_list.index(target))
    return torch.tensor(opp, dtype=torch.int64)


OPPOSITE = _build_opposite()


@functools.cache
def _c_on(device: torch.device) -> torch.Tensor:
    return C.to(device)


@functools.cache
def _w_on(device: torch.device) -> torch.Tensor:
    return W.to(device)


def equilibrium27(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute D3Q27 Maxwell-Boltzmann equilibrium distribution.

    Args:
        rho: Density field, shape ``(nz, ny, nx)``.
        ux: x-velocity field, shape ``(nz, ny, nx)``.
        uy: y-velocity field, shape ``(nz, ny, nx)``.
        uz: z-velocity field, shape ``(nz, ny, nx)``.
        device: Target device (inferred from *rho* if *None*).

    Returns:
        Equilibrium distribution of shape ``(27, nz, ny, nx)``.
    """
    if device is None:
        device = rho.device
    c = _c_on(device).float()
    w = _w_on(device).view(27, 1, 1, 1)

    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)

    u_sq = ux * ux + uy * uy + uz * uz
    cu = cx * ux + cy * uy + cz * uz
    return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))


def macroscopic27(
    f: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover (rho, ux, uy, uz) from D3Q27 distributions.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        device: Target device (inferred from *f* if *None*).

    Returns:
        Tuple ``(rho, ux, uy, uz)`` of shape ``(nz, ny, nx)`` each.
    """
    if device is None:
        device = f.device
    c = _c_on(device).float()
    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * cx).sum(dim=0) / rho_safe
    uy = (f * cy).sum(dim=0) / rho_safe
    uz = (f * cz).sum(dim=0) / rho_safe
    return rho, ux, uy, uz


def collide_bgk27(f: torch.Tensor, tau: float) -> torch.Tensor:
    """D3Q27 single-relaxation-time BGK collision.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Relaxation time τ > 0.5.

    Returns:
        Post-collision distribution of the same shape.
    """
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    return f - (f - feq) / tau


def _build_d3q27_mrt_matrices() -> tuple[list[list[float]], list[list[float]]]:
    """Compute and return (M, M_inv) for the D3Q27 MRT transformation.

    Constructs the 27×27 transformation matrix using the Gram–Schmidt
    orthogonalised polynomial basis over the D3Q27 velocity set
    ``{cx, cy, cz} ∈ {−1, 0, 1}³``.  The basis polynomials follow the
    Qian/d'Humières moment hierarchy:

    * Row 0:  1                               (mass)
    * Row 1:  cx                              (x-momentum)
    * Row 2:  cy                              (y-momentum)
    * Row 3:  cz                              (z-momentum)
    * Row 4:  cx² + cy² + cz²                (energy, e)
    * Row 5:  cx²                             (normal stress xx; raw)
    * Row 6:  cy² − cz²                       (normal stress yy–zz; raw)
    * Row 7:  cx·cy                           (shear stress xy)
    * Row 8:  cx·cz                           (shear stress xz)
    * Row 9:  cy·cz                           (shear stress yz)
    * Rows 10–26: higher-order moments via Gram–Schmidt orthogonalisation.

    The resulting matrix is verified to be full rank (rank 27).
    """
    import numpy as np

    c_np = np.array(_C_DATA, dtype=np.float64)  # (27, 3)
    cx, cy, cz = c_np[:, 0], c_np[:, 1], c_np[:, 2]
    e2 = cx**2 + cy**2 + cz**2

    # Define raw moment vectors (length 27 each) in physical significance order
    raw_rows: list[np.ndarray] = [
        np.ones(27),           # 0: mass
        cx,                    # 1: jx
        cy,                    # 2: jy
        cz,                    # 3: jz
        e2,                    # 4: energy e = |c|^2
        3.0 * cx**2 - e2,      # 5: Nxx  (normal stress xx)
        cy**2 - cz**2,         # 6: Nyy  (normal stress yy-zz)
        cx * cy,               # 7: Pxy  (shear stress xy)
        cx * cz,               # 8: Pxz  (shear stress xz)
        cy * cz,               # 9: Pyz  (shear stress yz)
        # 3rd-order raw moments
        cx * e2,               # 10: qx
        cy * e2,               # 11: qy
        cz * e2,               # 12: qz
        cx**2 * cy,            # 13
        cx**2 * cz,            # 14
        cy**2 * cx,            # 15
        cy**2 * cz,            # 16
        cz**2 * cx,            # 17
        cz**2 * cy,            # 18
        # 4th-order raw moments
        e2**2,                 # 19
        cx**2 * e2,            # 20
        cy**2 * e2,            # 21
        cz**2 * e2,            # 22
        cx**2 * cy**2,         # 23
        cx**2 * cz**2,         # 24
        cy**2 * cz**2,         # 25
        cx * cy * cz,          # 26
    ]

    # Gram–Schmidt orthogonalisation to ensure full rank
    orth_rows: list[np.ndarray] = []
    for row in raw_rows:
        v = row.copy()
        for prev in orth_rows:
            v = v - (np.dot(v, prev) / np.dot(prev, prev)) * prev
        norm = np.sqrt(np.dot(v, v))
        if norm < 1e-14:
            # Row is linearly dependent — replace with an orthogonal complement
            # by searching for a standard-basis vector not yet represented
            for i in range(27):
                e_i = np.zeros(27)
                e_i[i] = 1.0
                u = e_i.copy()
                for prev in orth_rows:
                    u = u - (np.dot(u, prev) / np.dot(prev, prev)) * prev
                if np.sqrt(np.dot(u, u)) > 1e-10:
                    v = u
                    break
        orth_rows.append(v)

    matrix = np.array(orth_rows, dtype=np.float64)
    assert np.linalg.matrix_rank(matrix) == 27, "D3Q27 MRT matrix is rank-deficient"
    matrix_inv = np.linalg.inv(matrix)
    return matrix.tolist(), matrix_inv.tolist()


_M_D3Q27_DATA, _M_D3Q27_INV_DATA = _build_d3q27_mrt_matrices()


@functools.cache
def _get_d3q27_mrt_matrices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.tensor(_M_D3Q27_DATA, dtype=torch.float32, device=device)
    matrix_inv = torch.tensor(_M_D3Q27_INV_DATA, dtype=torch.float32, device=device)
    return matrix, matrix_inv


def collide_mrt27(
    f: torch.Tensor,
    tau: float,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """D3Q27 multi-relaxation-time (MRT) collision step.

    Shear viscosity is controlled by *tau*: ν = (τ − ½)/3.  Independent
    relaxation rates for non-hydrodynamic moments improve stability at high
    Reynolds numbers.

    Relaxation rates:
        * Rows 0–3  (mass, momenta):  0 (conserved)
        * Row  4    (energy e):       s_e
        * Rows 5–9  (stress modes):   1/tau
        * Rows 10–18 (3rd-order):     s_q
        * Rows 19–26 (4th-order+):    s_pi (defaults to s_e)

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Relaxation time for shear stress (τ > ½).
        s_e: Relaxation rate for the energy moment.
        s_eps: Relaxation rate for the energy-square moment (row 19).
        s_q: Relaxation rate for 3rd-order heat-flux moments (rows 10–18).
        s_pi: Relaxation rate for 4th-order moments (rows 20–26);
              defaults to *s_e* when *None*.

    Returns:
        Updated distribution tensor of the same shape.
    """
    if s_pi is None:
        s_pi = s_e

    device = f.device
    matrix, matrix_inv = _get_d3q27_mrt_matrices(device)

    s_nu = 1.0 / tau
    s_vec = torch.tensor(
        [
            0.0,   # 0  mass
            0.0,   # 1  jx
            0.0,   # 2  jy
            0.0,   # 3  jz
            s_e,   # 4  energy
            s_nu,  # 5  Nxx
            s_nu,  # 6  Nyy
            s_nu,  # 7  Pxy
            s_nu,  # 8  Pxz
            s_nu,  # 9  Pyz
            s_q,   # 10 qx
            s_q,   # 11 qy
            s_q,   # 12 qz
            s_q,   # 13
            s_q,   # 14
            s_q,   # 15
            s_q,   # 16
            s_q,   # 17
            s_q,   # 18
            s_eps, # 19 e²
            s_pi,  # 20
            s_pi,  # 21
            s_pi,  # 22
            s_pi,  # 23
            s_pi,  # 24
            s_pi,  # 25
            s_pi,  # 26
        ],
        dtype=f.dtype,
        device=device,
    )

    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(27, -1)
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    feq_flat = feq.reshape(27, -1)

    moments = matrix @ f_flat
    moments_eq = matrix @ feq_flat
    moments_star = moments - s_vec.unsqueeze(1) * (moments - moments_eq)
    return (matrix_inv @ moments_star).reshape(27, nz, ny, nx)


def correct_mass27(f: torch.Tensor, target_mass: float) -> torch.Tensor:
    """Redistribute mass uniformly to correct global mass drift (D3Q27).

    Rescales the entire distribution tensor so that the sum of all
    populations equals *target_mass*. This corrects slow mass drift
    accumulated by inexact boundary conditions over many time steps.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        target_mass: Desired total mass (sum of all populations).

    Returns:
        Rescaled distribution tensor of the same shape.
    """
    current = f.sum()
    if current.abs() < 1e-30:
        return f
    return f * (target_mass / current)


def stream27(f: torch.Tensor) -> torch.Tensor:
    """Periodic gather streaming for D3Q27.

    Index tensors are cached per (shape, device) to avoid re-allocation on
    every call.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.

    Returns:
        Streamed distribution of the same shape.
    """
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    device = f.device
    c = _c_on(device)

    cache_key = (nz, ny, nx, device.type, device.index)
    if cache_key not in _stream27_cache:
        z_src = (torch.arange(nz, device=device).unsqueeze(0) - c[:, 2].unsqueeze(1)) % nz
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx
        q_idx = torch.arange(27, device=device).view(27, 1, 1, 1).expand(27, nz, ny, nx)
        z_idx = z_src.unsqueeze(2).unsqueeze(3).expand(27, nz, ny, nx)
        y_idx = y_src.unsqueeze(1).unsqueeze(3).expand(27, nz, ny, nx)
        x_idx = x_src.unsqueeze(1).unsqueeze(2).expand(27, nz, ny, nx)
        _stream27_cache[cache_key] = (q_idx, z_idx, y_idx, x_idx)

    q_idx, z_idx, y_idx, x_idx = _stream27_cache[cache_key]
    return f[q_idx, z_idx, y_idx, x_idx]


__all__ = [
    "C",
    "W",
    "OPPOSITE",
    "equilibrium27",
    "macroscopic27",
    "collide_bgk27",
    "collide_mrt27",
    "stream27",
    "correct_mass27",
    "_get_d3q27_mrt_matrices",
]
