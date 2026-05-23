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

import torch

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


def stream27(f: torch.Tensor) -> torch.Tensor:
    """Periodic gather streaming for D3Q27.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.

    Returns:
        Streamed distribution of the same shape.
    """
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    device = f.device
    c = _c_on(device)

    z_src = (torch.arange(nz, device=device).unsqueeze(0) - c[:, 2].unsqueeze(1)) % nz
    y_src = (torch.arange(ny, device=device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
    x_src = (torch.arange(nx, device=device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx

    q_idx = torch.arange(27, device=device).view(27, 1, 1, 1).expand(27, nz, ny, nx)
    z_idx = z_src.unsqueeze(2).unsqueeze(3).expand(27, nz, ny, nx)
    y_idx = y_src.unsqueeze(1).unsqueeze(3).expand(27, nz, ny, nx)
    x_idx = x_src.unsqueeze(1).unsqueeze(2).expand(27, nz, ny, nx)

    return f[q_idx, z_idx, y_idx, x_idx]


__all__ = [
    "C",
    "W",
    "OPPOSITE",
    "equilibrium27",
    "macroscopic27",
    "collide_bgk27",
    "stream27",
]
