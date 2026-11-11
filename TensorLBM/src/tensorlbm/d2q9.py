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


@functools.cache
def _c_on(device: torch.device) -> torch.Tensor:
    return C.to(device)


@functools.cache
def _w_for(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """D2Q9 weights built from exact double literals for ``dtype``."""
    return torch.tensor(W_EXACT, dtype=dtype, device=device)


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
