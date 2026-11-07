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


@functools.cache
def _c_on(device: torch.device) -> torch.Tensor:
    return C.to(device)


@functools.cache
def _w_on(device: torch.device) -> torch.Tensor:
    return W.to(device)


# Exact double-precision D2Q9 weights, same order as ``W``.
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
def _w_for(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """D2Q9 weights built from exact double literals for ``dtype``.

    ``W`` is stored as ``float32``; ``W.to(torch.float64)`` therefore keeps the
    ``float32`` rounding (its sum is ``1 + 7.45e-9``, not ``1``).  Used through
    :func:`equilibrium` that rounding biases the mass of every collision step by
    ``+7.45e-9 / tau`` and accumulates into a linear density ramp, which is
    fatal for long low-Mach steady-state measurements (measured 4.8e-10 per
    step at ``tau = 15.5``, i.e. 5e-5 after 1e5 steps).  For ``float32`` the
    returned table is bit-identical to ``W``.
    """
    return torch.tensor(W_EXACT, dtype=dtype, device=device)


def equilibrium(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute D2Q9 equilibrium distribution f_eq for rho and velocity fields."""
    if not (rho.shape == ux.shape == uy.shape):
        raise ValueError(
            "rho, ux, and uy shapes must match: "
            f"rho={tuple(rho.shape)}, ux={tuple(ux.shape)}, uy={tuple(uy.shape)}"
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
    """Recover rho, ux, uy from particle distributions."""
    if device is None:
        device = f.device
    c = _c_on(device)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * c[:, 0].view(9, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0) / rho_safe
    return rho, ux, uy
