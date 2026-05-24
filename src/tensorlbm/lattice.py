"""Lattice utilities for a minimal D2Q9 LBM."""

import torch

from .constants import D2Q9

# Precompute (cy, cx) shift pairs once at module load to avoid rebuilding per call.
_SHIFTS: list[tuple[int, int]] = [(int(cy), int(cx)) for cx, cy in D2Q9.c.tolist()]


def equilibrium(rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute D2Q9 equilibrium distribution.

    Args:
        rho: Density field of shape (...).
        u:   Velocity field of shape (..., 2). Must be on the same device as rho.

    Raises:
        ValueError: If rho and u are on different devices.
    """
    if rho.device != u.device:
        raise ValueError(
            f"rho and u must be on the same device (got {rho.device} and {u.device})"
        )
    c = D2Q9.c.to(device=rho.device, dtype=u.dtype)
    w = D2Q9.w.to(device=rho.device, dtype=u.dtype)

    cu = torch.einsum("qa,...a->...q", c, u)
    u2 = (u**2).sum(dim=-1, keepdim=True)
    return rho.unsqueeze(-1) * w * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)


def stream(f_post_collision: torch.Tensor) -> torch.Tensor:
    """Stream D2Q9 distributions with periodic boundaries.

    Expects a tensor of shape (ny, nx, 9) and rolls the two spatial
    dimensions (dim 0 = y, dim 1 = x) for each of the 9 directions.
    """
    streamed = torch.empty_like(f_post_collision)
    for i, (cy, cx) in enumerate(_SHIFTS):
        streamed[..., i] = torch.roll(
            f_post_collision[..., i], shifts=(cy, cx), dims=(-2, -1)
        )
    return streamed
