"""Lattice utilities for a minimal D2Q9 LBM."""

import torch

from .constants import D2Q9


def equilibrium(rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute D2Q9 equilibrium distribution."""
    c = D2Q9.c.to(device=rho.device, dtype=u.dtype)
    w = D2Q9.w.to(device=rho.device, dtype=u.dtype)

    cu = torch.einsum("qa,...a->...q", c, u)
    u2 = (u**2).sum(dim=-1, keepdim=True)
    return rho.unsqueeze(-1) * w * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)


def stream(f_post_collision: torch.Tensor) -> torch.Tensor:
    """Stream D2Q9 distributions with periodic boundaries."""
    streamed = torch.empty_like(f_post_collision)
    for i, (cx, cy) in enumerate(D2Q9.c.tolist()):
        streamed[..., i] = torch.roll(f_post_collision[..., i], shifts=(cy, cx), dims=(0, 1))
    return streamed
