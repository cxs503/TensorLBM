"""Nonperiodic D3Q19 fluid-to-solid link masks.

The mask is indexed at a FLUID node in direction c_i toward a SOLID node.
No periodic wrapping is permitted. q=0.5 is a deliberate halfway baseline,
NOT a curved-surface Bouzidi intersection fraction.
"""
from __future__ import annotations

import torch
from tensorlbm.d3q19 import C


def halfway_fluid_solid_links_3d(solid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (links, q) in (19,z,y,x) order for nonperiodic domains."""
    if solid.ndim != 3 or solid.dtype != torch.bool:
        raise ValueError("solid must be a boolean (z,y,x) tensor")
    shape = tuple(solid.shape)
    links = torch.zeros((19, *shape), dtype=torch.bool, device=solid.device)
    q = torch.full((19, *shape), 0.5, dtype=torch.float32, device=solid.device)
    for i in range(1, 19):
        cx, cy, cz = map(int, C[i].tolist())
        shifts = (cz, cy, cx)
        dst = tuple(slice(max(0, -d), min(n, n-d)) for n, d in zip(shape, shifts))
        src = tuple(slice(max(0, d), min(n, n+d)) for n, d in zip(shape, shifts))
        if any(s.start >= s.stop for s in dst):
            continue
        links[i][dst] = (~solid[dst]) & solid[src]
    return links, q
