"""Bounded-memory execution for strictly cell-local collision operators."""
from __future__ import annotations

from collections.abc import Callable

import torch

CellLocalCollision = Callable[[torch.Tensor], torch.Tensor]


def collide_in_z_chunks(
    populations: torch.Tensor,
    collision: CellLocalCollision,
    *,
    chunk_cells: int,
) -> torch.Tensor:
    """Apply a cell-local collision in z slabs with a bounded working set.

    This is valid only for collision operators whose result at a cell depends
    on populations at that same cell.  Streaming, gradient SGS closures and
    any stencil-based regularisation must remain outside this helper.
    """
    if (
        not isinstance(populations, torch.Tensor)
        or populations.ndim != 4
        or populations.shape[0] not in (19, 27)
    ):
        raise ValueError("populations must have shape (19|27,nz,ny,nx)")
    if isinstance(chunk_cells, bool) or chunk_cells < 1:
        raise ValueError("chunk_cells must be a positive integer")
    _, nz, ny, nx = populations.shape
    planes_per_chunk = max(1, chunk_cells // (ny * nx))
    if planes_per_chunk >= nz:
        result = collision(populations)
        if result.shape != populations.shape or result.device != populations.device:
            raise ValueError("collision must preserve population shape and device")
        return result

    output = torch.empty_like(populations)
    for start in range(0, nz, planes_per_chunk):
        stop = min(start + planes_per_chunk, nz)
        slab = populations[:, start:stop]
        collided = collision(slab)
        if collided.shape != slab.shape or collided.device != populations.device:
            raise ValueError("collision must preserve slab shape and device")
        output[:, start:stop] = collided
    return output


__all__ = ["CellLocalCollision", "collide_in_z_chunks"]
