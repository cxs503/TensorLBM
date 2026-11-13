"""Fused pack/unpack kernels for one ghost-exchange face.

waLBerla generates a packing kernel per communication direction whose buffer
holds exactly the crossing populations of a cell interval
(``PackingKernels.tmpl.h``: ``packDirection`` / ``unpackDirection`` /
``getSizeOfPackedCellInterval``).  Here the same contract is expressed with two
tensor ops: a single ``index_select`` over the crossing channels of the face
plane (pack) and a single fused index write into the ghost plane (unpack).
Both are one pass over the face, never a Python loop over populations -- that
fusion is the point of the OSS design, and the tests assert it is a lossless
permutation of exactly those entries.
"""

from __future__ import annotations

from typing import Sequence

import torch


def _selection(populations: Sequence[int], device: torch.device) -> torch.Tensor:
    if len(populations) == 0:
        raise ValueError("empty population set")
    return torch.as_tensor(tuple(populations), dtype=torch.long, device=device)


def pack_face(post: torch.Tensor, populations: Sequence[int], index: int) -> torch.Tensor:
    """Pack the crossing populations of owned plane ``index`` into a contiguous buffer.

    ``post`` has shape ``(Q, nz, ny, nx_local)``; the result has shape
    ``(len(populations), nz, ny)`` -- the layout waLBerla sends.
    """
    if post.ndim != 4:
        raise ValueError("post must be (Q, nz, ny, nx_local)")
    return post.index_select(0, _selection(populations, post.device))[..., index].contiguous()


def unpack_face(
    buffer: torch.Tensor,
    padded: torch.Tensor,
    populations: Sequence[int],
    ghost_index: int,
) -> None:
    """Unpack a packed buffer straight into one ghost plane (one fused write)."""
    if buffer.shape != (len(populations),) + padded.shape[1:3]:
        raise ValueError(
            f"buffer shape {tuple(buffer.shape)} does not match "
            f"({len(populations)},) + {tuple(padded.shape[1:3])}"
        )
    sel = _selection(populations, padded.device)
    if tuple(padded[sel, :, :, ghost_index].shape) != tuple(buffer.shape):
        raise ValueError(
            f"ghost slice {tuple(padded[sel, :, :, ghost_index].shape)} does not match "
            f"buffer {tuple(buffer.shape)}"
        )
    padded[sel, :, :, ghost_index] = buffer


def packed_elements(populations: Sequence[int], nz: int, ny: int) -> int:
    return len(populations) * nz * ny