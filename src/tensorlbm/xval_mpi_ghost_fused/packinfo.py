"""PackInfo for a D3Q19 x-slab MPI ghost exchange.

Semantics are taken from the OSS reference, not transcribed by hand:

``lbmpy/advanced_streaming/communication.py::_extend_dir`` (installed lbmpy 2.0,
line 233) expands a communication direction into the set of streaming
directions that have to be exchanged across that face; ``packinfo.py::
generate_lb_pack_info`` intersects that set with the method stencil and adds the
corresponding write accesses to ``spec[(comm_dir,)]``.  :func:`extend_dir` below
reproduces that function so the derivation can be machine-compared against the
OSS function itself (``tests/test_xval_mpi_ghost_fused.py``); nothing here is a
hand-written direction table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch

from ..d3q19 import C

Direction = Tuple[int, int, int]


def extend_dir(direction: Sequence[int]) -> Iterable[Direction]:
    """lbmpy ``_extend_dir``: fix non-zero components, expand zero components."""
    if len(direction) == 0:
        yield tuple()
        return
    head, rest = direction[0], direction[1:]
    if head == 0:
        for d in (-1, 0, 1):
            for tail in extend_dir(rest):
                yield (d,) + tail
    else:
        for tail in extend_dir(rest):
            yield (head,) + tail


def stencil_directions(stencil: torch.Tensor | None = None) -> Tuple[Direction, ...]:
    table = C if stencil is None else stencil
    return tuple(tuple(int(v) for v in row) for row in table.tolist())


def crossing_populations(
    comm_dir: Sequence[int], stencil: torch.Tensor | None = None
) -> Tuple[int, ...]:
    """Stencil indices exchanged across the face normal to ``comm_dir``.

    Mirrors ``set(_extend_dir(comm_dir)) & set(stencil)`` from
    ``lbmpy_walberla/packinfo.py``; ordering is the stencil order so the packed
    buffer layout is reproducible.
    """
    wanted = set(extend_dir(tuple(int(v) for v in comm_dir)))
    return tuple(
        index
        for index, direction in enumerate(stencil_directions(stencil))
        if direction in wanted
    )


def face_index(side: str, width: int) -> int:
    """Owned-cell index adjacent to ``side`` (waLBerla's 'slice before ghost layer')."""
    if side == "left":
        return 0
    if side == "right":
        return width - 1
    raise ValueError(f"unknown face {side!r}")


@dataclass(frozen=True)
class FacePackInfo:
    """The populations crossing one face, and where that face lives.

    ``pack_populations`` are read from the owned plane adjacent to the peer and
    written into the peer's ghost plane; ``ghost_populations`` are the
    populations this rank expects in its own ghost plane from that side.  For a
    periodic x-ring in a pull stream both sides are the same set, but they are
    kept explicit because a non-periodic face can insert an asymmetric halo.
    """

    side: str
    comm_dir: Direction
    populations: Tuple[int, ...]
    ghost_populations: Tuple[int, ...]

    @property
    def n_populations(self) -> int:
        return len(self.populations)

    def packed_elements(self, nz: int, ny: int) -> int:
        """waLBerla ``getSizeOfPackedCellInterval`` for a one-cell-thick face."""
        return self.n_populations * nz * ny

    def packed_bytes(self, nz: int, ny: int, itemsize: int) -> int:
        return self.packed_elements(nz, ny) * itemsize


def face_packinfos(
    selective: bool = True, stencil: torch.Tensor | None = None
) -> Tuple[FacePackInfo, FacePackInfo]:
    """(left, right) PackInfos for a periodic x-ring.

    A population with ``cx = +1`` travels to the right, so it is packed from the
    owned right plane and unpacked into the left ghost; ``cx = -1`` is the
    mirror image.  ``selective=False`` reproduces the shipped full-19-plane
    behaviour (the control arm).
    """
    directions = stencil_directions(stencil)
    if selective:
        to_right = tuple(i for i, d in enumerate(directions) if d[0] == 1)
        to_left = tuple(i for i, d in enumerate(directions) if d[0] == -1)
        if not to_right or not to_left:
            raise ValueError("stencil has no x-crossing populations")
    else:
        to_right = tuple(range(len(directions)))
        to_left = tuple(range(len(directions)))
    left = FacePackInfo("left", (-1, 0, 0), to_left, to_right)
    right = FacePackInfo("right", (1, 0, 0), to_right, to_left)
    return left, right