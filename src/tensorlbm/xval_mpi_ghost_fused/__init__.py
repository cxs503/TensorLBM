"""XVAL: lbmpy/waLBerla direction-selective, kernel-fused MPI ghost exchange.

Anchor ``XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED``.

The OSS side of this cross-validation is the PackInfo code-generation path of
``lssfau/lbmpy`` and its waLBerla binding ``lbmpy_walberla`` (both read from the
local checkout of the fork -- see ``predeclared.json:oss_reference``).  The
transferable idea is:

* an MPI ghost exchange is described by a *PackInfo* per communication
  direction, and the pack/unpack kernels are **generated from the LBM access
  pattern** -- so only the populations whose stencil direction crosses that
  face are transferred (D3Q19 x-face: 5 of 19, waLBerla's own docstring for
  ``packDirection``: "if dir == N, the pdfs streaming in directions NW, N, NE
  are packed");
* the direction-sized buffers are sized by ``getSizeOfPackedCellInterval`` and
  pack/unpack run as one fused pass over the face interval, not one pass per
  population;
* in-place streaming patterns get separate ``Even``/``Odd`` PackInfos.

TensorLBM's shipped ``D3Q19GlooTransport`` instead ships the full 19-population
boundary planes every step.  This package implements both paths behind one
transport so the only changed variable is the packed population set.
"""

from .packinfo import (
    FacePackInfo,
    crossing_populations,
    extend_dir,
    face_index,
    face_packinfos,
)
from .kernels import pack_face, unpack_face, packed_elements
from .transport import SlabExchange, replicate_shipped_exchange
from .case import TGV, collide_bgk, equilibrium3d, macroscopics, weights_for

__all__ = [
    "FacePackInfo",
    "TGV",
    "SlabExchange",
    "collide_bgk",
    "crossing_populations",
    "equilibrium3d",
    "extend_dir",
    "face_packinfos",
    "weights_for",
    "face_index",
    "macroscopics",
    "pack_face",
    "packed_elements",
    "replicate_shipped_exchange",
    "unpack_face",
]