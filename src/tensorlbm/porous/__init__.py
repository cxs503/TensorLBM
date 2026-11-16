"""Porous-media data layouts and kernels for TensorLBM.

``fluid_node_table`` implements the *compressed fluid-node neighbour table*
layout of porous-media LBM codes (fluid-node compaction + flat per-fluid-node
stencil with a solid sentinel) together with the dense full-lattice reference
of the same physics, so that a representation change can be shown to be
physics-neutral rather than asserted.

The list-based sparse flavour (waLBerla/lbmpy style, one gather index per
(fluid cell, direction)) lives in :mod:`tensorlbm.sparse_list_aos`; the
volume-averaged Kozeny-Carman/Ergun ladder lives in
:mod:`tensorlbm.porous_ladder`.

Transplant source (cited, not copied): PorousMediaSimulation/openLBMPM
``ShanChen2D/ShanChenD2Q9.py:589``, ``ShanChen2D/OptimizedD2Q9GPU.py:22``,
``ShanChen2D/OptimizedD2Q9GPU.py:452``.
See ``docs/xval_compressed_fluid_node_neighbour_table.md`` for the difference
list, the pre-declared verification ladder and its measured numbers.
"""

from .fluid_node_table import (
    DIRS,
    FluidNodeTable,
    build_fluid_node_table,
    collide_with_guo_forcing,
    compressed_step,
    dense_step,
    dense_stream,
    intrinsic_permeability,
    macroscopic_with_forcing,
    permeability,
)

__all__ = [
    "DIRS",
    "FluidNodeTable",
    "build_fluid_node_table",
    "collide_with_guo_forcing",
    "compressed_step",
    "dense_step",
    "dense_stream",
    "intrinsic_permeability",
    "macroscopic_with_forcing",
    "permeability",
]
