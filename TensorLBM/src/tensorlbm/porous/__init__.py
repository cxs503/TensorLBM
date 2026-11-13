"""Porous-media helpers for TensorLBM.

``fluid_node_table`` implements the *compressed fluid-node neighbour table*
data layout used by porous-media LBM codes (fluid-node compaction + a flat
per-fluid-node stencil array with a solid sentinel), plus a dense full-lattice
reference implementation of the very same physics so that the layout can be
cross-validated against the in-repo dense path.

Transplant source (cited, not copied): PorousMediaSimulation/openLBMPM
  ShanChen2D/ShanChenD2Q9.py:589  optimizeFluidArray()
  ShanChen2D/OptimizedD2Q9GPU.py:22 fillNeighboringNodes()
  ShanChen2D/OptimizedD2Q9GPU.py:452 calStreaming1GPU()
See docs/xval_compressed_fluid_node_neighbour_table.md for the full
difference list and the rationale of the transplantation choices.
"""

from .fluid_node_table import (
    DIRS,
    FluidNodeTable,
    build_fluid_node_table,
    collide_with_guo_forcing,
    dense_step,
    dense_stream,
    intrinsic_permeability,
    macroscopic_with_forcing,
    permeability,
    compressed_step,
)

__all__ = [
    "DIRS",
    "FluidNodeTable",
    "build_fluid_node_table",
    "collide_with_guo_forcing",
    "compressed_step",
    "dense_step",
    "intrinsic_permeability",
    "dense_stream",
    "macroscopic_with_forcing",
    "permeability",
]