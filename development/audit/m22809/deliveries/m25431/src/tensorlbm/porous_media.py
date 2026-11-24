"""Porous-media LBM helpers (compressed fluid-node neighbour table).

This module re-exports the compressed fluid-node neighbour table implementation
from :mod:`tensorlbm.porous.fluid_node_table` for backward compatibility and
convenience.

See :mod:`tensorlbm.porous` for the full API.
"""

from .porous.fluid_node_table import (
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
