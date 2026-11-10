"""TensorLBM: Lattice Boltzmann Method solver library.

Provides D3Q19 MRT collision with Smagorinsky SGS model for LES.
"""

from .d3q19 import C, W, OPPOSITE, CS2
from .solver import (
    equilibrium3d,
    macroscopic3d,
    stream3d,
    collide_mrt,
    collide_smagorinsky_mrt3d,
)

__all__ = [
    "C", "W", "OPPOSITE", "CS2",
    "equilibrium3d", "macroscopic3d", "stream3d",
    "collide_mrt", "collide_smagorinsky_mrt3d",
]
