"""Boundary condition modules for TensorLBM."""

from .suction import SuctionWallBC, apply_suction_wall
from .solver2d import (
    equilibrium2d,
    macroscopic2d,
    stream2d,
    collide_bgk2d,
    compute_wall_shear_stress,
)
from .d2q9 import C2Q9, W2Q9, OPPOSITE_2Q9, CS2_2Q9

__all__ = [
    "SuctionWallBC",
    "apply_suction_wall",
    "equilibrium2d",
    "macroscopic2d",
    "stream2d",
    "collide_bgk2d",
    "compute_wall_shear_stress",
    "C2Q9",
    "W2Q9",
    "OPPOSITE_2Q9",
    "CS2_2Q9",
]
