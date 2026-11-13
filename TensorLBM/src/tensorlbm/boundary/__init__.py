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
from .inamuro_wetnode import (
    compute_boundary_distance,
    compute_boundary_distance_from_levelset,
    apply_inamuro_wetnode,
    apply_inamuro_all_directions,
    create_curved_boundary_masks,
)

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
    "compute_boundary_distance",
    "compute_boundary_distance_from_levelset",
    "apply_inamuro_wetnode",
    "apply_inamuro_all_directions",
    "create_curved_boundary_masks",
]
