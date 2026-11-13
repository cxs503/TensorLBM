"""Boundary condition modules for TensorLBM."""

from .equilibrium_extrapolation import (
    EquilibriumExtrapolationSpec,
    apply_cavity_extrapolation,
    apply_channel_extrapolation,
    apply_equilibrium_extrapolation,
    oss_equilibrium_difference,
)
from .suction import SuctionWallBC, apply_suction_wall
from .solver2d import (
    equilibrium2d,
    macroscopic2d,
    stream2d,
    collide_bgk2d,
    compute_wall_shear_stress,
)
from .d2q9 import C2Q9, W2Q9, OPPOSITE_2Q9, CS2_2Q9
from .nebb import (
    NebbChannelConfig,
    NebbPoiseuilleChannel,
    analytic_poiseuille,
    collide_edm,
    collide_guo,
    force_corrected_velocity,
    guo_source,
    nebb_wall_plane,
    wall_density,
    wall_directions,
)

__all__ = [
    "EquilibriumExtrapolationSpec",
    "apply_cavity_extrapolation",
    "apply_channel_extrapolation",
    "apply_equilibrium_extrapolation",
    "oss_equilibrium_difference",
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
    "NebbChannelConfig",
    "NebbPoiseuilleChannel",
    "analytic_poiseuille",
    "collide_edm",
    "collide_guo",
    "force_corrected_velocity",
    "guo_source",
    "nebb_wall_plane",
    "wall_density",
    "wall_directions",
]
