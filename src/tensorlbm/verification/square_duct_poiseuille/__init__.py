"""3D square duct Poiseuille flow verification for TensorLBM.

Reference: Shah & London (1978), "Laminar Flow Forced Convection in Ducts",
Academic Press. Table/Chapter on rectangular ducts.
For square duct (aspect ratio = 1): f·Re = 56.91.
"""

from .solver import SquareDuctConfig, SquareDuctSolver, run_square_duct
from .analytical import shah_london_fRe, square_duct_velocity_profile

__all__ = [
    "SquareDuctConfig",
    "SquareDuctSolver",
    "run_square_duct",
    "shah_london_fRe",
    "square_duct_velocity_profile",
]
