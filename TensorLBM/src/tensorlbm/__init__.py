"""TensorLBM: Lattice Boltzmann Method solver library.

Provides:
- D3Q19 MRT collision with Smagorinsky SGS model for LES
- D2Q9 single-phase flow with regularized BGK (RLBM) for high-Re stability
- Shan-Chen multiphase model (SCMP) with EDM forcing
- Wetting/diffuse-interface verification (three-anchor: contact angle, Ca scaling, Young-Laplace)
- Adjoint / differentiable LBM (BPTT reference + out-of-core checkpointed gradient)
"""

from .d3q19 import C, W, OPPOSITE, CS2
from .solver import (
    equilibrium3d,
    macroscopic3d,
    stream3d,
    collide_mrt,
    collide_smagorinsky_mrt3d,
    collide_rlbm_2d,
)
from .d2q9 import (
    equilibrium,
    macroscopic,
    stream,
    collide_bgk,
    CS2 as CS2_2D,
)
from . import adjoint

__all__ = [
    # 3D
    "C", "W", "OPPOSITE", "CS2",
    "equilibrium3d", "macroscopic3d", "stream3d",
    "collide_mrt", "collide_smagorinsky_mrt3d",
    # 2D
    "equilibrium", "macroscopic", "stream", "collide_bgk", "CS2_2D",
    "collide_rlbm_2d",
    # Adjoint / differentiable
    "adjoint",
]
