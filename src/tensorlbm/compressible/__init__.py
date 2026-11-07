"""Compressible LBM module for high-Mach-number flow verification.

Provides:
- Rankine-Hugoniot exact shock relations
- Godunov solver for 1D shock verification
- D2Q16 lattice for compressible flow initialization
- Verification utilities
- Case configurations
"""

from .solver import (
    normal_shock_relations,
    oblique_shock_relations,
    GodunovSolver1D,
    D2Q16Lattice,
    NormalShockResult,
)
from .cases import (
    NormalShockConfig,
    DoubleMachReflectionConfig,
    ForwardStepConfig,
    NORMAL_SHOCK_SCAN,
)
from .verification import (
    compute_shock_profile_error,
    extract_shock_position,
)

__all__ = [
    "normal_shock_relations",
    "oblique_shock_relations",
    "GodunovSolver1D",
    "D2Q16Lattice",
    "NormalShockResult",
    "NormalShockConfig",
    "DoubleMachReflectionConfig",
    "ForwardStepConfig",
    "NORMAL_SHOCK_SCAN",
    "compute_shock_profile_error",
    "extract_shock_position",
]
