"""Compressible LBM module for high-Mach-number flow verification.

Provides:
- Rankine-Hugoniot exact shock relations
- Godunov solver for 1D shock verification
- D2Q16 lattice for compressible flow initialization
- NACA0012 airfoil simulation (subsonic/transonic)
- Sonic boom N-wave simulation (supersonic wedge)
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
from .naca0012 import (
    naca0012_thickness,
    naca0012_coordinates,
    simulate_naca0012,
    thin_airfoil_theory_C_L,
    thin_airfoil_theory_slope,
    ackeret_pressure_coefficient,
    ackeret_sonic_boom_dp,
    grid_convergence_study,
    GridConvergenceResult,
)
from .sonic_boom import (
    create_wedge_mask,
    simulate_sonic_boom,
    SonicBoomResult,
    oblique_shock_angle,
    oblique_shock_pressure_ratio,
)

__all__ = [
    # Solver
    "normal_shock_relations",
    "oblique_shock_relations",
    "GodunovSolver1D",
    "D2Q16Lattice",
    "NormalShockResult",
    # Cases
    "NormalShockConfig",
    "DoubleMachReflectionConfig",
    "ForwardStepConfig",
    "NORMAL_SHOCK_SCAN",
    # Verification
    "compute_shock_profile_error",
    "extract_shock_position",
    # NACA0012
    "naca0012_thickness",
    "naca0012_coordinates",
    "simulate_naca0012",
    "thin_airfoil_theory_C_L",
    "thin_airfoil_theory_slope",
    "ackeret_pressure_coefficient",
    "ackeret_sonic_boom_dp",
    "grid_convergence_study",
    "GridConvergenceResult",
    # Sonic boom
    "create_wedge_mask",
    "simulate_sonic_boom",
    "SonicBoomResult",
    "oblique_shock_angle",
    "oblique_shock_pressure_ratio",
]
