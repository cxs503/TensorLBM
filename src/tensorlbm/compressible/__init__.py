"""Compressible LBM module for high-Mach-number flow verification.

Provides:
- Rankine-Hugoniot exact shock relations
- Godunov solver for 1D shock verification
- D2Q16 lattice for compressible flow initialization
- NACA0012 airfoil geometry and C-mesh generation
- Ackeret linearized supersonic theory
- Isentropic flow relations
- 2D compressible LBM solver (DDF method)
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
from .airfoil import (
    naca0012_thickness,
    naca0012_coordinates,
    naca0012_surface_points,
    thin_airfoil_theory,
    prandtl_glauert_correction,
    CMeshConfig,
    generate_c_mesh,
    compute_surface_forces,
    AGARD_NACA0012_REFERENCE,
)
from .ackeret import (
    ackeret_pressure_coefficient,
    ackeret_pressure_jump,
    ackeret_lift_coefficient,
    ackeret_drag_coefficient,
    n_wave_signature,
    compare_ackeret_vs_oblique_shock,
    NWaveSignature,
)
from .isentropic import (
    isentropic_p_p0,
    isentropic_rho_rho0,
    isentropic_T_T0,
    isentropic_A_Astar,
    prandtl_meyer_angle,
    prandtl_meyer_from_nu,
    verify_isentropic_relations,
    critical_conditions,
    max_turning_angle,
    IsentropicVerification,
)
from .compressible_lbm import (
    CompressibleLBMConfig,
    CompressibleDDFSolver,
    run_flat_plate_shock,
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
    # Airfoil
    "naca0012_thickness",
    "naca0012_coordinates",
    "naca0012_surface_points",
    "thin_airfoil_theory",
    "prandtl_glauert_correction",
    "CMeshConfig",
    "generate_c_mesh",
    "compute_surface_forces",
    "AGARD_NACA0012_REFERENCE",
    # Ackeret
    "ackeret_pressure_coefficient",
    "ackeret_pressure_jump",
    "ackeret_lift_coefficient",
    "ackeret_drag_coefficient",
    "n_wave_signature",
    "compare_ackeret_vs_oblique_shock",
    "NWaveSignature",
    # Isentropic
    "isentropic_p_p0",
    "isentropic_rho_rho0",
    "isentropic_T_T0",
    "isentropic_A_Astar",
    "prandtl_meyer_angle",
    "prandtl_meyer_from_nu",
    "verify_isentropic_relations",
    "critical_conditions",
    "max_turning_angle",
    "IsentropicVerification",
    # Compressible LBM
    "CompressibleLBMConfig",
    "CompressibleDDFSolver",
    "run_flat_plate_shock",
]

# --- shock-capturing kinetic-flux solver (P153W1-N05) -----------------------
from .kinetic_flux import (  # noqa: E402,F401
    AnnulusMesh,
    KineticFluxSolver,
    NormalShock1D,
    ShockTube1D,
    SolverConfig,
    build_annulus_mesh,
    contour_loop,
    experimental_anchor_status,
    kfvs_flux,
    moving_shock_ratios,
    naca0012_half_thickness,
    normal_shock_ratios,
    radial_levels,
)

# --- NACA0012 LBM solver (P153W2-N04) ---------------------------------------
from .naca0012_lbm import (  # noqa: E402,F401
    NACA0012LBMConfig,
    NACA0012LBMSolver,
)
