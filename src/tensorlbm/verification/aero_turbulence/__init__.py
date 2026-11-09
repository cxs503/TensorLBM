"""Aero-turbulence dual verification surface for TensorLBM.

This package provides verification cases for:
1. External aerodynamics: NACA0012 airfoil (lift/drag polars)
2. Turbulence modeling: LES channel flow (Smagorinsky + van Driest)

References:
- Abbott & von Doenhoff (1945): NACA0012 experimental data
- Mos, Kim & Mansour (1999): DNS of turbulent channel at Re_tau=180
"""

from .grid_convergence import GCIResult, compute_gci, grid_convergence_study
from .les_channel_verification import (
    LESChannelConfig,
    ChannelStatsAccumulator,
    compute_energy_spectrum,
    compute_wall_distance_2d,
    fit_inertial_subrange,
    run_les_channel,
    van_driest_damping,
)
from .naca0012_verification import (
    AirfoilSimConfig,
    build_naca0012_mask,
    compute_lift_slope,
    naca0012_thickness,
    run_airfoil_sim,
    run_polar_sweep,
)
from .reference_data import (
    MKM1999Statistics,
    NACA0012_ABBOTT_CL,
    NACA0012_ABBOTT_CD,
    abbott_cd_interp,
    abbott_cl_interp,
    kolmogorov_spectrum,
    mkm1999_mean_velocity_profile,
    mkm1999_reynolds_stress_profile,
    mkm1999_tke_profile,
    thin_airfoil_cl,
    thin_airfoil_cl_slope,
)
from .run_verification import (
    run_airfoil_verification,
    run_full_verification,
    run_grid_convergence_study,
    run_les_verification,
)

__all__ = [
    # Reference data
    "NACA0012_ABBOTT_CL",
    "NACA0012_ABBOTT_CD",
    "abbott_cl_interp",
    "abbott_cd_interp",
    "thin_airfoil_cl",
    "thin_airfoil_cl_slope",
    "MKM1999Statistics",
    "mkm1999_mean_velocity_profile",
    "mkm1999_reynolds_stress_profile",
    "mkm1999_tke_profile",
    "kolmogorov_spectrum",
    # Airfoil
    "naca0012_thickness",
    "build_naca0012_mask",
    "AirfoilSimConfig",
    "run_airfoil_sim",
    "run_polar_sweep",
    "compute_lift_slope",
    # LES channel
    "van_driest_damping",
    "compute_wall_distance_2d",
    "LESChannelConfig",
    "ChannelStatsAccumulator",
    "run_les_channel",
    "compute_energy_spectrum",
    "fit_inertial_subrange",
    # Grid convergence
    "GCIResult",
    "compute_gci",
    "grid_convergence_study",
    # Runners
    "run_airfoil_verification",
    "run_les_verification",
    "run_grid_convergence_study",
    "run_full_verification",
]
