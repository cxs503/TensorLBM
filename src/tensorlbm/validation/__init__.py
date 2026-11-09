"""Verification & validation utilities for the high-Reynolds wall-treatment ladder.

Sub-modules
-----------
``gci``
    Roache Grid Convergence Index (three grids, unequal refinement ratios,
    asymptotic-range check).
``wall_law``
    Law-of-the-wall references, the wall-function inverse problem, Blasius
    similarity solution and skin-friction correlations.
``cf_scaling``
    ``C_f = A Re^{-n}`` scale-effect calibration against ITTC-1957 / Dean.
``wall_budget``
    Cell budget of wall-resolved vs wall-modelled meshes (negative control).
``ladder``
    Machine-checkable verdict assembly.
``dns_reference_data``
    DNS reference data for turbulent channel flow (MKM1999/Moser2021).
``turbulent_channel_analysis``
    Analysis tools for turbulent channel statistics, spectra, wall-law fitting.

The package is deliberately free of solver imports (pure ``math``/``numpy``
free code except where noted) so it can be used by pytest, by post-processing
scripts and by the campaign report generator without a GPU backend.
"""

from __future__ import annotations

from .cf_scaling import CfFit, CfScalingReport, calibrate_cf_scaling, fit_power_law
from .dns_reference_data import (
    DNSReferenceData,
    channel_bulk_velocity,
    channel_centerline_velocity,
    friction_coefficient,
    get_dns_data,
    mkm1999_reynolds_stresses,
    musker_profile,
    reichardt_profile,
    VAN_DRIEST_B,
    VAN_DRIEST_KAPPA,
)
from .gci import GCIReport, gci, observed_order, richardson_extrapolation
from .ladder import (
    Criterion,
    assemble_ladder,
    criterion,
    relative_error_pct,
    verdict,
    write_payload,
)
from .turbulent_channel_analysis import (
    ChannelStatistics,
    ComparisonResult,
    WallLawFit,
    compare_with_dns,
    compute_channel_statistics,
    compute_energy_spectrum,
    fit_log_law,
    fit_spectral_slope,
)
from .wall_budget import MeshBudget, comparison_report, plate_mesh_budget
from .wall_law import (
    B_CHANNEL,
    B_LOG,
    KAPPA,
    BlasiusSolution,
    blasius_plate_cf,
    blasius_solution,
    channel_cf_from_u_tau,
    dean_channel_cf,
    friction_velocity_from_cf,
    ittc57_cf,
    log_law_u_plus,
    reichardt_u_plus,
    schoenherr_cf,
    schultz_grunow_cf,
    solve_friction_velocity,
    spalding_u_plus,
    spalding_y_plus,
    y_plus,
)

__all__ = [
    "B_CHANNEL",
    "B_LOG",
    "DNSReferenceData",
    "KAPPA",
    "BlasiusSolution",
    "CfFit",
    "CfScalingReport",
    "ChannelStatistics",
    "ComparisonResult",
    "Criterion",
    "GCIReport",
    "MeshBudget",
    "VAN_DRIEST_B",
    "VAN_DRIEST_KAPPA",
    "WallLawFit",
    "assemble_ladder",
    "blasius_plate_cf",
    "blasius_solution",
    "calibrate_cf_scaling",
    "channel_bulk_velocity",
    "channel_centerline_velocity",
    "channel_cf_from_u_tau",
    "compare_with_dns",
    "comparison_report",
    "compute_channel_statistics",
    "compute_energy_spectrum",
    "criterion",
    "dean_channel_cf",
    "fit_log_law",
    "fit_power_law",
    "fit_spectral_slope",
    "friction_coefficient",
    "friction_velocity_from_cf",
    "gci",
    "get_dns_data",
    "ittc57_cf",
    "log_law_u_plus",
    "mkm1999_reynolds_stresses",
    "musker_profile",
    "observed_order",
    "plate_mesh_budget",
    "reichardt_profile",
    "reichardt_u_plus",
    "relative_error_pct",
    "richardson_extrapolation",
    "schoenherr_cf",
    "schultz_grunow_cf",
    "solve_friction_velocity",
    "spalding_u_plus",
    "spalding_y_plus",
    "verdict",
    "write_payload",
    "y_plus",
]
