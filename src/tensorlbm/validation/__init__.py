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

The package is deliberately free of solver imports (pure ``math``/``numpy``
free code except where noted) so it can be used by pytest, by post-processing
scripts and by the campaign report generator without a GPU backend.
"""

from __future__ import annotations

from .cf_scaling import CfFit, CfScalingReport, calibrate_cf_scaling, fit_power_law
from .gci import GCIReport, gci, observed_order, richardson_extrapolation
from .ladder import (
    Criterion,
    assemble_ladder,
    criterion,
    relative_error_pct,
    verdict,
    write_payload,
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
    "KAPPA",
    "BlasiusSolution",
    "CfFit",
    "CfScalingReport",
    "Criterion",
    "GCIReport",
    "MeshBudget",
    "assemble_ladder",
    "blasius_plate_cf",
    "blasius_solution",
    "calibrate_cf_scaling",
    "channel_cf_from_u_tau",
    "comparison_report",
    "criterion",
    "dean_channel_cf",
    "fit_power_law",
    "friction_velocity_from_cf",
    "gci",
    "ittc57_cf",
    "log_law_u_plus",
    "observed_order",
    "plate_mesh_budget",
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