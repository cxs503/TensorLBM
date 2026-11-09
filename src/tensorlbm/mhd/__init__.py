"""Low-magnetic-Reynolds-number MHD for TensorLBM (Hartmann ladder + thruster ledger).

Module map
----------
``hartmann``  closed-form Hartmann/Poiseuille solutions and a D2Q9 duct solver
              with a genuine ``F = sigma (E + u x B)`` Lorentz body force.
``ladder``    Hartmann-number ladder, grid-convergence (GCI) study, machine
              acceptance criteria and artefact writing.
``thruster``  channel (pump-jet) MHD thruster thrust/efficiency ledger, energy
              guard and same-size propeller comparison.

The module owns solver-local MHD verification only (see ``docs/PROJECT_CHARTER.md``);
campaign orchestration lives in TensorPhysics.
"""

from __future__ import annotations

from .hartmann import (
    HartmannConfig,
    HartmannResult,
    ha_from_params,
    hartmann_mean_factor,
    hartmann_mean_velocity_coefficient,
    hartmann_profile,
    hartmann_shape_factor,
    hartmann_u0_normalised_profile,
    hartmann_wall_shear_coefficient,
    poiseuille_profile,
    run_hartmann_channel,
)
from .ladder import (
    LadderResult,
    evaluate_ladder_criteria,
    gci,
    run_hartmann_ladder,
)
from .thruster import (
    MHDThrusterConfig,
    ThrusterLedger,
    propeller_ideal_efficiency,
    run_thruster_ledger,
)

__all__ = [
    "HartmannConfig",
    "HartmannResult",
    "LadderResult",
    "MHDThrusterConfig",
    "ThrusterLedger",
    "evaluate_ladder_criteria",
    "gci",
    "ha_from_params",
    "hartmann_mean_factor",
    "hartmann_profile",
    "hartmann_shape_factor",
    "hartmann_u0_normalised_profile",
    "hartmann_mean_velocity_coefficient",
    "hartmann_wall_shear_coefficient",
    "poiseuille_profile",
    "propeller_ideal_efficiency",
    "run_hartmann_channel",
    "run_hartmann_ladder",
    "run_thruster_ledger",
]
