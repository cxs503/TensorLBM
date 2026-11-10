"""Low-magnetic-Reynolds-number MHD for TensorLBM (Hartmann ladder + thruster ledger).

Module map
----------
``hartmann``  closed-form Hartmann/Poiseuille solutions and a D2Q9 duct solver
              with a genuine ``F = sigma (E + u x B)`` Lorentz body force.
``ladder``    Hartmann-number ladder, grid-convergence (GCI) study, machine
              acceptance criteria and artefact writing.
``thruster``  channel (pump-jet) MHD thruster thrust/efficiency ledger, energy
              guard, same-size propeller comparison and the ideal ``sigma|E|B V``
              thrust reference (``thruster_ideal_check``).
``shercliff`` insulating-wall (zero net current) branch: closed form, dynamic
              ``K = <u_x>`` closure, energy split and its own acceptance criteria.
``damping``   shear-wave magnetic damping: exact ``exp(-gamma t)`` decay-rate
              anchor with ``gamma = nu k^2 + nu Ha^2/h^2`` and the ``gamma ~ Ha^2``
              scaling ladder (node p315w1-n04).

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
from .damping import (
    ShearWaveCase,
    damping_ladder,
    evaluate_damping_criteria,
    ha_scaling_exponent,
    magnetic_decay_rate,
    run_damping_case,
    viscous_decay_rate,
    wave_number,
)
from .ladder import (
    LadderResult,
    evaluate_ladder_criteria,
    gci,
    run_hartmann_ladder,
)
from .shercliff import (
    ShercliffCase,
    evaluate_shercliff_criteria,
    shercliff_current_density,
    shercliff_grad_for_mean,
    shercliff_mean_velocity,
    shercliff_normalised_profile,
    shercliff_profile,
    shercliff_wall_shear_coefficient,
    run_shercliff_channel,
)
from .thruster import (
    MHDThrusterConfig,
    ThrusterLedger,
    ideal_lorentz_thrust,
    propeller_ideal_efficiency,
    run_thruster_ledger,
    thruster_ideal_check,
)

__all__ = [
    "HartmannConfig",
    "ShearWaveCase",
    "damping_ladder",
    "evaluate_damping_criteria",
    "ha_scaling_exponent",
    "magnetic_decay_rate",
    "run_damping_case",
    "viscous_decay_rate",
    "wave_number",
    "ShercliffCase",
    "evaluate_shercliff_criteria",
    "ideal_lorentz_thrust",
    "run_shercliff_channel",
    "shercliff_current_density",
    "shercliff_grad_for_mean",
    "shercliff_mean_velocity",
    "shercliff_normalised_profile",
    "shercliff_profile",
    "shercliff_wall_shear_coefficient",
    "thruster_ideal_check",
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
