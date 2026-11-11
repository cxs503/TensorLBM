"""Flexural-gravity wave dispersion under an elastic ice sheet (node p338w3-n05).

Public surface:

* :mod:`tensorlbm.wave_ice_dispersion.analytic` -- closed-form dispersion,
  phase/group velocity, limits and sensitivity;
* :mod:`tensorlbm.wave_ice_dispersion.tank` -- periodic linear potential-flow
  numerical tank with an elastic (massless) plate cover;
* :mod:`tensorlbm.wave_ice_dispersion.campaign` -- 5 x 10 cross-adjudication,
  negative controls, grid convergence, group-velocity measurement and report.
"""

from .analytic import (
    E_ICE,
    G,
    NU_ICE,
    RHO_I,
    RHO_W,
    critical_phase_speed,
    dispersion_omega,
    flexural_rigidity,
    free_surface_omega,
    group_speed,
    mass_loading,
    phase_speed,
    rigidity_sensitivity,
    shallow_water_speed,
    thickness_scaling_slope,
)
from .tank import FlexuralWaveTank, TankConfig, initial_mode, initial_packet

__all__ = [
    "E_ICE", "G", "NU_ICE", "RHO_I", "RHO_W",
    "critical_phase_speed", "dispersion_omega", "flexural_rigidity",
    "free_surface_omega", "group_speed", "mass_loading", "phase_speed",
    "rigidity_sensitivity", "shallow_water_speed", "thickness_scaling_slope",
    "FlexuralWaveTank", "TankConfig", "initial_mode", "initial_packet",
]