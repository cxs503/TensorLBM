"""LNG cargo-tank sloshing at partial fill: free-surface LBM + Abramson anchors.

New frontier capability (campaign node IN143-N07).  Distinct from
:mod:`tensorlbm.sloshing_tank`, which drives a *two-phase colour-gradient*
model; here the free surface is tracked with the Körner mass-ledger model
(:mod:`tensorlbm.free_surface_lbm_2d`) and the analytic anchor is Abramson's
(1966) potential-flow natural-frequency ladder plus the matching linear forced
response / wall-pressure theory.

Sub-modules
-----------
``abramson``  analytic eigen-frequencies, linear modal response, wall pressure,
              stagnation (Wagner) impact bound  (REAL, closed form)
``tank``      rectangular free-surface LBM tank: geometry, fill, sway forcing,
              modal free-surface elevation, wall pressure gauges
``analysis``  frequency/damping/harmonic/peak estimators on the time series
``campaign``  the acceptance campaign (fill sweep, pressure sweep, figures)
"""

from __future__ import annotations

from .abramson import (
    SP106_REFERENCE,
    abramson_angular_frequency,
    abramson_hydrostatic_wave_speed,
    abramson_mode_ratio,
    abramson_mode_shape,
    abramson_natural_frequency,
    abramson_wave_number,
    linear_forced_modal_amplitude,
    linear_response_ratio,
    linear_wall_pressure_amplitude,
    stagnation_impact_pressure,
)
from .analysis import (
    decay_damping,
    harmonic_amplitudes,
    pressure_peak,
    reconstruction_error,
    refined_frequency,
)
from .tank import (
    TankConfig,
    TankState,
    build_tank,
    column_liquid_height,
    fill_ratio_to_depth,
    modal_projection,
    simulate_tank,
    surface_elevation,
    wall_pressure_profile,
)

__all__ = [
    "SP106_REFERENCE",
    "TankConfig",
    "TankState",
    "abramson_angular_frequency",
    "abramson_hydrostatic_wave_speed",
    "abramson_mode_ratio",
    "abramson_mode_shape",
    "abramson_natural_frequency",
    "abramson_wave_number",
    "build_tank",
    "column_liquid_height",
    "decay_damping",
    "fill_ratio_to_depth",
    "harmonic_amplitudes",
    "linear_forced_modal_amplitude",
    "linear_response_ratio",
    "linear_wall_pressure_amplitude",
    "modal_projection",
    "pressure_peak",
    "reconstruction_error",
    "refined_frequency",
    "simulate_tank",
    "stagnation_impact_pressure",
    "surface_elevation",
    "wall_pressure_profile",
]