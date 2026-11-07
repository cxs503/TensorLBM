"""Flettner rotor-sail (spinning-cylinder Magnus) toolbox for TensorLBM.

Sub-packages
------------
``lbm2d``   D2Q9 rotating-cylinder solver with BFL moving-wall bounce-back,
            optional Smagorinsky LES, prescribed inlet profiles and two
            independent force estimators (wall momentum exchange and
            control-volume momentum balance).
``aero``    coefficient handling, ship-wake inflow models and the spanwise
            thrust integration / apparent-wind decomposition.
``voyage``  Weibull wind statistics -> annual thrust assist -> fuel and CO2.

Entry points
------------
``run_rotor_channel``, ``build_geometry`` (lbm2d); ``strip_thrust``,
``uniform_inflow_thrust``, ``rotor_sail_thrust``, ``deck_boundary_layer_profile``
(aero); ``run_voyage_assessment`` (voyage).
"""

from .aero import (  # noqa: F401
    AIR_DENSITY,
    REFERENCE_TABLE,
    RotorGeometrySpec,
    apparent_wind,
    combined_wake_profile,
    deck_boundary_layer_profile,
    interpolate_coefficients,
    propeller_race_profile,
    rotor_sail_thrust,
    strip_thrust,
    uniform_inflow_thrust,
)
from .lbm2d import (  # noqa: F401
    RotorChannelConfig,
    build_geometry,
    macroscopic,
    run_rotor_channel,
)
from .voyage import VoyageInputs, run_voyage_assessment, weibull_bins, weibull_pdf  # noqa: F401

__all__ = [
    "AIR_DENSITY",
    "REFERENCE_TABLE",
    "RotorChannelConfig",
    "RotorGeometrySpec",
    "VoyageInputs",
    "apparent_wind",
    "build_geometry",
    "combined_wake_profile",
    "deck_boundary_layer_profile",
    "interpolate_coefficients",
    "macroscopic",
    "propeller_race_profile",
    "rotor_sail_thrust",
    "run_rotor_channel",
    "run_voyage_assessment",
    "strip_thrust",
    "uniform_inflow_thrust",
    "weibull_bins",
    "weibull_pdf",
]
