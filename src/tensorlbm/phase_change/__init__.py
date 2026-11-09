"""Phase-change (evaporation/condensation) verification surface for TensorLBM.

Modules
-------
``stefan_anchors``
    Analytic anchors for the one-phase Stefan problem in three geometries
    (planar / 2-D cylindrical / 3-D spherical) plus the "heat-flux estimate"
    formula quoted in the campaign brief.

``lee_model``
    Lee-type phase-change mass/energy source terms together with a D2Q9
    thermal lattice Bhatnagar-Gross-Krook (LBE) solver and an enthalpy
    formulation of the interface update.

The package deliberately works in **lattice units** with the
constant-density (density-equal) Stefan idealisation so that the latent heat
per unit mass of the disappearing phase equals ``h_fg`` and the Jacob number
reduces to the Stefan number ``Ja = c_p dT / h_fg`` -- exactly the definition
used by the campaign brief.
"""

from .lee_model import (  # noqa: F401
    BubbleGrowthResult,
    LeePhaseChangeModel,
    PhaseChangeParameters,
    ThermalLBE2D,
    run_bubble_growth,
)
from .stefan_anchors import (  # noqa: F401
    exact_stefan_coefficient,
    heat_flux_estimate_coefficient,
    similarity_temperature_profile,
)

__all__ = [
    "BubbleGrowthResult",
    "LeePhaseChangeModel",
    "PhaseChangeParameters",
    "ThermalLBE2D",
    "run_bubble_growth",
    "exact_stefan_coefficient",
    "heat_flux_estimate_coefficient",
    "similarity_temperature_profile",
]