"""Phase-change (evaporation/condensation) verification surface for TensorLBM.

Modules
-------
``stefan_anchors``
    Analytic anchors for the one-phase Stefan problem in three geometries
    (planar / 2-D cylindrical / 3-D spherical) plus the "heat-flux estimate"
    formula quoted in the campaign brief.

``stefan_exact`` / ``stefan_verification``
    Exact similarity solutions (planar Neumann freezing/melting branches,
    cylindrical circular front) and the enthalpy thermal-LBE Stefan
    verification surface (1-D front vs Neumann, 2-D circular-front
    solidification vs an independent sharp-interface reference, melting energy
    audit, grid-convergence order).  See
    ``docs/P155W1_N01_STEFAN_PHASE_CHANGE.md``.

``stefan_two_phase`` / ``stefan_sweep``
    Two-phase (superheated-melt) planar Stefan similarity solution -- a coupled
    dual-``lambda`` coefficient, ``T_i > T_freeze`` ahead of the front -- plus the
    Stefan-number sweep driver that verifies the enthalpy LBM against it
    (front / profile / energy / liquid-fraction / grid-order acceptance).
    See ``docs/P272W2_N05_STEFAN_TWO_PHASE_SWEEP.md``.

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
from .stefan_exact import (  # noqa: F401
    CylindricalSimilarity,
    NeumannSimilarity,
)
from .stefan_sweep import (  # noqa: F401
    run_two_phase_grid_convergence,
    run_two_phase_stefan,
    run_two_phase_sweep,
)
from .stefan_two_phase import (  # noqa: F401
    TwoPhaseSimilarity,
    two_phase_lambda,
    two_phase_residual,
)
from .stefan_verification import (  # noqa: F401
    EnthalpyThermalLBM,
    run_cylindrical_wall_solidification,
    run_grid_convergence,
    run_planar_stefan,
)

__all__ = [
    "CylindricalSimilarity",
    "EnthalpyThermalLBM",
    "NeumannSimilarity",
    "run_cylindrical_wall_solidification",
    "run_grid_convergence",
    "run_planar_stefan",
    "BubbleGrowthResult",
    "LeePhaseChangeModel",
    "PhaseChangeParameters",
    "ThermalLBE2D",
    "run_bubble_growth",
    "exact_stefan_coefficient",
    "heat_flux_estimate_coefficient",
    "similarity_temperature_profile",
    "TwoPhaseSimilarity",
    "two_phase_lambda",
    "two_phase_residual",
    "run_two_phase_stefan",
    "run_two_phase_sweep",
    "run_two_phase_grid_convergence",
]