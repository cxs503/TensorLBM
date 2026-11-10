"""Ice--water phase-change interface surface: Stefan anchors + front tracking.

This package fills the "phase-change heat transfer" verification gap for the
*sharp* ice/water interface.  It is deliberately **complementary** to the
existing diffuse phase-change surface (:mod:`tensorlbm.phase_change`, which
covers Lee-model sources, enthalpy thermal-LBE and one-phase planar /
cylindrical similarity):

===============  =========================================================
module           content
===============  =========================================================
properties       SI thermophysical data for ice / water at 0 degC, derived
                 dimensionless groups (Ste, Ja, diffusivity ratio)
similarity       *two-phase* Neumann similarity solution (finite solid
                 subcooling, k_s != k_l, alpha_s != alpha_l) -- the
                 one-phase branches in ``phase_change.stefan_exact`` are the
                 ``dT_solid -> 0`` limit of this one -- plus the classical
                 quasi-steady cylindrical (London--Seban / Lunardini)
                 melting law used as the literature anchor for C2
front1d          1-D *sharp* front-tracking finite-volume solver (cut-cell
                 at the interface, ghost-node Dirichlet closure) with latent
                 heat / energy auditing
radial           axisymmetric (r) sharp front-tracking solver for the
                 2-D circular ice cylinder melting into warm water
enthalpy2d       genuine 2-D Cartesian enthalpy (diffuse) solver on the same
                 case -- used as the *interface-thickness contrast* to prove
                 the sharp tracker resolves the front within <= 2 cells
audit            energy / latent-heat conservation, interface thickness and
                 temperature-continuity metrics
verification     the C1..C7 acceptance driver returning a JSON-ready dict
===============  =========================================================

Notation
--------
``s``  interface position measured from the wall (1-D) or the front radius
(radial), ``alpha`` thermal diffusivity, ``L`` latent heat of fusion,
``Ste = c (T_wall - T_m) / L``.

All solvers are written in SI units and are *steady with respect to the
verification anchors*: the same linear conduction operator, the same
interface energy balance.  The density-equal idealisation
``rho_s = rho_l`` (documented limitation, ~9 % on the latent term for a real
ice/water interface) is used consistently in both the anchor and the solver,
so the numerical-vs-analytic comparison isolates discretisation error.
"""

from .properties import (  # noqa: F401
    ICE,
    WATER,
    IceWaterCase,
    PhaseProperties,
)
from .similarity import (  # noqa: F401
    TwoPhaseNeumann,
    quasi_steady_cylinder_time,
    quasi_steady_cylinder_front,
    quasi_steady_cylinder_error_estimate,
)
from .front1d import (  # noqa: F401
    FrontTracking1D,
    FrontTracking1DResult,
)
from .radial import (  # noqa: F401
    RadialFrontTracking,
    RadialFrontTrackingResult,
)
from .enthalpy2d import (  # noqa: F401
    Enthalpy2D,
    Enthalpy2DResult,
)
from .audit import (  # noqa: F401
    energy_audit,
    interface_thickness_cells,
    temperature_continuity,
)
from .verification import run_verification_suite  # noqa: F401

__all__ = [
    "ICE",
    "WATER",
    "IceWaterCase",
    "PhaseProperties",
    "TwoPhaseNeumann",
    "quasi_steady_cylinder_time",
    "quasi_steady_cylinder_front",
    "quasi_steady_cylinder_error_estimate",
    "FrontTracking1D",
    "FrontTracking1DResult",
    "RadialFrontTracking",
    "RadialFrontTrackingResult",
    "Enthalpy2D",
    "Enthalpy2DResult",
    "energy_audit",
    "interface_thickness_cells",
    "temperature_continuity",
    "run_verification_suite",
]