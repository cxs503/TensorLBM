"""Unsteady Kunz cavitation dynamics + sheet/cloud transition (mission P22W2-N09).

Why this package exists
-----------------------
The in-repo quasi-steady chain :mod:`tensorlbm.cavitation_chain` (mission
P151W1-N04) answers "how long is the cavity at cavitation number ``sigma``"
with an *isobaric contour* argument and answers "how fast does it shed" with an
**imposed** Strouhal number (``f = U/(k L_cav)`` with ``k = 4`` fixed).  Both
answers are therefore closed-form; the shedding frequency is an input, not an
output.

This package is the complementary, genuinely time-accurate face of the same
physics: vapour fraction is transported by a finite-volume discretisation of

.. math::

    \\frac{\\partial \\alpha_v}{\\partial t}
    + \\frac{\\partial (u\\,\\alpha_v)}{\\partial x} = S(\\alpha_v, p),

with the Kunz (2000) pressure-driven mass-transfer closure :math:`S`.  Given a
NACA0015 section, a free-stream speed and a cavitation number, the solver
returns

* the time-mean and instantaneous cavity length ``L_cav/c``,
* the **measured** cloud-shedding frequency ``f_shed`` (from the time series of
  the vapour extent, not imposed) and the resulting Strouhal numbers,
* the evaporation/condensation mass budget of the transport solve,
* the sheet-to-cloud transition number ``sigma_critical`` (bistability of the
  solve, cross-checked with the re-entrant-jet criterion ``L/c = 0.5``),
* the blade-element thrust derating ``dK_T/K_T0`` resolved per blade azimuth.

Sub-modules
-----------
``kunz_fv``     Kunz closure + 1-D unsteady finite-volume transport + shedding
``thrust``      blade-element propeller model with cavitation derating
``validation``  the E1-E7 acceptance criteria of the node, writes the artefact
``references``  digitised experimental data used by the acceptance criteria

Deviation note (declared): the mission statement writes the Kunz source without
availability factors.  Here the standard Kunz (2000) availability factors are
kept -- liquid fraction ``alpha_l = 1 - alpha_v`` for evaporation and
``alpha_v`` for condensation -- because the bare form is unbounded above
``alpha_v = 1``.  This is reported in the validation artefact as
``kunz_form``.
"""

from __future__ import annotations

from .kunz_fv import (
    CavityDynamics,
    KunzClosure,
    cavitation_number,
    dominant_frequency,
    naca0015_cp,
    run_dynamics,
    sigma_critical_dynamics,
    steady_cavity_length,
    strouhal,
    vapour_pressure,
    ventilation_number,
)
from .references import CLOUD_REFERENCE, SHEET_REFERENCE, sigma_critical_reference
from .thrust import BladeGeometry, PropellerDesign, thrust_derating, thrust_derating_vs_sigma

__all__ = [
    "BladeGeometry",
    "CLOUD_REFERENCE",
    "CavityDynamics",
    "KunzClosure",
    "PropellerDesign",
    "SHEET_REFERENCE",
    "cavitation_number",
    "dominant_frequency",
    "naca0015_cp",
    "run_dynamics",
    "sigma_critical_dynamics",
    "sigma_critical_reference",
    "steady_cavity_length",
    "strouhal",
    "thrust_derating",
    "thrust_derating_vs_sigma",
    "vapour_pressure",
    "ventilation_number",
]