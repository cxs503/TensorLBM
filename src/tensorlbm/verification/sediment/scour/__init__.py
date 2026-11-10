"""Suspended-sediment transport and bed-scour verification face.

Public surface
--------------
``Sediment``, ``shields_critical_soulsby``, ``rouse_profile``,
``total_load_transport``, ``exner_step``   -- physics kernels
``shields_case``, ``rouse_case``, ``pile_scour_case``  -- the three cases
``D2Q9Channel``                            -- the lean NumPy D2Q9 engine
``run_all`` / ``main``                     -- artifacts + figures (``runner``)

Evidence grading (see the node report):
  * Shields   : REAL -- LBM hydraulics, threshold from an empirical curve
  * Rouse     : REAL -- LBM hydraulics + three independent transport solvers
  * Exner mass: REAL -- discrete telescoping property
  * Pile scour: PARTIAL -- coupled LBM+Exner, but 2-D depth-averaged model form
                cannot resolve the horseshoe vortex, see the honest-FAIL note.
"""

from .cases import log_law_fit, pile_scour_case, rouse_case, shields_case
from .lattice import D2Q9Channel
from .sediment import (
    KAPPA,
    Sediment,
    avalanche_relax,
    exner_step,
    rouse_profile,
    shields_critical_soulsby,
    shields_curve_digitised,
    total_load_transport,
)

__all__ = [
    "D2Q9Channel",
    "KAPPA",
    "Sediment",
    "avalanche_relax",
    "exner_step",
    "log_law_fit",
    "pile_scour_case",
    "rouse_case",
    "rouse_profile",
    "shields_case",
    "shields_critical_soulsby",
    "shields_curve_digitised",
    "total_load_transport",
]
