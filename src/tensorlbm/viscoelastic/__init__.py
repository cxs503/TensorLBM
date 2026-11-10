"""Viscoelastic (polymer) constitutive branch for the D2Q9 lattice-Boltzmann solver.

Physics
-------
The polymer contribution to the stress is carried by the conformation tensor
``C`` (ensemble averaged dumbbell end-to-end tensor).  Two constitutive models
share one code path:

* **Oldroyd-B** -- upper-convected Maxwell stress of a Hookean dumbbell,

      dC/dt + u.grad(C) = L^T C + C L - (C - I) / lambda,
      tau_p = (eta_p / lambda) (C - I),

  with ``L = grad(u)``.  In steady homogeneous shear (rate ``gdot``, ``Wi =
  lambda*gdot``) the exact solution is ``C_xx = 1 + 2 Wi^2``, ``C_xy = Wi``,
  ``C_yy = C_zz = 1``, hence ``tau_xy = eta_p*gdot`` and the first normal
  stress difference is ``N1 = tau_xx - tau_yy = 2 eta_p lambda gdot^2``.

* **FENE-P** -- finitely extensible dumbbell,

      dC/dt + u.grad(C) = L^T C + C L - (f(tr C) C - a I) / lambda,
      tau_p = (eta_p / lambda) (f(tr C) C - a I),
      f(r) = 1 / (1 - r / L^2),      a = 1 / (1 - 3 / L^2).

  ``tr C < L^2`` replaces ``C >= I`` as the realizability domain.

Realizability
-------------
``C`` must stay symmetric positive semi-definite with eigenvalues >= 1
(Oldroyd-B) or in ``(0, L^2)`` (FENE-P).  The discrete evolution does *not*
guarantee this once the advective/strain terms are advanced explicitly, so
:func:`project_realizable` clips the eigenspectrum after every update (the
"realizability projection").  The projection is switchable so that a negative
control can demonstrate the projected/unprojected difference.

Layout
------
``conformation``   tensor algebra, stresses, realizability gate + projection,
                   constitutive time step (implicit in strain/relaxation).
``analytical``     exact Oldroyd-B plane-Poiseuille / Couette anchors.
``lbm``            coupled LBM: solvent BGK + Guo forcing + polymer stress
                   divergence, channel driver.
``energy``         discrete energy budget (driving power = solvent + polymer
                   dissipation + elastic storage).
``virk``           Virk three-segment law / MDR asymptote reference for the
                   turbulent drag-reduction side (reference only -- see the
                   module docstring for the evidence grade).
``study``          builds ``artifacts/viscoelastic/`` (study.json + figures).
"""

from __future__ import annotations

from .analytical import (
    oldroyd_b_poiseuille_conformation,
    oldroyd_b_poiseuille_fields,
    oldroyd_b_poiseuille_profile,
    oldroyd_b_steady_shear_conformation,
    poiseuille_max_velocity,
)
from .conformation import (
    ConformationState,
    ViscoelasticDivergenceError,
    advance_conformation,
    check_realizability,
    fene_p_steady_shear_conformation,
    fene_p_stress,
    oldroyd_b_stress,
    project_realizable,
)
from .energy import energy_budget
from .lbm import ViscoelasticParams, run_viscoelastic_channel
from .virk import (
    MDR_DELTA_B_MAX,
    check_delta_b_bound,
    virk_mdr_profile,
    virk_newtonian_profile,
    virk_three_segment,
)

__all__ = [
    "MDR_DELTA_B_MAX",
    "ConformationState",
    "ViscoelasticDivergenceError",
    "ViscoelasticParams",
    "advance_conformation",
    "check_delta_b_bound",
    "check_realizability",
    "energy_budget",
    "fene_p_steady_shear_conformation",
    "fene_p_stress",
    "oldroyd_b_poiseuille_conformation",
    "oldroyd_b_poiseuille_fields",
    "oldroyd_b_poiseuille_profile",
    "oldroyd_b_steady_shear_conformation",
    "oldroyd_b_stress",
    "poiseuille_max_velocity",
    "project_realizable",
    "run_viscoelastic_channel",
    "virk_mdr_profile",
    "virk_newtonian_profile",
    "virk_three_segment",
]