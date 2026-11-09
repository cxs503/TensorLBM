"""Wall-modelled LES (WMLES) capability for high-Reynolds-number wall bounds.

Modules
-------
``laws``      wall laws (log law, Spalding unified law, Reichardt) and the
              friction-velocity inversion;
``models``    equilibrium and non-equilibrium (thin-boundary-layer ODE) wall
              models that turn resolved velocity into wall shear stress;
``channel``   D3Q19 WMLES solver for the pressure-driven plane channel;
``budget``    Reynolds-stress budget post-processing and DNS anchors;
``flat_plate`` zero-pressure-gradient flat-plate boundary layer driven by the
              same wall law, with ITTC-1957 / Prandtl-Schlichting anchors;
``resistance``friction/form/residual resistance decomposition (KVLCC2 anchor).
"""

from __future__ import annotations

from .laws import (
    B_DEFAULT,
    KAPPA_DEFAULT,
    law_u_plus,
    log_law_u_plus,
    reichardt_u_plus,
    solve_u_tau,
    spalding_u_plus,
)
from .models import (
    EquilibriumWallModel,
    NonEquilibriumWallModel,
    WallModelResult,
    source_term_from_resolved,
)

__all__ = [
    "B_DEFAULT",
    "KAPPA_DEFAULT",
    "EquilibriumWallModel",
    "NonEquilibriumWallModel",
    "WallModelResult",
    "law_u_plus",
    "log_law_u_plus",
    "reichardt_u_plus",
    "source_term_from_resolved",
    "spalding_u_plus",
    "solve_u_tau",
]