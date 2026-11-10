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
``resistance``friction/form/residual resistance decomposition (KVLCC2 anchor);
``closure_channel`` one-dimensional wall-model-driven channel closure in wall
              units (a-priori grading of the closure itself);
``reference_gate`` integrity gate for channel-DNS reference tables (shape and
              stress plausibility before any error number is formed);
``peak_diagnostic`` sub-sample Reynolds-stress peak location, three-grid
              Richardson/GCI, and the wall-function peak-position correction
              map used to separate grid quantisation from wall-model error.
"""

from __future__ import annotations

from .closure_channel import (
    ChannelClosureSolution,
    log_layer_fit,
    solve_channel_closure,
)
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
from .peak_diagnostic import (
    GridConvergence,
    PeakEstimate,
    analytic_stress_profile,
    find_stress_peak,
    fit_offset_correction,
    load_channel_means_table,
    reynolds_stress_from_mean_profile,
    richardson_gci,
    spacing_for_tolerance,
)
from .reference_gate import validate_channel_dns_table

__all__ = [
    "B_DEFAULT",
    "GridConvergence",
    "PeakEstimate",
    "analytic_stress_profile",
    "find_stress_peak",
    "fit_offset_correction",
    "load_channel_means_table",
    "reynolds_stress_from_mean_profile",
    "richardson_gci",
    "spacing_for_tolerance",
    "ChannelClosureSolution",
    "KAPPA_DEFAULT",
    "EquilibriumWallModel",
    "NonEquilibriumWallModel",
    "WallModelResult",
    "law_u_plus",
    "log_layer_fit",
    "log_law_u_plus",
    "reichardt_u_plus",
    "source_term_from_resolved",
    "spalding_u_plus",
    "solve_channel_closure",
    "solve_u_tau",
    "validate_channel_dns_table",
]