"""Seakeeping validation surface for TensorLBM.

This package implements ship seakeeping (wave-induced motions) capabilities:

* :mod:`tensorlbm.seakeeping.wigley_hull` — Wigley III hull geometry,
  hydrostatics, and sectional properties for strip-theory integration.
* :mod:`tensorlbm.seakeeping.strip_theory` — Frequency-domain strip theory
  for heave/pitch added mass, damping, and excitation (BEM reference).
* :mod:`tensorlbm.seakeeping.lbm_body_coupling` — 2-D free-surface LBM
  coupled to a floating section (heave only), demonstrating the fluid–body
  coupling mechanism.
* :mod:`tensorlbm.seakeeping.rao_campaign` — Campaign driver: wave matrix
  λ/L ∈ {0.5, 0.75, 1.0, 1.5, 2.0}, H/λ = 1/60, 20 periods per case,
  FFT-based RAO extraction from the last 10 periods.

The validation target is Wigley III heave/pitch RAO in regular waves,
cross-validated against strip-theory frequency-domain results.
"""

from .wigley_hull import WigleyIII, sectional_properties
from .strip_theory import StripTheorySolver, compute_strip_theory_rao
from .lbm_body_coupling import LBMBodyCoupling2D, run_section_coupling
from .rao_campaign import SeakeepingCampaign, run_campaign

__all__ = [
    "WigleyIII",
    "sectional_properties",
    "StripTheorySolver",
    "compute_strip_theory_rao",
    "LBMBodyCoupling2D",
    "run_section_coupling",
    "SeakeepingCampaign",
    "run_campaign",
]
