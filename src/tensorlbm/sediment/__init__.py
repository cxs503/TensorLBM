"""Sediment transport & bed-scour verification surface (mission m7329/m8506).

This package closes the zero-verification gap for sediment dynamics on
TensorLBM.  It provides four machine-judgeable anchors:

* **Incipient-motion curve** — numerical critical Shields stress measured from
  a coupled LBM-shear + Exner/MPM morphodynamic response, compared against the
  Soulsby–Whitehouse (2005) reference curve (criterion C1).
* **Rouse suspended-load profile** — Rouse exponent ``Z = w_s/(kappa u*)``
  recovered by fitting a numerically solved 1-D advection–diffusion balance
  driven by the LBM wall-shear velocity (criterion C2).
* **Exner bed evolution** — conservative finite-difference update of the
  Meyer-Peter–Müller flux; global sediment mass drift < 1e-6 (C3) and a
  sub-critical negative control (C4).
* **Single-pile local scour** — 2-D vertical LBM flow over a circular pile on
  a movable bed; equilibrium scour depth ``S/D`` checked against the
  Melville–Chiew empirical band [1.0, 2.4] (C5).

The end-to-end runner (``verification.build_results``) emits the JSON
contract ``{shields_curve, rouse_profiles, bed_elevation, scour_depth,
mass_budget}`` required by criterion C6.

Engineering hook: ``bed_elevation`` (m, per column) is the interface consumed
by the ship platform UKC / siltation endpoint.
"""

from .exner import ExnerConfig, exner_step, mpm_flux
from .rouse import KAPPA, rouse_exponent, rouse_profile, solve_rouse_profile
from .scour import ScourConfig, estimate_scour_depth, run_single_pile_scour
from .shields import (
    critical_shields,
    dimensionless_grain_size,
    dimensionless_settling_velocity,
)
from .verification import D_STAR_POINTS, SedimentScenario, build_results

__all__ = [
    "D_STAR_POINTS",
    "ExnerConfig",
    "KAPPA",
    "ScourConfig",
    "SedimentScenario",
    "build_results",
    "critical_shields",
    "dimensionless_grain_size",
    "dimensionless_settling_velocity",
    "estimate_scour_depth",
    "exner_step",
    "mpm_flux",
    "rouse_exponent",
    "rouse_profile",
    "run_single_pile_scour",
    "solve_rouse_profile",
]
