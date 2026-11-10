"""Sediment transport & bed-scour verification surface (mission P84W2C-N04).

This package provides the minimum machine-judgeable chain for suspended-sediment
transport and bed evolution on TensorLBM:

* **incipient motion** -- ``theta_cr(d*)`` measured through the physical chain
  ``d -> d* -> theta_cr -> gate -> Exner response`` and compared with the
  Soulsby & Whitehouse (2005) reference curve (:mod:`.shields`, :mod:`.morpho`);
* **suspended load** -- passive-scalar D2Q9 lattice Boltzmann transport in
  Rouse-log coordinates, ``Z = w_s/(kappa u*)`` recovered to <1 %
  (:mod:`.passive_scalar`, :mod:`.rouse`);
* **bed evolution** -- conservative Meyer-Peter-Mueller / Exner update with an
  exact discrete mass budget and the no-motion gate (:mod:`.exner`);
* **single-pile scour** -- 2-D LBM flow past a cylinder + 2-D movable-bed
  relaxation, ``S/D`` against the Melville & Chiew band (:mod:`.scour`);
* **engineering hook** -- ``bed_elevation`` -> ship-platform UKC / siltation
  payload (:mod:`.ukc_hook`).

The end-to-end runner (:func:`.verification.build_results`) emits the JSON
contract ``{shields_curve, rouse_profiles, bed_elevation, scour_depth,
mass_budget}`` and is exercised by ``tests/test_sediment_scour.py``; the
artifact runner is :mod:`tensorlbm.sediment_benchmarks`.

Documented limitations (see the report): the C1 onset measurement is an
implementation-consistency check because the transport law needs the
incipient-motion parameter; the naive variable-tau scalar LBM is over-diffusive
(O(dD/dy) spurious flux) which is why the Rouse anchor uses the Rouse-log
graded formulation; the scour model is depth-averaged 2-D (no horseshoe vortex),
so ``S/D`` is judged against the empirical band only.
"""

from .exner import ExnerConfig, bed_volume, exner_step, mpm_flux, stable_dt
from .morpho import MorphoConfig, bed_response, measure_onset_shields
from .passive_scalar import (
    RouseLBMConfig,
    ScalarLBMConfig,
    column_exchange_budget,
    rouse_log_metric,
    run_rouse_lbm,
    run_scalar_channel,
)
from .rouse import (
    KAPPA,
    eddy_diffusivity,
    fit_rouse_exponent,
    rouse_exponent,
    rouse_profile,
    solve_rouse_profile,
)
from .scour import (
    PileConfig,
    ScourConfig,
    relax_scour,
    run_pile_flow,
    run_single_pile_scour,
    verify_flow_solver,
)
from .shields import (
    critical_shields,
    dimensionless_grain_size,
    dimensionless_settling_velocity,
    grain_diameter_from_dstar,
    settling_velocity,
    shear_velocity_from_shields,
    shields_parameter,
)
from .ukc_hook import CONTRACT as UKC_CONTRACT
from .ukc_hook import bed_elevation_to_ukc, build_payload, post_bed_elevation
from .verification import (
    D_STAR_POINTS,
    SedimentScenario,
    build_results,
    c1_incipient_curve,
    c2_rouse_profiles,
    c3_mass_budget,
    c4_negative_controls,
    c5_pile_scour,
)

__all__ = [
    "D_STAR_POINTS",
    "ExnerConfig",
    "KAPPA",
    "MorphoConfig",
    "PileConfig",
    "RouseLBMConfig",
    "ScalarLBMConfig",
    "ScourConfig",
    "SedimentScenario",
    "UKC_CONTRACT",
    "bed_elevation_to_ukc",
    "bed_response",
    "bed_volume",
    "build_payload",
    "build_results",
    "c1_incipient_curve",
    "c2_rouse_profiles",
    "c3_mass_budget",
    "c4_negative_controls",
    "c5_pile_scour",
    "column_exchange_budget",
    "critical_shields",
    "dimensionless_grain_size",
    "dimensionless_settling_velocity",
    "eddy_diffusivity",
    "exner_step",
    "fit_rouse_exponent",
    "grain_diameter_from_dstar",
    "measure_onset_shields",
    "mpm_flux",
    "post_bed_elevation",
    "relax_scour",
    "rouse_exponent",
    "rouse_log_metric",
    "rouse_profile",
    "run_pile_flow",
    "run_rouse_lbm",
    "run_scalar_channel",
    "run_single_pile_scour",
    "settling_velocity",
    "shear_velocity_from_shields",
    "shields_parameter",
    "solve_rouse_profile",
    "stable_dt",
    "verify_flow_solver",
]