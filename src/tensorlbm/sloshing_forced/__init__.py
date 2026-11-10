"""Forced-sway LNG-tank sloshing verification surface (campaign node P88W1-N01).

What this package is
--------------------
A *verification* add-on to the audited Körner free-surface LBM
(:mod:`tensorlbm.free_surface_lbm_2d` / :mod:`tensorlbm.sloshing_resonance`):
it runs a partially filled rectangular tank under **harmonic sway** and puts the
two engineering quantities of the sloshing load chain on a machine-checkable
footing

* the **resonance frequency** of the first two free-surface modes
  (analytic anchor: Abramson 1966 / Faltinsen & Timokha 2009 dispersion), and
* the **wall impact pressure** at resonance against closed-form bounds
  (linear wall pressure, Wagner stagnation ``0.5 rho V^2``, bore reflection,
  hydrostatic run-up).

Difference from the sibling verification nodes
----------------------------------------------
``tensorlbm.sloshing_resonance`` (P87W2-N01) validates *free decay* — one modal
release, damping identification.  ``tensorlbm.lng_sloshing`` (P142W3-N01) is a
1:1 campaign aimed at fatigue spectra.  This package validates the **forced
transfer function**: a frequency sweep over ``f/f_1`` that covers the first and
second natural frequencies, the measured parity null at ``f_2`` (horizontal
translation excites odd modes only), and the pressure/run-up response at
resonance, plus a 2x grid-refinement study of the pressure peak.

Evidence grading (per the platform evidence scale)
--------------------------------------------------
REAL        : the lattice solver, the dispersion anchor, the forced-sway RAO
              derivation, all pressure bounds.
WITHHELD    : experimental (model-test / ITTC-recommended-procedure) sloshing
              pressure data — the repository carries no digitised time series and
              none was fabricated here; every experimental comparison is marked
              as such in the campaign report.
"""

from __future__ import annotations

from .sweep import (
    CASE_DEFAULTS,
    TankSpec,
    build_case,
    harmonic_metrics,
    identify_resonance,
    pressure_metrics,
    run_decay_case,
    run_forced_case,
    wall_runup_series,
)
from .theory import (
    abramson_angular_frequency,
    abramson_frequency,
    abramson_period,
    abramson_wave_number,
    bore_reflection_pressure,
    gauge_height_from_swl,
    hydrostatic_runup_pressure,
    linear_wall_pressure_amplitude,
    mode_excitation_coefficient,
    modal_forcing_amplitude,
    odd_mode_fraction,
    peak_frequency,
    resonance_ratio,
    stagnation_impact_pressure,
    static_tilt_elevation,
    steady_modal_amplitude,
    steady_pressure_amplitude,
    steady_wall_amplitude,
)

__all__ = [
    "CASE_DEFAULTS",
    "TankSpec",
    "build_case",
    "harmonic_metrics",
    "identify_resonance",
    "pressure_metrics",
    "run_decay_case",
    "run_forced_case",
    "wall_runup_series",
    "abramson_angular_frequency",
    "abramson_frequency",
    "abramson_period",
    "abramson_wave_number",
    "bore_reflection_pressure",
    "gauge_height_from_swl",
    "hydrostatic_runup_pressure",
    "linear_wall_pressure_amplitude",
    "mode_excitation_coefficient",
    "modal_forcing_amplitude",
    "odd_mode_fraction",
    "peak_frequency",
    "resonance_ratio",
    "stagnation_impact_pressure",
    "static_tilt_elevation",
    "steady_modal_amplitude",
    "steady_pressure_amplitude",
    "steady_wall_amplitude",
]

__version__ = "1.0.0"