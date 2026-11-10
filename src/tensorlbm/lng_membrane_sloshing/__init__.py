"""LNG membrane-tank partial-fill sloshing: LBM x Wagner/MLM x Dirlik fatigue.

Campaign node **P142W3-N01** (TensorLBM domain).  The package extends the
verified free-surface sloshing machinery of
:mod:`tensorlbm.sloshing_resonance` with the three physics pieces that were
missing for a cargo-tank load chain:

``impact``    Wagner (1932) flat-plate/wedge impact solution and the MLM
              (modified Logvinovich, Korobkin 2004) stagnation-pressure anchor,
              both re-derived from the complex potential so that the plate
              boundary condition, the ``1/sqrt(1-xi^2)`` spray-root singularity
              and the integrated slam force are testable in-repo.
``fatigue``   Dirlik (1985) spectral rainflow-range model and the ASTM
              E1049-85 time-domain rainflow count, both on a Basquin S-N curve
              (IIW FAT80 default), with the Palmgren-Miner rule.
``tank``      the 174 000 m^3 LNG carrier (four 44 m x 38 m x 26 m prismatic
              membrane tanks), the LNG properties, the fill/similarity scaling
              and the declared linear pressure -> stress transfer of the loaded
              wall panel.
``audit``     grid-convergence index (Roache) and the mechanical-energy budget
              of a free-surface run (KE + PE vs excitation work and the
              dissipation measured from the non-equilibrium stress).
``campaign``  the acceptance campaign: free-decay resonance at 15/30/50 % fill,
              forced-resonance impact histories, controlled Wagner/MLM wedge
              impacts, and the Dirlik-vs-rainflow damage comparison.

Everything is expressed in SI units at the case boundary (metres, seconds, Pa)
and in lattice units inside the solver; ``LatticeScaling`` carries the
conversion.  The one non-similarity of the model is the viscosity: the lattice
cannot reach the LNG Reynolds number, so every case reports the equivalent
numerical viscosity and the resulting over-damping (see ``similarity_report``).
"""

from __future__ import annotations

from .audit import (
    energy_budget,
    gci,
    gci_family,
    mechanical_energy,
    richardson_observed_order,
    viscous_dissipation_2d,
)
from .fatigue import (
    IIW_FAT80,
    SnCurve,
    dirlik_damage,
    dirlik_range_pdf,
    narrowband_damage,
    rainflow_cycles,
    rainflow_damage,
    spectral_moments,
    stress_from_pressure,
    wideband_bandwidth_parameter,
)
from .impact import (
    LNG_SOUND_SPEED,
    MLM_BETA_LIMIT_DEG,
    WagnerImpact,
    dimensionless_anchors,
    impact_pressure_anchors,
    mlm_bounded_pressure,
    mlm_stagnation_pressure,
    stagnation_pressure,
    wagner_complex_potential,
    wagner_contact_time,
    wagner_plate_potential,
    wagner_pressure_profile,
    wagner_slam_force_per_length,
    wagner_wetted_halfwidth,
    water_hammer_pressure,
)
from .tank import (
    DEFAULT_FILLS,
    LNG_PROPERTIES,
    LNGProperties,
    MembraneTank174k,
    PanelModel,
    scaling_for_case,
    similarity_report,
)

__all__ = [
    "DEFAULT_FILLS",
    "IIW_FAT80",
    "LNG_PROPERTIES",
    "LNG_SOUND_SPEED",
    "MLM_BETA_LIMIT_DEG",
    "LNGProperties",
    "MembraneTank174k",
    "PanelModel",
    "SnCurve",
    "WagnerImpact",
    "dimensionless_anchors",
    "dirlik_damage",
    "dirlik_range_pdf",
    "energy_budget",
    "gci",
    "gci_family",
    "impact_pressure_anchors",
    "mechanical_energy",
    "mlm_bounded_pressure",
    "mlm_stagnation_pressure",
    "narrowband_damage",
    "rainflow_cycles",
    "rainflow_damage",
    "richardson_observed_order",
    "scaling_for_case",
    "similarity_report",
    "spectral_moments",
    "stagnation_pressure",
    "stress_from_pressure",
    "viscous_dissipation_2d",
    "wagner_complex_potential",
    "wagner_contact_time",
    "wagner_plate_potential",
    "wagner_pressure_profile",
    "wagner_slam_force_per_length",
    "wagner_wetted_halfwidth",
    "water_hammer_pressure",
    "wideband_bandwidth_parameter",
]