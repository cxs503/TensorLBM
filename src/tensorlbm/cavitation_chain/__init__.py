"""Cavitation + ventilation validation chain (mission P151W1-N04).

End-to-end physics chain: Kunz pressure-driven mass transfer ->
sheet/cloud transition -> blade-element thrust derating -> critical numbers.

Sub-modules
-----------
``sigma``        cavitation/ventilation numbers, regimes, power-law fitting
``kunz``         Kunz (2000) mass-transfer closure + 0-D transport
``section``      panel suction pressure, isobaric cavity, section lift loss
``sheet_cloud``  sheet-to-cloud transition number, re-entrant-jet shedding
``thrust``       blade-element ``K_T`` derating and the 5 %-loss critical number
``freecavity``   closed (Riabouchinsky) free-streamline cavity: reported WITHHELD
``validation``   acceptance criteria of the node, writes the artefact JSON
"""

from __future__ import annotations

from .kunz import (
    Kunz0D,
    KunzCoefficients,
    kunz_mass_transfer,
    kunz_timescale,
    steady_state_vapour_fraction,
)
from .section import SectionPressure, inception_sweep, isobaric_cavity, lift_loss, suction_pressure
from .sheet_cloud import SheddingEstimate, classify, critical_sigma, shedding, strouhal_band
from .sigma import (
    PowerLaw,
    cavitation_number,
    fit_power_law,
    inception_sigma,
    leading_edge_radius_ratio,
    regime,
    sigma_from_depth,
    ventilation_number,
    vapour_pressure,
)
from .thrust import (
    PropellerDesign,
    blade_element_solution,
    derating_curve,
    sigma_critical_thrust,
    thrust_derating,
)

__all__ = [
    "Kunz0D",
    "KunzCoefficients",
    "PowerLaw",
    "PropellerDesign",
    "SectionPressure",
    "SheddingEstimate",
    "blade_element_solution",
    "cavitation_number",
    "classify",
    "critical_sigma",
    "derating_curve",
    "fit_power_law",
    "inception_sigma",
    "inception_sweep",
    "isobaric_cavity",
    "kunz_mass_transfer",
    "kunz_timescale",
    "leading_edge_radius_ratio",
    "lift_loss",
    "regime",
    "shedding",
    "sigma_critical_thrust",
    "sigma_from_depth",
    "steady_state_vapour_fraction",
    "strouhal_band",
    "suction_pressure",
    "thrust_derating",
    "ventilation_number",
    "vapour_pressure",
]
