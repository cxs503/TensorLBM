"""High-density-ratio Shan-Chen + enthalpy phase-change fuzed verification surface.

This package is the *fusion* layer for the campaign node
``p348w3-n04-lbm-multiphase-phasechange``: one place where the ice/water
enthalpy phase-change model (Stefan condition, Neumann similarity anchor) and
the high-density-ratio Shan-Chen pseudopotential model (Laplace law up to
100:1) are verified together, plus a Rayleigh-Taylor linear-growth anchor for
the interfacial dynamics of the same two-layer configuration.

Modules
-------
``enthalpy``
    SI-unit finite-volume enthalpy solver (implicit interface, explicit Stefan
    audit) + two-phase Neumann similarity solution with subcooling.
``highdensity``
    Maxwell equal-area construction for the Carnahan-Starling / Peng-Robinson
    EOS and the static-droplet Laplace-law ladder at 10:1 / 50:1 / 100:1.
``rt_anchor``
    Single-mode Rayleigh-Taylor growth-rate anchor (vectorised diagnostics on
    top of :mod:`tensorlbm.rayleigh_taylor`) at three Atwood numbers.
``gates``
    The machine-readable acceptance gates C1-C7 and the verdict assembly.
"""

from .enthalpy import (  # noqa: F401
    ICE_WATER,
    EnthalpyPhaseChange1D,
    NeumannMelting,
    PhaseChangeProperties,
    PhaseChangeStep,
)

__all__ = [
    "ICE_WATER",
    "EnthalpyPhaseChange1D",
    "NeumannMelting",
    "PhaseChangeProperties",
    "PhaseChangeStep",
]
