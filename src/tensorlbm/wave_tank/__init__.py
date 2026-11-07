"""Numerical wave tank benchmark toolkit (Stokes-2nd generation/absorption).

Modules
-------
``stokes2``   second-order Stokes wave theory in finite depth (lattice units),
              dispersion solver, free-surface and velocity kinematics.
``gauges``    free-surface gauges (phase-field liquid-fraction integration),
              harmonic analysis, Goda & Suzuki two-point reflection split.
``tank``      phase-field LBM wave tank (relaxation-zone wavemaker + absorber)
              built on :class:`tensorlbm.cac_lbm.CACSim`.
``campaign``  kh x eps sweep driver, criteria evaluation, figures.
"""

from __future__ import annotations

from .stokes2 import Stokes2Wave, dispersion_solve
from .gauges import (
    HarmonicFit,
    ReflectionSplit,
    fit_harmonics,
    liquid_fraction,
    phase_speed_from_phase,
    surface_elevation,
    two_probe_split,
)
from .tank import WaveTankConfig, WaveTankResult, run_case

__all__ = [
    "Stokes2Wave",
    "dispersion_solve",
    "HarmonicFit",
    "ReflectionSplit",
    "fit_harmonics",
    "liquid_fraction",
    "phase_speed_from_phase",
    "surface_elevation",
    "two_probe_split",
    "WaveTankConfig",
    "WaveTankResult",
    "run_case",
]