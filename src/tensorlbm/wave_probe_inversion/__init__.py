"""Wave-probe inversion identifiability study.

This module establishes the *identifiability law* for wave-probe inversion:
given a probe protocol (number of probes, spacing, propagation length),
what is the achievable accuracy for reflection coefficient R, wavenumber k_r,
and viscous attenuation alpha?

Submodules
----------
synthetic_tank
    Analytical probe synthesis: Stokes 1st+2nd order + reflection + viscous decay.
inversion
    Inversion methods: two-point, three-point Goda, multi-probe least squares.
identifiability
    Identifiability scan over protocol parameters.
lbm_holdout
    Real LBM hold-out test against the TensorLBM wave tank.
stability_envelope
    Viscosity-stability envelope: N_max(nu).
"""

from .synthetic_tank import SyntheticWaveTank
from .inversion import (
    TwoPointInversion,
    ThreePointGodaInversion,
    MultiProbeLeastSquares,
    InversionResult,
)
from .identifiability import IdentifiabilityScan, IdentifiabilityResult

__all__ = [
    "SyntheticWaveTank",
    "TwoPointInversion",
    "ThreePointGodaInversion",
    "MultiProbeLeastSquares",
    "InversionResult",
    "IdentifiabilityScan",
    "IdentifiabilityResult",
]
