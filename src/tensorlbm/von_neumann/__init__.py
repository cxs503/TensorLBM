"""Von Neumann linear stability analysis for LBM collision operators.

This package provides:
- Analytical amplification matrices for D2Q9 BGK and MRT
- Spectral radius computation over (θ, τ, Ma) grids
- Experimental growth rate measurement from LBM simulations
- Equivalent equation (Chapman-Enskog) analysis
- Stability domain mapping

References:
  - Lallemand & Luo (2000), JCP 159, 1-37
  - Dubois & Lallemand (2008), JCP 227, 4600-4625
  - Malaspinas (2013), PhD thesis EPFL
"""

from .amplification import (
    AmplificationResult,
    compute_spectral_radius,
    scan_stability_domain,
)
from .d2q9_bgk import D2Q9BGKAmplification
from .d2q9_mrt import D2Q9MRTAmplification
from .equivalent_equations import EquivalentEquationsD2Q9
from .experimental import ExperimentalGrowthRate

__all__ = [
    "AmplificationResult",
    "D2Q9BGKAmplification",
    "D2Q9MRTAmplification",
    "EquivalentEquationsD2Q9",
    "ExperimentalGrowthRate",
    "compute_spectral_radius",
    "scan_stability_domain",
]
