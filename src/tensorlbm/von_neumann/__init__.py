"""Von Neumann linear stability analysis for LBM collision operators.

Provides machine-computable dispersion relations ω(k), dissipation Im(ω),
and stability envelopes for BGK, TRT, and MRT operators on D2Q9/D3Q19 lattices.

References:
    - Lallemand & Luo (2000), JCP 161, 649-678.
    - Dubois et al. (2006), Computers & Mathematics with Applications 51.
    - Ginzburg (2006), J. Stat. Phys. 125, 157-199 (TRT magic parameter Λ=3/16).
"""

from .analysis import (
    AmplificationMatrix,
    von_neumann_eigenvalues,
)
from .collision import BGKCollision, TRTCollision, MRTCollision
from .dispersion import (
    compute_dispersion_curve,
    compute_dissipation_curve,
    fit_viscosity_from_dissipation,
)
from .lattice import D2Q9Lattice, D3Q19Lattice, LatticeBase
from .stability import scan_stability_envelope, max_eigenvalue_vs_tau

__all__ = [
    "AmplificationMatrix",
    "von_neumann_eigenvalues",
    "BGKCollision",
    "TRTCollision",
    "MRTCollision",
    "D2Q9Lattice",
    "D3Q19Lattice",
    "LatticeBase",
    "compute_dispersion_curve",
    "compute_dissipation_curve",
    "fit_viscosity_from_dissipation",
    "scan_stability_envelope",
    "max_eigenvalue_vs_tau",
]
