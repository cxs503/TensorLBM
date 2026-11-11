"""Lid-driven cavity verification for TensorLBM D2Q9.

Classic benchmark: square cavity with moving lid, validated against
Ghia et al. (1982) JCP 48:387-411 DNS reference data.

Provides:
- Cavity solver (D2Q9 BGK, bounce-back walls, moving lid)
- Ghia 1982 reference data (Re=100, 400, 1000, 3200, 5000, 7500, 10000)
- Verification gates (velocity profile comparison)
- Grid convergence study
"""

from .solver import (
    CavityConfig,
    CavitySolver,
    run_cavity,
)
from .reference_data import (
    GhiaReference,
    get_ghia_reference,
    GHIA_RE100,
    GHIA_RE400,
    GHIA_RE1000,
)
from .verification import (
    VerificationResult,
    run_single_case,
    run_grid_convergence,
)

__all__ = [
    "CavityConfig",
    "CavitySolver",
    "run_cavity",
    "GhiaReference",
    "get_ghia_reference",
    "GHIA_RE100",
    "GHIA_RE400",
    "GHIA_RE1000",
    "VerificationResult",
    "run_single_case",
    "run_grid_convergence",
]
