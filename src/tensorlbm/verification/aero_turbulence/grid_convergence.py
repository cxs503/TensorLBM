"""Grid Convergence Index (GCI) analysis for verification studies.

Implements the ASME V&V 20 standard methodology for estimating
discretization error and grid convergence.

Reference:
- Roache, P.J. (1998). "Verification and Validation in Computational
  Science and Engineering." Hermosa Publishers.
- ASME V&V 20-2009: "Standard for Verification and Validation in
  Computational Fluid Dynamics and Heat Transfer."
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GCIResult:
    """Result of GCI analysis."""
    phi_fine: float       # Value on finest grid
    phi_medium: float     # Value on medium grid
    phi_coarse: float     # Value on coarse grid
    r: float              # Grid refinement ratio
    p: float              # Observed order of convergence
    e_fine: float         # Relative error on finest grid (%)
    gci_fine: float       # GCI on finest grid (%)
    gci_medium: float     # GCI on medium grid (%)
    convergence_status: str  # "converged", "oscillatory", "divergent"


def compute_gci(
    phi: list[float],
    grid_sizes: list[int],
    safety_factor: float = 1.25,
) -> GCIResult:
    """Compute Grid Convergence Index from three grid levels.
    
    Parameters
    ----------
    phi : list of float
        Solution values [coarse, medium, fine] (ordered coarse to fine).
    grid_sizes : list of int
        Characteristic grid sizes [N_coarse, N_medium, N_fine].
        Typically N = sqrt(nx*ny) for 2D or (nx*ny*nz)^(1/3) for 3D.
    safety_factor : float
        Safety factor F_s (1.25 for 3 grids, 3.0 for 2 grids).
    
    Returns
    -------
    GCIResult with convergence metrics.
    """
    if len(phi) != 3 or len(grid_sizes) != 3:
        raise ValueError("Need exactly 3 grid levels")
    
    phi_coarse, phi_medium, phi_fine = phi
    N_coarse, N_medium, N_fine = grid_sizes
    
    # Grid refinement ratio
    r21 = (N_medium / N_fine) ** 0.5  # medium to fine
    r32 = (N_coarse / N_medium) ** 0.5  # coarse to medium
    r = (r21 + r32) / 2.0  # average refinement ratio
    if r < 1.01:
        r = 2.0  # default if grids are too similar
    
    # Differences
    eps21 = phi_medium - phi_fine  # medium - fine
    eps32 = phi_coarse - phi_medium  # coarse - medium
    
    # Check for convergence (all values essentially the same)
    if abs(eps21) < 1e-12 and abs(eps32) < 1e-12:
        return GCIResult(
            phi_fine=phi_fine,
            phi_medium=phi_medium,
            phi_coarse=phi_coarse,
            r=r,
            p=2.0,
            e_fine=0.0,
            gci_fine=0.0,
            gci_medium=0.0,
            convergence_status="converged",
        )
    
    # Check for oscillatory convergence (different signs)
    if eps21 * eps32 < 0:
        # Oscillatory convergence
        e_fine = abs(eps21) / max(abs(phi_fine), 1e-15) * 100.0
        gci_fine = safety_factor * e_fine / (r**2 - 1)
        return GCIResult(
            phi_fine=phi_fine,
            phi_medium=phi_medium,
            phi_coarse=phi_coarse,
            r=r,
            p=0.0,  # undefined for oscillatory
            e_fine=e_fine,
            gci_fine=gci_fine,
            gci_medium=safety_factor * abs(eps32) / max(abs(phi_medium), 1e-15) * 100.0,
            convergence_status="oscillatory",
        )
    
    # Monotonic convergence: compute observed order
    ratio = abs(eps32 / eps21)
    if ratio > 1e-10 and r > 1.01:
        # p = ln(|eps32/eps21|) / ln(r)
        p = math.log(ratio) / math.log(r)
        p = max(0.05, min(p, 10.0))  # bound to reasonable range
    else:
        p = 2.0  # default to design order
    
    # Relative error on finest grid
    e_fine = abs(eps21) / max(abs(phi_fine), 1e-15) * 100.0
    
    # GCI on finest grid
    denom = r**p - 1.0
    if abs(denom) < 1e-10:
        denom = 1.0  # fallback
    gci_fine = safety_factor * e_fine / denom
    
    # GCI on medium grid
    e_medium = abs(eps32) / max(abs(phi_medium), 1e-15) * 100.0
    gci_medium = safety_factor * e_medium
    
    # Convergence status
    if gci_fine < 1.0:
        status = "converged"
    elif gci_fine < 5.0:
        status = "marginal"
    else:
        status = "not_converged"
    
    return GCIResult(
        phi_fine=phi_fine,
        phi_medium=phi_medium,
        phi_coarse=phi_coarse,
        r=r,
        p=p,
        e_fine=e_fine,
        gci_fine=gci_fine,
        gci_medium=gci_medium,
        convergence_status=status,
    )


def grid_convergence_study(
    results: dict[str, list[float]],
    grid_sizes: list[int],
) -> dict[str, GCIResult]:
    """Perform GCI analysis for multiple quantities.
    
    Parameters
    ----------
    results : dict
        {quantity_name: [value_coarse, value_medium, value_fine]}
    grid_sizes : list of int
        [N_coarse, N_medium, N_fine]
    
    Returns
    -------
    dict of {quantity_name: GCIResult}
    """
    gci_results = {}
    for name, values in results.items():
        if len(values) == 3:
            gci_results[name] = compute_gci(values, grid_sizes)
    return gci_results


__all__ = [
    "GCIResult",
    "compute_gci",
    "grid_convergence_study",
]
