"""Verification gates for lid-driven cavity against Ghia et al. (1982)."""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np

from .solver import CavityConfig, CavitySolver
from .reference_data import GhiaReference, get_ghia_reference, interpolate_ghia


@dataclasses.dataclass
class VerificationResult:
    """Result of cavity verification."""
    re: float
    n: int
    # L2 error norms
    l2_u: float
    l2_v: float
    # Max error
    max_u: float
    max_v: float
    # Centerline extrema
    u_min: float          # Minimum u on vertical centerline
    u_min_y: float        # y location of u_min
    v_max: float          # Maximum v on horizontal centerline
    v_max_x: float        # x location of v_max
    # Reference values
    ref_u_min: float
    ref_v_max: float
    # Pass/fail
    passed: bool
    details: Dict[str, str] = dataclasses.field(default_factory=dict)


def compute_errors(
    y_sim: np.ndarray, u_sim: np.ndarray,
    x_sim: np.ndarray, v_sim: np.ndarray,
    ref: GhiaReference
) -> Dict[str, float]:
    """Compute error metrics between simulation and Ghia reference."""
    # Interpolate reference at simulation points
    u_ref = interpolate_ghia(ref, y_sim, axis='u')
    v_ref = interpolate_ghia(ref, x_sim, axis='v')

    # L2 errors
    l2_u = np.sqrt(np.mean((u_sim - u_ref)**2))
    l2_v = np.sqrt(np.mean((v_sim - v_ref)**2))

    # Max errors
    max_u = np.max(np.abs(u_sim - u_ref))
    max_v = np.max(np.abs(v_sim - v_ref))

    return {
        'l2_u': l2_u,
        'l2_v': l2_v,
        'max_u': max_u,
        'max_v': max_v,
    }


def run_single_case(
    re: float,
    n: int = 129,
    max_steps: int = 30000,
    tolerance: float = 0.05,
) -> VerificationResult:
    """Run a single cavity case and verify against Ghia.

    Parameters
    ----------
    re : float
        Reynolds number
    n : int
        Grid resolution
    max_steps : int
        Maximum simulation steps
    tolerance : float
        Acceptable L2 error (5% default)

    Returns
    -------
    VerificationResult
    """
    cfg = CavityConfig(n=n, re=re, max_steps=max_steps, u_lid=0.1)
    solver = CavitySolver(cfg)

    # Run simulation
    for step in range(max_steps):
        solver.step()

    # Get profiles
    y_norm, u_norm, x_norm, v_norm = solver.get_profiles()

    # Get reference
    ref = get_ghia_reference(re)

    # Compute errors
    errors = compute_errors(y_norm, u_norm, x_norm, v_norm, ref)

    # Find extrema
    u_min_idx = np.argmin(u_norm)
    v_max_idx = np.argmax(v_norm)

    u_min = float(u_norm[u_min_idx])
    u_min_y = float(y_norm[u_min_idx])
    v_max = float(v_norm[v_max_idx])
    v_max_x = float(x_norm[v_max_idx])

    # Reference extrema
    ref_u_min = float(np.min(ref.u))
    ref_v_max = float(np.max(ref.v))

    # Check pass/fail
    passed = errors['l2_u'] < tolerance and errors['l2_v'] < tolerance

    return VerificationResult(
        re=re,
        n=n,
        l2_u=errors['l2_u'],
        l2_v=errors['l2_v'],
        max_u=errors['max_u'],
        max_v=errors['max_v'],
        u_min=u_min,
        u_min_y=u_min_y,
        v_max=v_max,
        v_max_x=v_max_x,
        ref_u_min=ref_u_min,
        ref_v_max=ref_v_max,
        passed=passed,
        details={
            'u_min_err': f"{abs(u_min - ref_u_min) / abs(ref_u_min) * 100:.1f}%",
            'v_max_err': f"{abs(v_max - ref_v_max) / abs(ref_v_max) * 100:.1f}%",
        }
    )


def run_grid_convergence(
    re: float,
    resolutions: List[int] = None,
    max_steps: int = 30000,
) -> List[VerificationResult]:
    """Run grid convergence study."""
    if resolutions is None:
        resolutions = [33, 65, 129, 257]

    results = []
    for n in resolutions:
        result = run_single_case(re=re, n=n, max_steps=max_steps)
        results.append(result)

    return results
