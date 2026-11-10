"""Stability envelope analysis for LBM.

Scans the (τ, Ma) parameter space to determine the stability boundary
where max|λ| ≤ 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from .analysis import AmplificationMatrix
from .collision import BGKCollision, MRTCollision, TRTCollision
from .lattice import LatticeBase


@dataclass
class StabilityResult:
    """Result of stability scan."""

    tau_values: np.ndarray
    max_eigenvalue: np.ndarray  # max|λ| for each τ
    is_stable: np.ndarray  # boolean array
    k_direction: np.ndarray  # direction used for scan


def max_eigenvalue_vs_tau(
    lattice: LatticeBase,
    collision_class: type,
    tau_values: np.ndarray,
    k_direction: np.ndarray,
    k_max: float = np.pi,
    n_k_points: int = 50,
    **collision_kwargs,
) -> StabilityResult:
    """Compute max|λ| as a function of τ for a given k-direction.

    Args:
        lattice: lattice definition
        collision_class: BGKCollision, TRTCollision, or MRTCollision
        tau_values: array of τ values to scan
        k_direction: direction in k-space
        k_max: maximum |k| to check
        n_k_points: number of k points along direction
        **collision_kwargs: additional arguments for collision constructor

    Returns:
        StabilityResult with max|λ| vs τ
    """
    k_dir = k_direction / np.linalg.norm(k_direction)
    k_mags = np.linspace(1e-6, k_max, n_k_points)

    max_eigs = []
    for tau in tau_values:
        # Create collision operator
        if collision_class == BGKCollision:
            coll = collision_class(lattice, tau=tau)
        elif collision_class == TRTCollision:
            coll = collision_class(lattice, tau_plus=tau, **collision_kwargs)
        elif collision_class == MRTCollision:
            # For MRT, set all relaxation times based on tau
            coll = collision_class(
                lattice,
                tau_e=tau,
                tau_eps=tau,
                tau_qj=tau,
                tau_pxx=tau,
                tau_pxy=tau,
            )
        else:
            raise ValueError(f"Unknown collision class: {collision_class}")

        # Find max|λ| over all k
        amp = AmplificationMatrix(lattice, coll)
        max_lambda = 0.0
        for km in k_mags:
            k = km * k_dir
            M = amp(k)
            eigs = np.linalg.eigvals(M)
            max_lambda = max(max_lambda, np.max(np.abs(eigs)))

        max_eigs.append(max_lambda)

    max_eigs = np.array(max_eigs)
    is_stable = max_eigs <= 1.0 + 1e-12

    return StabilityResult(
        tau_values=tau_values,
        max_eigenvalue=max_eigs,
        is_stable=is_stable,
        k_direction=k_direction,
    )


def scan_stability_envelope(
    lattice: LatticeBase,
    collision_class: type,
    tau_range: tuple = (0.5001, 2.0),
    n_tau: int = 50,
    n_k_angles: int = 8,
    n_k_mags: int = 50,
    **collision_kwargs,
) -> dict:
    """Scan the full stability envelope in (τ, k) space.

    Checks multiple k-directions to capture anisotropy.

    Args:
        lattice: lattice definition
        collision_class: collision operator class
        tau_range: (τ_min, τ_max)
        n_tau: number of τ values
        n_k_angles: number of k-directions to check
        n_k_mags: number of k magnitudes per direction
        **collision_kwargs: additional collision parameters

    Returns:
        dict with stability information
    """
    tau_values = np.linspace(tau_range[0], tau_range[1], n_tau)

    # Generate k-directions (uniform in angle for 2D, or specific directions for 3D)
    if lattice.dim == 2:
        angles = np.linspace(0, np.pi / 2, n_k_angles)
        k_directions = [np.array([np.cos(a), np.sin(a)]) for a in angles]
    else:  # 3D
        # Use high-symmetry directions
        k_directions = [
            np.array([1, 0, 0]),
            np.array([1, 1, 0]) / np.sqrt(2),
            np.array([1, 1, 1]) / np.sqrt(3),
            np.array([2, 1, 0]) / np.sqrt(5),
        ]
        if n_k_angles > 4:
            # Add more random directions
            rng = np.random.default_rng(42)
            for _ in range(n_k_angles - 4):
                k_dir = rng.standard_normal(lattice.dim)
                k_dir /= np.linalg.norm(k_dir)
                k_directions.append(k_dir)

    # Scan for each direction
    results_per_direction = []
    for k_dir in k_directions:
        result = max_eigenvalue_vs_tau(
            lattice, collision_class, tau_values, k_dir,
            n_k_points=n_k_mags, **collision_kwargs
        )
        results_per_direction.append(result)

    # Find the worst case (max over all directions)
    max_eigs_all = np.array([r.max_eigenvalue for r in results_per_direction])
    worst_case = np.max(max_eigs_all, axis=0)
    is_stable = worst_case <= 1.0 + 1e-12

    return {
        "tau_values": tau_values,
        "max_eigenvalue_worst_case": worst_case,
        "is_stable": is_stable,
        "results_per_direction": results_per_direction,
        "k_directions": k_directions,
        "stable_tau_range": (
            tau_values[is_stable][0] if np.any(is_stable) else None,
            tau_values[is_stable][-1] if np.any(is_stable) else None,
        ),
    }


def check_stability_at_point(
    lattice: LatticeBase,
    collision: Union[BGKCollision, TRTCollision, MRTCollision],
    k: np.ndarray,
    tolerance: float = 1e-12,
) -> dict:
    """Check stability at a specific (τ, k) point.

    Args:
        lattice: lattice definition
        collision: collision operator
        k: wavevector
        tolerance: tolerance for |λ| ≤ 1 check

    Returns:
        dict with stability information
    """
    amp = AmplificationMatrix(lattice, collision)
    M = amp(k)
    eigs = np.linalg.eigvals(M)
    max_lambda = np.max(np.abs(eigs))

    return {
        "k": k,
        "eigenvalues": eigs,
        "max_abs_lambda": max_lambda,
        "is_stable": max_lambda <= 1.0 + tolerance,
        "tolerance": tolerance,
    }
