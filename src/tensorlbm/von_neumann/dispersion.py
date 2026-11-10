"""Dispersion and dissipation analysis for LBM.

Computes ω(k) curves and fits the numerical viscosity from the dissipation rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy.optimize import curve_fit

from .analysis import AmplificationMatrix, eigenvalues_to_omega, identify_physical_modes
from .collision import BGKCollision, MRTCollision, TRTCollision
from .lattice import LatticeBase


@dataclass
class DispersionResult:
    """Result of dispersion analysis along a wavevector path."""

    k_magnitudes: np.ndarray  # |k| values
    omega_re_acoustic: np.ndarray  # Re(ω) for acoustic mode
    omega_im_acoustic: np.ndarray  # Im(ω) for acoustic mode
    omega_im_shear: np.ndarray  # Im(ω) for shear mode
    all_eigenvalues: list  # all eigenvalues at each k
    k_vectors: list  # k vectors used


def compute_dispersion_curve(
    lattice: LatticeBase,
    collision: Union[BGKCollision, TRTCollision, MRTCollision],
    k_direction: np.ndarray,
    k_max: float = np.pi,
    n_points: int = 100,
) -> DispersionResult:
    """Compute dispersion relation along a given direction in k-space.

    Args:
        lattice: lattice definition
        collision: collision operator
        k_direction: unit vector in k-space (will be normalized)
        k_max: maximum |k| (default π for Nyquist)
        n_points: number of k points

    Returns:
        DispersionResult with ω(k) for physical modes
    """
    # Normalize direction
    k_dir = k_direction / np.linalg.norm(k_direction)

    # k magnitudes (avoid k=0 exactly)
    k_mags = np.linspace(1e-6, k_max, n_points)

    k_vectors = []
    all_eigenvalues = []
    omega_re_acoustic = []
    omega_im_acoustic = []
    omega_im_shear = []

    for km in k_mags:
        k = km * k_dir
        k_vectors.append(k)

        # Compute amplification matrix and eigenvalues
        amp = AmplificationMatrix(lattice, collision)
        M = amp(k)
        eigs = np.linalg.eigvals(M)
        all_eigenvalues.append(eigs)

        # Identify physical modes (pass k for better identification)
        modes = identify_physical_modes(eigs, lattice, k)
        omega = modes["omega"]

        # Acoustic mode: largest |Re(ω)| among physical modes
        if len(modes["acoustic"]) > 0:
            acoustic_idx = modes["acoustic"][np.argmax(np.abs(omega[modes["acoustic"]].real))]
            omega_re_acoustic.append(omega[acoustic_idx].real)
            omega_im_acoustic.append(omega[acoustic_idx].imag)
        else:
            omega_re_acoustic.append(np.nan)
            omega_im_acoustic.append(np.nan)

        # Shear mode: physical mode with Re(ω) ≈ 0
        if len(modes["shear"]) > 0:
            shear_idx = modes["shear"][np.argmax(omega[modes["shear"]].imag)]  # least damped
            omega_im_shear.append(omega[shear_idx].imag)
        else:
            omega_im_shear.append(np.nan)

    return DispersionResult(
        k_magnitudes=k_mags,
        omega_re_acoustic=np.array(omega_re_acoustic),
        omega_im_acoustic=np.array(omega_im_acoustic),
        omega_im_shear=np.array(omega_im_shear),
        all_eigenvalues=all_eigenvalues,
        k_vectors=k_vectors,
    )


def compute_dissipation_curve(
    lattice: LatticeBase,
    collision: Union[BGKCollision, TRTCollision, MRTCollision],
    k_direction: np.ndarray,
    k_max: float = 0.5,  # small k for viscous regime
    n_points: int = 50,
) -> tuple:
    """Compute dissipation rate Im(ω) vs k² for viscosity fitting.

    Args:
        lattice: lattice definition
        collision: collision operator
        k_direction: direction in k-space
        k_max: maximum |k| (keep small for linear viscous regime)
        n_points: number of points

    Returns:
        (k_squared, omega_im_shear): arrays for fitting
    """
    result = compute_dispersion_curve(lattice, collision, k_direction, k_max, n_points)
    k_squared = result.k_magnitudes ** 2
    return k_squared, result.omega_im_shear


def fit_viscosity_from_dissipation(
    lattice: LatticeBase,
    collision: Union[BGKCollision, TRTCollision, MRTCollision],
    k_direction: np.ndarray,
    k_max: float = 0.3,
    n_points: int = 30,
) -> dict:
    """Fit numerical viscosity from dissipation rate.

    For small k, the shear mode has:
        Im(ω) ≈ -ν_num k²

    We fit this linear relation to extract ν_num.

    Args:
        lattice: lattice definition
        collision: collision operator
        k_direction: direction in k-space
        k_max: maximum |k| for fitting
        n_points: number of points

    Returns:
        dict with:
            - nu_num: numerical viscosity
            - nu_theory: theoretical viscosity from collision operator
            - relative_error: |nu_num - nu_theory| / nu_theory
            - k_squared: k² values used
            - omega_im: Im(ω) values
            - fit_quality: R² of the fit
    """
    k_squared, omega_im = compute_dissipation_curve(
        lattice, collision, k_direction, k_max, n_points
    )

    # Remove NaN values
    valid = ~np.isnan(omega_im)
    k_squared = k_squared[valid]
    omega_im = omega_im[valid]

    if len(k_squared) < 3:
        return {
            "nu_num": np.nan,
            "nu_theory": collision.nu,
            "relative_error": np.nan,
            "k_squared": k_squared,
            "omega_im": omega_im,
            "fit_quality": 0.0,
            "fit_range": "insufficient data",
        }

    # Fit: omega_im = -nu * k^2
    # Use only small k (first half) for better linear fit
    n_fit = min(len(k_squared) // 2, 15)
    k2_fit = k_squared[:n_fit]
    omega_fit = omega_im[:n_fit]

    # Linear fit: omega_im = a * k^2
    def linear_model(k2, nu):
        return -nu * k2

    popt, pcov = curve_fit(linear_model, k2_fit, omega_fit, p0=[collision.nu])
    nu_num = popt[0]

    # Compute R²
    omega_pred = linear_model(k2_fit, nu_num)
    ss_res = np.sum((omega_fit - omega_pred) ** 2)
    ss_tot = np.sum((omega_fit - np.mean(omega_fit)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    nu_theory = collision.nu
    rel_error = abs(nu_num - nu_theory) / nu_theory

    return {
        "nu_num": nu_num,
        "nu_theory": nu_theory,
        "relative_error": rel_error,
        "k_squared": k_squared,
        "omega_im": omega_im,
        "fit_quality": r_squared,
        "fit_range": f"k ∈ [0, {k_max}], using first {n_fit} points",
    }
