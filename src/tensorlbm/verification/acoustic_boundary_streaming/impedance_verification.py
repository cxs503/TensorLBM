"""Impedance boundary verification: Delany-Bazley + TMM reflection coefficient.

Verifies that the LBM impedance boundary condition reproduces the analytical
TMM reflection coefficient for porous absorbers within 3% error.

Acceptance criteria addressed:
  (a) |R_LBM - R_TMM| / |R_TMM| < 3% across frequency band
  (d) Rigid wall limit: |R| -> 1 as Z_surf -> infinity
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...acoustics_porous_absorber import (
    DEFAULT_C0,
    DEFAULT_RHO0,
    Layer,
    Lining,
    absorption_coefficient,
    delany_bazley,
    measure_reflection,
    reflection_coefficient,
    surface_impedance,
    sweep_absorption,
)

__all__ = [
    "ImpedanceVerificationResult",
    "run_impedance_verification",
    "verify_rigid_wall_limit",
    "verify_single_layer_reflection",
    "verify_multilayer_reflection",
]


@dataclass
class ImpedanceVerificationResult:
    """Result of impedance boundary verification."""
    freqs: np.ndarray
    R_lbm: np.ndarray  # complex
    R_tmm: np.ndarray  # complex
    alpha_lbm: np.ndarray
    alpha_tmm: np.ndarray
    max_R_error: float  # max |R_lbm - R_tmm| / |R_tmm|
    mean_R_error: float
    max_alpha_error: float
    rigid_wall_R: complex
    rigid_wall_error: float  # |1 - |R_rigid||
    all_pass: bool
    details: Dict = field(default_factory=dict)


def verify_single_layer_reflection(
    freqs: Sequence[float],
    sigma: float = 25000.0,
    thickness: float = 0.05,
    rho0: float = DEFAULT_RHO0,
    c0: float = DEFAULT_C0,
) -> Dict:
    """Verify reflection coefficient for single Delany-Bazley layer.

    Returns dict with R_lbm, R_tmm, errors at each frequency.
    """
    lining = Lining([Layer(thickness, sigma)])
    result = sweep_absorption(lining, freqs, c0=c0, rho0=rho0)
    
    R_lbm = result["R_lbm"]
    R_tmm = result["R_tmm"]
    
    # Relative error (avoiding division by zero)
    mag_tmm = np.abs(R_tmm)
    mask = mag_tmm > 0.01  # only compare where TMM reflection is significant
    if mask.any():
        rel_errors = np.abs(R_lbm[mask] - R_tmm[mask]) / mag_tmm[mask]
        max_error = float(np.max(rel_errors))
        mean_error = float(np.mean(rel_errors))
    else:
        max_error = 0.0
        mean_error = 0.0
    
    return {
        "freqs": result["freqs"],
        "R_lbm": R_lbm,
        "R_tmm": R_tmm,
        "alpha_lbm": result["alpha_lbm"],
        "alpha_tmm": result["alpha_tmm"],
        "max_R_error": max_error,
        "mean_R_error": mean_error,
    }


def verify_multilayer_reflection(
    freqs: Sequence[float],
    layers: List[Tuple[float, float]],  # (thickness, sigma) pairs
    rho0: float = DEFAULT_RHO0,
    c0: float = DEFAULT_C0,
) -> Dict:
    """Verify reflection for multilayer lining."""
    py_layers = [Layer(d, sig) for d, sig in layers]
    lining = Lining(py_layers)
    result = sweep_absorption(lining, freqs, c0=c0, rho0=rho0)
    
    R_lbm = result["R_lbm"]
    R_tmm = result["R_tmm"]
    
    mag_tmm = np.abs(R_tmm)
    mask = mag_tmm > 0.01
    if mask.any():
        rel_errors = np.abs(R_lbm[mask] - R_tmm[mask]) / mag_tmm[mask]
        max_error = float(np.max(rel_errors))
    else:
        max_error = 0.0
    
    return {
        "freqs": result["freqs"],
        "R_lbm": R_lbm,
        "R_tmm": R_tmm,
        "max_R_error": max_error,
    }


def verify_rigid_wall_limit(
    freq: float = 500.0,
    rho0: float = DEFAULT_RHO0,
    c0: float = DEFAULT_C0,
) -> Dict:
    """Negative control: rigid wall should give |R| = 1.

    A rigid wall has Z_surf -> infinity, so R = (Z-Z0)/(Z+Z0) -> 1.
    """
    # Very high impedance approximates rigid wall
    Z_rigid = complex(1e6, 0)  # effectively rigid
    result = measure_reflection(Z_rigid, freq, c0=c0, rho0=rho0)
    
    R_meas = result["R_lbm"]
    error = abs(1.0 - abs(R_meas))
    
    return {
        "R_rigid": R_meas,
        "abs_R": abs(R_meas),
        "error_from_unity": error,
    }


def run_impedance_verification(
    freqs: Optional[Sequence[float]] = None,
    sigma: float = 25000.0,
    thickness: float = 0.05,
) -> ImpedanceVerificationResult:
    """Run full impedance boundary verification suite.
    
    Parameters
    ----------
    freqs : array of frequencies to test (Hz)
    sigma : flow resistivity (Pa*s/m^2)
    thickness : layer thickness (m)
    
    Returns
    -------
    ImpedanceVerificationResult with all metrics
    """
    if freqs is None:
        freqs = np.array([200, 400, 600, 800, 1000, 1500, 2000])
    
    # Single layer verification
    single = verify_single_layer_reflection(freqs, sigma, thickness)
    
    # Rigid wall limit
    rigid = verify_rigid_wall_limit(freq=800.0)
    
    # Determine pass/fail
    max_R_err = single["max_R_error"]
    rigid_err = rigid["error_from_unity"]
    
    # Criterion (a): R error < 3%
    criterion_a = max_R_err < 0.03
    # Criterion (d): rigid wall |R| -> 1 (within 1%)
    criterion_d = rigid_err < 0.01
    
    all_pass = criterion_a and criterion_d
    
    return ImpedanceVerificationResult(
        freqs=single["freqs"],
        R_lbm=single["R_lbm"],
        R_tmm=single["R_tmm"],
        alpha_lbm=single["alpha_lbm"],
        alpha_tmm=single["alpha_tmm"],
        max_R_error=max_R_err,
        mean_R_error=single["mean_R_error"],
        max_alpha_error=float(np.max(np.abs(
            single["alpha_lbm"] - single["alpha_tmm"]
        ))),
        rigid_wall_R=rigid["R_rigid"],
        rigid_wall_error=rigid_err,
        all_pass=all_pass,
        details={
            "single_layer": single,
            "rigid_wall": rigid,
            "criterion_a_pass": criterion_a,
            "criterion_d_pass": criterion_d,
        },
    )
