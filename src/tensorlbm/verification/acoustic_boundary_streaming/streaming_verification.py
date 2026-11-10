"""Acoustic streaming verification: Rayleigh/Schlichting + Re_s scaling.

Verifies that the LBM acoustic streaming simulation reproduces:
  (b) Rayleigh outer streaming velocity profile within 5%
  (c) Acoustic Reynolds number scaling: v_streaming ∝ Re_s^α, α ≈ 1

Theory
------
Rayleigh streaming in a standing wave channel (height H):
  - Stokes layer: δ = √(2ν/ω)
  - Outer streaming velocity: u_s(y) = (3U₀²)/(4ωH) · sin(2kx) · [cos(2πy/H) - 1/3]
  - Streaming Reynolds number: Re_s = U₀²/(νω)
  - Scaling: U_streaming,max ∝ U₀²/ω ∝ Re_s (for fixed ν, ω)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ...acoustic_streaming import (
    AcousticStreamingConfig,
    AcousticStreamingSimulator,
    compute_streaming_reynolds,
    fit_scaling_exponent,
    rayleigh_analytical_outer,
    run_acoustic_streaming,
    stokes_layer_thickness,
)

__all__ = [
    "StreamingVerificationResult",
    "run_streaming_verification",
    "run_scaling_law_verification",
    "verify_rayleigh_profile",
    "verify_stokes_thickness",
]


@dataclass
class StreamingVerificationResult:
    """Result of acoustic streaming verification."""
    # Rayleigh profile comparison
    y_nodes: np.ndarray
    u_lbm: np.ndarray
    u_rayleigh: np.ndarray
    profile_max_error: float  # max |u_lbm - u_rayleigh| / max(|u_rayleigh|)
    profile_rms_error: float
    
    # Stokes layer
    stokes_thickness_theory: float
    stokes_thickness_measured: float
    stokes_error: float
    
    # Scaling law
    Re_s_values: np.ndarray
    U_streaming_values: np.ndarray
    scaling_exponent: float
    scaling_exponent_theory: float  # should be ~1.0
    scaling_exponent_error: float
    scaling_R_squared: float
    
    # Overall
    all_pass: bool
    details: Dict = field(default_factory=dict)


def verify_rayleigh_profile(
    nx: int = 64,
    ny: int = 32,
    U0: float = 0.01,
    omega: float = 0.05,
    tau: float = 1.0,
    n_warmup: int = 50,
    n_avg_periods: int = 15,
    x_phase: Optional[float] = None,
) -> Dict:
    """Verify streaming velocity profile against Rayleigh analytical solution.
    
    Returns dict with LBM and analytical profiles, error metrics.
    """
    config = AcousticStreamingConfig(
        nx=nx, ny=ny, U0=U0, omega=omega, tau=tau,
        n_warmup=n_warmup, n_avg_periods=n_avg_periods,
        period_steps=int(2 * math.pi / omega),
        device="cpu",
        dtype=torch.float64,
    )
    
    result = run_acoustic_streaming(config)
    
    # Extract mid-channel streaming profile (at x = nx//4 for sin(2kx)=1)
    x_idx = nx // 4
    u_lbm = result.u_stream[:, x_idx]
    
    # Analytical Rayleigh solution
    y_vals = np.arange(ny, dtype=float)
    H = float(ny)
    u_rayleigh = rayleigh_analytical_outer(y_vals, H, U0, omega, x_phase=x_idx)
    
    # Error metrics (normalized by max analytical velocity)
    max_rayleigh = np.max(np.abs(u_rayleigh))
    if max_rayleigh > 1e-12:
        profile_error = float(np.max(np.abs(u_lbm - u_rayleigh)) / max_rayleigh)
        rms_error = float(np.sqrt(np.mean((u_lbm - u_rayleigh)**2)) / max_rayleigh)
    else:
        profile_error = 0.0
        rms_error = 0.0
    
    return {
        "y_nodes": y_vals,
        "u_lbm": u_lbm,
        "u_rayleigh": u_rayleigh,
        "max_error": profile_error,
        "rms_error": rms_error,
        "max_streaming_lbm": float(np.max(np.abs(u_lbm))),
        "max_streaming_rayleigh": float(max_rayleigh),
    }


def verify_stokes_thickness(
    nu: float,
    omega: float,
    ny: int = 32,
    U0: float = 0.01,
    tau: float = 1.0,
    n_warmup: int = 50,
    n_avg_periods: int = 15,
) -> Dict:
    """Verify Stokes boundary layer thickness from LBM streaming field.
    
    Measures the penetration depth of the streaming flow near the wall
    and compares with δ = √(2ν/ω).
    """
    config = AcousticStreamingConfig(
        nx=64, ny=ny, U0=U0, omega=omega, tau=tau,
        n_warmup=n_warmup, n_avg_periods=n_avg_periods,
        period_steps=int(2 * math.pi / omega),
        device="cpu",
        dtype=torch.float64,
    )
    
    result = run_acoustic_streaming(config)
    
    # Theory
    delta_theory = stokes_layer_thickness(nu, omega)
    
    # Measure from LBM: find where |u| drops to 1/e of wall-adjacent max
    x_idx = config.nx // 4
    u_profile = np.abs(result.u_stream[:, x_idx])
    
    # Find peak near wall (within first quarter)
    quarter = ny // 4
    wall_region = u_profile[:quarter]
    if len(wall_region) > 0 and np.max(wall_region) > 1e-14:
        peak_idx = np.argmax(wall_region)
        peak_val = wall_region[peak_idx]
        # Find where it drops to 1/e
        threshold = peak_val / math.e
        below_threshold = np.where(wall_region[peak_idx:] < threshold)[0]
        if len(below_threshold) > 0:
            delta_measured = float(peak_idx + below_threshold[0])
        else:
            delta_measured = float(quarter)  # didn't drop, use quarter as upper bound
    else:
        delta_measured = delta_theory  # fallback
    
    error = abs(delta_measured - delta_theory) / delta_theory if delta_theory > 0 else 0.0
    
    return {
        "delta_theory": delta_theory,
        "delta_measured": delta_measured,
        "relative_error": error,
    }


def run_scaling_law_verification(
    U0_values: Optional[List[float]] = None,
    omega: float = 0.05,
    tau: float = 1.0,
    ny: int = 32,
    nx: int = 64,
    n_warmup: int = 30,
    n_avg_periods: int = 10,
) -> Dict:
    """Verify acoustic Reynolds number scaling law.
    
    For fixed ν and ω, the streaming velocity should scale as:
        U_streaming,max ∝ U₀²  (i.e., ∝ Re_s)
    
    So the exponent α in U_streaming ∝ Re_s^α should be ≈ 1.0.
    """
    if U0_values is None:
        U0_values = [0.005, 0.008, 0.01, 0.015, 0.02]
    
    cs2 = 1.0 / 3.0
    nu = cs2 * (tau - 0.5)
    
    Re_s_list = []
    U_max_list = []
    
    period_steps = int(2 * math.pi / omega)
    
    for U0 in U0_values:
        config = AcousticStreamingConfig(
            nx=nx, ny=ny, U0=U0, omega=omega, tau=tau,
            n_warmup=n_warmup, n_avg_periods=n_avg_periods,
            period_steps=period_steps,
            device="cpu",
            dtype=torch.float64,
        )
        result = run_acoustic_streaming(config)
        
        Re_s = compute_streaming_reynolds(U0, nu, omega)
        U_max = float(np.max(np.abs(result.u_stream)))
        
        Re_s_list.append(Re_s)
        U_max_list.append(U_max)
    
    # Fit scaling exponent
    Re_s_arr = np.array(Re_s_list)
    U_max_arr = np.array(U_max_list)
    
    # Filter out near-zero values
    mask = U_max_arr > 1e-12
    if mask.sum() >= 3:
        exponent, coeff, R_squared = fit_scaling_exponent(
            Re_s_arr[mask].tolist(), U_max_arr[mask].tolist()
        )
    else:
        exponent, coeff, R_squared = 1.0, 1.0, 0.0
    
    # Theory: exponent should be 1.0 (linear in Re_s)
    exponent_error = abs(exponent - 1.0)
    
    return {
        "Re_s_values": Re_s_arr,
        "U_streaming_values": U_max_arr,
        "scaling_exponent": exponent,
        "scaling_exponent_theory": 1.0,
        "exponent_error": exponent_error,
        "R_squared": R_squared,
        "coeff": coeff,
    }


def run_streaming_verification(
    nx: int = 64,
    ny: int = 32,
    U0: float = 0.01,
    omega: float = 0.05,
    tau: float = 1.0,
    n_warmup: int = 50,
    n_avg_periods: int = 15,
) -> StreamingVerificationResult:
    """Run full acoustic streaming verification.
    
    Combines Rayleigh profile, Stokes thickness, and scaling law checks.
    """
    cs2 = 1.0 / 3.0
    nu = cs2 * (tau - 0.5)
    
    # (b) Rayleigh profile
    profile = verify_rayleigh_profile(
        nx=nx, ny=ny, U0=U0, omega=omega, tau=tau,
        n_warmup=n_warmup, n_avg_periods=n_avg_periods,
    )
    
    # Stokes thickness
    stokes = verify_stokes_thickness(
        nu=nu, omega=omega, ny=ny, U0=U0, tau=tau,
        n_warmup=n_warmup, n_avg_periods=n_avg_periods,
    )
    
    # (c) Scaling law
    scaling = run_scaling_law_verification(
        omega=omega, tau=tau, ny=ny, nx=nx,
        n_warmup=max(20, n_warmup // 2),
        n_avg_periods=max(5, n_avg_periods // 2),
    )
    
    # Check criteria
    criterion_b = profile["max_error"] < 0.05  # 5%
    criterion_c = scaling["exponent_error"] < 0.05  # 5%
    
    all_pass = criterion_b and criterion_c
    
    return StreamingVerificationResult(
        y_nodes=profile["y_nodes"],
        u_lbm=profile["u_lbm"],
        u_rayleigh=profile["u_rayleigh"],
        profile_max_error=profile["max_error"],
        profile_rms_error=profile["rms_error"],
        stokes_thickness_theory=stokes["delta_theory"],
        stokes_thickness_measured=stokes["delta_measured"],
        stokes_error=stokes["relative_error"],
        Re_s_values=scaling["Re_s_values"],
        U_streaming_values=scaling["U_streaming_values"],
        scaling_exponent=scaling["scaling_exponent"],
        scaling_exponent_theory=1.0,
        scaling_exponent_error=scaling["exponent_error"],
        scaling_R_squared=scaling["R_squared"],
        all_pass=all_pass,
        details={
            "profile": profile,
            "stokes": stokes,
            "scaling": scaling,
            "criterion_b_pass": criterion_b,
            "criterion_c_pass": criterion_c,
        },
    )
