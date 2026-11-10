"""Phase-averaged analysis for perforated plate oscillatory flow.

Extracts:
    C_d(KC, β) – discharge coefficient
    C_m(KC, β) – added-mass coefficient
    Φ          – energy dissipation per cycle
    ζ_eff      – equivalent damping ratio

Method:
    Phase-average the pressure drop Δp(t) and flow rate Q(t) over
    multiple oscillation cycles. Decompose Δp into inertial and
    dissipative components:

        Δp(t) = C_m · ρ · L · dQ/dt + ½ ρ C_d⁻² (Q/(Aβ))² · sign(Q)

    where:
        L = plate thickness (2 lu)
        A = channel cross-section area (ny - 2 in 2D)
        β = open area ratio
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal


def phase_average(
    time: np.ndarray,
    signal: np.ndarray,
    omega: float,
    n_bins: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-average a time series over oscillation cycles.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    signal : np.ndarray
        Signal to phase-average.
    omega : float
        Angular frequency.
    n_bins : int
        Number of phase bins per cycle.

    Returns
    -------
    phase : np.ndarray
        Phase angles [0, 2π).
    averaged : np.ndarray
        Phase-averaged signal.
    """
    period = 2.0 * math.pi / omega
    phases = (time % period) / period * 2.0 * math.pi

    phase_bins = np.linspace(0, 2 * math.pi, n_bins + 1)
    phase_centers = 0.5 * (phase_bins[:-1] + phase_bins[1:])
    averaged = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phases >= phase_bins[i]) & (phases < phase_bins[i + 1])
        if mask.any():
            averaged[i] = np.mean(signal[mask])

    return phase_centers, averaged


def extract_cd_cm(
    phase_q: np.ndarray,
    phase_dp: np.ndarray,
    omega: float,
    beta: float,
    ny: int,
    rho0: float = 1.0,
    plate_thickness: float = 2.0,
) -> dict:
    """Extract C_d and C_m from phase-averaged Q and Δp.

    Uses least-squares fit to decompose:
        Δp = C_m · ρ · L · dQ/dt + ½ ρ C_d⁻² (Q/(Aβ))² · sign(Q)

    Parameters
    ----------
    phase_q : np.ndarray
        Phase-averaged flow rate.
    phase_dp : np.ndarray
        Phase-averaged pressure drop.
    omega : float
        Angular frequency.
    beta : float
        Open area ratio.
    ny : int
        Channel height in lattice units.
    rho0 : float
        Reference density.
    plate_thickness : float
        Plate thickness in lattice units.

    Returns
    -------
    dict with C_d, C_m, and fit quality.
    """
    A = max(ny - 2, 1)  # cross-section area (2D: height)
    u_hole = phase_q / (A * beta + 1e-15)  # velocity through holes

    # Numerical derivative of Q
    dqdt = np.gradient(phase_q, 2 * math.pi / omega / len(phase_q))

    # Inertial term: ρ · L · dQ/dt
    inertial = rho0 * plate_thickness * dqdt

    # Dissipative term: ½ ρ (Q/(Aβ))² · sign(Q)
    dissipative = 0.5 * rho0 * u_hole * np.abs(u_hole)

    # Least-squares fit: Δp = C_m · inertial + C_d⁻² · dissipative
    # Build design matrix
    X = np.column_stack([inertial, dissipative])
    y = phase_dp

    # Solve using pseudo-inverse
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        c_m = coeffs[0]
        cd_inv_sq = coeffs[1]

        # C_d from inverse square
        if cd_inv_sq > 0:
            c_d = 1.0 / math.sqrt(cd_inv_sq)
        else:
            c_d = float('nan')

        # R² of fit
        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-15)

        return {
            "C_d": c_d,
            "C_m": c_m,
            "r_squared": r_squared,
            "cd_inv_sq": cd_inv_sq,
        }
    except Exception:
        return {
            "C_d": float('nan'),
            "C_m": float('nan'),
            "r_squared": 0.0,
            "cd_inv_sq": float('nan'),
        }


def compute_energy_dissipation(
    phase_q: np.ndarray,
    phase_dp: np.ndarray,
    omega: float,
) -> float:
    """Compute energy dissipation per cycle: Φ = ∮ Δp · Q dt.

    Parameters
    ----------
    phase_q : np.ndarray
        Phase-averaged flow rate.
    phase_dp : np.ndarray
        Phase-averaged pressure drop.
    omega : float
        Angular frequency.

    Returns
    -------
    phi : float
        Energy dissipation per cycle.
    """
    dt = 2.0 * math.pi / omega / len(phase_q)
    phi = np.trapz(phase_dp * phase_q, dx=dt)
    return abs(phi)  # dissipation is always positive


def compute_zeta_eff(
    phi: float,
    omega: float,
    e_kin: float,
) -> float:
    """Compute equivalent damping ratio for sloshing.

    ζ_eff = Φ / (2π ω · 2 E_kin)

    Parameters
    ----------
    phi : float
        Energy dissipation per cycle.
    omega : float
        Angular frequency.
    e_kin : float
        Kinetic energy scale.

    Returns
    -------
    zeta_eff : float
        Equivalent damping ratio.
    """
    denom = 2.0 * math.pi * omega * 2.0 * max(e_kin, 1e-15)
    return phi / denom


def compute_phase_lag(
    phase_q: np.ndarray,
    phase_dp: np.ndarray,
    omega: float,
) -> dict:
    """Compute phase lag between Δp and Q using cross-correlation.

    Returns phase lag in degrees (positive = Δp leads Q).
    """
    # Normalize
    q_norm = phase_q - np.mean(phase_q)
    dp_norm = phase_dp - np.mean(phase_dp)

    q_mag = np.sqrt(np.sum(q_norm ** 2))
    dp_mag = np.sqrt(np.sum(dp_norm ** 2))

    if q_mag < 1e-15 or dp_mag < 1e-15:
        return {"phase_lag_deg": 0.0, "correlation": 0.0}

    q_norm /= q_mag
    dp_norm /= dp_mag

    # Cross-correlation
    n = len(phase_q)
    corr = np.correlate(dp_norm, q_norm, mode='full')
    corr = corr / n  # normalize

    # Find peak
    peak_idx = np.argmax(corr)
    lag = peak_idx - (n - 1)

    # Convert to phase angle
    phase_lag_rad = 2.0 * math.pi * lag / n
    phase_lag_deg = math.degrees(phase_lag_rad)

    # Correlation at zero lag
    corr_zero = np.mean(dp_norm * q_norm)

    return {
        "phase_lag_deg": phase_lag_deg,
        "correlation": corr_zero,
    }


def analyze_simulation(
    result: dict,
    omega: float,
    beta: float,
    ny: int,
    n_phase_bins: int = 32,
) -> dict:
    """Full analysis pipeline for a perforated plate simulation.

    Parameters
    ----------
    result : dict
        Output from run_perforated_plate().
    omega : float
        Angular frequency.
    beta : float
        Open area ratio.
    ny : int
        Channel height.
    n_phase_bins : int
        Number of phase bins.

    Returns
    -------
    dict with all analysis results.
    """
    ts = result["time_series"]
    time = ts["time"]
    q = ts["flow_rate"]
    dp = ts["delta_p"]

    # Phase average
    phase_q, avg_q = phase_average(time, q, omega, n_phase_bins)
    phase_dp, avg_dp = phase_average(time, dp, omega, n_phase_bins)

    # Extract C_d, C_m
    cd_cm = extract_cd_cm(avg_q, avg_dp, omega, beta, ny)

    # Energy dissipation
    phi = compute_energy_dissipation(avg_q, avg_dp, omega)

    # Phase lag
    phase_lag = compute_phase_lag(avg_q, avg_dp, omega)

    # Estimate kinetic energy scale
    u_max = np.max(np.abs(avg_q)) / max(ny - 2, 1)
    e_kin = 0.5 * u_max ** 2  # per unit mass

    # Equivalent damping
    zeta_eff = compute_zeta_eff(phi, omega, e_kin)

    return {
        "phase_q": phase_q,
        "avg_q": avg_q,
        "phase_dp": phase_dp,
        "avg_dp": avg_dp,
        "C_d": cd_cm["C_d"],
        "C_m": cd_cm["C_m"],
        "r_squared": cd_cm["r_squared"],
        "phi": phi,
        "zeta_eff": zeta_eff,
        "phase_lag_deg": phase_lag["phase_lag_deg"],
        "correlation": phase_lag["correlation"],
        "e_kin": e_kin,
    }
