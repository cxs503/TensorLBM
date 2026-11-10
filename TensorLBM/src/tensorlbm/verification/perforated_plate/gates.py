"""Acceptance criteria gates for perforated plate oscillatory flow verification.

Gates:
    C1: Quasi-steady limit (KC ≥ 15): C_d → Idelchik ±10%
    C2: Phase monotonicity: Δp-Q phase lag decreases with KC (Spearman ρ = -1)
    C3: C_d vs β monotonic: C_d increases with β (Spearman ρ = 1), range 0.4–0.9
    C4: Grid convergence: GCI(C_d) < 5% across three resolutions
    C5: Negative controls:
        NC1: No plate (β=1): pressure drop vs analytical < 3%
        NC2: Stokes layer: wall shear stress < 5%
    C6: Energy budget: Φ from momentum exchange vs ⟨Δp·Q⟩ closure < 3%
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

from .reference import idelchik_cd_quasi_steady, quasi_steady_pressure_drop


# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------

GATE_THRESHOLDS = {
    "C1_quasi_steady_error": 0.10,       # 10%
    "C2_spearman_rho": -1.0,             # perfect negative correlation
    "C3_spearman_rho": 1.0,              # perfect positive correlation
    "C3_cd_min": 0.4,
    "C3_cd_max": 0.9,
    "C4_gci_threshold": 0.05,            # 5%
    "C5_nc1_error": 0.03,                # 3%
    "C5_nc2_error": 0.05,                # 5%
    "C6_energy_closure": 0.03,           # 3%
}


# ---------------------------------------------------------------------------
# C1: Quasi-steady limit
# ---------------------------------------------------------------------------

def check_c1_quasi_steady(
    cd_measured: float,
    beta: float,
    kc: float,
    threshold: float = GATE_THRESHOLDS["C1_quasi_steady_error"],
) -> dict:
    """C1: At high KC (≥15), C_d should approach Idelchik quasi-steady value.

    Parameters
    ----------
    cd_measured : float
        Measured discharge coefficient.
    beta : float
        Open area ratio.
    kc : float
        Keulegan-Carpenter number.
    threshold : float
        Maximum relative error (default 10%).

    Returns
    -------
    dict with passed, error, and details.
    """
    cd_ref = idelchik_cd_quasi_steady(beta)

    if kc < 15:
        return {
            "passed": False,
            "reason": f"KC={kc:.1f} < 15, not in quasi-steady regime",
            "cd_measured": cd_measured,
            "cd_reference": cd_ref,
            "kc": kc,
        }

    if math.isnan(cd_measured) or cd_ref <= 0:
        return {
            "passed": False,
            "reason": "Invalid C_d or reference value",
            "cd_measured": cd_measured,
            "cd_reference": cd_ref,
        }

    error = abs(cd_measured - cd_ref) / cd_ref
    passed = error < threshold

    return {
        "passed": passed,
        "error": error,
        "threshold": threshold,
        "cd_measured": cd_measured,
        "cd_reference": cd_ref,
        "kc": kc,
    }


# ---------------------------------------------------------------------------
# C2: Phase monotonicity
# ---------------------------------------------------------------------------

def check_c2_phase_monotonic(
    kc_values: list[float],
    phase_lag_values: list[float],
) -> dict:
    """C2: Phase lag between Δp and Q should decrease monotonically with KC.

    At low KC: inertia dominates → phase lag ≈ 90°
    At high KC: dissipation dominates → phase lag ≈ 0°

    Spearman rank correlation should be ρ = -1 (perfect negative).

    Parameters
    ----------
    kc_values : list[float]
        KC values (at least 8).
    phase_lag_values : list[float]
        Corresponding phase lag values in degrees.

    Returns
    -------
    dict with passed, spearman_rho, and details.
    """
    if len(kc_values) < 2:
        return {
            "passed": False,
            "reason": "Need at least 2 data points",
        }

    kc_arr = np.array(kc_values)
    pl_arr = np.array(phase_lag_values)

    # Spearman rank correlation
    rho, p_value = scipy_stats.spearmanr(kc_arr, pl_arr)

    # Check monotonic decrease
    is_monotonic = all(
        pl_arr[i] >= pl_arr[i + 1]
        for i in range(len(pl_arr) - 1)
    )

    # Accept if rho ≈ -1 (allow small deviation)
    passed = rho < -0.9 and is_monotonic

    return {
        "passed": passed,
        "spearman_rho": rho,
        "p_value": p_value,
        "is_monotonic": is_monotonic,
        "kc_values": kc_values,
        "phase_lag_values": phase_lag_values,
    }


# ---------------------------------------------------------------------------
# C3: C_d vs β monotonic
# ---------------------------------------------------------------------------

def check_c3_cd_vs_beta(
    beta_values: list[float],
    cd_values: list[float],
) -> dict:
    """C3: C_d should increase monotonically with β.

    Spearman rank correlation should be ρ = 1 (perfect positive).
    C_d values should be in range [0.4, 0.9].

    Parameters
    ----------
    beta_values : list[float]
        Open area ratio values (at least 5).
    cd_values : list[float]
        Corresponding C_d values.

    Returns
    -------
    dict with passed, spearman_rho, and details.
    """
    if len(beta_values) < 2:
        return {
            "passed": False,
            "reason": "Need at least 2 data points",
        }

    beta_arr = np.array(beta_values)
    cd_arr = np.array(cd_values)

    # Spearman rank correlation
    rho, p_value = scipy_stats.spearmanr(beta_arr, cd_arr)

    # Check monotonic increase
    is_monotonic = all(
        cd_arr[i] <= cd_arr[i + 1]
        for i in range(len(cd_arr) - 1)
    )

    # Check physical range
    in_range = all(
        GATE_THRESHOLDS["C3_cd_min"] <= c <= GATE_THRESHOLDS["C3_cd_max"]
        for c in cd_arr
        if not math.isnan(c)
    )

    passed = rho > 0.9 and is_monotonic and in_range

    return {
        "passed": passed,
        "spearman_rho": rho,
        "p_value": p_value,
        "is_monotonic": is_monotonic,
        "in_range": in_range,
        "beta_values": beta_values,
        "cd_values": cd_values,
    }


# ---------------------------------------------------------------------------
# C4: Grid convergence (GCI)
# ---------------------------------------------------------------------------

def check_c4_gci(
    cd_values: list[float],
    refinement_ratios: list[float],
    threshold: float = GATE_THRESHOLDS["C4_gci_threshold"],
) -> dict:
    """C4: Grid Convergence Index for C_d < 5%.

    Uses Richardson extrapolation with three grid resolutions.

    Parameters
    ----------
    cd_values : list[float]
        C_d values at three resolutions [coarse, medium, fine].
    refinement_ratios : list[float]
        Grid refinement ratios [r21, r32].
    threshold : float
        Maximum GCI (default 5%).

    Returns
    -------
    dict with passed, gci, and details.
    """
    if len(cd_values) < 3:
        return {
            "passed": False,
            "reason": "Need 3 grid resolutions",
        }

    f1, f2, f3 = cd_values[2], cd_values[1], cd_values[0]  # fine, medium, coarse
    r21 = refinement_ratios[0] if len(refinement_ratios) > 0 else 2.0
    r32 = refinement_ratios[1] if len(refinement_ratios) > 1 else 2.0

    # Differences
    eps21 = f2 - f1
    eps32 = f3 - f2

    # Apparent order of convergence
    if abs(eps32) < 1e-15 or abs(eps21) < 1e-15:
        return {
            "passed": True,
            "gci": 0.0,
            "reason": "Negligible differences",
        }

    q = abs(eps32 / eps21)
    try:
        p = math.log(q) / math.log(r21)
    except (ValueError, ZeroDivisionError):
        p = 2.0  # assume second order

    # Richardson extrapolated value
    f_ext = f1 + eps21 / (r21 ** p - 1)

    # GCI (fine grid)
    fs = 1.25  # safety factor
    gci = fs * abs(eps21 / f_ext) / (r21 ** p - 1)

    passed = gci < threshold

    return {
        "passed": passed,
        "gci": gci,
        "threshold": threshold,
        "apparent_order": p,
        "f_ext": f_ext,
        "cd_coarse": f3,
        "cd_medium": f2,
        "cd_fine": f1,
    }


# ---------------------------------------------------------------------------
# C5: Negative controls
# ---------------------------------------------------------------------------

def check_c5_negative_controls(
    nc1_dp_measured: float = 0.0,
    nc1_dp_analytical: float = 0.0,
    nc2_shear_measured: float = 0.0,
    nc2_shear_analytical: float = 0.0,
) -> dict:
    """C5: Negative controls.

    NC1: No plate (β=1): pressure drop should match analytical channel flow.
    NC2: Stokes layer: wall shear stress should match analytical solution.

    Parameters
    ----------
    nc1_dp_measured : float
        Measured pressure drop without plate.
    nc1_dp_analytical : float
        Analytical pressure drop for channel flow.
    nc2_shear_measured : float
        Measured wall shear stress.
    nc2_shear_analytical : float
        Analytical wall shear stress.

    Returns
    -------
    dict with results for each negative control.
    """
    results = {}

    # NC1: No plate
    if abs(nc1_dp_analytical) > 1e-15:
        nc1_error = abs(nc1_dp_measured - nc1_dp_analytical) / abs(nc1_dp_analytical)
    else:
        nc1_error = abs(nc1_dp_measured)

    nc1_passed = nc1_error < GATE_THRESHOLDS["C5_nc1_error"]
    results["nc1_no_plate"] = {
        "passed": nc1_passed,
        "error": nc1_error,
        "threshold": GATE_THRESHOLDS["C5_nc1_error"],
        "measured": nc1_dp_measured,
        "analytical": nc1_dp_analytical,
    }

    # NC2: Stokes layer
    if abs(nc2_shear_analytical) > 1e-15:
        nc2_error = abs(nc2_shear_measured - nc2_shear_analytical) / abs(nc2_shear_analytical)
    else:
        nc2_error = abs(nc2_shear_measured)

    nc2_passed = nc2_error < GATE_THRESHOLDS["C5_nc2_error"]
    results["nc2_stokes_layer"] = {
        "passed": nc2_passed,
        "error": nc2_error,
        "threshold": GATE_THRESHOLDS["C5_nc2_error"],
        "measured": nc2_shear_measured,
        "analytical": nc2_shear_analytical,
    }

    all_passed = nc1_passed and nc2_passed

    return {
        "all_passed": all_passed,
        "results": results,
    }


# ---------------------------------------------------------------------------
# C6: Energy budget closure
# ---------------------------------------------------------------------------

def check_c6_energy_budget(
    phi_momentum: float,
    phi_pressure_flow: float,
    threshold: float = GATE_THRESHOLDS["C6_energy_closure"],
) -> dict:
    """C6: Energy budget closure.

    Two independent paths to compute energy dissipation:
    1. Φ from wall momentum exchange integral
    2. Φ from ⟨Δp · Q⟩ over one cycle

    These should agree within 3%.

    Parameters
    ----------
    phi_momentum : float
        Energy dissipation from momentum exchange.
    phi_pressure_flow : float
        Energy dissipation from pressure-flow product.
    threshold : float
        Maximum relative difference (default 3%).

    Returns
    -------
    dict with passed, error, and details.
    """
    avg_phi = 0.5 * (abs(phi_momentum) + abs(phi_pressure_flow))

    if avg_phi < 1e-15:
        return {
            "passed": True,
            "error": 0.0,
            "reason": "Negligible dissipation",
        }

    error = abs(phi_momentum - phi_pressure_flow) / avg_phi
    passed = error < threshold

    return {
        "passed": passed,
        "error": error,
        "threshold": threshold,
        "phi_momentum": phi_momentum,
        "phi_pressure_flow": phi_pressure_flow,
    }
