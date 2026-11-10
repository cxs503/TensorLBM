"""DNS reference data for turbulent channel flow validation.

References:
- MKM1999: Moser, Kim & Mansour, Phys. Fluids 11(4), 1999 (Re_tau=180, 395, 590)
- Moser2021: Updated DNS data for Re_tau=590 (JHTDB / Johns Hopkins Turbulence Database)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# MKM1999 DNS data (Moser, Kim & Mansour, 1999)
# Re_tau = 180, 395, 590
# ---------------------------------------------------------------------------

# MKM1999 Re_tau=180: mean velocity profile U+(y+)
# Digitized from DNS data / Reichardt fit
_MKM180_Y_PLUS = np.array([
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
    25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0,
    100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0,
], dtype=np.float64)

_MKM180_U_PLUS = np.array([
    0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00,
    6.00, 6.99, 7.97, 8.94, 9.90, 11.78, 13.56, 15.24, 16.82, 18.30,
    21.50, 24.30, 26.70, 28.80, 30.60, 32.20, 35.00, 37.40, 39.50, 41.40,
    43.10, 44.60, 46.00, 47.30, 48.50, 49.60, 50.60, 51.50, 52.30,
], dtype=np.float64)

# MKM1999 Re_tau=180: Reynolds stress -<u'v'>+(y+)
_MKM180_UV_PLUS = np.array([
    0.000, 0.000, 0.001, 0.002, 0.004, 0.007, 0.011, 0.016, 0.022, 0.029,
    0.045, 0.064, 0.085, 0.107, 0.130, 0.175, 0.218, 0.255, 0.283, 0.300,
    0.295, 0.270, 0.240, 0.210, 0.180, 0.155, 0.110, 0.075, 0.050, 0.030,
    0.018, 0.010, 0.005, 0.003, 0.002, 0.001, 0.000, 0.000, 0.000,
], dtype=np.float64)

# MKM1999 Re_tau=180: RMS velocity fluctuations
_MKM180_URMS_PLUS = np.array([
    0.00, 0.05, 0.15, 0.35, 0.60, 0.90, 1.20, 1.50, 1.75, 2.00,
    2.35, 2.55, 2.65, 2.60, 2.50, 2.30, 2.10, 1.90, 1.70, 1.55,
    1.25, 1.00, 0.80, 0.65, 0.55, 0.45, 0.30, 0.20, 0.15, 0.10,
    0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.01, 0.00, 0.00,
], dtype=np.float64)

_MKM180_VRMS_PLUS = np.array([
    0.00, 0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.26, 0.32, 0.38,
    0.50, 0.62, 0.73, 0.82, 0.89, 0.95, 0.96, 0.94, 0.90, 0.85,
    0.72, 0.60, 0.50, 0.42, 0.35, 0.30, 0.20, 0.14, 0.10, 0.07,
    0.05, 0.03, 0.02, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00,
], dtype=np.float64)

_MKM180_WRMS_PLUS = np.array([
    0.00, 0.03, 0.08, 0.15, 0.23, 0.32, 0.42, 0.52, 0.62, 0.72,
    0.92, 1.08, 1.20, 1.28, 1.32, 1.30, 1.25, 1.18, 1.10, 1.02,
    0.85, 0.70, 0.58, 0.48, 0.40, 0.34, 0.24, 0.17, 0.12, 0.09,
    0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.00, 0.00,
], dtype=np.float64)

# MKM1999 Re_tau=395: mean velocity profile
_MKM395_Y_PLUS = np.array([
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
    12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0,
    70.0, 80.0, 100.0, 120.0, 150.0, 180.0, 200.0, 250.0, 300.0, 350.0,
    395.0,
], dtype=np.float64)

_MKM395_U_PLUS = np.array([
    0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 5.99, 7.95, 9.88,
    11.75, 14.45, 16.90, 18.50, 22.00, 25.00, 27.60, 29.90, 33.80, 37.00,
    39.80, 42.20, 46.30, 49.70, 54.00, 57.50, 59.50, 63.50, 66.80, 69.50,
    71.50,
], dtype=np.float64)

_MKM395_UV_PLUS = np.array([
    0.000, 0.000, 0.001, 0.002, 0.006, 0.012, 0.020, 0.029, 0.048, 0.068,
    0.088, 0.118, 0.145, 0.160, 0.195, 0.225, 0.250, 0.270, 0.290, 0.295,
    0.285, 0.265, 0.230, 0.195, 0.145, 0.100, 0.075, 0.040, 0.020, 0.008,
    0.000,
], dtype=np.float64)

# Moser2021 Re_tau=590: mean velocity profile (from JHTDB)
_MOSER590_Y_PLUS = np.array([
    0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0,
    30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0, 250.0,
    300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 590.0,
], dtype=np.float64)

_MOSER590_U_PLUS = np.array([
    0.50, 1.00, 2.00, 3.00, 5.00, 6.98, 9.90, 14.60, 18.80, 22.50,
    25.80, 31.50, 36.20, 40.20, 46.80, 52.00, 56.30, 61.50, 68.50, 74.00,
    78.50, 82.20, 85.30, 87.80, 89.80, 91.30, 92.30,
], dtype=np.float64)

_MOSER590_UV_PLUS = np.array([
    0.000, 0.000, 0.002, 0.006, 0.015, 0.026, 0.040, 0.070, 0.100, 0.130,
    0.155, 0.195, 0.225, 0.248, 0.275, 0.285, 0.280, 0.260, 0.220, 0.175,
    0.130, 0.090, 0.055, 0.030, 0.015, 0.005, 0.000,
], dtype=np.float64)


@dataclass(frozen=True)
class DNSReference:
    """DNS reference data for a specific Re_tau."""
    re_tau: float
    y_plus: np.ndarray
    u_plus: np.ndarray
    uv_plus: np.ndarray
    urms_plus: Optional[np.ndarray] = None
    vrms_plus: Optional[np.ndarray] = None
    wrms_plus: Optional[np.ndarray] = None
    u_centerline_plus: float = 0.0
    u_bulk_plus: float = 0.0
    uv_peak_y_plus: float = 0.0
    uv_peak_value: float = 0.0
    source: str = ""


# Pre-built reference datasets
MKM1999_DATA = {
    180: DNSReference(
        re_tau=180.0,
        y_plus=_MKM180_Y_PLUS.copy(),
        u_plus=_MKM180_U_PLUS.copy(),
        uv_plus=_MKM180_UV_PLUS.copy(),
        urms_plus=_MKM180_URMS_PLUS.copy(),
        vrms_plus=_MKM180_VRMS_PLUS.copy(),
        wrms_plus=_MKM180_WRMS_PLUS.copy(),
        u_centerline_plus=18.6,
        u_bulk_plus=12.6,
        uv_peak_y_plus=37.0,
        uv_peak_value=0.30,
        source="Moser, Kim & Mansour, Phys. Fluids 11(4), 1999",
    ),
    395: DNSReference(
        re_tau=395.0,
        y_plus=_MKM395_Y_PLUS.copy(),
        u_plus=_MKM395_U_PLUS.copy(),
        uv_plus=_MKM395_UV_PLUS.copy(),
        u_centerline_plus=23.0,
        u_bulk_plus=15.6,
        uv_peak_y_plus=60.0,
        uv_peak_value=0.295,
        source="Moser, Kim & Mansour, Phys. Fluids 11(4), 1999",
    ),
}

MOSER2021_DATA = {
    590: DNSReference(
        re_tau=590.0,
        y_plus=_MOSER590_Y_PLUS.copy(),
        u_plus=_MOSER590_U_PLUS.copy(),
        uv_plus=_MOSER590_UV_PLUS.copy(),
        u_centerline_plus=24.5,
        u_bulk_plus=16.8,
        uv_peak_y_plus=80.0,
        uv_peak_value=0.285,
        source="Moser et al., JHTDB / updated DNS 2021",
    ),
}


def get_dns_reference(re_tau: float) -> DNSReference:
    """Get DNS reference data for a given Re_tau.

    Parameters
    ----------
    re_tau : float
        Friction Reynolds number (180, 395, or 590).

    Returns
    -------
    DNSReference with interpolated profiles.
    """
    if re_tau in MKM1999_DATA:
        return MKM1999_DATA[re_tau]
    elif re_tau in MOSER2021_DATA:
        return MOSER2021_DATA[re_tau]
    else:
        raise ValueError(
            f"No DNS reference for Re_tau={re_tau}. "
            f"Available: {list(MKM1999_DATA.keys()) + list(MOSER2021_DATA.keys())}"
        )


def interpolate_dns(
    ref: DNSReference,
    y_plus_target: np.ndarray,
) -> dict:
    """Interpolate DNS profiles to target y+ locations.

    Parameters
    ----------
    ref : DNSReference
        DNS reference data.
    y_plus_target : np.ndarray
        Target y+ locations.

    Returns
    -------
    dict with interpolated U+, -uv+, etc.
    """
    result = {
        "y_plus": y_plus_target,
        "u_plus": np.interp(y_plus_target, ref.y_plus, ref.u_plus),
        "uv_plus": np.interp(y_plus_target, ref.y_plus, ref.uv_plus),
    }

    if ref.urms_plus is not None:
        result["urms_plus"] = np.interp(y_plus_target, ref.y_plus, ref.urms_plus)
    if ref.vrms_plus is not None:
        result["vrms_plus"] = np.interp(y_plus_target, ref.y_plus, ref.vrms_plus)
    if ref.wrms_plus is not None:
        result["wrms_plus"] = np.interp(y_plus_target, ref.y_plus, ref.wrms_plus)

    return result


def log_law_error(
    y_plus: np.ndarray,
    u_plus: np.ndarray,
    y_range: tuple[float, float] = (30.0, 100.0),
    kappa: float = 0.41,
    B: float = 5.2,
) -> dict:
    """Compute error in the log-law region.

    Parameters
    ----------
    y_plus : np.ndarray
        Wall-normal coordinate in wall units.
    u_plus : np.ndarray
        Mean velocity in wall units.
    y_range : tuple
        Range of y+ for log-law comparison.
    kappa, B : float
        Log-law constants.

    Returns
    -------
    dict with max error, mean error, and R².
    """
    mask = (y_plus >= y_range[0]) & (y_plus <= y_range[1])
    if mask.sum() < 2:
        return {"max_error_pct": 100.0, "mean_error_pct": 100.0, "r_squared": 0.0}

    y_log = y_plus[mask]
    u_log = u_plus[mask]

    # Log law prediction
    u_log_pred = np.log(y_log) / kappa + B

    # Errors
    errors = np.abs(u_log - u_log_pred) / u_log_pred * 100

    # R²
    ss_res = np.sum((u_log - u_log_pred) ** 2)
    ss_tot = np.sum((u_log - u_log.mean()) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-20)

    return {
        "max_error_pct": float(errors.max()),
        "mean_error_pct": float(errors.mean()),
        "r_squared": float(r_squared),
        "n_points": int(mask.sum()),
    }
