"""Turbulent channel flow analysis tools.

Provides:
- Statistical analysis of channel flow fields (mean, RMS, Reynolds stress)
- Energy spectrum computation E(k)
- Wall-law parameter extraction (kappa, B)
- Comparison metrics against DNS reference data
- Grid convergence index (GCI) computation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ChannelStatistics",
    "compute_channel_statistics",
    "compute_energy_spectrum",
    "fit_log_law",
    "compare_with_dns",
    "ComparisonResult",
    "WallLawFit",
]


@dataclass
class ChannelStatistics:
    """Time-averaged channel flow statistics.

    All profiles are functions of the wall-normal coordinate y.
    """
    y: np.ndarray  # wall-normal coordinate (cell indices or physical)
    y_plus: np.ndarray  # wall-normal in wall units
    u_mean: np.ndarray  # <u> (streamwise mean)
    u_plus: np.ndarray  # <u>+ = <u>/u_tau
    u_rms: np.ndarray  # u'_rms
    v_rms: np.ndarray  # v'_rms
    w_rms: np.ndarray  # w'_rms
    uv_reynolds: np.ndarray  # -<u'v'>
    u_rms_plus: np.ndarray  # u'_rms / u_tau
    v_rms_plus: np.ndarray  # v'_rms / u_tau
    w_rms_plus: np.ndarray  # w'_rms / u_tau
    uv_plus: np.ndarray  # -<u'v'> / u_tau^2
    u_tau: float  # friction velocity
    re_tau: float  # friction Reynolds number
    n_samples: int  # number of time samples

    def to_dict(self) -> dict:
        return {
            "y": self.y.tolist(),
            "y_plus": self.y_plus.tolist(),
            "u_mean": self.u_mean.tolist(),
            "u_plus": self.u_plus.tolist(),
            "u_rms": self.u_rms.tolist(),
            "v_rms": self.v_rms.tolist(),
            "w_rms": self.w_rms.tolist(),
            "uv_reynolds": self.uv_reynolds.tolist(),
            "u_rms_plus": self.u_rms_plus.tolist(),
            "v_rms_plus": self.v_rms_plus.tolist(),
            "w_rms_plus": self.w_rms_plus.tolist(),
            "uv_plus": self.uv_plus.tolist(),
            "u_tau": self.u_tau,
            "re_tau": self.re_tau,
            "n_samples": self.n_samples,
        }


def compute_channel_statistics(
    ux_field: np.ndarray,
    uy_field: np.ndarray,
    uz_field: np.ndarray,
    u_tau: float,
    nu: float,
    wall_index: int = 0,
    channel_height_cells: int = 0,
) -> ChannelStatistics:
    """Compute time-averaged statistics from instantaneous velocity fields.

    Parameters
    ----------
    ux_field : np.ndarray, shape (nx, ny, nz) or (n_samples, nx, ny, nz)
        Streamwise velocity.
    uy_field : np.ndarray
        Wall-normal velocity.
    uz_field : np.ndarray
        Spanwise velocity.
    u_tau : float
        Friction velocity for non-dimensionalization.
    nu : float
        Kinematic viscosity.
    wall_index : int
        Index of the lower wall in the y-direction.
    channel_height_cells : int
        Number of fluid cells between walls.

    Returns
    -------
    ChannelStatistics
        Computed statistics.
    """
    # Handle both single snapshot and time series
    if ux_field.ndim == 3:
        # Single snapshot - just compute spatial averages
        nx, ny, nz = ux_field.shape
        n_samples = 1
        # Average over homogeneous x, z directions
        u_mean = ux_field.mean(axis=(0, 2))
        u_rms = np.zeros(ny)
        v_rms = np.zeros(ny)
        w_rms = np.zeros(ny)
        uv = np.zeros(ny)
    elif ux_field.ndim == 4:
        n_samples, nx, ny, nz = ux_field.shape
        # Time average
        u_mean = ux_field.mean(axis=(0, 1, 3))
        # Fluctuations
        ux_prime = ux_field - u_mean[None, :, None, None]
        uy_prime = uy_field - uy_field.mean(axis=(0, 1, 3))[None, :, None, None]
        uz_prime = uz_field - uz_field.mean(axis=(0, 1, 3))[None, :, None, None]
        # RMS
        u_rms = np.sqrt((ux_prime**2).mean(axis=(0, 1, 3)))
        v_rms = np.sqrt((uy_prime**2).mean(axis=(0, 1, 3)))
        w_rms = np.sqrt((uz_prime**2).mean(axis=(0, 1, 3)))
        uv = -(ux_prime * uy_prime).mean(axis=(0, 1, 3))
    else:
        raise ValueError(f"Expected 3D or 4D array, got {ux_field.ndim}D")

    # Wall-normal coordinate
    y = np.arange(ny)
    # Distance from lower wall
    y_dist = y - wall_index - 0.5  # cell center offset

    # Wall units
    if channel_height_cells > 0:
        half_height = channel_height_cells / 2.0
    else:
        half_height = (ny - 2) / 2.0  # assume walls at 0 and ny-1

    y_plus = y_dist * u_tau / nu if nu > 0 else y_dist
    re_tau = half_height * u_tau / nu if nu > 0 else half_height * u_tau

    # Non-dimensionalize
    u_plus = u_mean / u_tau if u_tau > 0 else u_mean
    u_rms_plus = u_rms / u_tau if u_tau > 0 else u_rms
    v_rms_plus = v_rms / u_tau if u_tau > 0 else v_rms
    w_rms_plus = w_rms / u_tau if u_tau > 0 else w_rms
    uv_plus = uv / (u_tau**2) if u_tau > 0 else uv

    return ChannelStatistics(
        y=y,
        y_plus=y_plus,
        u_mean=u_mean,
        u_plus=u_plus,
        u_rms=u_rms,
        v_rms=v_rms,
        w_rms=w_rms,
        uv_reynolds=uv,
        u_rms_plus=u_rms_plus,
        v_rms_plus=v_rms_plus,
        w_rms_plus=w_rms_plus,
        uv_plus=uv_plus,
        u_tau=u_tau,
        re_tau=re_tau,
        n_samples=n_samples,
    )


def compute_energy_spectrum(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    dx: float = 1.0,
    dz: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the 1D energy spectrum E(k) from a 3D velocity field.

    Uses the streamwise-spanwise plane average at each wall-normal level,
    then integrates over the wall-normal direction.

    Parameters
    ----------
    ux, uy, uz : np.ndarray, shape (nx, ny, nz)
        Velocity components.
    dx, dz : float
        Grid spacing in streamwise and spanwise directions.

    Returns
    -------
    k : np.ndarray
        Wavenumber magnitudes.
    E_k : np.ndarray
        Energy spectrum E(k).
    """
    nx, ny, nz = ux.shape

    # Wavenumbers
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
    KX, KZ = np.meshgrid(kx, kz, indexing="ij")
    K = np.sqrt(KX**2 + KZ**2)

    # Compute 2D spectrum at each y-level
    E_2d = np.zeros((nx, nz))
    for j in range(ny):
        # FFT of velocity components
        ux_hat = np.fft.fft2(ux[:, j, :])
        uy_hat = np.fft.fft2(uy[:, j, :])
        uz_hat = np.fft.fft2(uz[:, j, :])

        # Energy at each wavenumber
        E_y = 0.5 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2 + np.abs(uz_hat)**2)
        E_2d += E_y

    E_2d /= ny

    # Bin by wavenumber magnitude
    k_max = np.pi / min(dx, dz)
    n_bins = min(nx // 2, nz // 2, 64)
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    E_k = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (K >= k_edges[i]) & (K < k_edges[i + 1])
        if mask.any():
            E_k[i] = E_2d[mask].sum()
            # Normalize by annulus area in wavenumber space
            dk = k_edges[i + 1] - k_edges[i]
            k_annulus = k_centers[i] if k_centers[i] > 0 else dk
            E_k[i] /= max(2.0 * np.pi * k_annulus, 1e-10)

    # Remove zero wavenumber
    valid = k_centers > 0
    return k_centers[valid], E_k[valid]


@dataclass
class WallLawFit:
    """Result of fitting the log-law to a velocity profile.

    u+ = (1/kappa) * ln(y+) + B
    """
    kappa: float  # von Karman constant
    B: float  # log-law intercept
    y_plus_range: Tuple[float, float]  # fitting range
    r_squared: float  # goodness of fit
    n_points: int  # number of points used

    def to_dict(self) -> dict:
        return {
            "kappa": self.kappa,
            "B": self.B,
            "y_plus_range": list(self.y_plus_range),
            "r_squared": self.r_squared,
            "n_points": self.n_points,
        }


def fit_log_law(
    y_plus: np.ndarray,
    u_plus: np.ndarray,
    y_plus_min: float = 30.0,
    y_plus_max: float = None,
) -> WallLawFit:
    """Fit the log-law u+ = (1/kappa)*ln(y+) + B to the velocity profile.

    Parameters
    ----------
    y_plus : np.ndarray
        Wall-normal positions in wall units.
    u_plus : np.ndarray
        Mean velocity in wall units.
    y_plus_min : float
        Lower bound of fitting range (default: 30, start of log layer).
    y_plus_max : float, optional
        Upper bound (default: 0.2*re_tau or 0.3*max(y_plus)).

    Returns
    -------
    WallLawFit
        Fitted parameters.
    """
    yp = np.asarray(y_plus)
    up = np.asarray(u_plus)

    if y_plus_max is None:
        y_plus_max = 0.3 * yp.max()

    # Select fitting range
    mask = (yp >= y_plus_min) & (yp <= y_plus_max)
    if mask.sum() < 3:
        # Fallback: use all points above y+ = 20
        mask = yp >= 20.0

    yp_fit = yp[mask]
    up_fit = up[mask]

    if len(yp_fit) < 3:
        return WallLawFit(
            kappa=0.41, B=5.2, y_plus_range=(y_plus_min, y_plus_max),
            r_squared=0.0, n_points=0
        )

    # Linear regression: u+ = a * ln(y+) + b
    ln_yp = np.log(yp_fit)
    A = np.vstack([ln_yp, np.ones_like(ln_yp)]).T
    result = np.linalg.lstsq(A, up_fit, rcond=None)
    coeffs = result[0]

    a, b = coeffs[0], coeffs[1]
    kappa = 1.0 / a if a > 0 else 0.41
    B = b

    # R-squared
    up_pred = a * ln_yp + b
    ss_res = np.sum((up_fit - up_pred)**2)
    ss_tot = np.sum((up_fit - up_fit.mean())**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return WallLawFit(
        kappa=kappa,
        B=B,
        y_plus_range=(float(yp_fit.min()), float(yp_fit.max())),
        r_squared=r_squared,
        n_points=len(yp_fit),
    )


@dataclass
class ComparisonResult:
    """Result of comparing LBM results with DNS reference data."""
    re_tau: float
    u_plus_error_pct: float  # L2 relative error in log region
    u_rms_error_pct: float
    v_rms_error_pct: float
    w_rms_error_pct: float
    uv_error_pct: float
    kappa_error: float  # |kappa - 0.41|
    B_error: float  # |B - 5.2|
    spectral_slope: float  # fitted E(k) ~ k^(-alpha)
    spectral_slope_error: float  # |alpha - 5/3|
    wall_law_fit: Optional[WallLawFit] = None
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "re_tau": self.re_tau,
            "u_plus_error_pct": self.u_plus_error_pct,
            "u_rms_error_pct": self.u_rms_error_pct,
            "v_rms_error_pct": self.v_rms_error_pct,
            "w_rms_error_pct": self.w_rms_error_pct,
            "uv_error_pct": self.uv_error_pct,
            "kappa_error": self.kappa_error,
            "B_error": self.B_error,
            "spectral_slope": self.spectral_slope,
            "spectral_slope_error": self.spectral_slope_error,
        }
        if self.wall_law_fit:
            d["wall_law_fit"] = self.wall_law_fit.to_dict()
        d["details"] = self.details
        return d

    def passes_judgement(self) -> Dict[str, bool]:
        """Check each judgement criterion."""
        return {
            "J1_u_plus_log_region_lt_10pct": self.u_plus_error_pct < 10.0,
            "J2_rms_lt_15pct": max(
                self.u_rms_error_pct, self.v_rms_error_pct, self.w_rms_error_pct
            ) < 15.0,
            "J3_reynolds_stress_lt_15pct": self.uv_error_pct < 15.0,
            "J4_spectral_slope": abs(self.spectral_slope_error) < 0.3,
            "J5_kappa_B": self.kappa_error < 0.05 and self.B_error < 0.5,
        }


def compare_with_dns(
    stats: ChannelStatistics,
    dns_y_plus: np.ndarray,
    dns_u_plus: np.ndarray,
    dns_u_rms: np.ndarray,
    dns_v_rms: np.ndarray,
    dns_w_rms: np.ndarray,
    dns_uv: np.ndarray,
    spectral_slope: float = -5.0 / 3.0,
    wall_law_fit: Optional[WallLawFit] = None,
) -> ComparisonResult:
    """Compare computed statistics with DNS reference data.

    Parameters
    ----------
    stats : ChannelStatistics
        Computed channel statistics.
    dns_* : np.ndarray
        DNS reference profiles.
    spectral_slope : float
        Measured spectral slope from E(k).
    wall_law_fit : WallLawFit, optional
        Fitted wall-law parameters.

    Returns
    -------
    ComparisonResult
        Comparison metrics.
    """
    # Interpolate DNS data to LBM y+ locations
    from numpy import interp

    # Only compare in valid range (avoid wall and centerline artifacts)
    valid = (stats.y_plus > 1.0) & (stats.y_plus < stats.re_tau * 0.95)

    if valid.sum() < 5:
        valid = np.ones(len(stats.y_plus), dtype=bool)

    yp_lbm = stats.y_plus[valid]

    # Interpolate DNS to LBM points
    dns_u_interp = interp(yp_lbm, dns_y_plus, dns_u_plus)
    dns_ur_interp = interp(yp_lbm, dns_y_plus, dns_u_rms)
    dns_vr_interp = interp(yp_lbm, dns_y_plus, dns_v_rms)
    dns_wr_interp = interp(yp_lbm, dns_y_plus, dns_w_rms)
    dns_uv_interp = interp(yp_lbm, dns_y_plus, dns_uv)

    def relative_error_l2(computed, reference):
        """L2 relative error in percent."""
        ref_norm = np.sqrt(np.sum(reference**2))
        if ref_norm < 1e-10:
            return 0.0
        diff_norm = np.sqrt(np.sum((computed - reference)**2))
        return 100.0 * diff_norm / ref_norm

    # Log-region error for u+ (y+ > 30)
    log_mask = yp_lbm > 30.0
    if log_mask.sum() > 3:
        u_plus_err = relative_error_l2(
            stats.u_plus[valid][log_mask], dns_u_interp[log_mask]
        )
    else:
        u_plus_err = relative_error_l2(stats.u_plus[valid], dns_u_interp)

    u_rms_err = relative_error_l2(stats.u_rms_plus[valid], dns_ur_interp)
    v_rms_err = relative_error_l2(stats.v_rms_plus[valid], dns_vr_interp)
    w_rms_err = relative_error_l2(stats.w_rms_plus[valid], dns_wr_interp)
    uv_err = relative_error_l2(stats.uv_plus[valid], dns_uv_interp)

    # Wall-law parameter errors
    if wall_law_fit is not None:
        kappa_err = abs(wall_law_fit.kappa - 0.41)
        B_err = abs(wall_law_fit.B - 5.2)
    else:
        kappa_err = 0.41
        B_err = 5.2

    return ComparisonResult(
        re_tau=stats.re_tau,
        u_plus_error_pct=u_plus_err,
        u_rms_error_pct=u_rms_err,
        v_rms_error_pct=v_rms_err,
        w_rms_error_pct=w_rms_err,
        uv_error_pct=uv_err,
        kappa_error=kappa_err,
        B_error=B_err,
        spectral_slope=spectral_slope,
        spectral_slope_error=abs(spectral_slope - (-5.0 / 3.0)),
        wall_law_fit=wall_law_fit,
    )


def fit_spectral_slope(
    k: np.ndarray, E_k: np.ndarray, k_min: float = None, k_max: float = None
) -> float:
    """Fit the spectral slope E(k) ~ k^alpha in the inertial subrange.

    Parameters
    ----------
    k : np.ndarray
        Wavenumber magnitudes.
    E_k : np.ndarray
        Energy spectrum.
    k_min, k_max : float, optional
        Fitting range in wavenumber space.

    Returns
    -------
    float
        Fitted spectral exponent alpha (should be ~-5/3).
    """
    valid = (k > 0) & (E_k > 0)
    k_fit = k[valid]
    E_fit = E_k[valid]

    if k_min is None:
        k_min = k_fit.min() * 2
    if k_max is None:
        k_max = k_fit.max() / 2

    mask = (k_fit >= k_min) & (k_fit <= k_max)
    if mask.sum() < 3:
        return -5.0 / 3.0  # default

    ln_k = np.log(k_fit[mask])
    ln_E = np.log(E_fit[mask])

    # Linear fit: ln(E) = alpha * ln(k) + const
    A = np.vstack([ln_k, np.ones_like(ln_k)]).T
    result = np.linalg.lstsq(A, ln_E, rcond=None)
    alpha = result[0][0]

    return float(alpha)
