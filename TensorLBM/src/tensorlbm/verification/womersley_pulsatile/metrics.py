"""Phase-averaging and phasor extraction for steady-periodic LBM signals.

The acceptance criterion of ``CASE-LBM-WOMERSLEY-PULSATILE`` is stated as
"centreline velocity amplitude ratio and phase lag vs the Womersley analytic
solution (alpha=5, whole-period phase averaged)".  This module implements
exactly that estimator:

1. ``fold_phase`` folds the time series into ``n_bins`` phase bins of the
   driving period ``T = 2*pi/omega`` (whole-period phase averaging, so the
   startup transient and any non-periodic residual average out).
2. ``fit_phasor`` least-squares fits ``u(t) = A cos(omega t) + B sin(omega t)``
   to the phase-averaged samples, returning the complex amplitude
   ``Z = A - i B`` such that ``u = Re{Z e^{i omega t}}``.
3. ``harmonic_metrics`` compares ``Z`` with the analytic phasor:
   amplitude ratio ``|Z_num|/|Z_ana|`` and phase lag difference in degrees
   (also normalised by the analytic lag, since the tolerance is relative).

All routines are pure numpy and independent of the solver, so they can be
tested on synthetic signals.
"""

from __future__ import annotations

import math

import numpy as np


def fold_phase(
    t: np.ndarray, u: np.ndarray, omega: float, n_bins: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whole-period phase average of ``u(t)``.

    Parameters
    ----------
    t, u : array_like
        Sample times (lattice steps) and values; ``u`` may be 1-D or have a
        trailing sample axis ``u[..., k]`` matching ``t[k]``.
    omega : float
        Driving angular frequency (rad per lattice time unit).
    n_bins : int
        Number of phase bins spanning one period.

    Returns
    -------
    phase_centers : (n_bins,) array
        Bin-centre phases in ``[0, 2*pi)``.
    u_mean : (..., n_bins) array
        Phase-averaged values (``nan`` for empty bins).
    counts : (n_bins,) int array
        Number of samples averaged into each bin.
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    phase = np.mod(omega * t, 2.0 * math.pi)
    idx = np.minimum((phase / (2.0 * math.pi) * n_bins).astype(np.int64), n_bins - 1)

    counts = np.bincount(idx, minlength=n_bins).astype(np.int64)
    if u.ndim == 1:
        sums = np.bincount(idx, weights=u, minlength=n_bins)
        u_mean = np.full(n_bins, np.nan)
        good = counts > 0
        u_mean[good] = sums[good] / counts[good]
    else:
        flat = u.reshape((-1, u.shape[-1]))
        acc = np.zeros((flat.shape[0], n_bins), dtype=float)
        for k in range(flat.shape[0]):
            acc[k] = np.bincount(idx, weights=flat[k], minlength=n_bins)
        u_mean = np.full((u.shape[:-1]) + (n_bins,), np.nan)
        good = counts > 0
        u_mean[..., good] = (acc[:, good] / counts[good]).reshape(u.shape[:-1] + (int(good.sum()),))
    phase_centers = (np.arange(n_bins) + 0.5) * (2.0 * math.pi / n_bins)
    return phase_centers, u_mean, counts


def fit_phasor(
    t: np.ndarray, u: np.ndarray, omega: float, weights: np.ndarray | None = None
) -> dict:
    """Least-squares harmonic fit ``u ~ A cos(omega t) + B sin(omega t) + C``.

    Returns ``Z = A - i B`` (so ``u = Re{Z e^{i omega t}} + C``), the mean
    offset ``C`` and the RMS residual on the fitted samples.
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    design = np.column_stack(
        [np.cos(omega * t), np.sin(omega * t), np.ones_like(t)]
    )
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=float))
        design = design * w[:, None]
        u = u * w
    coef, *_ = np.linalg.lstsq(design, u, rcond=None)
    z = complex(coef[0], -coef[1])
    resid = design @ coef - u
    rms = float(np.sqrt(np.mean(resid**2))) if resid.size else float("nan")
    return {"Z": z, "offset": float(coef[2]), "rms": rms, "A": float(coef[0]), "B": float(coef[1])}


def wrap_deg(angle_deg: float) -> float:
    """Wrap an angle to ``(-180, 180]`` degrees."""
    a = math.fmod(angle_deg, 360.0)
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    return a


def harmonic_metrics(z_num: complex, z_ana: complex) -> dict:
    """Amplitude ratio / phase-lag comparison of a measured vs analytic phasor."""
    amp_num = abs(z_num)
    amp_ana = abs(z_ana)
    lag_num = -math.degrees(math.atan2(z_num.imag, z_num.real))
    lag_ana = -math.degrees(math.atan2(z_ana.imag, z_ana.real))
    dlag = wrap_deg(lag_num - lag_ana)
    return {
        "amplitude_num": amp_num,
        "amplitude_ana": amp_ana,
        "amplitude_ratio": amp_num / amp_ana,
        "amplitude_rel_pct": abs(amp_num / amp_ana - 1.0) * 100.0,
        "phase_lag_num_deg": lag_num,
        "phase_lag_ana_deg": lag_ana,
        "phase_diff_deg": dlag,
        "phase_rel_pct": abs(dlag) / abs(lag_ana) * 100.0,
        "complex_rel_pct": abs(z_num / z_ana - 1.0) * 100.0,
    }


def waveform_error(
    phase_centers: np.ndarray,
    u_num: np.ndarray,
    u_ana_waveform: np.ndarray,
    scale: float,
) -> dict:
    """L2 / max error of a phase-averaged waveform against the analytic one.

    ``scale`` is the normalisation speed (typically the analytic peak speed),
    so the reported percentages are independent of the profile passing through
    zero at the flow-reversal phases.
    """
    du = np.asarray(u_num, dtype=float) - np.asarray(u_ana_waveform, dtype=float)
    good = np.isfinite(du)
    du = du[good]
    return {
        "l2_pct": float(np.sqrt(np.mean(du**2)) / scale * 100.0),
        "max_pct": float(np.max(np.abs(du)) / scale * 100.0),
    }