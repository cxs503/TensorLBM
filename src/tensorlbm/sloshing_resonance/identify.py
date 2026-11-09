"""Time-series estimators for sloshing free-decay records.

All estimators are pure ``numpy`` and side-effect free so they can be unit
tested against synthetic damped sinusoids.

Definitions used by the acceptance campaign
-------------------------------------------
For a linear damped standing wave the modal record is

    a(t) = A e^{-zeta omega_n t} cos(omega_d t + phi),
    lambda = zeta omega_n,   omega_d = omega_n sqrt(1 - zeta^2)      (I)

so the *natural* (undamped) frequency is ``omega_n = sqrt(lambda^2 +
omega_d^2)`` — the frequency Abramson's inviscid theory predicts.  Two numbers
are therefore always reported:

``f_damped``  ``omega_d / 2 pi``   — the raw, directly measured frequency;
``f_natural`` ``omega_n / 2 pi``   — the undamped-frequency estimate (the one
              compared with Abramson in criteria (a)/(b)).

``zeta`` is the damping ratio of (I).  Reporting both keeps the (large) lattice
damping visible instead of hiding it inside a single "measured" number.

Estimators
----------
``zero_crossings``        upward zero crossings with linear interpolation
``period_from_crossings`` damped period ``T_d`` from consecutive crossings
``envelope_decay_rate``   ``lambda`` from a log-linear fit of the |extrema|
``fit_damped_sinusoid``   (lambda, omega_d) -> (A, phi), plus residual quality
``identify_mode``         one-call bundle of the above + phase error
``fft_peak``              Hann-windowed FFT peak (independent cross-check)
``phase_error_deg``       phase error against ``A cos(omega_n t)`` at a
                          reference quarter-period crossing
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ModeFit",
    "harmonic_amplitude",
    "zero_crossings",
    "period_from_crossings",
    "envelope_decay_rate",
    "fit_damped_sinusoid",
    "fft_peak",
    "phase_error_deg",
    "identify_mode",
    "pressure_peak",
]


@dataclass(frozen=True)
class ModeFit:
    """Identification result for one modal time series."""

    f_damped: float
    f_natural: float
    zeta: float
    amplitude: float
    phase_deg: float
    rms_residual: float
    n_crossings: int
    period_damped: float
    phase_error_deg: float = float("nan")

    @property
    def decay_rate(self) -> float:
        return self.zeta * 2.0 * math.pi * self.f_natural


def zero_crossings(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Upward zero crossings of ``y(t)`` (linear interpolation in time)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    out = []
    for i in range(1, len(y)):
        if y[i - 1] < 0.0 <= y[i]:
            span = y[i] - y[i - 1]
            frac = 0.0 if span == 0.0 else -y[i - 1] / span
            out.append(t[i - 1] + frac * (t[i] - t[i - 1]))
    return np.asarray(out, dtype=float)


def period_from_crossings(t: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Damped period ``T_d`` (mean spacing of upward crossings) and their count."""
    zc = zero_crossings(t, y)
    if len(zc) < 2:
        return float("nan"), 0
    return float(np.mean(np.diff(zc))), len(zc)


def _extrema(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interior local extrema (used for the envelope fit)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    idx = [i for i in range(1, len(y) - 1) if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) < 0.0]
    return t[idx], y[idx]


def envelope_decay_rate(t: np.ndarray, y: np.ndarray, *, min_amp: float = 0.0) -> float:
    """``lambda`` from a log-linear fit of ``|extrema|``; NaN if unresolvable."""
    te, ye = _extrema(t, y)
    if len(te) < 3:
        return float("nan")
    keep = np.abs(ye) > max(min_amp, 1e-12)
    if keep.sum() < 3:
        return float("nan")
    te, ye = te[keep], np.abs(ye[keep])
    A = np.stack([np.ones_like(te), te], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.log(ye), rcond=None)
    return float(-coef[1])


def fit_damped_sinusoid(
    t: np.ndarray, y: np.ndarray, omega_d: float, lam: float
) -> tuple[float, float, float]:
    """Least-squares ``(A, B)`` for ``y ~ e^{-lam t}(A cos + B sin)``; returns
    amplitude, phase (rad) and the RMS residual."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    env = np.exp(-lam * t)
    S = np.stack([env * np.cos(omega_d * t), env * np.sin(omega_d * t)], axis=1)
    coef, *_ = np.linalg.lstsq(S, y, rcond=None)
    res = y - S @ coef
    amp = float(np.hypot(*coef))
    phase = float(math.atan2(-coef[1], coef[0]))
    return amp, phase, float(np.sqrt(np.mean(res**2)))


def fft_peak(
    t: np.ndarray, y: np.ndarray, *, min_f: float = 0.0, max_f: float | None = None
) -> float:
    """Hann-windowed spectral peak frequency (Hz) — independent cross-check."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(y)
    if n < 8:
        return float("nan")
    dt = float(np.mean(np.diff(t)))
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft((y - y.mean()) * w))
    freqs = np.fft.rfftfreq(n, d=dt)
    mask = freqs >= min_f
    if max_f is not None:
        mask &= freqs <= max_f
    if not mask.any():
        return float("nan")
    idx = np.argmax(spec[mask])
    return float(freqs[mask][idx])


def phase_error_deg(
    t: np.ndarray, y: np.ndarray, f_reference: float, *, n_crossings: int = 1
) -> float:
    """Phase error (degrees) of the first ``n_crossings`` upward crossings.

    The reference signal of eq. (2) starts at its *maximum*, ``y = A cos(2 pi f
    t)``, and therefore crosses zero upward at ``t_k = (4k - 1) / (4
    f_reference)`` (``3T/4, 7T/4, ...``).  The measured ``k``-th crossing is
    compared with that grid; the error is normalised by the reference period, so
    it is a true phase error and not a timing error.
    """
    zc = zero_crossings(t, y)
    if len(zc) < n_crossings:
        return float("nan")
    tm = zc[n_crossings - 1]
    tt = (4 * n_crossings - 1) / (4.0 * f_reference)
    return float(360.0 * f_reference * (tm - tt))


def identify_mode(
    t: np.ndarray,
    y: np.ndarray,
    f_reference: float,
    *,
    phase_window_periods: float = 1.0,
) -> ModeFit:
    """Bundle: damped period, decay rate, natural frequency, phase error.

    ``f_reference`` is the inviscid Abramson frequency of the mode; it is only
    used for the *phase* metric (criterion (c)) and as the FFT search bound.
    """
    T_d, n_x = period_from_crossings(t, y)
    f_d = float("nan") if not np.isfinite(T_d) else 1.0 / T_d
    lam = envelope_decay_rate(t, y)
    if not np.isfinite(lam) or lam < 0.0:
        lam = 0.0
    omega_d = 2.0 * math.pi * f_d if np.isfinite(f_d) else float("nan")
    omega_n = math.sqrt(lam * lam + omega_d * omega_d) if np.isfinite(omega_d) else float("nan")
    f_n = omega_n / (2.0 * math.pi) if np.isfinite(omega_n) else float("nan")
    zeta = lam / omega_n if np.isfinite(omega_n) and omega_n > 0 else float("nan")
    if np.isfinite(omega_d) and len(y) >= 8:
        amp, phase, res = fit_damped_sinusoid(t, y, omega_d, lam)
    else:
        amp, phase, res = float("nan"), float("nan"), float("nan")
    # phase error measured on the first reference period only (accumulated drift
    # is reported separately by the campaign as a diagnostic)
    win = t <= phase_window_periods / f_reference if np.isfinite(f_reference) else None
    if win is not None and win.sum() >= 4:
        ph_err = phase_error_deg(t[win], y[win], f_reference, n_crossings=1)
    else:
        ph_err = float("nan")
    return ModeFit(
        f_damped=f_d,
        f_natural=f_n,
        zeta=zeta,
        amplitude=amp,
        phase_deg=math.degrees(phase) if np.isfinite(phase) else float("nan"),
        rms_residual=res,
        n_crossings=int(n_x),
        period_damped=T_d,
        phase_error_deg=ph_err,
    )


def harmonic_amplitude(t: np.ndarray, y: np.ndarray, frequency: float) -> tuple[float, float]:
    """Least-squares amplitude and phase (deg) of the component at ``frequency``.

    Equivalent to a single-bin DFT but exact for non-integer record lengths;
    used to compare the LBM wall pressure against the linear (Faltinsen) wall
    pressure amplitude, which is a *first-harmonic* quantity.  The raw
    instantaneous peak of the same signal is spike-sensitive (interface cells
    entering/leaving the gauge row) and is therefore reported separately.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 8:
        return float("nan"), float("nan")
    w = 2.0 * math.pi * frequency
    S = np.stack([np.cos(w * t), np.sin(w * t)], axis=1)
    coef, *_ = np.linalg.lstsq(S, y - y.mean(), rcond=None)
    return float(np.hypot(*coef)), float(math.degrees(math.atan2(-coef[1], coef[0])))


def pressure_peak(t: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Peak pressure (max, and time of the peak) ignoring NaNs."""
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    if not ok.any():
        return float("nan"), float("nan")
    i = int(np.argmax(np.where(ok, p, -np.inf)))
    return float(p[i]), float(t[i])
