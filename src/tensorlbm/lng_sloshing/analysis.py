"""Signal analysis for the LNG-tank sloshing campaign.

Everything here operates on plain numpy arrays (float64) so the numbers in the
report can be reproduced from the saved time series.

Estimators
----------
``refined_frequency``  zero-padded periodogram peak followed by a least-squares
                       frequency scan (relative accuracy ~1e-3 for a decaying
                       sinusoid sampled over >= 5 periods).  Used for the
                       Abramson frequency comparison, where a bias would leak
                       straight into the acceptance criterion.
``decay_damping``      log-envelope fit of the analytic (Hilbert) amplitude ->
                       damping ratio ``zeta = -slope / omega``.
``harmonic_amplitudes``  linear least squares fit of ``DC + sum_k a_k cos + b_k sin``
                       -> ``A_k = hypot(a_k, b_k)`` for the 1st/2nd/3rd harmonic,
                       used for the Faltinsen second-order check.
``pressure_peak``      peak |p|, its time, and the RMS over a window.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "refined_frequency",
    "decay_damping",
    "harmonic_amplitudes",
    "pressure_peak",
    "detrend",
    "band_limited_rms",
    "reconstruction_error",
]


def detrend(t: np.ndarray, x: np.ndarray, order: int = 1) -> np.ndarray:
    """Remove a polynomial trend (mean level / slow bookkeeping drift)."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.size < order + 2:
        return x - x.mean() if x.size else x
    a = np.polyfit(t - t.mean(), x, order)
    return x - np.polyval(a, t - t.mean())


def _ls_periodogram(t: np.ndarray, x: np.ndarray, freqs: np.ndarray, nuisance: int = 2) -> np.ndarray:
    """Least-squares sinusoid amplitude after projecting out a polynomial trend.

    The free-surface mass ledger has a slow (non-physical) bookkeeping drift of
    order 0.4 % of the liquid inventory; a plain periodogram lets that drift
    win the peak search.  Projecting a polynomial nuisance basis out of both the
    signal and the trial sinusoid removes it without touching the sloshing band.
    """
    tau = (t - t.mean()) / max(t.max() - t.min(), 1e-30)
    P = np.vander(tau, nuisance + 1)
    Q, _ = np.linalg.qr(P)
    yv = x - Q @ (Q.T @ x)
    out = np.empty(freqs.size)
    for i, f in enumerate(freqs):
        c = np.cos(2.0 * np.pi * f * t)
        s = np.sin(2.0 * np.pi * f * t)
        c = c - Q @ (Q.T @ c)
        s = s - Q @ (Q.T @ s)
        cc, ss, cs = c @ c, s @ s, c @ s
        cy, sy = c @ yv, s @ yv
        det = cc * ss - cs * cs
        if abs(det) < 1e-20:
            out[i] = 0.0
            continue
        a = (ss * cy - cs * sy) / det
        b = (cc * sy - cs * cy) / det
        out[i] = float(np.hypot(a, b))
    return out


def refined_frequency(
    t,
    x,
    f_guess: float | None = None,
    rel_band: float = 0.35,
    n_scan: int = 1201,
    nuisance: int = 2,
    f_min: float | None = None,
    f_max: float | None = None,
) -> dict:
    """Estimate the dominant frequency of a (possibly decaying) sinusoid.

    The estimator is a least-squares periodogram that projects a polynomial
    nuisance basis (order ``nuisance``) out of *both* the data and the trial
    sinusoid, so the slow mass-ledger drift of the free-surface model cannot win
    the peak search.  With ``f_guess=None`` the search range is prior-free
    (``[3/T_record, f_max]``); ``f_guess`` only widens/narrows that window and
    is reported separately as ``f_window`` for cross-checking.

    Returns ``{f, f_window, f_global, amplitude, residual_rms, snr, ...}``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 12:
        raise ValueError("need at least 12 samples")
    dt = float(np.median(np.diff(t)))
    span = float(t[-1] - t[0])
    if f_min is None:
        f_min = 3.0 / max(span, 1e-30)
    if f_max is None:
        f_max = 0.25 / dt
    f_min = max(float(f_min), 1e-9)
    f_max = max(float(f_max), 2.0 * f_min)

    def _scan(lo, hi, m):
        lo = min(max(lo, f_min), f_max)
        hi = min(max(hi, f_min), f_max)
        if hi <= lo * 1.001:
            hi = min(lo * 1.001 + 1e-12, f_max)
        grid = np.linspace(lo, hi, m)
        mag = _ls_periodogram(t, x, grid, nuisance=nuisance)
        j = int(np.argmax(mag))
        f = float(grid[j])
        if 0 < j < grid.size - 1:
            y0, y1, y2 = mag[j - 1], mag[j], mag[j + 1]
            den = y0 - 2.0 * y1 + y2
            if abs(den) > 1e-18:
                f += 0.5 * (y0 - y2) / den * (grid[1] - grid[0])
        return f, float(mag[j]), grid

    f_global, _, _ = _scan(f_min, f_max, n_scan)
    if f_guess is None:
        f = f_global
        f_window = float("nan")
    else:
        f, _, _ = _scan(f_guess * (1.0 - rel_band), f_guess * (1.0 + rel_band), n_scan)
        f_window = f

    # amplitude / residual of the fitted single sinusoid (+ polynomial nuisance)
    w = 2.0 * np.pi * f
    tau = (t - t.mean()) / max(span, 1e-30)
    design = np.column_stack([np.cos(w * t), np.sin(w * t), np.vander(tau, nuisance + 1)])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    amp = float(np.hypot(coef[0], coef[1]))
    resid = x - design @ coef
    resid_rms = float(np.sqrt(np.mean(resid**2)))
    return {
        "f": float(f),
        "f_global": float(f_global),
        "f_window": float(f_window),
        "amplitude": amp,
        "residual_rms": resid_rms,
        "snr": float(amp / max(resid_rms, 1e-15)),
        "n_samples": int(n),
        "dt": dt,
        "span": span,
    }


def _hilbert_envelope(x: np.ndarray) -> np.ndarray:
    n = x.size
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(X * h))


def decay_damping(t, x, f: float, n_windows: int = 6, skip_frac: float = 0.05) -> dict:
    """Damping ratio from the decay of the fitted sinusoid amplitude.

    The record is split into ``n_windows`` windows; inside each window the
    amplitude at ``f`` is a least-squares fit (with a linear nuisance term, which
    a short window keeps distinguishable from the sinusoid).  ``log(A)`` is then
    fitted against the window centre time, so ``zeta = -slope / (2 pi f)``.

    This is deliberately *not* a Hilbert-envelope fit: an envelope over the whole
    (drifting) record, or a global polynomial detrend, biases a strongly damped
    exponential by 30 % or more (the campaign's first version did exactly that).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    keep = t >= t[0] + skip_frac * (t[-1] - t[0])
    t, x = t[keep], x[keep]
    if t.size < 8 * n_windows:
        return {"zeta": float("nan"), "envelope_slope": float("nan"), "r2": float("nan"), "envelope0": float("nan")}
    w = 2.0 * np.pi * f
    edges = np.linspace(0, t.size, n_windows + 1).astype(int)
    tc, amp = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        tt, xx = t[a:b], x[a:b]
        if tt.size < 6:
            continue
        tau = (tt - tt.mean()) / max(tt.max() - tt.min(), 1e-30)
        design = np.column_stack([np.cos(w * tt), np.sin(w * tt), np.vander(tau, 2)])
        coef, *_ = np.linalg.lstsq(design, xx, rcond=None)
        a1 = float(np.hypot(coef[0], coef[1]))
        if a1 <= 0:
            continue
        tc.append(0.5 * (tt[0] + tt[-1]))
        amp.append(a1)
    if len(amp) < 3:
        return {"zeta": float("nan"), "envelope_slope": float("nan"), "r2": float("nan"), "envelope0": float("nan")}
    tc = np.asarray(tc)
    la = np.log(np.asarray(amp))
    slope, intercept = np.polyfit(tc, la, 1)
    fit = slope * tc + intercept
    ss_res = float(np.sum((la - fit) ** 2))
    ss_tot = float(np.sum((la - la.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    omega = 2.0 * np.pi * f
    return {
        "zeta": float(-slope / omega),
        "envelope_slope": float(slope),
        "r2": float(r2),
        "envelope0": float(math.exp(intercept - slope * float(t[0]))),
        "n_windows": len(amp),
    }


def harmonic_amplitudes(
    t, x, f1: float, n_harm: int = 3, t_start: float | None = None, trend_order: int = 1
) -> dict:
    """Least-squares harmonic decomposition at ``f1, 2 f1, ...``.

    Returns ``{A1, A2, ..., phases, dc, residual_rms, ratios}``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t_start is not None:
        m = t >= t_start
        t, x = t[m], x[m]
    if t.size < 4 * n_harm + 4:
        raise ValueError("too few samples for harmonic fit")
    tau = (t - t.mean()) / max(t.max() - t.min(), 1e-30)
    cols = [np.vander(tau, trend_order + 1)]
    w = 2.0 * np.pi * f1
    for k in range(1, n_harm + 1):
        cols.append(np.cos(k * w * t))
        cols.append(np.sin(k * w * t))
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    amp = {}
    phase = {}
    off = trend_order + 1
    for k in range(1, n_harm + 1):
        a, b = coef[off + 2 * (k - 1)], coef[off + 2 * (k - 1) + 1]
        amp[k] = float(np.hypot(a, b))
        phase[k] = float(math.atan2(-b, a))
    resid = x - design @ coef
    ratios = {
        f"A{k}/A1": float(amp[k] / amp[1]) if amp[1] > 1e-15 else float("nan")
        for k in range(2, n_harm + 1)
    }
    return {
        **{f"A{k}": amp[k] for k in amp},
        **{f"phase{k}": phase[k] for k in phase},
        "dc": float(coef[trend_order]),
        "residual_rms": float(np.sqrt(np.mean(resid**2))),
        "ratios": ratios,
    }


def pressure_peak(p, t=None, t_start: float | None = None) -> dict:
    """Peak absolute pressure, its time, and window RMS/mean."""
    p = np.asarray(p, dtype=float)
    if t is not None:
        t = np.asarray(t, dtype=float)
        if t_start is not None:
            m = t >= t_start
            p = p[m]
            t = t[m]
    if p.size == 0:
        return {"p_max": float("nan"), "t_peak": float("nan"), "p_rms": float("nan"), "p_mean": float("nan")}
    finite = np.isfinite(p)
    if not finite.any():
        return {"p_max": float("nan"), "t_peak": float("nan"), "p_rms": float("nan"), "p_mean": float("nan")}
    pf = p[finite]
    idx = int(np.nanargmax(np.abs(np.where(finite, p, np.nan))))
    return {
        "p_max": float(abs(p[idx])),
        "p_signed_max": float(p[idx]),
        "t_peak": float(t[idx]) if t is not None else float("nan"),
        "p_rms": float(np.sqrt(np.mean(pf**2))),
        "p_mean": float(pf.mean()),
    }


def band_limited_rms(t, x, f_lo: float, f_hi: float) -> float:
    """RMS of ``x`` inside a frequency band (FFT mask)."""
    t = np.asarray(t, dtype=float)
    x = detrend(t, np.asarray(x, dtype=float))
    dt = float(np.median(np.diff(t)))
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, d=dt)
    mask = (f >= f_lo) & (f <= f_hi)
    Xm = np.where(mask, X, 0.0)
    y = np.fft.irfft(Xm, n=x.size)
    return float(np.sqrt(np.mean(y**2)))


def reconstruction_error(model: np.ndarray, measured: np.ndarray, mask=None) -> dict:
    """Relative L2 / max errors of ``model`` vs ``measured`` (both same shape).

    Returns ``{rel_l2, rel_max, max_abs_error, rms_measured, norm}`` where
    ``rel_l2 = ||model-measured||_2 / ||measured||_2`` and ``rel_max`` uses the
    peak of the measured signal as the scale.
    """
    model = np.asarray(model, dtype=float)
    measured = np.asarray(measured, dtype=float)
    if mask is not None:
        model = model[mask]
        measured = measured[mask]
    if model.shape != measured.shape:
        raise ValueError("model and measured must have the same shape")
    diff = model - measured
    l2_meas = float(np.sqrt(np.mean(measured**2)))
    return {
        "rel_l2": float(np.sqrt(np.mean(diff**2)) / max(l2_meas, 1e-15)),
        "rel_max": float(np.max(np.abs(diff)) / max(np.max(np.abs(measured)), 1e-15)),
        "max_abs_error": float(np.max(np.abs(diff))),
        "rms_measured": l2_meas,
    }