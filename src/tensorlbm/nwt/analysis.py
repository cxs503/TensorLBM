"""Post-processing for the numerical wave tank: harmonic content, phase speed, budgets.

All estimators here are *independent* of the solver: they consume only the
recorded probe time series ``eta_j(t)`` (from
:func:`tensorlbm.nwt.wavemaker.column_free_surface`) and the analytic reference
from :mod:`tensorlbm.nwt.stokes2`.

Free-surface harmonic decomposition
-----------------------------------
For a second-order Stokes wave the elevation is

.. math::

    \\eta = A_1 \\cos\\theta + A_2 \\cos 2\\theta + O((ka)^3 H)

so a three-term least-squares fit (``cos omega t``, ``sin omega t``,
``cos 2 omega t``, ``sin 2 omega t``, ``1``) recovers ``A_1`` and ``A_2``
without requiring an integer number of periods in the window, and the fit
residual quantifies how much energy sits outside the two Stokes harmonics
(nonlinear contamination, noise from interface cell conversion, ...).

Phase speed
-----------
Two independent estimators (both reported, agreement is part of the evidence):

* **phase difference** between two probes at the driving frequency —
  ``k = -arg(Z_2 conj(Z_1)) / Delta x`` (unambiguous while
  ``|k Delta x| < pi``), hence ``c = omega / k``;
* **cross-correlation travel time** — the lag of the normalised
  cross-correlation peak (parabolic sub-step refinement), hence
  ``c = Delta x / (lag * dt)``.

The measured ``c`` is compared against the closed form
``c = sqrt(g/k tanh(k h))`` *evaluated at the measured* ``k``, which makes the
test a genuine dispersion check: a solver that propagates at the wrong speed
cannot satisfy it while the closed form is pinned to its own measured ``k``.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "fit_harmonics",
    "wavenumber_from_probes",
    "lag_from_cross_correlation",
    "inventory_drift",
]


def _series(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.reshape(-1).to(torch.float64)
    return torch.as_tensor(x, dtype=torch.float64).reshape(-1)


def fit_harmonics(eta, dt: float, omega: float) -> dict:
    """Least-squares Stokes harmonic fit at ``omega`` and ``2 omega``.

    Returns ``A1`` (fundamental amplitude), ``A2`` (second-harmonic amplitude),
    the mean level, residual RMS, and the fit's total RMS for context.
    """
    y = _series(eta)
    n = y.numel()
    t = torch.arange(n, dtype=torch.float64) * dt
    A = torch.stack(
        [
            torch.cos(omega * t),
            torch.sin(omega * t),
            torch.cos(2.0 * omega * t),
            torch.sin(2.0 * omega * t),
            torch.ones_like(t),
        ],
        dim=1,
    )
    coef, *_ = torch.linalg.lstsq(A, y.unsqueeze(1))
    c1, s1, c2, s2, mean = (float(v) for v in coef[:, 0])
    resid = A @ coef[:, 0] - y
    return {
        "A1": math.hypot(c1, s1),
        "A2": math.hypot(c2, s2),
        "phase1": math.atan2(-s1, c1),
        "phase2": math.atan2(-s2, c2),
        "mean_level": mean,
        "residual_rms": float(torch.sqrt((resid**2).mean())),
        "signal_rms": float(torch.sqrt((y**2).mean())),
    }


def wavenumber_from_probes(eta_probes, x_probes, dt: float, omega: float, k_sign: float = 1.0) -> dict:
    """Wave number from the spatial phase of the probe records.

    Each record is reduced to a complex amplitude ``Z_j`` at ``omega``; for a
    train travelling toward ``+x`` the phase decreases linearly,
    ``arg Z_j = arg Z_0 - k x_j``.  The phases are unwrapped across the probe
    array (unambiguous while ``|k Delta x| < pi`` between adjacent probes) and a
    least-squares straight line is fitted, which uses the whole probe span and is
    therefore appreciably more accurate than a single phase difference.  The
    adjacent-pair wave numbers are also returned for a spread/uncertainty check.
    """
    from .reflection import complex_amplitude

    Z = [complex_amplitude(e, dt, omega) for e in eta_probes]
    x = [float(v) for v in x_probes]
    if len(x) < 2:
        raise ValueError("need at least two distinct probe positions")
    phases = [math.atan2(z.imag, z.real) for z in Z]
    unwrapped = [phases[0]]
    for j in range(1, len(phases)):
        d = phases[j] - phases[j - 1]
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        unwrapped.append(unwrapped[-1] + d)
    n = len(x)
    mx = sum(x) / n
    mp = sum(unwrapped) / n
    num = sum((x[j] - mx) * (unwrapped[j] - mp) for j in range(n))
    den = sum((x[j] - mx) ** 2 for j in range(n))
    slope = num / den if den else 0.0
    k_meas = -slope * (1.0 if k_sign >= 0 else -1.0)
    ks = []
    for j in range(n - 1):
        dx = x[j + 1] - x[j]
        if dx == 0:
            continue
        ks.append(-(unwrapped[j + 1] - unwrapped[j]) / dx)
    return {
        "k_measured": k_meas,
        "k_spread": (max(ks) - min(ks)) if len(ks) > 1 else 0.0,
        "k_pairs": ks,
        "phase_unwrapped": unwrapped,
        "c_measured": omega / abs(k_meas) if k_meas else float("inf"),
    }


def _upsample(x: torch.Tensor, up: int) -> torch.Tensor:
    """Band-limited (Fourier) interpolation onto ``up`` sub-samples per step."""
    n = x.numel()
    if up == 1:
        return x
    X = torch.fft.rfft(x)
    m = n * up
    Y = torch.zeros(m // 2 + 1, dtype=X.dtype)
    Y[: X.numel()] = X
    return torch.fft.irfft(Y, m) * up


def lag_from_cross_correlation(eta1, eta2, dt: float, upsample: int = 8) -> dict:
    """Travel time between two probes (cross-correlation peak + carrier phase).

    Two stages, because each alone is inadequate for a narrowband wave record:

    1. **coarse lag** — the peak of the normalised cross-correlation, evaluated
       by FFT on a grid refined by ``upsample`` sub-samples per step (both
       records are Fourier-interpolated first).  This resolves the ``2 pi``
       ambiguity of the phase measurement without any prior knowledge of the
       period;
    2. **sub-sample refinement** — the carrier (spectral peak) phase difference,
       ``tau = -arg(F_b conj(F_a)) / omega_0``, unwrapped to the coarse lag.  A
       quadratic fit of the correlation peak is *not* used: for a narrowband
       signal the peak is so flat that the finite-record envelope tilt biases the
       parabola vertex by O(1) sample.

    Returns the lag in time steps, the lag in time units and the peak correlation.
    """
    a = _series(eta1)
    b = _series(eta2)
    a = a - a.mean()
    b = b - b.mean()
    a_u = _upsample(a, upsample)
    b_u = _upsample(b, upsample)
    n = a_u.numel()
    nfft = 1 << int(math.ceil(math.log2(max(2 * n, 2))))
    fa_u = torch.fft.rfft(a_u, nfft)
    fb_u = torch.fft.rfft(b_u, nfft)
    cc = torch.fft.irfft(fb_u * torch.conj(fa_u), nfft)
    norm = float(torch.sqrt((a_u * a_u).sum() * (b_u * b_u).sum()))
    cc = cc / (norm if norm > 0 else 1.0)
    search = torch.cat([cc[nfft - n :], cc[:n]])
    lags = torch.arange(-n, n, dtype=torch.float64) / upsample
    idx = int(torch.argmax(search))
    lag_coarse = float(lags[idx])

    # carrier phase refinement on the un-interpolated records
    ns = a.numel()
    Fa = torch.fft.rfft(a)
    Fb = torch.fft.rfft(b)
    k0 = int(torch.argmax(Fa.abs()[1:]).item()) + 1
    omega0 = 2.0 * math.pi * k0 / ns
    cross = Fb[k0] * torch.conj(Fa[k0])
    phase = math.atan2(float(cross.imag), float(cross.real))
    period = 2.0 * math.pi / omega0
    tau = -phase / omega0
    # unwrap to the branch closest to the coarse correlation lag
    tau += period * round((lag_coarse - tau) / period)
    return {
        "lag_samples": tau,
        "lag_time": tau * dt,
        "lag_coarse_samples": lag_coarse,
        "peak_correlation": float(search[idx]),
        "carrier_period": period,
        "carrier_frequency": omega0,
    }


def inventory_drift(inventory, steps_per_period: int, inventory_initial: float | None = None) -> dict:
    """Relative drift of the total liquid inventory over a record.

    Reports the drift at each sampled period boundary, the maximum relative
    drift, and the maximum **per-period** drift (which is the quantity bounded
    by the NWT mass-conservation criterion).
    """
    v = _series(inventory)
    v0 = inventory_initial if inventory_initial is not None else float(v[0])
    rel = (v - v0) / v0
    n_periods = max(1, (v.numel() - 1) // steps_per_period)
    per_period = []
    for p in range(1, n_periods + 1):
        i = min(p * steps_per_period, v.numel() - 1)
        prev = min((p - 1) * steps_per_period, v.numel() - 1)
        per_period.append(float((v[i] - v[prev]) / v0))
    return {
        "max_relative_drift": float(rel.abs().max()),
        "final_relative_drift": float(rel[-1]),
        "max_per_period_drift": max(abs(x) for x in per_period) if per_period else 0.0,
        "per_period_drift": per_period,
        "n_periods": n_periods,
    }