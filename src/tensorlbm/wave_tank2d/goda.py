"""Goda–Suzuki two-point (and multi-probe least-squares) wave reflection analysis.

Physical setting
----------------
A regular wave train propagates toward a reflecting structure and the reflected
train travels back, so the measured surface elevation at a probe located a
distance ``s`` seaward of the reflector is

.. math::

    \\eta(s,t) = \\mathrm{Re}\\left[ I e^{i(\\omega t + k s)}
                                 + R e^{i(\\omega t - k s)} \\right]

with ``I`` the complex incident amplitude (at the reflector) and ``R`` the
complex reflected amplitude.  With two probes at ``s1`` and ``s2`` the two
phasors are two linear equations in ``(I, R)``; solving them is the
Goda–Suzuki two-point separation method (Goda & Suzuki, 1976, *Estimation of
incident and reflected waves in random wave experiments*, Proc. 15th ICCE,
pp. 828–845).  The classical closed-form of that paper is the same algebraic
solution written out for the specific trigonometric identities of a
two-probe system; here the exact linear solve is used, which is equivalent
but also permits arbitrary probe spacings and arbitrary numbers of probes.

The method is ill-conditioned when ``k (s2 - s1) = n*pi`` (the two probe
signals are then linearly dependent); Goda & Suzuki recommend
``0.05 < ds/lambda < 0.45``.  :func:`conditioning` returns the amplification
factor ``1/|sin(k ds)|`` so a campaign can reject bad probe layouts instead of
silently reporting noise.

All functions take/return complex phasors with the ``Re[Z e^{i omega t}]``
convention, matching the ``a cos(wt) + b sin(wt)`` Fourier convention used by
:func:`phasor_from_series`.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "phasor_from_series",
    "fit_harmonics",
    "two_point_reflection",
    "least_squares_reflection",
    "apparent_reflection",
    "conditioning",
    "wave_number_from_phasors",
    "spatiotemporal_harmonics",
    "spatial_decay_rate",
    "phase_speed_from_phasors",
    "recommended_spacing",
]


# ---------------------------------------------------------------------------
# spectral analysis of probe time series
# ---------------------------------------------------------------------------
def phasor_from_series(t, eta, omega: float) -> complex:
    """Least-squares phasor ``Z`` with ``eta(t) = Re[Z e^{i omega t}] + const``.

    A constant (mean level) is fitted simultaneously so a slowly drifting mean
    water level does not leak into the harmonic amplitude.  Any number of
    samples at arbitrary spacing is accepted (uniform sampling is *not*
    required), which keeps the analysis usable for checkpointed runs.
    """
    t = np.asarray(t, dtype=float)
    eta = np.asarray(eta, dtype=float)
    wt = omega * t
    A = np.stack([np.cos(wt), -np.sin(wt), np.ones_like(wt)], axis=1)
    coef, *_ = np.linalg.lstsq(A, eta, rcond=None)
    return complex(coef[0], coef[1])


def fit_harmonics(t, eta, omega: float, n_harm: int = 2, detrend: bool = True) -> dict:
    """Fit the first ``n_harm`` harmonics of a single probe record.

    Returns ``{"phasors": [Z1, Z2, ...], "amplitudes": [...], "mean": float,
    "rms_residual": float}``.  A single simultaneous least-squares fit is used
    so the harmonics stay orthogonal even for a partially filled last period.
    """
    t = np.asarray(t, dtype=float)
    eta = np.asarray(eta, dtype=float)
    cols = []
    for n in range(1, n_harm + 1):
        wt = n * omega * t
        cols += [np.cos(wt), -np.sin(wt)]
    if detrend:
        cols.append(np.ones_like(t))
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, eta, rcond=None)
    phasors = [complex(coef[2 * i], coef[2 * i + 1]) for i in range(n_harm)]
    resid = eta - A @ coef
    return {
        "phasors": phasors,
        "amplitudes": [abs(z) for z in phasors],
        "mean": float(coef[-1]) if detrend else 0.0,
        "rms_residual": float(np.sqrt(np.mean(resid ** 2))),
    }


# ---------------------------------------------------------------------------
# two-point separation (Goda–Suzuki)
# ---------------------------------------------------------------------------
def two_point_reflection(z1: complex, z2: complex, k: float,
                         s1: float, s2: float) -> dict:
    """Separate incident and reflected phasors from two probes.

    Parameters
    ----------
    z1, z2:
        Complex phasors at the two probes.
    k:
        Wave number used for the separation (normally the measured/incident one).
    s1, s2:
        Distances of the probes from the reflecting surface, in the same unit as
        ``1/k`` (any monotone convention works as long as it is consistent).

    Returns
    -------
    dict with ``I``, ``R`` (complex, at the reflector), ``K_R = |R|/|I|``,
    ``phase`` (reflection phase in radians), ``conditioning`` and the raw
    probe phasors.
    """
    ds = float(s2) - float(s1)
    s = abs(math.sin(k * ds))
    if s < 1e-12:
        raise ValueError(
            f"probe spacing is degenerate: k*(s2-s1) = {k * ds:.6f} rad "
            "(multiple of pi); change the spacing"
        )
    M = np.array([[np.exp(1j * k * s1), np.exp(-1j * k * s1)],
                  [np.exp(1j * k * s2), np.exp(-1j * k * s2)]], dtype=complex)
    inc, R = np.linalg.solve(M, np.array([complex(z1), complex(z2)], dtype=complex))
    k_r = abs(R) / abs(inc) if abs(inc) > 0 else float("inf")
    return {
        "I": complex(inc),
        "R": complex(R),
        "K_R": float(k_r),
        "phase": float(np.angle(R / inc)) if abs(inc) > 0 else float("nan"),
        "conditioning": float(1.0 / s),
        "k_ds": float(k * ds),
        "probe_phasors": [complex(z1), complex(z2)],
    }


def least_squares_reflection(z, s, k: float) -> dict:
    """Multi-probe least-squares separation ``Z_j = I e^{iks_j} + R e^{-iks_j}``.

    Over-determined (``len(z) >= 3``) and therefore insensitive to the
    ``k ds = n pi`` degeneracy of the two-probe method; the residual of the fit
    is a direct measure of how well the two-wave model describes the record
    (i.e. of higher-harmonic or evanescent contamination).
    """
    z = np.asarray(z, dtype=complex)
    s = np.asarray(s, dtype=float)
    if z.size != s.size or z.size < 2:
        raise ValueError("z and s must have the same length >= 2")
    M = np.stack([np.exp(1j * k * s), np.exp(-1j * k * s)], axis=1)
    coef, *_ = np.linalg.lstsq(M, z, rcond=None)
    inc, R = complex(coef[0]), complex(coef[1])
    resid = z - M @ coef
    return {
        "I": inc,
        "R": R,
        "K_R": float(abs(R) / abs(inc)) if abs(inc) > 0 else float("inf"),
        "phase": float(np.angle(R / inc)) if abs(inc) > 0 else float("nan"),
        "rms_residual": float(np.sqrt(np.mean(np.abs(resid) ** 2))),
        "n_probes": int(z.size),
    }


def apparent_reflection(eta_max: float, eta_min: float) -> float:
    """Envelope estimate ``K_R = (eta_max - eta_min)/(eta_max + eta_min)``.

    Only exact for a pure standing pattern with a node antinode pair; provided
    as an independent sanity cross-check of the probe separation.
    """
    hi, lo = float(eta_max), float(eta_min)
    if hi + lo == 0.0:
        return float("nan")
    return (hi - lo) / (hi + lo)


def conditioning(k: float, ds: float) -> float:
    """Error amplification factor ``1/|sin(k ds)|`` of the two-point method."""
    s = abs(math.sin(k * ds))
    return float("inf") if s < 1e-12 else float(1.0 / s)


def recommended_spacing(k: float, n: int = 4) -> dict:
    """Probe spacings safe for the two-point method (``0.05 <= ds/lambda <= 0.45``)."""
    lam = 2.0 * math.pi / k
    out = {}
    for j in range(1, n + 1):
        ds = lam * (0.05 + 0.40 * (j - 1) / max(n - 1, 1))
        out[f"ds_over_lambda_{0.05 + 0.40 * (j - 1) / max(n - 1, 1):.2f}"] = {
            "ds": ds,
            "ds_over_lambda": ds / lam,
            "conditioning": conditioning(k, ds),
        }
    return out


# ---------------------------------------------------------------------------
# wave-number / phase-speed measurement from phasors along x
# ---------------------------------------------------------------------------
def wave_number_from_phasors(z, s, detrend_phase: bool = True) -> dict:
    """Recover ``k`` (and phase speed with a known ``omega``) from a phasor row.

    The phase of the travelling wave is linear in ``s``; a least-squares line
    through the unwrapped phase gives ``k`` without any assumption on the
    amplitude or on the presence of a reflected component (the reflected part
    shows up as scatter of the phase, reported as ``phase_scatter``).
    """
    z = np.asarray(z, dtype=complex)
    s = np.asarray(s, dtype=float)
    phase = np.unwrap(np.angle(z))
    slope, intercept = np.polyfit(s, phase, 1)
    resid = phase - (slope * s + intercept)
    return {
        "k": float(slope),
        "phase_scatter": float(np.std(resid)),
        "n_probes": int(z.size),
    }


def phase_speed_from_phasors(z, s, omega: float) -> dict:
    """Phase speed from a phasor row: ``c = omega / k``."""
    res = wave_number_from_phasors(z, s)
    k = res["k"]
    return {
        "k": k,
        "c": float(omega / k) if k != 0 else float("inf"),
        "phase_scatter": res["phase_scatter"],
        "n_probes": res["n_probes"],
    }

# ---------------------------------------------------------------------------
# bound vs free harmonic separation (spatio-temporal decomposition)
# ---------------------------------------------------------------------------
def spatiotemporal_harmonics(eta_row_matrix, x, t, omega: float, k_bound: float,
                             n_max: int = 2, wavelength_count: int | None = None) -> dict:
    """Split each temporal harmonic into its *bound* and *free* wave numbers.

    For every harmonic ``n`` the fitted phasor row ``Z_n(x)`` (``eta =
    Re[Z_n e^{i n omega t}]``) is Fourier transformed along ``x``.  In a
    horizontally periodic box of length ``L = n_lambda * lambda`` the admissible
    wave numbers are integer multiples of ``2 pi / L``; the Stokes *bound*
    harmonic sits at ``|k| = n k_bound`` while a spurious *free* wave of the same
    frequency sits at a different wave number (for gravity waves ``k(2 omega) !=
    2 k``).  Reporting the two separately is the only way to tell a genuine
    second-order wave from numerical contamination.

    Returns ``{"bound": {...}, "free": {...}, "spectrum": {...}}`` where the
    amplitudes are in the same units as ``eta``.
    """
    eta = np.asarray(eta_row_matrix, dtype=float)
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    nt, nx = eta.shape
    length = x[-1] - x[0] + (x[1] - x[0])
    m0 = wavelength_count if wavelength_count is not None else int(round(length * k_bound / (2 * np.pi)))
    out = {"bound": {}, "free": {}, "spectrum": {}}
    for n in range(1, n_max + 1):
        zn = np.array([phasor_from_series(t, eta[:, j], n * omega) for j in range(nx)])
        coef = np.fft.fft(zn) / nx
        amps = np.abs(coef)
        m_bound = (-n * m0) % nx                       # +x propagation -> e^{-i n k x}
        bound = float(amps[m_bound])
        others = np.delete(amps, m_bound)
        free_rms = float(np.sqrt(np.sum(others ** 2)))
        out["bound"][n] = {"amplitude": bound, "wavenumber_index": int(m_bound),
                           "wavenumber": 2 * np.pi * m_bound / length}
        out["free"][n] = {"rms_amplitude": free_rms,
                          "max_other_amplitude": float(others.max()) if others.size else 0.0,
                          "wavenumber_of_max": float(
                              2 * np.pi * np.argmax(amps) / length)}
        top = np.argsort(amps)[::-1][:4]
        out["spectrum"][n] = [
            {"m": int(i), "k": float(2 * np.pi * i / length),
             "amplitude": float(amps[i]),
             "k_over_kbound": float((2 * np.pi * i / length) / k_bound)} for i in top]
    return out


def spatial_decay_rate(phasors, x) -> dict:
    """Fit ``|Z(x)| = A exp(-beta x)`` — the spatial damping rate per lattice cell."""
    z = np.asarray(phasors, dtype=complex)
    x = np.asarray(x, dtype=float)
    amp = np.abs(z)
    ok = amp > 1e-12
    slope, intercept = np.polyfit(x[ok], np.log(amp[ok]), 1)
    pred = slope * x[ok] + intercept
    resid = np.log(amp[ok]) - pred
    return {
        "beta_per_cell": float(-slope),
        "A0": float(np.exp(intercept)),
        "fit_rms": float(np.std(resid)),
        "amplitude_ratio_end_to_start": float(amp[-1] / amp[0]) if amp[0] > 0 else float("nan"),
    }
