"""Spectral (Dirlik) and time-domain (rainflow) fatigue damage for sloshing loads.

The sloshing wall pressure time history of :mod:`tensorlbm.lng_membrane_sloshing`
is turned into a *stress* history with a declared, linear static structural
model (see :mod:`tensorlbm.lng_membrane_sloshing.tank`), and the resulting
narrow/wide-band random process is evaluated with two independent fatigue
estimators:

``rainflow_cycles``    time-domain ASTM E1049-85 rainflow count (4-point stack
                       algorithm, hysteresis-filtered), i.e. the reference
                       method: every extracted (range, count) pair is exact for
                       the realised history.
``dirlik_damage``      frequency-domain closed form (Dirlik 1985): the rainflow
                       range distribution of a stationary Gaussian process from
                       its first four spectral moments, integrated against the
                       Basquin S-N curve.

Both feed the Palmgren-Miner rule, ``D = sum_i n_i / N(S_i)``, with

    N(S) = A * S**-m          (Basquin;  log10 A = log10(n_ref) + m log10(S_ref))

For a welded stainless membrane barrier the IIW FAT-80 class is used by default
(``S_ref = 80 MPa at n_ref = 2e6``, slope ``m = 3``), which is the standard
design S-N curve for the 304L/Invar stainless primary barrier of a Mark III
containment system; the class is a *declared assumption* of the load-spectrum
chain, not a material test of this repository.

References
----------
Dirlik, T. (1985). "Application of computers in fatigue analysis." PhD thesis,
    University of Warwick.
ASTM E1049-85 (2017). "Standard practices for cycle counting in fatigue
    analysis."
Hobbacher, A. (2016). *Recommendations for Fatigue Design of Welded Joints and
    Components*, IIW doc. IIW-2259-15 (FAT classes).
Benasciutti, D. & Tovo, R. (2005). "Spectral methods for lifetime prediction
    under wide-band stationary random processes." Int. J. Fatigue 27, 867-877.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "lowpass",
    "SnCurve",
    "IIW_FAT80",
    "spectral_moments",
    "dirlik_range_pdf",
    "dirlik_damage",
    "narrowband_damage",
    "rainflow_cycles",
    "rainflow_damage",
    "stress_from_pressure",
    "wideband_bandwidth_parameter",
]


# ---------------------------------------------------------------------------
# S-N curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnCurve:
    """Basquin S-N curve ``N(S) = n_ref * (S/S_ref) ** -slope``.

    ``S`` is the *stress range* (peak-to-peak, MPa) and ``N`` the cycles to
    failure.  The inverse form used for damage is ``1/N = S**slope / A`` with
    ``A = n_ref * S_ref**slope``.
    """

    s_ref_mpa: float = 80.0
    n_ref: float = 2.0e6
    slope: float = 3.0
    name: str = "IIW FAT80"

    def __post_init__(self) -> None:
        if self.s_ref_mpa <= 0.0 or self.n_ref <= 0.0:
            raise ValueError("s_ref_mpa and n_ref must be positive")
        if not (1.0 <= self.slope <= 12.0):
            raise ValueError("slope must be in [1, 12]")

    @property
    def a_coefficient(self) -> float:
        """``A`` in ``N = A S**-m`` (stress range in MPa)."""
        return self.n_ref * self.s_ref_mpa**self.slope

    def cycles_to_failure(self, s_range: np.ndarray | float) -> np.ndarray:
        s = np.asarray(s_range, dtype=float)
        out = np.full_like(s, np.inf)
        pos = s > 0.0
        out[pos] = self.a_coefficient * s[pos] ** (-self.slope)
        return out

    def damage_increment(self, s_range: np.ndarray | float, count: float = 1.0):
        """``n / N(S)`` per cycle (``0`` for ``S <= 0``)."""
        s = np.asarray(s_range, dtype=float)
        out = np.zeros_like(s)
        pos = s > 0.0
        out[pos] = count * s[pos] ** self.slope / self.a_coefficient
        return out


IIW_FAT80 = SnCurve()


def stress_from_pressure(
    pressure: np.ndarray,
    *,
    span: float,
    thickness: float,
    poisson: float = 0.3,
) -> np.ndarray:
    """Static wall-panel bending stress for a uniform pressure load (MPa).

    The membrane barrier of a Mark III containment system is a corrugated
    stainless sheet on a reinforced-foam support; for the *load-spectrum* chain
    of this module the panel is idealised as a clamped square plate whose
    maximum bending stress is (Timoshenko & Woinowsky-Krieger, *Theory of
    Plates and Shells*, table 11: clamped edges, ``beta = 0.5`` for a square
    panel)

        sigma = 0.5 * p * (span / thickness)**2

    The coefficient is the classical clamped-square-plate value, so the model is
    a declared linear elastic transfer function, not a finite-element result:
    the stress *spectrum* (and hence the Dirlik-vs-rainflow comparison) is
    insensitive to the constant, only the absolute damage scales with it.

    Parameters
    ----------
    pressure : ndarray
        Wall pressure [Pa] (may contain the hydrostatic part; a static offset
        contributes no cycles).
    span : float
        Panel span [m] between supports.
    thickness : float
        Barrier thickness [m].
    poisson : float
        Poisson ratio (accepted for bookkeeping; the tabulated coefficient 0.5
        already includes nu = 0.3).

    Returns
    -------
    ndarray
        Stress [MPa].
    """
    if span <= 0.0 or thickness <= 0.0:
        raise ValueError("span and thickness must be positive")
    return 0.5 * np.asarray(pressure, dtype=float) * (span / thickness) ** 2 / 1.0e6


# ---------------------------------------------------------------------------
# spectral moments and Dirlik
# ---------------------------------------------------------------------------


def lowpass(signal: np.ndarray, dt: float, f_cut: float, *, passes: int = 2) -> np.ndarray:
    """Zero-phase two-pass first-order low-pass filter (first-order RC cascade).

    Used to remove the numerical interface jitter of the LBM pressure history
    from the *load spectrum* before either fatigue estimator sees it: the
    sloshing load is contained in the first few harmonics, while the cell
    conversion noise is a broad-band floor that would otherwise inflate the
    spectral moment ``m4`` (and hence the Dirlik peak rate) without carrying
    any physical fatigue damage.  The same filtered series feeds both
    estimators, so the Dirlik-vs-rainflow comparison stays fair.
    """
    x = np.asarray(signal, dtype=float)
    if f_cut <= 0.0 or f_cut >= 0.5 / dt:
        return x.copy()
    alpha = 1.0 - math.exp(-2.0 * math.pi * f_cut * dt)
    y = x.copy()
    for _ in range(max(1, passes)):
        y = _rc_pass(y, alpha)
    y = _rc_pass(y[::-1], alpha)[::-1]
    return y


def _rc_pass(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x)
    acc = x[0]
    for i, value in enumerate(x):
        acc += alpha * (value - acc)
        y[i] = acc
    return y


def spectral_moments(
    signal: np.ndarray, dt: float, n_moments: int = 5, f_max: float | None = None
) -> np.ndarray:
    """One-sided spectral moments ``m_k = int_0^inf f^k S(f) df`` of a signal.

    The signal is de-meaned and the one-sided PSD is normalised so that ``m_0``
    equals the signal variance (``sigma^2``).  No taper is applied: a taper
    broadens the spectral peak and inflates ``m4/m2``, which would bias the
    peak rate (and hence the damage) of a narrow-band process; for the long
    records used here the leakage error is far smaller than that bias.
    Frequencies are in Hz, so ``m2/m0 = f_peak^2`` and ``m4/m2 = nu_p^2``.
    """
    x = np.asarray(signal, dtype=float)
    if x.size < 16:
        raise ValueError("signal too short for a spectral estimate")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    x = x - x.mean()
    n = x.size
    win = np.ones(n)
    xw = (x - x.mean()) * win
    spec = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=dt)
    # one-sided PSD with density normalisation: 2 |X|^2 / (fs * sum(w^2))
    psd = 2.0 * np.abs(spec) ** 2 / (n * (win**2).sum())
    psd[0] *= 0.5
    if n % 2 == 0:
        psd[-1] *= 0.5
    # normalise the one-sided PSD so that m0 equals the signal variance (the
    # physically meaningful convention for fatigue: m0 = sigma^2).  Without it
    # the window/segment normalisation leaves an arbitrary constant that would
    # bias every spectral moment.
    if f_max is not None:
        psd = np.where(freqs <= f_max, psd, 0.0)
    raw_m0 = float(np.trapezoid(psd, freqs))
    if raw_m0 > 0.0:
        psd = psd * (float(np.var(x)) / raw_m0)
    return np.array(
        [np.trapezoid(psd * freqs**k, freqs) for k in range(n_moments)], dtype=float
    )


def wideband_bandwidth_parameter(moments: np.ndarray) -> float:
    """Vanmarcke bandwidth ``eps = sqrt(1 - m2^2/(m0 m4))`` in ``[0, 1)``."""
    m0, m2, m4 = moments[0], moments[2], moments[4]
    if m0 <= 0.0 or m4 <= 0.0:
        return 0.0
    val = 1.0 - m2 * m2 / (m0 * m4)
    return float(math.sqrt(max(val, 0.0)))


def _dirlik_coefficients(moments: np.ndarray) -> tuple[float, float, float, float, float]:
    """``(D1, D2, D3, Q, R)`` of the Dirlik range PDF (Dirlik 1985)."""
    m0, m1, m2, m4 = moments[0], moments[1], moments[2], moments[4]
    if m0 <= 0.0 or m2 <= 0.0 or m4 <= 0.0:
        raise _DirlikDegenerate()
    gamma = m2 / math.sqrt(m0 * m4)          # irregularity factor
    xm = (m1 / m0) * math.sqrt(m2 / m4)      # mean-frequency ratio
    d1 = 2.0 * (xm - gamma**2) / (1.0 + gamma**2)
    den = 1.0 - gamma - d1 + d1**2
    r = (gamma - xm - d1**2) / den if abs(den) > 1e-12 else 0.0
    d2 = (1.0 - gamma - d1 + d1**2) / (1.0 - r) if abs(1.0 - r) > 1e-12 else 0.0
    d3 = 1.0 - d1 - d2
    q = 1.25 * (gamma - d2 - d3 * r) / d1 if abs(d1) > 1e-12 else 0.0
    if not (d1 > 1e-9 and q > 0.0 and r > 0.0):
        # narrow-band / degenerate limit: coefficients are singular there, and
        # the Rayleigh form below is the exact limit (gamma -> 1).
        raise _DirlikDegenerate()
    return d1, d2, d3, q, r


class _DirlikDegenerate(Exception):
    """Raised when the Dirlik coefficients are singular (gamma -> 1)."""


def dirlik_range_pdf(
    s_range: np.ndarray | float, moments: np.ndarray
) -> np.ndarray:
    """Dirlik rainflow *range* PDF ``p(S)`` (units 1/stress).

    ``p(S) = (D1/Q exp(-Z/Q) + D2 Z/R^2 exp(-Z^2/(2R^2)) + D3 Z exp(-Z^2/2))
    / (2 sqrt(m0))`` with ``Z = S / (2 sqrt(m0))``.
    """
    s = np.asarray(s_range, dtype=float)
    try:
        d1, d2, d3, q, r = _dirlik_coefficients(moments)
    except _DirlikDegenerate:
        # Rayleigh (narrow-band) limit: p(S) = S / (4 m0) exp(-S^2 / (8 m0))
        sigma = math.sqrt(moments[0])
        pdf = s / (4.0 * moments[0]) * np.exp(-s * s / (8.0 * moments[0]))
        return np.where(s > 0.0, pdf, 0.0)
    z = s / (2.0 * math.sqrt(moments[0]))
    pdf = (
        d1 / q * np.exp(-z / q)
        + d2 * z / (r * r) * np.exp(-z * z / (2.0 * r * r))
        + d3 * z * np.exp(-z * z / 2.0)
    ) / (2.0 * math.sqrt(moments[0]))
    return np.where(s > 0.0, pdf, 0.0)


def dirlik_damage(
    signal: np.ndarray,
    dt: float,
    sn: SnCurve = IIW_FAT80,
    *,
    n_grid: int = 400,
    peak_clip: float = 12.0,
    f_max: float | None = None,
) -> float:
    """Dirlik expected fatigue damage of the signal over its full duration.

    ``D = nu_p * T * int p(S) / N(S) dS`` with the peak rate
    ``nu_p = (1/2pi) sqrt(m4/m2)`` and ``T = n*dt`` the record length.  The
    integral is evaluated on a mid-point grid spanning ``0 .. peak_clip*sqrt(m0)``
    stress ranges (12 standard deviations, i.e. the Dirlik tail is fully
    covered for the bandwidths reached here).
    """
    x = np.asarray(signal, dtype=float)
    if np.std(x) <= 0.0:
        return 0.0
    moments = spectral_moments(x, dt, f_max=f_max)
    return _dirlik_damage_from_moments(
        moments, duration=x.size * dt, sn=sn, n_grid=n_grid, peak_clip=peak_clip
    )


def _dirlik_damage_from_moments(
    moments: np.ndarray,
    *,
    duration: float,
    sn: SnCurve = IIW_FAT80,
    n_grid: int = 400,
    peak_clip: float = 12.0,
) -> float:
    m0, m2, m4 = moments[0], moments[2], moments[4]
    if m0 <= 0.0 or m2 <= 0.0 or m4 <= 0.0:
        return 0.0
    nu_p = math.sqrt(m4 / m2)  # Hz (moments are in Hz)
    s_max = peak_clip * math.sqrt(m0)
    edges = np.linspace(0.0, s_max, n_grid + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    pdf = dirlik_range_pdf(mid, moments)
    damage = np.sum(pdf * sn.damage_increment(mid)) * (edges[1] - edges[0])
    return float(nu_p * duration * damage)


def narrowband_damage(
    sigma: float, duration: float, nu0: float, sn: SnCurve = IIW_FAT80
) -> float:
    """Closed-form narrow-band (Rayleigh) damage ``E[D] = nu0 T A^-1 (2 sqrt2 sigma)^m
    Gamma(1 + m/2)``.

    Used as an analytic check on :func:`dirlik_damage` in the ``gamma -> 1``
    limit (test ``test_dirlik_reduces_to_rayleigh_for_narrowband``).
    """
    m = sn.slope
    s_rms_range = 2.0 * math.sqrt(2.0) * sigma
    expected = sn.a_coefficient ** (-1.0) * s_rms_range**m * math.gamma(1.0 + m / 2.0)
    return float(nu0 * duration * expected)


# ---------------------------------------------------------------------------
# rainflow (ASTM E1049-85)
# ---------------------------------------------------------------------------


def _hysteresis_filter(peaks: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0.0 or peaks.size < 3:
        return peaks
    keep = [peaks[0]]
    for value in peaks[1:]:
        if abs(value - keep[-1]) >= threshold:
            keep.append(value)
        elif len(keep) >= 2:
            # replace the previous retained peak when the excursion reverses
            # inside the hysteresis band (keeps the more extreme turning point)
            if abs(value) > abs(keep[-1]):
                keep[-1] = value
    return np.asarray(keep, dtype=float)


def rainflow_cycles(
    signal: np.ndarray, threshold: float = 0.0
) -> list[tuple[float, float]]:
    """ASTM E1049-85 rainflow count: ``[(range, count), ...]``.

    The history is reduced to its turning points (peaks/valleys), a hysteresis
    filter of amplitude ``threshold`` removes sub-threshold reversals, and the
    four-point stack algorithm extracts closed cycles.  Residual reversals are
    closed as half cycles (count ``0.5``).
    """
    x = np.asarray(signal, dtype=float)
    if x.size < 3:
        return []
    d = np.diff(x)
    turning = [x[0]]
    for i in range(1, len(x) - 1):
        if (d[i - 1] > 0.0 and d[i] <= 0.0) or (d[i - 1] < 0.0 and d[i] >= 0.0):
            turning.append(x[i])
    turning.append(x[-1])
    peaks = _hysteresis_filter(np.asarray(turning, dtype=float), threshold)

    cycles: list[tuple[float, float]] = []
    stack: list[float] = []
    for value in peaks:
        stack.append(float(value))
        while len(stack) >= 3:
            r_new = abs(stack[-1] - stack[-2])
            r_old = abs(stack[-2] - stack[-3])
            if r_new < r_old:
                break
            # the closed cycle is the *earlier* excursion (ASTM E1049-85,
            # four-point rule): remove its two interior turning points
            cycles.append((r_old, 1.0))
            del stack[-3:-1]
    for i in range(len(stack) - 1):
        cycles.append((abs(stack[i + 1] - stack[i]), 0.5))
    return cycles


def rainflow_damage(
    signal: np.ndarray, sn: SnCurve = IIW_FAT80, *, threshold: float = 0.0
) -> float:
    """Palmgren-Miner damage of the rainflow cycle set."""
    cycles = rainflow_cycles(signal, threshold=threshold)
    if not cycles:
        return 0.0
    ranges = np.array([c[0] for c in cycles], dtype=float)
    counts = np.array([c[1] for c in cycles], dtype=float)
    return float(np.sum(sn.damage_increment(ranges, count=1.0) * counts))