"""Free-surface gauges and wave-analysis tools for the phase-field wave tank.

Two ingredients:

* :func:`surface_elevation` turns a phase field ``phi`` (``1`` = liquid,
  ``0`` = gas, ``tanh`` interface of width ``W``) into a *sub-cell accurate*
  elevation per column by integrating the liquid fraction
  ``f(y) = (1 + phi)/2`` over the water column.  For a symmetric ``tanh``
  interface the integral equals the interface centre exactly, so the estimator
  has no half-cell bias and is insensitive to the interface width -- unlike a
  ``phi = 0.5`` iso-contour crossing, which quantises the surface to the grid.

* :func:`two_probe_split` is the Goda & Suzuki two-point reflection split:
  with complex amplitudes ``eta_1``, ``eta_2`` at two gauges a distance
  ``dx`` apart, the incident/reflected pair follows from the exact 2x2
  linear system (no small-reflection approximation).  ``dx`` should avoid
  multiples of ``lambda/2`` where the system is singular.

All functions are numpy-based.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "HarmonicFit",
    "ReflectionSplit",
    "fit_harmonics",
    "liquid_fraction",
    "phase_speed_from_phase",
    "surface_elevation",
    "two_probe_split",
]


def liquid_fraction(phi: np.ndarray) -> np.ndarray:
    """Map the phase field to a liquid volume fraction in ``[0, 1]``.

    This repository's phase-field convention
    (:func:`tensorlbm.cac_lbm.init_phi_layers`) is ``phi = 1`` in the liquid
    and ``phi = 0`` in the gas, so the liquid fraction *is* the phase field;
    it is only clipped to guard against small Allen-Cahn undershoots.
    """
    return np.clip(np.asarray(phi, dtype=np.float64), 0.0, 1.0)


def surface_elevation(
    phi: np.ndarray,
    *,
    y_start: int = 1,
    y_stop: int | None = None,
    y_ref: float = 0.0,
) -> np.ndarray:
    """Return the free-surface elevation ``eta(x)`` from a 2-D phase field.

    Args:
        phi: ``(ny, nx)`` phase field, ``phi = 1`` liquid near the bottom.
        y_start: first row (bottom wall is row 0 and is excluded).
        y_stop: one past the last row to integrate (default ``ny - 1``; the top
            wall row is excluded).
        y_ref: elevation datum in row units, i.e. the still-water row index.
            ``eta = sum_y f(y) - y_ref``.

    Returns:
        ``(nx,)`` elevation array in lattice cells.
    """
    phi = np.asarray(phi, dtype=np.float64)
    ny = phi.shape[0]
    if y_stop is None:
        y_stop = ny - 1
    fl = liquid_fraction(phi[y_start:y_stop, :])
    return fl.sum(axis=0) - float(y_ref)


@dataclass(frozen=True)
class HarmonicFit:
    """Least-squares harmonic decomposition of a gauge record."""

    omega: float
    mean: float
    a1: float  # amplitude at omega
    phi1: float  # phase (rad) of the omega component, signal = a1 cos(omega t - phi1)
    a2: float  # amplitude at 2 omega
    phi2: float
    c1: complex  # complex amplitude a1 * exp(-i phi1) with convention exp(+i omega t)
    c2: complex
    residual_rms: float

    def to_dict(self) -> dict:
        return {
            "omega": self.omega,
            "mean": self.mean,
            "a1": self.a1,
            "phase1": self.phi1,
            "a2": self.a2,
            "phase2": self.phi2,
            "residual_rms": self.residual_rms,
        }


def fit_harmonics(
    t: np.ndarray,
    eta: np.ndarray,
    omega: float,
    *,
    n_harmonics: int = 2,
) -> HarmonicFit:
    """Fit ``mean + sum_n a_n cos(n omega t - phi_n)`` to a gauge record.

    The window should contain an integer number of periods so that the basis
    functions are orthogonal and the fit is unbiased (a partially covered
    period leaks power into neighbouring harmonics).
    """
    t = np.asarray(t, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    if t.shape != eta.shape:
        raise ValueError("t and eta must have the same shape")
    cols = [np.ones_like(t)]
    for n in range(1, n_harmonics + 1):
        cols.append(np.cos(n * omega * t))
        cols.append(np.sin(n * omega * t))
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, eta, rcond=None)
    fit = A @ coef
    resid = float(np.sqrt(np.mean((eta - fit) ** 2)))
    mean = float(coef[0])
    amps: list[float] = []
    phs: list[float] = []
    cmplx: list[complex] = []
    for n in range(1, n_harmonics + 1):
        c = coef[2 * n - 1]
        s = coef[2 * n]
        amp = float(math.hypot(c, s))
        ph = float(math.atan2(s, c))
        amps.append(amp)
        phs.append(ph)
        # eta = amp cos(n w t - ph) = Re[ amp e^{i(n w t - ph)} ]
        cmplx.append(amp * np.exp(-1j * ph))
    a1 = amps[0]
    phi1 = phs[0]
    a2 = amps[1] if len(amps) > 1 else 0.0
    phi2 = phs[1] if len(phs) > 1 else 0.0
    c1 = cmplx[0]
    c2 = cmplx[1] if len(cmplx) > 1 else 0j
    return HarmonicFit(
        omega=omega,
        mean=mean,
        a1=a1,
        phi1=phi1,
        a2=a2,
        phi2=phi2,
        c1=c1,
        c2=c2,
        residual_rms=resid,
    )


@dataclass(frozen=True)
class ReflectionSplit:
    """Incident/reflected decomposition at one frequency."""

    k: float
    dx: float
    incident: complex  # A  (travels +x)
    reflected: complex  # B  (travels -x)
    kr: float  # |B| / |A|
    a_incident: float
    a_reflected: float
    condition: float  # |sin(k dx)|; small values => ill-conditioned split

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "dx": self.dx,
            "kr": self.kr,
            "a_incident": self.a_incident,
            "a_reflected": self.a_reflected,
            "cond_sin_kdx": self.condition,
            "incident_re": self.incident.real,
            "incident_im": self.incident.imag,
            "reflected_re": self.reflected.real,
            "reflected_im": self.reflected.imag,
        }


def two_probe_split(
    eta1: complex,
    eta2: complex,
    x1: float,
    x2: float,
    k: float,
) -> ReflectionSplit:
    """Goda & Suzuki two-point incident/reflected split.

    **Time convention.** :func:`fit_harmonics` returns the complex amplitude
    ``c`` of ``eta(t) = Re[c e^{+i omega t}]``, i.e. ``c = a e^{-i phi}`` for
    ``a cos(omega t - phi)``.  For a wave travelling in ``+x``,
    ``a cos(kx - omega t) = Re[a e^{-ikx} e^{+i omega t}]``, so the spatial
    dependence of ``c`` is ``e^{-ikx}``.  Writing the complex amplitude field as

        c(x) = A e^{-i k x} + B e^{+i k x}     (A: +x incident, B: -x reflected)

    the two gauges give the exact 2x2 system

        A e^{-i k x1} + B e^{+i k x1} = c1
        A e^{-i k x2} + B e^{+i k x2} = c2

    which is what this function solves.  (Mixing up this sign conjugates the
    split: incident and reflected amplitudes are exchanged and the recovered
    phase speed flips sign -- a trap that cost one sweep here.)

    Returns ``kr = |B|/|A|``.
    """
    if abs(x2 - x1) < 1e-12:
        raise ValueError("gauge separation must be non-zero")
    # M = [[e^{-i k x1}, e^{+i k x1}], [e^{-i k x2}, e^{+i k x2}]]
    e1m = np.exp(-1j * k * x1)
    e1p = np.exp(1j * k * x1)
    e2m = np.exp(-1j * k * x2)
    e2p = np.exp(1j * k * x2)
    det = e1m * e2p - e1p * e2m  # = 2i sin(k dx)
    if abs(det) < 1e-12:
        raise ValueError("degenerate gauge spacing (k dx is a multiple of pi)")
    A = (eta1 * e2p - e1p * eta2) / det
    B = (e1m * eta2 - eta1 * e2m) / det
    a_inc = float(abs(A))
    a_ref = float(abs(B))
    kr = float(a_ref / a_inc) if a_inc > 0 else float("inf")
    return ReflectionSplit(
        k=k,
        dx=float(x2 - x1),
        incident=complex(A),
        reflected=complex(B),
        kr=kr,
        a_incident=a_inc,
        a_reflected=a_ref,
        condition=float(abs(np.sin(k * (x2 - x1)))),
    )


def phase_speed_from_phase(
    eta1: complex,
    eta2: complex,
    dx: float,
    omega: float,
    *,
    k_expected: float | None = None,
) -> tuple[float, float]:
    """Phase speed from the phase shift between two gauges.

    With the ``e^{+i omega t}`` convention of :func:`fit_harmonics` a ``+x``
    travelling wave has ``arg(c) = -k x``, so ``k_meas = -(arg(eta2)-arg(eta1))/dx``.
    The phase difference is unwrapped to the branch closest to
    ``-k_expected * dx`` (default: the principal branch ``|dphi| < pi``), then
    ``c = omega / k_meas``.

    Returns ``(c_measured, k_measured)``.
    """
    dphi = float(np.angle(eta2 * np.conj(eta1)))
    if k_expected is not None:
        target = -k_expected * dx
        dphi = dphi + 2.0 * math.pi * round((target - dphi) / (2.0 * math.pi))
    k_meas = -dphi / dx
    if abs(k_meas) < 1e-15:
        return float("inf"), k_meas
    return float(omega / k_meas), float(k_meas)