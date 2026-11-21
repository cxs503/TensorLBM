"""Analytic steady-periodic solutions for pulsatile (Womersley) duct flow.

Two geometries share the same complex-phasor structure

    u_x(y, t) = Re{ U(y) e^{i omega t} },      a(t) = a0 cos(omega t)

driven by a uniform, time-harmonic body force a(t) (equivalently a pressure
gradient dp/dx = -rho a):

* ``channel`` -- pressure-driven oscillatory flow between parallel plates
  (Sexl 1930; the parallel-plate counterpart of Womersley's tube solution):

      U(y) = (a0 / (i omega)) [ 1 - cosh(lam * eta) / cosh(lam) ],
      eta  = (y - H/2) / (H/2),   lam = sqrt(i) * alpha,
      alpha = (H/2) * sqrt(omega / nu).

* ``pipe`` -- axisymmetric pulsatile flow in a circular tube
  (Womersley 1955, J. Physiol. 127:553, the reference of this case):

      U(r) = (a0 / (i omega)) [ 1 - J0(lam * r/R) / J0(lam) ],
      alpha = R * sqrt(omega / nu).

``alpha`` is the Womersley number of the geometry (half-gap based for the
channel, radius based for the pipe): ``alpha^2`` is the ratio of the
unsteady/inertial term to the viscous term, i.e. the Stokes layer thickness
is ``H/(2 alpha)`` (channel) or ``R/alpha`` (pipe).

Sign/phase convention: ``U`` is complex; the physical velocity is
``Re{U e^{i omega t}}``.  The *phase lag* of the flow behind the driving force
is ``-arg(U)`` in degrees (positive = lagging).

Limit checks used by the tests: ``alpha -> 0`` recovers the steady Poiseuille
profile ``u_max = a0 H^2 / (8 nu)`` (channel) / ``a0 R^2 / (4 nu)`` (pipe).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import jv

GEOMETRIES = ("channel", "pipe")


def omega_from_alpha(geometry: str, alpha: float, nu: float, length: float) -> float:
    """Angular frequency giving Womersley number ``alpha``.

    ``length`` is the full gap ``H`` for the channel and the radius ``R`` for
    the pipe.
    """
    if geometry == "channel":
        return 4.0 * alpha * alpha * nu / (length * length)
    if geometry == "pipe":
        return alpha * alpha * nu / (length * length)
    raise ValueError(f"unknown geometry {geometry!r}")


def alpha_from_omega(geometry: str, omega: float, nu: float, length: float) -> float:
    """Womersley number for a given angular frequency."""
    if geometry == "channel":
        return 0.5 * length * math.sqrt(omega / nu)
    if geometry == "pipe":
        return length * math.sqrt(omega / nu)
    raise ValueError(f"unknown geometry {geometry!r}")


def womersley_lam(alpha: float, geometry: str = "channel") -> complex:
    """Complex Stokes parameter of the geometry.

    * channel: ``lam = sqrt(i) alpha``   (``lam^2 = +i alpha^2``)
    * pipe:    ``lam = sqrt(-i) alpha``  (``lam^2 = -i alpha^2``)

    The sign difference follows from the radial Bessel operator,
    ``lap J0(lam r/R) = -(lam^2/R^2) J0``: the momentum equation
    ``i omega U = a0 + nu lap U`` needs ``lam^2 = -i alpha^2`` in the tube,
    while the 1D plate problem needs ``+i alpha^2``.  With the wrong sign the
    pipe solution still has the right amplitude at alpha=5 (complex conjugate
    up to the 1/i rotation) but the wrong sign in the quasi-steady limit.
    Caught by ``test_quasi_steady_limit_recovers_poiseuille``.
    """
    if geometry == "channel":
        return np.sqrt(1j) * alpha
    if geometry == "pipe":
        return np.sqrt(-1j) * alpha
    raise ValueError(f"unknown geometry {geometry!r}")


def _shape(geometry: str, lam: complex, coord: np.ndarray) -> np.ndarray:
    """Dimensionless (``a0/omega`` normalised) complex profile shape."""
    coord = np.asarray(coord, dtype=float)
    if geometry == "channel":
        return 1.0 - np.cosh(lam * coord) / np.cosh(lam)
    if geometry == "pipe":
        # scipy jv supports complex arguments; J0(0) == 1
        return 1.0 - jv(0, lam * coord) / jv(0, lam)
    raise ValueError(f"unknown geometry {geometry!r}")


def phasor(
    geometry: str,
    coord: np.ndarray,
    length: float,
    alpha: float,
    a0: float,
    omega: float,
) -> np.ndarray:
    """Complex velocity amplitude ``U`` at normalised wall coordinate.

    ``coord`` is the physical ordinate: ``y`` in ``[0, H]`` for the channel,
    radius ``r`` in ``[0, R]`` for the pipe.
    """
    coord = np.asarray(coord, dtype=float)
    lam = womersley_lam(alpha, geometry)
    if geometry == "channel":
        eta = (coord - 0.5 * length) / (0.5 * length)
    elif geometry == "pipe":
        eta = coord / length
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    return (a0 / (1j * omega)) * _shape(geometry, lam, eta)


def centerline_phasor(
    geometry: str, length: float, alpha: float, a0: float, omega: float
) -> complex:
    """Complex velocity amplitude on the centreline (y=H/2 or r=0)."""
    lam = womersley_lam(alpha, geometry)
    if geometry == "channel":
        shape = 1.0 - 1.0 / np.cosh(lam)
    elif geometry == "pipe":
        shape = 1.0 - 1.0 / jv(0, lam)
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    return complex((a0 / (1j * omega)) * shape)


def centerline_amplitude_unit(geometry: str, alpha: float) -> float:
    """``|U(0)|`` in units of ``a0/omega`` (independent of a0 and omega)."""
    lam = womersley_lam(alpha)
    if geometry == "channel":
        return float(abs(1.0 - 1.0 / np.cosh(lam)))
    if geometry == "pipe":
        return float(abs(1.0 - 1.0 / jv(0, lam)))
    raise ValueError(f"unknown geometry {geometry!r}")


def centerline_phase_lag_deg(geometry: str, alpha: float) -> float:
    """Phase lag of the centreline velocity behind the driving force [deg].

    Uses the same complex phasor as :func:`centerline_phasor` (with
    ``a0/omega = 1``), i.e. the ``1/i`` rotation of the pressure-gradient
    phasor is included -- the velocity lags the force by ~90 deg at alpha=5.
    """
    lam = womersley_lam(alpha, geometry)
    if geometry == "channel":
        shape = 1.0 - 1.0 / np.cosh(lam)
    elif geometry == "pipe":
        shape = 1.0 - 1.0 / jv(0, lam)
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    u = shape / 1j
    return float(-math.degrees(math.atan2(u.imag, u.real)))


def force_amplitude_for_peak(
    geometry: str, length: float, alpha: float, omega: float, u_peak: float
) -> float:
    """Force amplitude ``a0`` such that the analytic peak speed equals ``u_peak``."""
    return float(u_peak * omega / centerline_amplitude_unit(geometry, alpha))


def poiseuille_centerline(geometry: str, length: float, a0: float, nu: float) -> float:
    """Steady (``omega -> 0``) centreline speed for a uniform force ``a0``."""
    if geometry == "channel":
        return a0 * length * length / (8.0 * nu)
    if geometry == "pipe":
        return a0 * length * length / (4.0 * nu)
    raise ValueError(f"unknown geometry {geometry!r}")


def poiseuille_profile(
    geometry: str, coord: np.ndarray, length: float, a0: float, nu: float
) -> np.ndarray:
    """Steady Poiseuille profile (channel: parabolic in y; pipe: parabolic in r)."""
    coord = np.asarray(coord, dtype=float)
    if geometry == "channel":
        eta = (coord - 0.5 * length) / (0.5 * length)
        return a0 * length * length / (8.0 * nu) * (1.0 - eta * eta)
    if geometry == "pipe":
        return a0 * length * length / (4.0 * nu) * (1.0 - (coord / length) ** 2)
    raise ValueError(f"unknown geometry {geometry!r}")


# ---------------------------------------------------------------------------
# Tabulated reference values (computed once with scipy high precision; used by
# the unit tests as an independent cross-check of the formulas above).
# ---------------------------------------------------------------------------
REFERENCE_ALPHA5 = {
    "channel": {
        "alpha": 5.0,
        "abs_U0_over_a0_w": 1.0540395420190274,
        "phase_lag_deg": 91.21371227934286,
    },
    "pipe": {
        "alpha": 5.0,
        "abs_U0_over_a0_w": 1.1604597069127809,
        "phase_lag_deg": 89.85244921580414,
    },
}