"""Second-order Stokes wave theory — analytical reference for the 2-D wave tank.

Conventions
-----------
* Still-water level at ``z = 0``, sea bed at ``z = -h`` (``z`` points up).
* Wave propagates in ``+x``: phase argument :math:`\\theta = kx - \\omega t`.
* ``a`` is the first-order amplitude (half wave height), ``k`` the wave number,
  ``h`` the still-water depth, ``g`` gravity.

Free-surface elevation (Stokes second order, e.g. Dean & Dalrymple §11.2)::

    eta(x,t) = a cos(theta)
             + (k a^2 / 4) * (3 - sigma^2) / sigma^3 * cos(2 theta),
    sigma = tanh(k h)

Velocity potential of the same order gives the kinematics::

    u = a w cosh(k (z+h)) / sinh(k h) cos(theta)
        + (3/4) a^2 w k cosh(2k (z+h)) / sinh^4(k h) cos(2 theta)
    w = a w sinh(k (z+h)) / sinh(k h) sin(theta)
        + (3/4) a^2 w k sinh(2k (z+h)) / sinh^4(k h) sin(2 theta)

and the linear dispersion relation of the gravity-wave problem, which the
lattice-Boltzmann tank must reproduce::

    w^2 = g k tanh(k h)   <=>   c^2 = (g/k) tanh(k h)

Everything in this module is pure ``numpy``/``math`` so it can be used as the
reference in tests and as the driving signal for the numerical tank.  All
formulas are evaluated with ``second_order=True`` by default; the
``second_order=False`` branch returns the linear (Airy) solution which is the
``eps = k a -> 0`` limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "GRAVITY",
    "Stokes2Wave",
    "dispersion_omega",
    "dispersion_k",
    "phase_speed",
    "second_order_amplitude",
    "stokes_validity",
    "kinematic_residual",
]

GRAVITY = 9.80665


# ---------------------------------------------------------------------------
# dispersion relation
# ---------------------------------------------------------------------------
def dispersion_omega(k: float, h: float, g: float = GRAVITY) -> float:
    """Angular frequency of a gravity wave, ``w^2 = g k tanh(k h)``."""
    k = float(k)
    h = float(h)
    if k <= 0.0 or h <= 0.0:
        raise ValueError("k and h must be positive")
    return math.sqrt(g * k * math.tanh(k * h))


def dispersion_k(omega: float, h: float, g: float = GRAVITY, tol: float = 1e-15,
                 max_iter: int = 200) -> float:
    """Invert ``w^2 = g k tanh(k h)`` for ``k`` by safeguarded Newton iteration.

    ``omega`` is monotone increasing in ``k`` for a fixed depth, so a
    bisection bracket is always available and the iteration cannot wander.
    """
    omega = float(omega)
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    lo, hi = 1e-12, max(omega * omega / g, 1e-12)  # hi ~ deep-water k
    while dispersion_omega(hi, h, g) < omega:
        hi *= 2.0
    k = 0.5 * (lo + hi)
    for _ in range(max_iter):
        f = dispersion_omega(k, h, g) - omega
        if abs(f) <= tol * omega:
            break
        if f < 0.0:
            lo = k
        else:
            hi = k
        # Newton step guarded by the bracket
        kh = k * h
        dk = g * (math.tanh(kh) + kh / math.cosh(kh) ** 2) / (2.0 * dispersion_omega(k, h, g))
        k_new = k - f / dk
        k = k_new if lo < k_new < hi else 0.5 * (lo + hi)
    return k


def phase_speed(k: float, h: float, g: float = GRAVITY) -> float:
    """Closed-form phase speed ``c^2 = (g/k) tanh(k h)``."""
    return dispersion_omega(k, h, g) / float(k)


def second_order_amplitude(a: float, k: float, h: float) -> float:
    """Amplitude of the bound second harmonic, ``(k a^2 / 4)(3 - s^2)/s^3``."""
    s = math.tanh(k * h)
    return 0.25 * k * a * a * (3.0 - s * s) / s ** 3


def stokes_validity(a: float, k: float, h: float) -> dict:
    """Dimensionless numbers controlling the validity of Stokes second order.

    * steepness ``eps = k a`` — truncation of the Stokes expansion is ``O(eps^2)``;
    * relative depth ``k h`` — the second-order term carries ``sigma^-3`` so the
      series degrades for ``k h -> 0`` (shallow water);
    * Ursell number ``U_r = a / (k h)^3 = eps / (k h)^3`` — the classical
      shallow-water ordering parameter.  ``U_r << 1`` keeps the second-order
      bound wave small compared with the first harmonic, ``U_r >> 1`` means
      cnoidal/stream-function theory is needed instead.
    """
    kh = k * h
    eps = k * a
    ursell = eps / kh ** 3
    sigma = math.tanh(kh)
    ratio = 0.0 if a == 0.0 else second_order_amplitude(a, k, h) / a
    return {
        "kh": kh,
        "eps": eps,
        "ursell": ursell,
        "sigma": sigma,
        "second_to_first_ratio": ratio,
        # Ordering assumptions of the Stokes expansion.
        "second_order_ok": bool(ratio < 0.25 and ursell < 1.0),
        "reason": (
            "ok: eps << 1 and Ursell << 1 (deep/intermediate water)"
            if ratio < 0.25 and ursell < 1.0
            else "second-order bound wave not small: use cnoidal / stream-function theory"
        ),
    }


# ---------------------------------------------------------------------------
# wave object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stokes2Wave:
    """A single regular wave train defined by amplitude, depth and wave number."""

    a: float
    k: float
    h: float
    g: float = GRAVITY

    # -- derived quantities -------------------------------------------------
    @property
    def kh(self) -> float:
        return self.k * self.h

    @property
    def sigma(self) -> float:
        return math.tanh(self.kh)

    @property
    def omega(self) -> float:
        return dispersion_omega(self.k, self.h, self.g)

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    @property
    def wavelength(self) -> float:
        return 2.0 * math.pi / self.k

    @property
    def c(self) -> float:
        return self.omega / self.k

    @property
    def eps(self) -> float:
        return self.k * self.a

    @property
    def a2(self) -> float:
        """Bound second-harmonic amplitude."""
        return second_order_amplitude(self.a, self.k, self.h)

    # -- fields -------------------------------------------------------------
    def eta(self, x, t, second_order: bool = True):
        """Free-surface elevation ``eta(x, t)`` (still-water level at 0)."""
        x = np.asarray(x, dtype=float)
        th = self.k * x - self.omega * t
        eta = self.a * np.cos(th)
        if second_order:
            eta = eta + self.a2 * np.cos(2.0 * th)
        return eta

    def kinematics(self, x, z, t, second_order: bool = True):
        """Horizontal/vertical velocity ``(u, w)`` at ``(x, z)``."""
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        th = self.k * x - self.omega * t
        sh = math.sinh(self.kh)
        cosh_kzh = np.cosh(self.k * (z + self.h))
        sinh_kzh = np.sinh(self.k * (z + self.h))
        u = self.a * self.omega * cosh_kzh / sh * np.cos(th)
        w = self.a * self.omega * sinh_kzh / sh * np.sin(th)
        if second_order:
            sh4 = sh ** 4
            u = u + 0.75 * self.a * self.a * self.omega * self.k * np.cosh(
                2.0 * self.k * (z + self.h)) / sh4 * np.cos(2.0 * th)
            w = w + 0.75 * self.a * self.a * self.omega * self.k * np.sinh(
                2.0 * self.k * (z + self.h)) / sh4 * np.sin(2.0 * th)
        return u, w

    def pressure_head(self, x, z, t):
        """Dynamic pressure head ``p/(rho g)`` (first order) plus hydrostatic."""
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        th = self.k * x - self.omega * t
        return -z + self.a * np.cosh(self.k * (z + self.h)) / math.cosh(self.kh) * np.cos(th)

    def harmonics(self) -> tuple:
        """``(a1, a2)`` — the first- and second-harmonic amplitudes."""
        return self.a, self.a2

    def validity(self) -> dict:
        return stokes_validity(self.a, self.k, self.h)

    def summary(self) -> dict:
        return {
            "a": self.a,
            "k": self.k,
            "kh": self.kh,
            "h": self.h,
            "g": self.g,
            "omega": self.omega,
            "period": self.period,
            "wavelength": self.wavelength,
            "phase_speed": self.c,
            "eps": self.eps,
            "a2": self.a2,
            "a2_over_a": self.a2 / self.a if self.a else 0.0,
            "sigma": self.sigma,
            **{k: v for k, v in self.validity().items() if k in ("ursell", "second_order_ok")},
        }


# ---------------------------------------------------------------------------
# diagnostics used by the verification suite
# ---------------------------------------------------------------------------
def kinematic_residual(wave: Stokes2Wave, x, t, second_order: bool = True) -> dict:
    """Residual of the kinematic free-surface boundary condition.

    For an exact solution of the free-surface problem

        F(x,t) = d(eta)/dt + u_s d(eta)/dx - w_s = 0 ,   u_s, w_s evaluated at z = eta.

    The derivatives are taken by central differences, so the finite-difference
    truncation error is separated from the physical residual by refining the
    step.  The residual of the *truncated* second-order solution is ``O(eps^2)``
    which is the quantity reported here: it must tend to zero as ``eps -> 0``
    (that is what test_stokes2 asserts).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    dt = 1e-6 * wave.period
    dx = 1e-6 * wave.wavelength
    eta = wave.eta(x, t, second_order)
    eta_t = (wave.eta(x, t + dt, second_order) - wave.eta(x, t - dt, second_order)) / (2 * dt)
    eta_x = (wave.eta(x + dx, t, second_order) - wave.eta(x - dx, t, second_order)) / (2 * dx)
    u_s, w_s = wave.kinematics(x, eta, t, second_order)
    residual = eta_t + u_s * eta_x - w_s
    scale = wave.a * wave.omega
    return {
        "residual": residual,
        "max_abs": float(np.max(np.abs(residual))),
        "normalised": float(np.max(np.abs(residual)) / scale),
    }