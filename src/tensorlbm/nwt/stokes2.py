"""Closed-form second-order Stokes wave theory (finite water depth).

This module is the *analytic anchor* for the numerical-wave-tank (NWT)
validation surface in :mod:`tensorlbm.nwt`.  Everything here is a closed-form
expression; it contains no numerical solver and no fitted constant.

Notation (SI or self-consistent lattice units — the formulae are dimensionless
in the sense that ``k``, ``h``, ``a``, ``g``, ``omega`` must share a length/time
system):

* ``h``   still-water depth (bed to still-water level, SWL)
* ``k``   wave number, ``k = 2 pi / lambda``
* ``a``   first-order amplitude, ``H/2``
* ``eps`` wave steepness ``k a``
* ``sigma`` ``tanh(k h)``
* ``omega`` angular frequency from the *linear* dispersion relation
  ``omega**2 = g k tanh(k h)``
* ``theta`` ``k x - omega t``  (propagation towards ``+x``)
* ``z``   vertical coordinate measured **upwards from the bed**, so the SWL is
  ``z = h`` and the free surface is ``z = h + eta``.

Second-order Stokes solution used (Dean & Dalrymple 1991, eqs. 3.51-3.56;
Fenton 1990):

.. math::

    \\eta(x,t) = a\\cos\\theta + \\frac{k a^2}{4}\\frac{3-\\sigma^2}{\\sigma^3}\\cos 2\\theta

.. math::

    \\phi = \\frac{a g}{\\omega}\\frac{\\cosh k(z+h)}{\\cosh kh}\\sin\\theta
         + \\frac{3}{8}a^2\\omega\\frac{\\cosh 2k(z+h)}{\\sinh^4(kh)}\\sin 2\\theta

with velocity components :math:`u=\\partial_x\\phi`, :math:`w=\\partial_z\\phi`.

Only arrays are duplicated where the caller needs them; all scalar helpers are
plain floats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "dispersion_omega",
    "dispersion_k",
    "phase_speed",
    "group_speed",
    "Stokes2Wave",
]


# ---------------------------------------------------------------------------
# linear dispersion relation and its derivatives
# ---------------------------------------------------------------------------


def dispersion_omega(k: float, h: float, g: float) -> float:
    """Linear angular frequency ``omega = sqrt(g k tanh(k h))``."""
    return math.sqrt(g * k * math.tanh(k * h))


def phase_speed(k: float, h: float, g: float) -> float:
    """Linear phase speed ``c = omega / k``; equivalently ``c**2 = (g/k) tanh(k h)``."""
    return dispersion_omega(k, h, g) / k


def group_speed(k: float, h: float, g: float) -> float:
    """Group speed ``c_g = n c`` with ``n = (1 + 2 k h / sinh 2kh) / 2``."""
    kh = k * h
    # 2 kh / sinh(2 kh) -> 0 for deep water; guard against sinh overflow
    t = 0.0 if 2.0 * kh > 300.0 else 2.0 * kh / math.sinh(2.0 * kh)
    n = 0.5 * (1.0 + t)
    return n * phase_speed(k, h, g)


def dispersion_k(omega: float, h: float, g: float, tol: float = 1e-15, itmax: int = 200) -> float:
    """Solve ``omega**2 = g k tanh(k h)`` for ``k`` by Newton iteration.

    The iteration is started from the deep-water estimate
    ``k0 = omega**2 / g`` and is safeguarded by bisection-free damping
    (``k <- k - resid / d resid``) until ``|resid| < tol * omega**2``.
    """
    k = omega * omega / g  # deep-water start
    target = omega * omega
    for _ in range(itmax):
        kh = k * h
        th = math.tanh(kh)
        f = g * k * th - target
        if abs(f) <= tol * target:
            break
        df = g * (th + kh * (1.0 - th * th))
        if df <= 0.0:
            raise ValueError("non-monotone dispersion residual derivative")
        k = max(k - f / df, 1e-12 * k + 1e-300)
    return k


# ---------------------------------------------------------------------------
# second-order Stokes wave
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stokes2Wave:
    """A second-order Stokes wave in finite depth.

    Construct with :meth:`from_steepness`, :meth:`from_amplitude` or
    :meth:`from_period`; all derived quantities are exact closed forms.
    """

    amplitude: float  # a = H/2
    wavenumber: float  # k
    depth: float  # h
    gravity: float  # g

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_steepness(cls, steepness: float, wavenumber: float, depth: float, gravity: float):
        """Build from ``eps = k a`` (the usual non-dimensional wave steepness)."""
        return cls(amplitude=steepness / wavenumber, wavenumber=wavenumber, depth=depth, gravity=gravity)

    @classmethod
    def from_amplitude(cls, amplitude: float, wavenumber: float, depth: float, gravity: float):
        return cls(amplitude=amplitude, wavenumber=wavenumber, depth=depth, gravity=gravity)

    @classmethod
    def from_period(cls, period: float, amplitude: float, depth: float, gravity: float):
        """Build from the wave period, solving the dispersion relation for ``k``."""
        omega = 2.0 * math.pi / period
        return cls(
            amplitude=amplitude,
            wavenumber=dispersion_k(omega, depth, gravity),
            depth=depth,
            gravity=gravity,
        )

    # -- derived linear quantities ----------------------------------------
    @property
    def steepness(self) -> float:
        return self.wavenumber * self.amplitude

    @property
    def sigma(self) -> float:
        """``tanh(k h)``."""
        return math.tanh(self.wavenumber * self.depth)

    @property
    def omega(self) -> float:
        return dispersion_omega(self.wavenumber, self.depth, self.gravity)

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    @property
    def wavelength(self) -> float:
        return 2.0 * math.pi / self.wavenumber

    @property
    def phase_speed(self) -> float:
        return phase_speed(self.wavenumber, self.depth, self.gravity)

    @property
    def group_speed(self) -> float:
        return group_speed(self.wavenumber, self.depth, self.gravity)

    @property
    def kh(self) -> float:
        return self.wavenumber * self.depth

    @property
    def second_harmonic_amplitude(self) -> float:
        """Amplitude of the ``cos 2 theta`` term, ``(k a^2 / 4)(3 - sigma^2)/sigma^3``."""
        s = self.sigma
        return 0.25 * self.wavenumber * self.amplitude**2 * (3.0 - s * s) / s**3

    @property
    def skewness_bound(self) -> float:
        """Second-order/first-order amplitude ratio ``A2/A1`` (diagnostic)."""
        return abs(self.second_harmonic_amplitude) / self.amplitude

    @property
    def c2_correction(self) -> float:
        """Second-order phase-speed correction ``c2 = c (1 + (ka)^2 C2)``.

        ``C2 = (9 - 10 sigma^2 + 9 sigma^4) / (8 sigma^4)`` (Stokes third-order
        dispersion, Dean & Dalrymple eq. 3.60).  Reported for context only: the
        NWT acceptance criterion compares against the *linear* closed form, so
        this bound tells the reviewer how large the neglected correction is.
        """
        s = self.sigma
        return (9.0 - 10.0 * s * s + 9.0 * s**4) / (8.0 * s**4)

    @property
    def second_order_phase_speed(self) -> float:
        return self.phase_speed * (1.0 + self.steepness**2 * self.c2_correction)

    # -- field evaluation --------------------------------------------------
    def phase(self, x, t: float):
        """``theta = k x - omega t`` (accepts python floats or tensors)."""
        return self.wavenumber * x - self.omega * t

    def elevation(self, x, t: float):
        """Free-surface elevation ``eta(x, t)`` including the ``cos 2 theta`` term."""
        th = self.phase(x, t)
        return self.amplitude * torch.cos(th) + self.second_harmonic_amplitude * torch.cos(2.0 * th)

    def potential(self, x, z, t: float):
        """Velocity potential ``phi(x, z, t)`` with ``z`` measured up from the bed."""
        k, h, a, w = self.wavenumber, self.depth, self.amplitude, self.omega
        th = self.phase(x, t)
        shkh = math.sinh(k * h)
        phi1 = a * self.gravity / w * torch.cosh(k * z) / math.cosh(k * h) * torch.sin(th)
        phi2 = (
            0.375
            * a
            * a
            * w
            * torch.cosh(2.0 * k * z)
            / shkh**4
            * torch.sin(2.0 * th)
        )
        return phi1 + phi2

    def potential_t(self, x, z, t: float):
        """Time derivative ``d phi / dt`` (needed for the Bernoulli surface condition)."""
        k, h, a, w = self.wavenumber, self.depth, self.amplitude, self.omega
        th = self.phase(x, t)
        shkh = math.sinh(k * h)
        d1 = -a * self.gravity * torch.cosh(k * z) / math.cosh(k * h) * torch.cos(th)
        d2 = (
            -0.75
            * a
            * a
            * w
            * w
            * torch.cosh(2.0 * k * z)
            / shkh**4
            * torch.cos(2.0 * th)
        )
        return d1 + d2

    def kinematics(self, x, z, t: float):
        """Return ``(u, w)`` — horizontal/vertical orbital velocity to second order."""
        k, h, a, w = self.wavenumber, self.depth, self.amplitude, self.omega
        th = self.phase(x, t)
        shkh = math.sinh(k * h)
        u1 = a * w * torch.cosh(k * z) / shkh * torch.cos(th)
        w1 = a * w * torch.sinh(k * z) / shkh * torch.sin(th)
        c2 = 0.75 * a * a * w * k / shkh**4
        u2 = c2 * torch.cosh(2.0 * k * z) * torch.cos(2.0 * th)
        w2 = c2 * torch.sinh(2.0 * k * z) * torch.sin(2.0 * th)
        return u1 + u2, w1 + w2

    def horizontal_velocity_amplitude_at_swl(self) -> float:
        """``max|u|`` on the SWL (``z = h``): ``a omega / tanh(kh)``.

        This is the LBM Mach-number driver: the free-surface orbital velocity
        must stay small compared with the lattice sound speed ``c_s = 1/sqrt3``
        for the solver to be in the incompressible limit.
        """
        return self.amplitude * self.omega / self.sigma

    def harmonic_amplitudes(self) -> tuple[float, float]:
        """``(A1, A2)`` — analytic cos-theta and cos-2theta amplitudes."""
        return self.amplitude, self.second_harmonic_amplitude

    # -- analytic self-check ----------------------------------------------
    def surface_bc_residual(self, n: int = 400, t: float = 0.0) -> dict[str, float]:
        """Normalised residual of the two exact free-surface conditions.

        The second-order Stokes solution satisfies the kinematic and dynamic
        surface conditions only up to the retained order, so the residual is a
        *known* function of the steepness:

        * kinematic  ``R_k = (eta_t + u eta_x - w) / (a omega)`` at ``z=h+eta``;
        * dynamic    ``R_d = (g eta + phi_t + (u^2+w^2)/2) / (g a)`` at ``z=h+eta``.

        ``R_d`` carries a **constant** offset of order ``(ka)^2 / 2`` (relative to
        ``g a``, i.e. first order in ``eps``): it is the second-order Bernoulli
        constant ``(1/2) a^2 omega^2`` that in the classical solution is balanced
        by a second-order *mean water level* term, which the two-harmonic anchor
        deliberately omits (it is a constant and cannot affect any harmonic
        amplitude).  The harmonic part of the dynamic residual — the part that a
        wrong second-order coefficient would corrupt — is returned as
        ``dynamic_rms_harmonic`` and is second order in ``eps``.

        Numerically both are evaluated with a fourth-order finite difference in
        ``x``.  The residuals must decay like ``eps**2`` (relative), which is
        exactly what the closed-form anchor of the NWT is tested against.
        """
        x = torch.linspace(0.0, 2.0 * self.wavelength, n, dtype=torch.float64)
        dx = float(x[1] - x[0])
        eta = self.elevation(x, t)
        d_eta_dt = (self.elevation(x, t + 1e-6) - self.elevation(x, t - 1e-6)) / 2e-6
        # fourth-order centred first derivative in x
        f = self.elevation(x, t)
        d_eta_dx = (-f[4:] + 8.0 * f[3:-1] - 8.0 * f[1:-3] + f[:-4]) / (12.0 * dx)
        xs = x[2:-2]
        etas = eta[2:-2]
        z = self.depth + etas
        u, w = self.kinematics(xs, z, t)
        r_kin = (d_eta_dt[2:-2] + u * d_eta_dx - w) / (self.amplitude * self.omega)
        g = self.gravity
        r_dyn = (g * etas + self.potential_t(xs, z, t) + 0.5 * (u * u + w * w)) / (g * self.amplitude)
        r_dyn_h = r_dyn - r_dyn.mean()
        return {
            "kinematic_rms": float(torch.sqrt((r_kin**2).mean())),
            "kinematic_max": float(r_kin.abs().max()),
            "dynamic_rms": float(torch.sqrt((r_dyn**2).mean())),
            "dynamic_max": float(r_dyn.abs().max()),
            "dynamic_mean": float(r_dyn.mean()),
            "dynamic_rms_harmonic": float(torch.sqrt((r_dyn_h**2).mean())),
            "dynamic_max_harmonic": float(r_dyn_h.abs().max()),
        }

    def as_dict(self) -> dict:
        return {
            "amplitude_a": self.amplitude,
            "wavenumber_k": self.wavenumber,
            "wavelength": self.wavelength,
            "depth_h": self.depth,
            "kh": self.kh,
            "kh_over_pi": self.kh / math.pi,
            "gravity_g": self.gravity,
            "omega": self.omega,
            "period": self.period,
            "steepness_eps": self.steepness,
            "sigma_tanh_kh": self.sigma,
            "phase_speed_closed_form": self.phase_speed,
            "group_speed": self.group_speed,
            "second_harmonic_amplitude": self.second_harmonic_amplitude,
            "second_harmonic_ratio": self.skewness_bound,
            "phase_speed_second_order": self.second_order_phase_speed,
            "max_orbital_velocity_swl": self.horizontal_velocity_amplitude_at_swl(),
        }