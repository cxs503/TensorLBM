"""Critical Shields stress and dimensionless sediment parameters.

Conventions
-----------
All quantities are SI (m, s) unless stated otherwise.

Dimensionless grain size (Soulsby 1997, eq. 2.13; Soulsby & Whitehouse 2005)
---------------------------------------------------------------------------
::

    d* = d [ g (s - 1) / nu^2 ]^(1/3)          (dimensionless)

**Unit check** (this is the bug that was fixed in mission m8016): the cube-root
form is the only one that is dimensionally consistent, because

    [ g (s-1) / nu^2 ] = (m s^-2) / (m^4 s^-2) = m^-3   ->  ^(1/3) = m^-1,

so ``d * m^-1`` is dimensionless.  The previously shipped expression
``d * sqrt((s-1) g) / nu`` has units ``m^-0.5`` and is *not* dimensionless;
for a 0.1 mm grain it returned ``d* = 252`` instead of ``d* = 1.86`` (a factor
136 error), which silently moved every evaluation of the critical-Shields
curve into the wrong hydraulic regime.

Critical Shields stress (Soulsby & Whitehouse 2005)
---------------------------------------------------
::

    theta_cr(d*) = 0.30 / (1 + 1.2 d*) + 0.055 [1 - exp(-0.02 d*)]

This is the curve used as the *reference* for criterion C1.  Asymptotes:
``theta_cr -> 0.30`` for ``d* -> 0`` (fine/cohesive-like limit, matching the
classical Shields diagram at low Re*) and ``theta_cr -> 0.055`` as ``d* -> inf``
(0.050 at ``d* = 100``; the classical rough-turbulent gravel value is 0.047-0.06).
The minimum (``theta_cr ~ 0.033``) sits near ``d* ~ 10``, which is the
well-known minimum of the Shields diagram (Re* ~ 8-10).  Soulsby (1997) also
quotes the large-``d*`` branch ``0.24/d* + 0.055[1-exp(-0.02 d*)]``; the two
branches agree to better than 2 % for ``d* >= 10``, the single expression is
used here for continuity.

Settling velocity (van Rijn 1984, simplified form)
--------------------------------------------------
For ``1 < d* < 10`` (the sand range of the C1/C2 anchors)
::

    w_s = 10 nu/d * [ (1 + 0.01 d*^3)^(1/2) - 1 ]

which reduces to the Stokes value ``g (s-1) d^2 / (18 nu)`` in the viscous
limit (verified in ``tests/test_sediment_scour.py::test_settling_stokes_limit``).
"""

from __future__ import annotations

import math

__all__ = [
    "critical_shields",
    "dimensionless_grain_size",
    "grain_diameter_from_dstar",
    "dimensionless_settling_velocity",
    "settling_velocity",
    "shields_parameter",
    "shear_velocity_from_shields",
]

#: Standard sediment density contrast (s = rho_s / rho).
DEFAULT_S = 1.65
#: Gravitational acceleration (m s^-2).
G = 9.81
#: von Karman constant.
KAPPA = 0.41


def dimensionless_grain_size(d: float, s: float, g: float, nu: float) -> float:
    """Dimensionless grain size ``d* = d [g(s-1)/nu^2]^(1/3)``.

    Args:
        d: Grain diameter (m).
        s: Density ratio ``rho_s / rho``.
        g: Gravitational acceleration (m s^-2).
        nu: Kinematic viscosity of the fluid (m^2 s^-1).
    """
    if d <= 0.0 or nu <= 0.0 or s <= 1.0:
        raise ValueError("require d > 0, nu > 0, s > 1")
    return d * (g * (s - 1.0) / (nu * nu)) ** (1.0 / 3.0)


def grain_diameter_from_dstar(d_star: float, s: float, g: float, nu: float) -> float:
    """Inverse of :func:`dimensionless_grain_size` -- ``d`` from ``d*``."""
    if d_star <= 0.0:
        raise ValueError("d* must be positive")
    return d_star * (nu * nu / (g * (s - 1.0))) ** (1.0 / 3.0)


def dimensionless_settling_velocity(
    w_s: float, d: float, s: float, g: float, nu: float = 1.0e-6
) -> float:
    """Dimensionless settling velocity ``w* = w_s / [ (s-1) g nu ]^(1/3)``."""
    if w_s < 0.0 or d <= 0.0 or s <= 1.0:
        raise ValueError("require w_s >= 0, d > 0, s > 1")
    return w_s / ((s - 1.0) * g * nu) ** (1.0 / 3.0)


def settling_velocity(d: float, s: float, g: float, nu: float) -> float:
    """van Rijn (1984) simplified settling velocity (m s^-1).

    ``w_s = 10 nu/d [ (1 + 0.01 d*^3)^(1/2) - 1 ]``; reduces to Stokes
    ``(s-1) g d^2 / (18 nu)`` for small ``d*``.
    """
    d_star = dimensionless_grain_size(d, s, g, nu)
    return 10.0 * nu / d * (math.sqrt(1.0 + 0.01 * d_star**3) - 1.0)


def critical_shields(d_star: float) -> float:
    """Soulsby & Whitehouse (2005) critical Shields stress for ``d*``.

    ``theta_cr = 0.30/(1 + 1.2 d*) + 0.055 (1 - exp(-0.02 d*))``

    Args:
        d_star: Dimensionless grain size (see
            :func:`dimensionless_grain_size`).

    Returns:
        The critical Shields stress ``theta_cr`` at incipient motion.
    """
    if d_star <= 0.0:
        raise ValueError("d* must be positive")
    return 0.30 / (1.0 + 1.2 * d_star) + 0.055 * (1.0 - math.exp(-0.02 * d_star))


def shields_parameter(tau_b: float, d: float, s: float, g: float) -> float:
    """Shields number ``theta = tau_b / ((rho_s - rho) g d)`` from bed stress.

    Using ``tau_b = rho u*^2`` this is ``u*^2 / ((s-1) g d)``.
    """
    if d <= 0.0 or s <= 1.0 or tau_b < 0.0:
        raise ValueError("require d > 0, s > 1, tau_b >= 0")
    return tau_b / ((s - 1.0) * g * d)


def shear_velocity_from_shields(theta: float, d: float, s: float, g: float) -> float:
    """Inverse of :func:`shields_parameter`: ``u* = sqrt(theta (s-1) g d)``."""
    if theta < 0.0 or d <= 0.0 or s <= 1.0:
        raise ValueError("require theta >= 0, d > 0, s > 1")
    return math.sqrt(theta * (s - 1.0) * g * d)