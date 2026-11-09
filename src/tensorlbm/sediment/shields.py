"""Critical Shields stress and dimensionless sediment parameters.

Reference curve
---------------
Soulsby, R. L. and Whitehouse, R. J. (2005). *A practical model of critical
bedload transport*.  The critical Shields stress is a piecewise function of
the dimensionless grain size ``d*``::

    d*  = d * sqrt((s-1) g) / nu          (dimensionless grain size)
    w*  = w_s * sqrt(d / ((s-1) g))       (dimensionless settling velocity)

    d* < 0.3 :  theta_cr = 0.30/(1 + 1.2 d*) + 0.055 (1 - exp(-0.02 d*))
    0.3<=d*<10: theta_cr = 0.02 + 0.14/(1 + 200 d*^2) + 0.014 (1 - exp(-0.045 d*))
    d* >= 10 :  theta_cr = 0.02 + 0.14/(1 + 200 d*^2)

The curve is the *reference* for criterion C1; the *numerical* theta_cr is
measured independently from the coupled LBM + Exner response (see
``verification.py``), so the comparison is not circular.
"""

from __future__ import annotations

import math

__all__ = [
    "critical_shields",
    "dimensionless_grain_size",
    "dimensionless_settling_velocity",
]

#: Standard sediment density contrast (s = rho_s / rho).
DEFAULT_S = 1.65
#: Gravitational acceleration (m s^-2).
G = 9.81


def dimensionless_grain_size(d: float, s: float, g: float, nu: float) -> float:
    """Dimensionless grain size ``d* = d sqrt((s-1) g) / nu``.

    Args:
        d: Grain diameter (m).
        s: Density ratio ``rho_s / rho``.
        g: Gravitational acceleration (m s^-2).
        nu: Kinematic viscosity of the fluid (m^2 s^-1).
    """
    if d <= 0.0 or nu <= 0.0 or s <= 1.0:
        raise ValueError("require d > 0, nu > 0, s > 1")
    return d * math.sqrt((s - 1.0) * g) / nu


def dimensionless_settling_velocity(
    w_s: float, d: float, s: float, g: float
) -> float:
    """Dimensionless settling velocity ``w* = w_s sqrt(d / ((s-1) g))``."""
    if d <= 0.0 or s <= 1.0:
        raise ValueError("require d > 0, s > 1")
    return w_s * math.sqrt(d / ((s - 1.0) * g))


def critical_shields(d_star: float) -> float:
    """Soulsby–Whitehouse (2005) critical Shields stress for ``d*``.

    Returns the dimensionless critical bed shear stress ``theta_cr`` at which
    incipient motion begins.  Valid for ``d* > 0``.
    """
    if d_star <= 0.0:
        raise ValueError("d* must be positive")
    if d_star < 0.3:
        return 0.30 / (1.0 + 1.2 * d_star) + 0.055 * (1.0 - math.exp(-0.02 * d_star))
    if d_star < 10.0:
        return (
            0.02
            + 0.14 / (1.0 + 200.0 * d_star * d_star)
            + 0.014 * (1.0 - math.exp(-0.045 * d_star))
        )
    return 0.02 + 0.14 / (1.0 + 200.0 * d_star * d_star)
