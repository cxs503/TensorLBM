"""Analytical reference solutions for the perforated-plate verification face.

Contents
--------
* :func:`sharp_edge_cd` - quasi-steady discharge coefficient of a thin,
  sharp-edged perforated plate (Borda-Carnot loss model, derived below).
* :func:`quasi_steady_pressure_drop` - quadratic pressure-drop law.
* :func:`womersley_channel_u`, :func:`womersley_wall_shear` - exact solution of
  the plane channel driven by a harmonic body force (used for the two negative
  controls: unobstructed channel pressure budget and wall shear / Stokes layer).
* :func:`stokes_layer_wall_shear` - semi-infinite Stokes-layer limit (delta << h),
  provided for the Stokes-layer leg of negative control 2.
* :func:`stokes_oscillating_wall_shear`, :func:`stokes_layer_thickness` - legacy
  helpers kept for API compatibility.

Sharp-edged plate reference
---------------------------
Control volume between the upstream station (1), the vena contracta (c) and the
downstream station (2) of a thin plate in a duct of area ``A_ch`` with hole area
``A_h = beta A_ch``:

* mass:      ``u_c = u_h / C_c``, ``u_1 = u_2 = beta u_h``
* Bernoulli 1 -> c (loss free): ``p_1 + u_1^2/2 = p_c + u_c^2/2``
* Borda-Carnot expansion c -> 2: ``p_c + u_c^2/2 = p_2 + u_2^2/2 + (u_c-u_2)^2/2``

Adding the two relations gives the classical result

    dp = 0.5 rho u_h^2 (1/C_c - beta)^2   =>   C_d = C_c / (1 - beta C_c)

with the sharp-edge contraction coefficient ``C_c = 0.61`` (von Mises / Rouse,
2-D slit).  Limits: ``C_d(beta->0) -> 0.61`` (classical sharp-edged orifice),
and ``C_d`` grows monotonically with ``beta`` because of the velocity-of-approach
recovery.
"""

from __future__ import annotations

import math

import numpy as np

#: contraction coefficient of a sharp-edged 2-D slit (von Mises / Rouse)
C_C_SHARP = 0.61


# ---------------------------------------------------------------------------
# Quasi-steady perforated plate
# ---------------------------------------------------------------------------

def sharp_edge_cd(beta: float, cc: float = C_C_SHARP) -> float:
    """Quasi-steady discharge coefficient ``C_d = C_c / (1 - beta C_c)``."""
    if not (0.0 < beta < 1.0):
        return float("nan")
    return cc / (1.0 - beta * cc)


def idelchik_cd_quasi_steady(beta: float, re: float = 1e5,
                             edge_type: str = "sharp") -> float:
    """Backward-compatible wrapper around :func:`sharp_edge_cd`.

    ``re`` is accepted for API compatibility; the reference is the
    high-Reynolds-number (quasi-steady) limit in which the loss coefficient
    ``zeta_h = 1/C_d^2`` is Reynolds independent.
    """
    if edge_type == "sharp":
        cd = sharp_edge_cd(beta)
    elif edge_type == "rounded":
        # rounded edge: contraction is suppressed, C_c -> ~0.98
        cd = sharp_edge_cd(beta, cc=0.98)
    else:
        raise ValueError(f"Unknown edge_type: {edge_type}")
    if not math.isnan(cd):
        cd = max(0.4, min(1.0, cd))
    return cd


def quasi_steady_pressure_drop(q: float, beta: float, area: float,
                              rho: float = 1.0, c_d: float = 0.61) -> float:
    """``dp = 0.5 rho C_d^-2 (Q/(A beta))^2 sign(Q)``."""
    if beta <= 0 or area <= 0 or c_d <= 0:
        return 0.0
    u_hole = q / (area * beta)
    return 0.5 * rho * (1.0 / c_d) ** 2 * u_hole * abs(u_hole)


def loss_coefficient(c_d: float) -> float:
    """Loss coefficient referred to the hole velocity, ``zeta = 1/C_d^2``."""
    if not (c_d > 0):
        return float("inf")
    return 1.0 / c_d ** 2


# ---------------------------------------------------------------------------
# Harmonic body-force driven plane channel (Womersley)
# ---------------------------------------------------------------------------

def _alpha(omega: float, nu: float) -> complex:
    return complex(math.sqrt(omega / (2.0 * nu)), math.sqrt(omega / (2.0 * nu)))


def womersley_channel_u(y, t: float, height: float, nu: float, f0: float,
                        omega: float) -> np.ndarray:
    """Exact velocity of a plane channel driven by ``F(t) = f0 sin(omega t)``.

    Walls at ``y = +-height/2`` (no-slip), ``F`` is a body-force density
    (force per unit mass, rho = 1).  The solution carries no mean drift::

        u(y,t) = Im{ (f0/(i omega)) [1 - cosh(a y)/cosh(a h/2)] e^{i omega t} }

    with ``a = sqrt(i omega / nu)``.
    """
    y = np.asarray(y, dtype=np.float64)
    a = complex(math.sqrt(omega / (2.0 * nu)), math.sqrt(omega / (2.0 * nu)))
    u_hat = (f0 / (1j * omega)) * (1.0 - np.cosh(a * y) / np.cosh(a * height / 2.0))
    return np.imag(u_hat * np.exp(1j * omega * t))


def womersley_channel_amplitude(y, height: float, nu: float, f0: float,
                                omega: float) -> np.ndarray:
    """Amplitude of the fundamental of :func:`womersley_channel_u`."""
    y = np.asarray(y, dtype=np.float64)
    a = complex(math.sqrt(omega / (2.0 * nu)), math.sqrt(omega / (2.0 * nu)))
    u_hat = (f0 / (1j * omega)) * (1.0 - np.cosh(a * y) / np.cosh(a * height / 2.0))
    return np.abs(u_hat)


def womersley_wall_shear(t: float, height: float, nu: float, f0: float,
                         omega: float, rho: float = 1.0) -> float:
    """Wall shear stress ``tau_w = mu du/dy`` at the lower wall (y = -h/2)."""
    a = complex(math.sqrt(omega / (2.0 * nu)), math.sqrt(omega / (2.0 * nu)))
    # du_hat/dy = -(f0/(i omega)) a sinh(a y)/cosh(a h/2)
    y = -height / 2.0
    du_hat = -(f0 / (1j * omega)) * a * np.sinh(a * y) / np.cosh(a * height / 2.0)
    tau_hat = rho * nu * du_hat
    return float(np.imag(tau_hat * np.exp(1j * omega * t)))


def womersley_wall_shear_amplitude(height: float, nu: float, f0: float,
                                   omega: float, rho: float = 1.0) -> float:
    a = complex(math.sqrt(omega / (2.0 * nu)), math.sqrt(omega / (2.0 * nu)))
    y = -height / 2.0
    du_hat = -(f0 / (1j * omega)) * a * np.sinh(a * y) / np.cosh(a * height / 2.0)
    return float(abs(rho * nu * du_hat))


def womersley_pressure_budget(height: float, nu: float, f0: float, omega: float,
                              t: float, width: float, u_bar: float,
                              du_bar_dt: float, wall_shear: float,
                              rho: float = 1.0) -> float:
    """Analytic ``dp_stations`` of the slab momentum budget (used by NC1).

    ``dp = rho W du_bar/dt - rho F W + (tau_lower + tau_upper)/A`` written with
    the same sign convention as the LBM probes: ``tau`` positive in +x on the
    wall and ``A`` the channel height (2-D, unit depth).
    """
    area = height
    tau_sum = 2.0 * wall_shear
    return rho * width * du_bar_dt - rho * f0 * math.sin(omega * t) * width + tau_sum / area


# ---------------------------------------------------------------------------
# Stokes layer (semi-infinite, delta << h)
# ---------------------------------------------------------------------------

def stokes_layer_thickness(omega: float, nu: float) -> float:
    """Stokes boundary-layer thickness ``delta = sqrt(2 nu / omega)``."""
    if nu <= 0 or omega <= 0:
        return float("inf")
    return math.sqrt(2.0 * nu / omega)


def stokes_layer_u(y, t: float, f0: float, omega: float, nu: float) -> np.ndarray:
    """Semi-infinite Stokes layer driven by ``F = f0 sin(omega t)``.

    ``u(y,t) = (f0/omega)[sin(omega t) - exp(-y/delta) sin(omega t - y/delta)]``
    (zero mean, no-slip at ``y = 0``).
    """
    y = np.asarray(y, dtype=np.float64)
    delta = stokes_layer_thickness(omega, nu)
    return (f0 / omega) * (np.sin(omega * t)
                           - np.exp(-y / delta) * np.sin(omega * t - y / delta))


def stokes_layer_wall_shear(t: float, f0: float, omega: float, nu: float,
                            rho: float = 1.0) -> float:
    """Wall shear of the semi-infinite Stokes layer at ``y = 0``."""
    a = complex(math.sqrt(omega / (2.0 * nu)), math.sqrt(omega / (2.0 * nu)))
    du_hat = (f0 / (1j * omega)) * a          # d/dy at y=0 of the Stokes solution
    return float(np.imag(rho * nu * du_hat * np.exp(1j * omega * t)))


def stokes_oscillating_wall_shear(y: float, t: float, u0: float, omega: float,
                                  nu: float) -> float:
    """Legacy helper: velocity of the Stokes second problem."""
    if nu <= 0 or omega <= 0:
        return 0.0
    delta = math.sqrt(2.0 * nu / omega)
    eta = y / delta
    return u0 * math.exp(-eta) * math.cos(omega * t - eta)


def stokes_wall_shear_stress(t: float, u0: float, omega: float, nu: float,
                             mu: float = 1.0) -> float:
    """Legacy helper: ``tau_w = -mu u0/delta (cos + sin)``."""
    if nu <= 0 or omega <= 0:
        return 0.0
    delta = math.sqrt(2.0 * nu / omega)
    return -mu * u0 / delta * (math.cos(omega * t) + math.sin(omega * t))