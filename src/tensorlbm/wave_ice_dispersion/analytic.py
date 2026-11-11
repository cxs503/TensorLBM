"""Flexural-gravity wave dispersion beneath an elastic ice sheet (node p338w3-n05).

Two forms are provided, and the difference matters for interpretation:

**Canonical massless-plate form** (the form specified by campaign node
``p338w3-n05``, target of the numerical/analytic cross-adjudication)::

    omega^2 = (g k + D k^5 / rho_w) * tanh(k h)
    D = E h_i^3 / (12 (1 - nu^2))

**Mass-loaded form** (physically standard for a floating ice sheet, kept here as
an explicit limitation of the massless form; see ``docs/P338W3_N05_*.md`` §5): it
shifts the curve down by ``1 + m k``, lowers the growth exponent from 3/2 to 1,
and moves the minimum phase speed ``c_min`` (no steady wave below ``c_min``)::

    omega^2 = (g k + D k^5 / rho_w) / (1 + m k) * tanh(k h),  m = rho_i h_i / rho_w

Both are derived from the linearised potential-flow problem
``Laplace(phi) = 0`` in ``-h < z < 0`` with bottom ``phi_z = 0``, kinematic
condition ``eta_t = phi_z`` and dynamic condition
``phi_t + g eta - (D/rho_w) eta_xxxx = 0`` (massless plate) at ``z = 0``.

All quantities are SI.  Nothing here is fitted; the only inputs are the elastic
constants, the water depth and the wavenumber.
"""

from __future__ import annotations

import numpy as np

#: gravitational acceleration [m/s^2]
G = 9.81
#: sea-water density [kg/m^3]
RHO_W = 1025.0
#: fresh/sea ice density [kg/m^3] (Timco & Weeks 2010 range 900-940)
RHO_I = 917.0
#: effective Young's modulus of sea ice [Pa] (Timco & Weeks 2010: 0.1-10 GPa)
E_ICE = 5.0e9
#: Poisson's ratio of ice [-]
NU_ICE = 0.3


def flexural_rigidity(E: float, h_i: float, nu: float = NU_ICE) -> float:
    """Plate bending stiffness ``D = E h_i^3 / (12 (1 - nu^2))`` [N m]."""
    return E * h_i**3 / (12.0 * (1.0 - nu**2))


def mass_loading(h_i: float, rho_i: float = RHO_I, rho_w: float = RHO_W) -> float:
    """Ice mass-loading parameter ``m = rho_i h_i / rho_w`` [m]."""
    return rho_i * h_i / rho_w


def dispersion_omega(
    k,
    depth: float,
    D: float,
    rho_w: float = RHO_W,
    g: float = G,
    mass_loading_param: float = 0.0,
):
    """Angular frequency ``omega(k)`` of a flexural-gravity wave.

    ``mass_loading_param = 0`` gives the canonical massless-plate relation of
    node p338w3-n05; pass ``m = rho_i h_i / rho_w`` for the mass-loaded plate.
    """
    k = np.asarray(k, dtype=float)
    numerator = g * k + D * k**5 / rho_w
    denominator = 1.0 + mass_loading_param * k
    return np.sqrt(numerator / denominator * np.tanh(k * depth))


def phase_speed(k, depth: float, D: float, **kw):
    """Phase speed ``c_p = omega / k`` [m/s]."""
    k = np.asarray(k, dtype=float)
    return dispersion_omega(k, depth, D, **kw) / k


def group_speed(k, depth: float, D: float, rho_w: float = RHO_W, g: float = G,
                mass_loading_param: float = 0.0):
    """Group velocity ``c_g = d omega / d k`` [m/s], analytic derivative.

    With ``N = g k + D k^5 / rho_w``, ``M = 1 + m k``, ``Q = N / M`` and
    ``F = Q tanh(k h)`` (so ``omega = sqrt(F)``)::

        dF/dk = [ (N' M - N M') / M^2 ] tanh(kh) + Q h sech^2(kh)
        c_g   = dF/dk / (2 omega)
    """
    k = np.asarray(k, dtype=float)
    N = g * k + D * k**5 / rho_w
    Np = g + 5.0 * D * k**4 / rho_w
    M = 1.0 + mass_loading_param * k
    Mp = mass_loading_param * np.ones_like(k) if np.ndim(k) else mass_loading_param
    Q = N / M
    Qp = (Np * M - N * Mp) / M**2
    th = np.tanh(k * depth)
    sch2 = 1.0 / np.cosh(k * depth) ** 2
    Fp = Qp * th + Q * depth * sch2
    return Fp / (2.0 * np.sqrt(Q * th))


def free_surface_omega(k, depth: float, g: float = G):
    """Open-water dispersion ``omega^2 = g k tanh(k h)`` (h_i -> 0 limit)."""
    k = np.asarray(k, dtype=float)
    return np.sqrt(g * k * np.tanh(k * depth))


def shallow_water_speed(depth: float, g: float = G) -> float:
    """Shallow-water limit ``c_p -> sqrt(g h)`` as ``k -> 0`` [m/s]."""
    return float(np.sqrt(g * depth))


def critical_phase_speed(D: float, depth: float | None = None,
                         mass_loading_param: float = 0.0,
                         rho_w: float = RHO_W, g: float = G) -> float:
    """Minimum phase speed of the flexural-gravity branch [m/s].

    ``c_p^2 = (g/k + D k^4 / rho_w) / (1 + m k)`` is minimised numerically on a
    log grid of ``k`` (deep water, ``tanh -> 1``).  The minimum exists for both
    the massless and the mass-loaded branch; mass loading shifts it.  Below
    ``c_min`` no steady propagating wave exists, which is why ice-covered ship
    resistance has a sharp critical-speed peak.
    """
    h = 1.0e3 if depth is None else depth          # deep water by default
    kk = np.logspace(-6, 3, 20001)
    cp = phase_speed(kk, h, D, rho_w=rho_w, g=g,
                     mass_loading_param=mass_loading_param)
    return float(np.min(cp))


def thickness_scaling_slope(h_i_values, k: float, depth: float,
                            E: float = E_ICE, nu: float = NU_ICE,
                            rho_w: float = RHO_W, g: float = G) -> float:
    """log-log slope ``d ln(omega) / d ln(h_i)`` of the massless-plate branch.

    Negative control (2) requires ``omega -> inf`` as ``h_i -> inf``; since
    ``D ~ h_i^3`` and ``omega^2 ~ D k^5 tanh(kh) / rho_w`` in that limit, the
    expected slope is exactly ``1.5``.
    """
    h = np.asarray(h_i_values, dtype=float)
    om = np.array([dispersion_omega(k, depth, flexural_rigidity(E, hi, nu),
                                    rho_w=rho_w, g=g) for hi in h])
    slope = np.polyfit(np.log(h), np.log(om), 1)
    return float(slope[0])


def rigidity_sensitivity(k, depth: float, D: float, rel_dD: float = 0.10,
                         rho_w: float = RHO_W, g: float = G):
    """Relative change of ``c_p`` when ``D`` is perturbed by ``rel_dD``."""
    k = np.asarray(k, dtype=float)
    base = phase_speed(k, depth, D, rho_w=rho_w, g=g)
    pert = phase_speed(k, depth, D * (1.0 + rel_dD), rho_w=rho_w, g=g)
    return pert / base - 1.0