"""Analytic / literature references for the FCM particle-suspension campaign.

Every function here is an *independent* reference value against which the
force-coupling LBM results in :mod:`tensorlbm.fcm` are compared.  Nothing in
this module is fitted to the simulations; the only calibrated quantity in the
campaign (the FCM effective radius) lives in the driver script and is declared
in the report.

References
----------
Stokes (1851); Schiller & Naumann (1933) *VDI-Z* 77, 318.
Batchelor & Green (1972). "The determination of the bulk stress in a suspension
    of spherical particles." *J. Fluid Mech.* 56, 375.
Einstein (1906) *Ann. Phys.* 19, 289.
Brenner (1961). "The slow motion of a sphere through a viscous fluid towards a
    plane surface." *Chem. Eng. Sci.* 16, 242.
Kim & Karrila (2005). *Microhydrodynamics*. Dover, ch. 4 (lubrication).
Hasimoto (1959). *J. Fluid Mech.* 5, 317 (periodic array of spheres).
Taylor & Green (1937). "Mechanism of the production of small eddies from large
    ones." *Proc. R. Soc. A* 158, 499.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "dynamic_viscosity_from_tau",
    "stokes_drag_force",
    "stokes_cd",
    "schiller_naumann_cd",
    "stokes_terminal_velocity",
    "finite_re_terminal_velocity",
    "stokes_number",
    "lubrication_force_sphere_sphere",
    "lubrication_force_sphere_plane",
    "hasimoto_drag_correction",
    "einstein_viscosity_ratio",
    "batchelor_green_viscosity_ratio",
    "taylor_green_decay_rate",
    "taylor_green_decay_rate_2d",
    "taylor_green_energy",
    "volume_fraction",
]


def dynamic_viscosity_from_tau(tau: float, rho: float = 1.0) -> float:
    """``mu = rho * nu`` with ``nu = (tau - 1/2)/3``."""
    return rho * (tau - 0.5) / 3.0


def stokes_drag_force(mu: float, radius: float, velocity: float) -> float:
    """``F = 6 pi mu a U`` (Stokes 1851)."""
    return 6.0 * math.pi * mu * radius * velocity


def stokes_cd(re: float) -> float:
    """``C_D = 24/Re``."""
    if re <= 0:
        raise ValueError("Re must be positive")
    return 24.0 / re


def schiller_naumann_cd(re: float) -> float:
    """Schiller-Naumann drag correlation, valid for ``Re <~ 800``."""
    if re <= 0:
        raise ValueError("Re must be positive")
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def stokes_terminal_velocity(
    diameter: float, rho_p: float, rho_f: float, mu: float, g: float
) -> float:
    """``v_t = (rho_p - rho_f) g d^2 / (18 mu)``."""
    return (rho_p - rho_f) * g * diameter**2 / (18.0 * mu)


def finite_re_terminal_velocity(
    diameter: float,
    rho_p: float,
    rho_f: float,
    mu: float,
    g: float,
    *,
    cd_model=schiller_naumann_cd,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Terminal velocity from a drag-balance fixed point.

    Solves ``(rho_p - rho_f) g pi d^3 / 6 = 0.5 rho_f v^2 (pi d^2/4) C_D(Re)``
    with ``Re = rho_f v d / mu`` by successive substitution, seeded with the
    Stokes velocity.  Returns the *finite-Re* terminal velocity, i.e. the
    Stokes velocity multiplied by ``f(Re)`` implicitly.
    """
    v = stokes_terminal_velocity(diameter, rho_p, rho_f, mu, g)
    area = 0.25 * math.pi * diameter**2
    weight = (rho_p - rho_f) * g * math.pi * diameter**3 / 6.0
    for _ in range(max_iter):
        re = rho_f * max(v, 1e-30) * diameter / mu
        cd = cd_model(re)
        v_new = math.sqrt(2.0 * weight / (rho_f * area * cd))
        if abs(v_new - v) < tol * max(abs(v_new), 1e-30):
            v = v_new
            break
        v = v_new
    return v


def stokes_number(rho_p: float, diameter: float, mu: float, velocity: float) -> float:
    """Particle Stokes number ``St = rho_p d^2 U / (18 mu)`` / (d/U)."""
    return rho_p * diameter**2 * velocity / (18.0 * mu) / (diameter / velocity)


def lubrication_force_sphere_sphere(
    radius1: float, radius2: float, gap: float, u_rel: float, mu: float
) -> float:
    """Leading-order normal lubrication force between two spheres.

    ``F = 6 pi mu a_eff^2 U_rel / h`` with ``a_eff = a1 a2 / (a1 + a2)``.
    In the limit ``a2 -> inf`` this reduces to Brenner's sphere-plane result
    ``F = 6 pi mu a1^2 V / h``; for equal radii ``a_eff = a/2`` and
    ``F = (3/2) pi mu a^2 U_rel / h``.  Relative error is ``O(h/a)``.
    """
    a_eff = radius1 * radius2 / (radius1 + radius2)
    return 6.0 * math.pi * mu * a_eff**2 * abs(u_rel) / gap


def lubrication_force_sphere_plane(radius: float, gap: float, velocity: float, mu: float) -> float:
    """Brenner (1961) leading-order sphere-plane lubrication force."""
    return 6.0 * math.pi * mu * radius**2 * abs(velocity) / gap


def hasimoto_drag_correction(phi: float) -> float:
    """Periodic simple-cubic array drag enhancement ``F/F_Stokes``.

    Hasimoto (1959): ``F/F_S = 1 / (1 - 1.7601 phi^{1/3} + phi + O(phi^2))``.
    Truncation error is ``O(phi^2)`` (``< 1e-5`` for the ``phi <= 2e-3`` boxes
    used in the campaign); the expansion is not used above ``phi = 0.05``.
    """
    if phi <= 0:
        return 1.0
    return 1.0 / (1.0 - 1.7601 * phi ** (1.0 / 3.0) + phi)


def einstein_viscosity_ratio(phi: float) -> float:
    """``mu_eff/mu = 1 + 5 phi / 2`` (dilute limit)."""
    return 1.0 + 2.5 * phi


def batchelor_green_viscosity_ratio(phi: float) -> float:
    """``mu_eff/mu = 1 + 5 phi/2 + 5.2 phi^2`` (Batchelor & Green 1972)."""
    return 1.0 + 2.5 * phi + 5.2 * phi**2


def taylor_green_decay_rate(nu: float, k: float) -> float:
    """Velocity decay rate of a single Fourier mode: ``gamma = nu k^2``."""
    return nu * k**2


def taylor_green_decay_rate_2d(nu: float, k: float) -> float:
    """Velocity decay rate of the 2-D Taylor-Green mode on a periodic box.

    ``u = U0 (sin kx cos ky, -cos kx sin ky, 0)`` has vorticity
    ``omega_z = 2 k sin kx sin ky``, i.e. Laplacian eigenvalue
    ``|k|^2 = 2 k^2`` with ``k = 2 pi / L``; the velocity therefore decays as
    ``exp(-2 nu k^2 t)`` and the kinetic energy as ``exp(-4 nu k^2 t)``.  The
    convective term of this particular mode vanishes identically, so the decay
    is exponential at any amplitude (Taylor 1923).
    """
    return 2.0 * nu * k**2


def taylor_green_energy(e0: float, nu: float, k: float, t) -> np.ndarray:
    """``E(t) = E0 exp(-2 nu k^2 t)``."""
    t = np.asarray(t, dtype=float)
    return e0 * np.exp(-2.0 * taylor_green_decay_rate(nu, k) * t)


def volume_fraction(radii, box_volume: float) -> float:
    radii = np.asarray(radii, dtype=float)
    return float(np.sum(4.0 / 3.0 * math.pi * radii**3) / box_volume)