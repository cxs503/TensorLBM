"""Numerical fluxes for the compressible (kinetic) finite-volume solver.

Three fluxes are provided so that every production number can be cross-checked
against an independent discretisation:

``hllc_flux``
    Toro's HLLC approximate Riemann solver with Einfeldt/Roe wave-speed
    estimates.  Positivity-preserving, contact-exact; the production flux.
``rusanov_flux``
    Local Lax-Friedrichs (Rusanov).  Heavily diffusive, used only as a
    discretisation-independent cross-check of converged shock positions.
``kfvs_flux``
    Boltzmann-level kinetic flux-vector splitting: the Euler flux is the first
    moment of a local Maxwellian, and the interface flux is the half-space
    integral of that distribution (upwind direction set by the sign of the
    normal molecular velocity).  This is the LBM-lineage flux: it is the
    continuum (Kn -> 0) limit of the discrete-velocity Boltzmann flux, which is
    why it is the "kinetic" member of this family.  ``kfvs_flux_quadrature``
    evaluates the same integral by direct velocity-space quadrature and is used
    in tests to prove the closed form.

State layout
------------
The conserved state is ``U = [rho, rho*u, rho*v, E]`` (2-D) with
``E = p/(gamma-1) + 0.5*rho*(u^2+v^2)``; the 1-D fluxes simply drop the
transverse component.  All functions are vectorised over the leading axis
(``U`` shaped ``(n, 4)`` and returns shaped ``(n, 4)``).
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

GAMMA = 1.4

__all__ = [
    "GAMMA",
    "primitive",
    "conserved",
    "euler_flux",
    "hllc_flux",
    "rusanov_flux",
    "kfvs_flux",
    "kfvs_flux_quadrature",
    "max_wave_speed",
]

SQRT_2PI = np.sqrt(2.0 * np.pi)


def primitive(U: np.ndarray, gamma: float = GAMMA) -> tuple:
    """Split conserved ``U=(...,4)`` into ``(rho, u, v, p)``."""
    rho = U[..., 0]
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    p = (gamma - 1.0) * (U[..., 3] - 0.5 * rho * (u * u + v * v))
    return rho, u, v, p


def conserved(rho, u, v, p, gamma: float = GAMMA) -> np.ndarray:
    """Assemble the conserved vector from primitives (broadcasting)."""
    rho = np.asarray(rho, dtype=float)
    shape = rho.shape + (4,)
    U = np.empty(shape, dtype=float)
    U[..., 0] = rho
    U[..., 1] = rho * u
    U[..., 2] = rho * v
    U[..., 3] = p / (gamma - 1.0) + 0.5 * rho * (np.asarray(u) ** 2 + np.asarray(v) ** 2)
    return U


def euler_flux(U: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Exact x-direction Euler flux ``F(U)`` (the continuum limit of the flux)."""
    rho, u, v, p = primitive(U, gamma)
    F = np.empty_like(U)
    F[..., 0] = rho * u
    F[..., 1] = rho * u * u + p
    F[..., 2] = rho * u * v
    F[..., 3] = u * (U[..., 3] + p)
    return F


def max_wave_speed(U: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Spectral radius ``|u| + a`` (used for the CFL condition)."""
    rho, u, v, p = primitive(U, gamma)
    a = np.sqrt(gamma * np.maximum(p, 1e-300) / rho)
    return np.abs(u) + a


def rusanov_flux(UL: np.ndarray, UR: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Local Lax-Friedrichs flux (diffusive reference discretisation)."""
    FL = euler_flux(UL, gamma)
    FR = euler_flux(UR, gamma)
    rhoL, uL, vL, pL = primitive(UL, gamma)
    rhoR, uR, vR, pR = primitive(UR, gamma)
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)
    smax = np.maximum(np.abs(uL) + aL, np.abs(uR) + aR)
    return 0.5 * (FL + FR) - 0.5 * smax[..., None] * (UR - UL)


def _roe_averages(UL, UR, gamma=GAMMA):
    rhoL, uL, vL, pL = primitive(UL, gamma)
    rhoR, uR, vR, pR = primitive(UR, gamma)
    sL = np.sqrt(rhoL)
    sR = np.sqrt(rhoR)
    den = sL + sR
    u = (sL * uL + sR * uR) / den
    v = (sL * vL + sR * vR) / den
    H = (sL * (UL[..., 3] + pL) / rhoL + sR * (UR[..., 3] + pR) / rhoR) / den
    a2 = (gamma - 1.0) * (H - 0.5 * (u * u + v * v))
    a = np.sqrt(np.maximum(a2, 1e-300))
    return u, v, a, H


def hllc_flux(UL: np.ndarray, UR: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """HLLC flux with Einfeldt (Roe) wave-speed bounds.

    Contact and shear waves are resolved exactly, which is what makes the
    slip line of a Mach-reflection pattern survive at moderate resolution.
    """
    rhoL, uL, vL, pL = primitive(UL, gamma)
    rhoR, uR, vR, pR = primitive(UR, gamma)
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)
    u_t, v_t, a_t, _ = _roe_averages(UL, UR, gamma)
    SL = np.minimum(uL - aL, u_t - a_t)
    SR = np.maximum(uR + aR, u_t + a_t)

    # contact speed
    num = pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)
    den = rhoL * (SL - uL) - rhoR * (SR - uR)
    den = np.where(np.abs(den) < 1e-300, 1e-300, den)
    SM = num / den

    FL = euler_flux(UL, gamma)
    FR = euler_flux(UR, gamma)

    def star(U, rho, u, v, p, S, SM):
        # HLLC star state (Toro eq. 10.39)
        fac = rho * (S - u) / (S - SM)
        out = np.empty_like(U)
        out[..., 0] = fac
        out[..., 1] = fac * SM
        out[..., 2] = fac * v
        out[..., 3] = fac * (U[..., 3] / rho + (SM - u) * (SM + p / (rho * (S - u))))
        return out

    USL = star(UL, rhoL, uL, vL, pL, SL, SM)
    USR = star(UR, rhoR, uR, vR, pR, SR, SM)
    FSL = FL + SL[..., None] * (USL - UL)
    FSR = FR + SR[..., None] * (USR - UR)

    out = np.where((SL >= 0.0)[..., None], FL, FSL)
    out = np.where(((SL < 0.0) & (SM >= 0.0))[..., None], out, out)
    out = np.where(((SL < 0.0) & (SM >= 0.0))[..., None], FSL, out)
    out = np.where(((SM < 0.0) & (SR > 0.0))[..., None], FSR, out)
    out = np.where((SR <= 0.0)[..., None], FR, out)
    return out


def kfvs_flux(UL: np.ndarray, UR: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Kinetic flux-vector split (half-space Maxwellian moments), closed form.

    For a state ``(rho, u, v, p)`` define ``RT = p/rho``, ``s = u/sqrt(2RT)``,
    ``k = rho exp(-s^2)/sqrt(2 pi)``.  The positive half-space moments of the
    local Maxwellian are

        F_rho^+ = rho u G(s) + sqrt(RT) k
        F_rhou^+ = rho u^2 G(s) + 2 u sqrt(RT) k + p G(s)
        F_rhov^+ = v F_rho^+
        F_E^+ = rho G(s) (u^3/2 + (5/2) u RT) + 2 sqrt(RT) (u^2+RT) k

    with ``G(s) = (1+erf(s))/2``; the negative split uses ``G(-s) = 1-G(s)`` and
    flipped signs on the ``k`` terms.  A single-temperature Maxwellian is
    monatomic, so the energy split carries the standard ``gamma`` correction
    ``+(1/(gamma-1) - 3/2) RT F_rho^+-`` which vanishes at ``gamma = 5/3`` and
    restores the exact Euler energy flux at ``gamma = 1.4``.
    """
    out = np.zeros_like(UL)
    for eps, U in ((+1.0, UL), (-1.0, UR)):
        rho, u, v, p = primitive(U, gamma)
        RT = p / rho
        sig = np.sqrt(RT)
        s = u / np.sqrt(2.0 * RT)
        G = 0.5 * (1.0 + eps * erf(s))
        k = rho * np.exp(-s * s) / SQRT_2PI
        Frho = rho * u * G + eps * sig * k
        Frhou = rho * u * u * G + eps * 2.0 * u * sig * k + p * G - eps * sig * u * k
        Frhov = v * Frho
        FE = rho * G * (0.5 * u**3 + 2.5 * u * RT) + eps * sig * k * (0.5 * u * u + 2.0 * RT)
        # transverse kinetic energy flux (v enters the Maxwellian as its mean)
        FE = FE + 0.5 * v * v * Frho
        # gamma correction: a Maxwellian carries 3/2 RT of internal energy only
        FE = FE + (1.0 / (gamma - 1.0) - 1.5) * RT * Frho
        out[..., 0] += Frho
        out[..., 1] += Frhou
        out[..., 2] += Frhov
        out[..., 3] += FE
    return out


def kfvs_flux_quadrature(
    UL: np.ndarray,
    UR: np.ndarray,
    gamma: float = GAMMA,
    n_quad: int = 401,
    xi_max: float = 12.0,
) -> np.ndarray:
    """Same kinetic split evaluated by brute-force velocity-space quadrature.

    Only used as an independent check of :func:`kfvs_flux` (trapezoid rule on
    ``xi in [-xi_max*sqrt(RT), +xi_max*sqrt(RT)]`` around the mean velocity).
    Slow: not for production use.
    """
    out = np.zeros_like(UL)
    for sign, U in ((+1, UL), (-1, UR)):
        rho, u, v, p = primitive(U, gamma)
        RT = p / rho
        sig = np.sqrt(RT)
        # nodes scaled per state (shape (n, n_quad))
        z = np.linspace(-xi_max, xi_max, n_quad)
        xi = u[..., None] + sig[..., None] * z
        f = (
            rho[..., None]
            / (sig[..., None] * SQRT_2PI)
            * np.exp(-0.5 * (xi - u[..., None]) ** 2 / (RT[..., None]))
        )
        w = np.gradient(z, axis=-1)  # trapezoid weights in z
        mask = (sign * xi) > 0.0
        fw = np.where(mask, f * sig[..., None] * w, 0.0)
        out[..., 0] += np.sum(fw * xi, axis=-1)
        out[..., 1] += np.sum(fw * xi * xi, axis=-1)
        out[..., 2] += v * np.sum(fw * xi, axis=-1)
        # energy: (xi^2 + 2 RT)/2 per unit mass (transverse DOF integrated out)
        out[..., 3] += np.sum(fw * xi * (0.5 * xi * xi + RT[..., None]), axis=-1)
        mflux = np.sum(fw * xi, axis=-1)
        out[..., 3] += 0.5 * v * v * mflux
        # gamma correction so the split sums to the perfect-gas Euler flux
        out[..., 3] += (1.0 / (gamma - 1.0) - 1.5) * RT * mflux
    return out
