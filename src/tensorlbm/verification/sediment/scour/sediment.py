"""Sediment physics: properties, Shields threshold, Rouse profile, transport,
Exner bed evolution.

Everything here is closed-form or a small ODE/iteration, so it can be reused by
the three verification cases and unit-tested in isolation.  Bibliographic
anchors are stated per function; nothing is fitted to the acceptance targets of
the campaign node (see ``docs/reports`` for the honesty discussion).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.81
KAPPA = 0.41  # von Karman constant

__all__ = [
    "G",
    "KAPPA",
    "Sediment",
    "shields_critical_soulsby",
    "shields_curve_digitised",
    "rouse_profile",
    "rouse_profile_numeric",
    "van_rijn_reference_concentration",
    "total_load_transport",
    "exner_step",
    "avalanche_relax",
]


@dataclass
class Sediment:
    """Quartz sand in fresh water (defaults = the node's test material)."""

    d: float = 2.0e-4  # grain diameter [m]
    rho_s: float = 2650.0
    rho_f: float = 1000.0
    nu: float = 1.0e-6
    porosity: float = 0.40
    repose_deg: float = 32.0
    w_settling: float | None = None

    @property
    def s(self) -> float:
        return self.rho_s / self.rho_f

    @property
    def d_star(self) -> float:
        """Dimensionless grain size D* = d [(s-1) g / nu^2]^(1/3)."""
        return self.d * ((self.s - 1.0) * G / self.nu**2) ** (1.0 / 3.0)

    @property
    def submerged_weight(self) -> float:
        """Submerged weight of one grain [N]."""
        return (self.rho_s - self.rho_f) * G * np.pi * self.d**3 / 6.0

    def settling_velocity(self) -> float:
        """Terminal settling velocity with the Schiller-Naumann drag law.

        ``w^2 = 4 (rho_s - rho_f) g d / (3 Cd rho_f)``,
        ``Cd = 24/Re (1 + 0.15 Re^0.687)``, solved by fixed-point iteration on
        ``Re = w d / nu``.  (Schiller & Naumann 1933; the standard correction
        used for sand-sized grains, cf. Soulsby 1997 eq. 2.24.)
        """
        if self.w_settling is not None:
            return float(self.w_settling)
        d, nu = self.d, self.nu
        w = (self.s - 1.0) * G * d * d / (18.0 * nu)  # Stokes start value
        for _ in range(200):
            re = max(w * d / nu, 1e-12)
            cd = 24.0 / re * (1.0 + 0.15 * re**0.687)
            w_new = np.sqrt(4.0 * (self.s - 1.0) * G * d / (3.0 * cd))
            w = 0.5 * (w + w_new)
        return float(w)

    def shields_number(self, tau_b: float) -> float:
        return tau_b / ((self.rho_s - self.rho_f) * G * self.d)

    def critical_shear_stress(self, theta_cr: float) -> float:
        return theta_cr * (self.rho_s - self.rho_f) * G * self.d


def shields_critical_soulsby(d_star) -> np.ndarray:
    """Critical Shields parameter, Soulsby & Whitehouse (1991) fit::

        theta_cr = 0.30 / (1 + 1.2 D*) + 0.055 (1 - exp(-0.02 D*))

    This is the parameterisation of the Shields (1936) curve recommended by
    Soulsby (1997, *Dynamics of Marine Sands*, eq. 4.14); it reproduces the
    classical minimum ``theta_cr ~ 0.045-0.05`` for ``D* ~ 5``.
    """
    d_star = np.asarray(d_star, dtype=float)
    return 0.30 / (1.0 + 1.2 * d_star) + 0.055 * (1.0 - np.exp(-0.02 * d_star))


# Approximate readings from the Shields (1936) diagram as reproduced in
# Soulsby (1997, fig. 4.5).  Digitised to ~1 decimal; used ONLY as a
# sanity band (tolerance 25 %), never as a fit target.
SHIELDS_1936_DSTAR = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0])
SHIELDS_1936_THETA = np.array([0.100, 0.070, 0.048, 0.043, 0.040, 0.042, 0.048, 0.052, 0.058, 0.060])


def shields_curve_digitised() -> tuple[np.ndarray, np.ndarray]:
    """Digitised Shields (1936) curve, see :data:`SHIELDS_1936_DSTAR`."""
    return SHIELDS_1936_DSTAR.copy(), SHIELDS_1936_THETA.copy()


def rouse_profile(z, h: float, z_a: float, c_a: float, z_rouse: float) -> np.ndarray:
    """Rouse (1937) equilibrium suspended-sediment profile::

        C(z)/C_a = [ ((h - z)/z) * (a/(h-a)) ]^Z ,   Z = w_s/(kappa u_*)

    ``Z`` is the Rouse number.  Derivation: vertical balance of settling and
    turbulent diffusion, ``w_s C + eps_s dC/dz = 0`` with the parabolic
    diffusivity ``eps_s = kappa u_* z (1 - z/h)``.
    """
    z = np.asarray(z, dtype=float)
    ratio = np.clip((h - z) / np.maximum(z, 1e-12) * (z_a / (h - z_a)), 1e-30, None)
    return c_a * ratio**z_rouse


def rouse_profile_numeric(
    h: float, z_a: float, c_a: float, z_rouse: float, n: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically integrate ``dC/dz = -w_s C / (kappa u_* z (1 - z/h))``.

    Used as an *independent* check of :func:`rouse_profile` (and as the target
    of the finite-volume advection-diffusion solve that the Lagrangian random
    walk is compared against).
    """
    z = np.linspace(z_a, 0.999 * h, n)
    eps = KAPPA * (z_rouse / KAPPA) * 0.0  # placeholder to keep signature explicit
    del eps
    integrand = z_rouse / (z * (1.0 - z / h))
    log_c = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(z))])
    return z, c_a * np.exp(-log_c)


def van_rijn_reference_concentration(
    sed: Sediment, theta: float, theta_cr: float, a_ref: float
) -> float:
    """Van Rijn (1984) reference concentration at height ``a_ref`` above bed::

        C_a = 0.015 * (d/a) * T^1.5 / D*^0.3 ,   T = (theta - theta_cr)/theta_cr

    Returns 0 below the threshold, which is exactly what makes the *total*
    transport vanish at incipient motion (needed for a well-posed scour
    equilibrium).
    """
    if theta <= theta_cr:
        return 0.0
    t = (theta - theta_cr) / theta_cr
    return 0.015 * (sed.d / a_ref) * t**1.5 / sed.d_star**0.3


def bedload_meyer_peter_muller(sed: Sediment, theta: float, theta_cr: float) -> float:
    """Meyer-Peter-Muller (1948) volumetric bed-load rate per unit width [m^2/s].

    ``q_b* = 8 (theta - theta_cr)^1.5``, ``q_b = q_b* sqrt((s-1) g d^3)``.
    """
    exc = max(theta - theta_cr, 0.0)
    if exc <= 0.0:
        return 0.0
    return 8.0 * exc**1.5 * np.sqrt((sed.s - 1.0) * G * sed.d**3)


def total_load_transport(
    sed: Sediment,
    *,
    theta: float,
    theta_cr: float,
    u_star: float,
    h: float,
    a_ref: float,
    n_z: int = 24,
) -> tuple[float, float, float]:
    """Total (bed-load + suspended) volumetric transport per unit width.

    Returns ``(q_total, q_bed, q_susp)`` in m^2/s.  Suspended load is the
    Rouse profile integrated against the log-law velocity::

        q_susp = int_a^h u(z) C(z) dz ,  u(z) = (u_*/kappa) ln(z/z0)

    with ``C`` from :func:`rouse_profile` and ``C_a`` from
    :func:`van_rijn_reference_concentration`.  Both contributions vanish at
    ``theta = theta_cr`` so that ``q = 0`` defines morphodynamic equilibrium.
    """
    q_bed = bedload_meyer_peter_muller(sed, theta, theta_cr)
    c_a = van_rijn_reference_concentration(sed, theta, theta_cr, a_ref)
    if c_a <= 0.0 or u_star <= 0.0:
        return q_bed, q_bed, 0.0
    w_s = sed.settling_velocity()
    z_rouse = w_s / (KAPPA * u_star)
    z = np.linspace(a_ref, h * (1.0 - 1e-3), n_z)
    z0 = max(sed.nu / (9.0 * u_star), 1e-9)
    u = (u_star / KAPPA) * np.log(np.maximum(z, 1.5 * z0) / z0)
    c = rouse_profile(z, h, a_ref, c_a, z_rouse)
    q_susp = float(np.trapezoid(u * c, z)) if hasattr(np, "trapezoid") else float(np.trapz(u * c, z))
    return q_bed + q_susp, q_bed, q_susp


def total_load_transport_vec(
    sed: Sediment,
    *,
    theta,
    theta_cr: float,
    u_star,
    h,
    a_ref: float,
    n_z: int = 40,
):
    """Vectorised :func:`total_load_transport` over an array of cells.

    Same physics (Meyer-Peter-Mueller bed load + Rouse-integrated suspended
    load), ``n_z`` point Gauss-like midpoint quadrature in the vertical.  Used by
    the 2-D scour case where the per-cell Python loop would dominate the run
    time.
    """
    theta = np.asarray(theta, dtype=float)
    u_star = np.asarray(u_star, dtype=float)
    h = np.asarray(h, dtype=float)
    exc = np.maximum(theta - theta_cr, 0.0)
    q_bed = 8.0 * exc**1.5 * np.sqrt((sed.s - 1.0) * G * sed.d**3)

    t = (np.arange(n_z) + 0.5) / n_z
    depth = np.maximum(h - a_ref, 1e-6)
    z = a_ref + depth[..., None] * t
    t_ratio = np.maximum((theta - theta_cr) / theta_cr, 0.0)
    c_a = 0.015 * (sed.d / a_ref) * t_ratio**1.5 / sed.d_star**0.3
    z0 = np.maximum(sed.nu / (9.0 * np.maximum(u_star, 1e-12)), 1e-9)
    u = (np.maximum(u_star, 0.0)[..., None] / KAPPA) * np.log(
        np.maximum(z, 1.5 * z0[..., None]) / z0[..., None]
    )
    ratio = np.clip(
        (h[..., None] - z)
        / np.maximum(z, 1e-12)
        * (a_ref / np.maximum(h - a_ref, 1e-9))[..., None],
        1e-30,
        None,
    )
    z_rouse = sed.settling_velocity() / (KAPPA * np.maximum(u_star, 1e-12))
    c = c_a[..., None] * ratio ** z_rouse[..., None]
    q_susp = (u * c).sum(axis=-1) * (depth / n_z)
    return q_bed + q_susp


def exner_step(
    z_b: np.ndarray,
    q: np.ndarray,
    dx: float,
    dt: float,
    porosity: float,
    *,
    dz_max: float | None = None,
) -> np.ndarray:
    """One conservative Exner update::

        dz_b/dt = -(1/(1-lambda_p)) * dq/dx

    Face fluxes are the arithmetic mean of the two adjacent cell-centre fluxes
    and the *same* face fluxes are reported for the domain mass balance, so the
    scheme is telescoping: the volume of bed change equals the net flux through
    the domain boundaries to round-off.  ``dz_max`` (cells) clips the increment
    for stability; the clipping is applied as a symmetric flux limiter so that
    conservation is preserved.
    """
    n = z_b.size
    q_face = 0.5 * (q + np.roll(q, -1))  # face i+1/2 between i and i+1
    dq = (q_face - np.roll(q_face, 1)) / dx
    dz = -dq / (1.0 - porosity) * dt

    if dz_max is not None:
        limit = dz_max * dx
        # symmetric limiter: scale the magnitude, keeps antiderivative form
        scale = np.minimum(1.0, limit / np.maximum(np.abs(dz), 1e-30))
        # applying a per-cell scale would break telescoping, so instead clip the
        # *flux* thickness consistently: rescale the whole flux field by the
        # worst-case factor (globally conservative because a single scalar
        # multiplies a telescoping field).
        s = float(scale.min())
        dz = dz * s
    return z_b + dz


def avalanche_relax(
    z_b: np.ndarray, dx: float, repose_deg: float, *, iters: int = 8
) -> np.ndarray:
    """Mass-conserving angle-of-repose limiter (relaxation toward ``tan(repose)``).

    Every link steeper than ``tan(repose)`` gets a diffusive face flux of
    ``0.25 * slope`` (the explicit-Laplacian stability limit); the flux is shared
    between the two cells, so ``sum(z_b)`` is invariant to round-off.  This is the
    discrete counterpart of gravity-driven grain avalanching.

    Note: this is a *relaxation*, not an exact projection -- after ``iters``
    sweeps the steepest link is typically within ~20 % of ``tan(repose)``.  A
    "greedy" per-link transfer (0.5 of the excess, applied simultaneously) is
    anti-diffusive and diverges; do not reintroduce it.
    """
    z = z_b.copy()
    tan_phi = np.tan(np.deg2rad(repose_deg))
    for _ in range(iters):
        slope = np.roll(z, -1) - z  # z[i+1] - z[i]
        coeff = 0.25 * (np.abs(slope) > tan_phi * dx)
        flux = coeff * slope  # face flux i+1/2, positive = downslope transfer
        z = z + flux - np.roll(flux, 1)
    return z
