"""Analytic anchors for the ice/water phase-change interface.

Two families are provided:

1. :class:`TwoPhaseNeumann` -- the *exact* self-similar solution of the
   one-dimensional two-phase Stefan problem with finite solid subcooling
   (or liquid superheat), unequal conductivities and unequal diffusivities.
   This is the anchor for criterion **C1** and generalises the one-phase
   branches of :mod:`tensorlbm.phase_change.stefan_exact` (the
   ``dT_bulk -> 0`` limit).

2. :func:`quasi_steady_cylinder_time` / :func:`quasi_steady_cylinder_front`
   -- the classical quasi-steady (zero sensible heat) solution for the
   outward melting of an ice *cylinder* into warm water; the literature
   anchor (London & Seban 1943; Carslaw & Jaeger 1959 sec. 11.2; Lunardini
   1991 ch. 4) used for criterion **C2**.

Derivation (two-phase Neumann)
------------------------------
Energy equation in each phase, ``dT/dt = alpha d2T/dx2``; interface at
``s(t) = 2 lam sqrt(alpha_in t)`` with ``T(s,t) = T_m``, wall at ``T_wall``,
bulk at ``T_bulk``.  With ``eta = x/(2 sqrt(alpha_in t))`` and
``xi = x/(2 sqrt(alpha_out t))`` the profiles that satisfy all four boundary
conditions are

    inner (0 <= x <= s):  T = T_wall + (T_m - T_wall) erf(eta)/erf(lam)
    outer (x >= s):       T = T_bulk + (T_m - T_bulk) erfc(xi)/erfc(xi_lam),
                          xi_lam = lam sqrt(alpha_in/alpha_out)

(the inner phase uses the ``erf`` combination because it is pinned at *both*
ends -- fixed wall temperature and interface; the outer phase uses ``erfc``
because its far end sits at the bulk temperature -- the standard Neumann
structure, cf. Alexiades & Solomon 1993 ch. 2).

Substituting into the interface energy balance

    rho L V = -[k dT/dx] = k_out dT_out/dx|+ - k_in dT_in/dx|-

and using ``V = lam sqrt(alpha_in/t)`` gives the transcendental relation

    lam = (Ste_in/sqrt(pi)) exp(-lam^2)/erf(lam)
          - (Ste_out/sqrt(pi)) sqrt(alpha_r) exp(-lam^2/alpha_r)
                                        /erfc(lam/sqrt(alpha_r))

with ``Ste_in = c_in |T_wall - T_m| / L``, ``Ste_out = c_out |T_m - T_bulk|/L``
and ``alpha_r = alpha_out / alpha_in``.  Both the ``Ste_out -> 0`` limit
(``lam exp(lam^2) erf(lam) = Ste_in/sqrt(pi)``) and the small-Ste limit
(``lam -> sqrt(Ste_in/2)``) are classical; both are asserted in
``tests/test_ice_water_front.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, erfc

from .properties import IceWaterCase

_SQRT_PI = math.sqrt(math.pi)

__all__ = [
    "TwoPhaseNeumann",
    "two_phase_lambda",
    "quasi_steady_cylinder_time",
    "quasi_steady_cylinder_front",
    "quasi_steady_cylinder_error_estimate",
]


# ---------------------------------------------------------------------------
# two-phase Neumann similarity solution
# ---------------------------------------------------------------------------
def _lambda_residual(lam: float, ste_in: float, ste_out: float, alpha_r: float) -> float:
    """Residual of the two-phase Neumann relation (zero at the solution)."""
    k_in = math.exp(-lam * lam) / erf(lam)
    xo = lam / math.sqrt(alpha_r)
    k_out = math.sqrt(alpha_r) * math.exp(-lam * lam / alpha_r) / erfc(xo)
    return lam - (ste_in / _SQRT_PI) * k_in + (ste_out / _SQRT_PI) * k_out


def two_phase_lambda(case: IceWaterCase) -> float:
    """Solve the two-phase Neumann relation for the similarity coefficient.

    Returns ``lam`` such that ``s(t) = 2 lam sqrt(alpha_inner t)``.
    """
    ste_in = case.stefan_inner
    ste_out = abs(case.outer.cp * (case.t_melt - case.t_initial) / case.latent_heat)
    alpha_r = case.diffusivity_ratio
    if ste_in <= 0:
        raise ValueError("Stefan number must be positive (wall at T_m?)")

    # The residual is -inf as lam->0 and +inf as lam->big; bracket robustly.
    lo, hi = 1.0e-12, 1.0
    f_lo = _lambda_residual(lo, ste_in, ste_out, alpha_r)
    for _ in range(200):
        f_hi = _lambda_residual(hi, ste_in, ste_out, alpha_r)
        if f_lo * f_hi < 0.0:
            return float(brentq(
                _lambda_residual, lo, hi, args=(ste_in, ste_out, alpha_r),
                xtol=1e-14, rtol=1e-15, maxiter=200,
            ))
        lo, f_lo = hi, f_hi
        hi *= 2.0
    raise RuntimeError("failed to bracket the Neumann similarity coefficient")


@dataclass
class TwoPhaseNeumann:
    """Exact similarity solution of the 1-D two-phase Stefan problem.

    ``s(t) = 2 lam sqrt(alpha_in t)``; ``case.inner`` is the phase adjacent to
    the wall (liquid when melting, ice when freezing), ``case.outer`` the bulk
    phase ahead of the front.
    """

    case: IceWaterCase
    lam: float = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.lam is None:
            self.lam = two_phase_lambda(self.case)

    # -- interface kinematics -----------------------------------------
    def front(self, t):
        """Interface position [m] at time ``t`` [s] (scalar or array)."""
        return 2.0 * self.lam * np.sqrt(self.case.alpha_inner * np.asarray(t, dtype=float))

    def front_velocity(self, t):
        """``ds/dt = lam sqrt(alpha/t)`` [m/s]."""
        t = np.asarray(t, dtype=float)
        return self.lam * np.sqrt(self.case.alpha_inner / t)

    @property
    def xi_lambda(self) -> float:
        return self.lam / math.sqrt(self.case.diffusivity_ratio)

    # -- fields --------------------------------------------------------
    def temperature(self, x, t):
        """Similarity temperature field [K] (vectorised over ``x``)."""
        case = self.case
        x = np.asarray(x, dtype=float)
        s = self.front(t)
        out = np.empty_like(x)

        eta = x / (2.0 * math.sqrt(case.alpha_inner * t))
        inner = x <= s
        if case.dT_inner != 0.0:
            out[inner] = case.t_wall + (case.t_melt - case.t_wall) * (
                erf(eta[inner]) / erf(self.lam)
            )
        else:
            out[inner] = case.t_melt

        dT_bulk = case.t_initial - case.t_melt
        if abs(dT_bulk) > 0.0:
            xi = x / (2.0 * math.sqrt(case.outer.alpha * t))
            out[~inner] = case.t_initial + (case.t_melt - case.t_initial) * (
                erfc(xi[~inner]) / erfc(self.xi_lambda)
            )
        else:
            out[~inner] = case.t_melt
        return out

    # -- interface fluxes ---------------------------------------------
    def interface_fluxes(self, t: float):
        """Return ``(q_in, q_out)`` in W/m^2, both counted in the +x sense.

        ``q_in = -k_in dT_in/dx|s-`` (heat arriving at the interface from the
        inner phase) and ``q_out = -k_out dT_out/dx|s+`` (heat leaving the
        interface into the outer phase; negative when the bulk is colder than
        ``T_m``).  The latent heat production is ``rho L ds/dt = q_in - |q_out|``
        when the outer phase absorbs heat (melting into a subcooled solid).
        """
        case = self.case
        tau = math.sqrt(case.alpha_inner * t)
        q_in = (
            case.inner.k
            * case.dT_inner
            * math.exp(-self.lam ** 2)
            / (tau * _SQRT_PI * erf(self.lam))
        )
        if case.dT_outer != 0.0:
            # dT/dx just outside the interface is positive (the bulk is colder
            # than T_m), so the +x flux -k dT/dx is negative, i.e. heat leaves
            # the interface and is stored in the subcooled solid.
            q_out = -(
                case.outer.k
                * case.dT_outer
                * math.exp(-self.xi_lambda ** 2)
                / (math.sqrt(case.outer.alpha * t) * _SQRT_PI * erfc(self.xi_lambda))
            )
        else:
            q_out = 0.0
        return q_in, q_out

    def stefan_residual(self, t: float) -> float:
        """Relative residual of the interface energy balance (should be ~0)."""
        q_in, q_out = self.interface_fluxes(t)
        lhs = self.case.latent_volumetric * float(self.front_velocity(t))
        rhs = q_in - abs(q_out)
        scale = max(abs(q_in), abs(q_out), 1e-30)
        return (lhs - rhs) / scale

    def pde_residual(self, x, t, eps: float = 1e-6):
        """Max |dT/dt - alpha d2T/dx2| normalised by ``dT/(t)``-scale.

        Evaluated with central differences inside each phase (nodes closer
        than ``eps*sqrt(alpha t)`` to either boundary are skipped).
        """
        case = self.case
        x = np.asarray(x, dtype=float)
        s = float(self.front(t))
        ht = eps * t
        hx = eps * float(np.sqrt(case.alpha_inner * t)) * 10.0
        mask = (x > 2 * hx) & (np.abs(x - s) > 2 * hx)
        mask &= x < s + 20 * float(np.sqrt(max(case.outer.alpha, case.alpha_inner) * t))
        xm = x[mask]
        if xm.size == 0:
            return 0.0
        t_dot = (self.temperature(xm, t + ht) - self.temperature(xm, t - ht)) / (2 * ht)
        lap = (
            self.temperature(xm + hx, t) - 2 * self.temperature(xm, t)
            + self.temperature(xm - hx, t)
        ) / hx ** 2
        alpha = np.where(xm <= s, case.alpha_inner, case.outer.alpha)
        resid = np.abs(t_dot - alpha * lap)
        scale = np.max(np.abs(t_dot)) + 1e-30
        return float(np.max(resid) / scale)


# ---------------------------------------------------------------------------
# quasi-steady cylinder (literature anchor for C2)
# ---------------------------------------------------------------------------
def _cyl_g(r: float, r_outer: float) -> float:
    """Antiderivative of ``r ln(R/r)``: ``r^2/2 ln(R/r) + r^2/4``."""
    return 0.5 * r * r * math.log(r_outer / r) + 0.25 * r * r


def quasi_steady_cylinder_time(
    case: IceWaterCase,
    r_f: float,
    r0: float,
    r_outer: float,
) -> float:
    """Time for an ice cylinder to melt from ``r0`` down to ``r_f``.

    Quasi-steady (zero sensible heat) melting of an ice cylinder in warm water:
    conduction across the growing liquid annulus ``r_f < r < R`` with the outer
    boundary held at ``T_wall``.  The interface balance

        rho L (-dr_f/dt) = k dT / (r_f ln(R/r_f))

    integrates to

        t(r_f) = rho L / (k dT) * { [r0^2/2 ln(R/r0) + r0^2/4]
                                    - [r_f^2/2 ln(R/r_f) + r_f^2/4] }

    which is the classical cylindrical phase-change solution (London & Seban
    1943; Carslaw & Jaeger 1959 sec. 11.2; Lunardini 1991 ch. 4).  It is the
    literature anchor of criterion C2 -- an *analytic* literature solution, not
    raw experimental data -- and is valid in the conduction-limited
    (small Stefan number) regime; it neglects the sensible heat of the water
    layer and any natural convection, both of which enhance real melting.
    """
    if not (0.0 < r_f <= r0 < r_outer):
        raise ValueError("require 0 < r_f <= r0 < r_outer")
    pref = case.latent_volumetric / (case.water.k * abs(case.dT_inner))
    return pref * (_cyl_g(r0, r_outer) - _cyl_g(r_f, r_outer))


def quasi_steady_cylinder_front(
    case: IceWaterCase,
    t: float,
    r0: float,
    r_outer: float,
) -> float:
    """Front radius of the quasi-steady cylindrical melting solution [m].

    Inverts :func:`quasi_steady_cylinder_time`; the ice core shrinks, so
    ``r_f(t) <= r0`` monotonically.
    """
    if t <= 0:
        return r0
    pref = case.latent_volumetric / (case.water.k * abs(case.dT_inner))
    target = _cyl_g(r0, r_outer) - t / pref
    if target <= _cyl_g(1e-12 * r_outer, r_outer):
        return 0.0
    return float(brentq(lambda r: _cyl_g(r, r_outer) - target,
                        1e-9 * r_outer, r0, xtol=1e-14, rtol=1e-15))


def quasi_steady_cylinder_error_estimate(case: IceWaterCase) -> float:
    """Order-of-magnitude relative error of the quasi-steady front [--].

    The quasi-steady solution neglects the sensible heat stored in the liquid
    layer; the neglected term is ``O(Ste)`` relative to the latent term.  Half
    the Stefan number is used as the (deliberately conservative) order
    estimate -- it is an *indicator* of validity, not a rigorous bound.
    """
    return 0.5 * case.stefan_inner