r"""Two-phase (superheated-melt) planar Stefan similarity solution.

Why a *second* Stefan surface
-----------------------------
``stefan_exact`` provides the **one-phase** Neumann branches: the phase change
takes place into a medium that is *isothermal* at ``T_freeze`` (no sensible heat
in the far phase).  The classical Stefan problem stated in most textbooks --
and in the campaign brief for this node -- is the **two-phase** one::

    solid  0 <= x < s(t) :  conduction, wall at T_0 < T_freeze
    liquid    x > s(t)   :  conduction, initial / far field T_i > T_freeze

so latent heat released at the front has to be conducted away through *both*
phases and the far liquid is superheated (``T_i > T_m``).  This is a strictly
larger problem class: the front coefficient solves a *coupled* transcendental
equation with two Stefan numbers instead of one, and the liquid-phase energy
budget has to be closed as well.

Similarity solution (Neumann 1860; Carslaw & Jaeger §11.2)
----------------------------------------------------------
With the Boltzmann variable ``eta_s = x/(2 sqrt(alpha_s t))`` for the solid and
``eta_l = x/(2 sqrt(alpha_l t))`` for the liquid::

    T_s(x,t) = T_0 + dT_s * erf(eta_s) / erf(lam)
    T_l(x,t) = T_i - dT_l * erfc(eta_l) / erfc(lam_l),   lam_l = lam*sqrt(alpha_s/alpha_l)
    s(t)     = 2*lam*sqrt(alpha_s*t)

where ``dT_s = T_freeze - T_0 > 0``, ``dT_l = T_i - T_freeze > 0`` and ``lam``
solves the interface energy balance ``rho*L*ds/dt = k_s dT_s/dx - k_l dT_l/dx``
evaluated at ``eta = lam``::

    lam = (Ste_s/sqrt(pi)) * exp(-lam^2)/erf(lam)
          - (Ste_l/sqrt(pi)) * sqrt(alpha_l/alpha_s)
            * exp(-lam^2*alpha_s/alpha_l) / erfc(lam_l)

with ``Ste_s = cp_s*dT_s/L`` and ``Ste_l = cp_l*dT_l/L``.  Setting
``Ste_l = 0`` collapses this to the one-phase relation
``lam*exp(lam^2)*erf(lam) = Ste_s/sqrt(pi)`` used by :mod:`stefan_exact`
(audited in the test-suite), which is why the two surfaces are consistent
instead of duplicated.

Energy bookkeeping (per unit area, ``rho = 1``, semi-infinite domain)
---------------------------------------------------------------------
Relative to the initial state ``T = T_i`` everywhere (all liquid)::

    latent    = -L*s(t)
    sensible_solid = cp_s * [ (T_0 - T_i)*s + dT_s * 2*sqrt(alpha_s*t)*I1(lam)/erf(lam) ]
    sensible_liquid = -cp_l * dT_l * 2*sqrt(alpha_l*t)*J1(lam_l)/erfc(lam_l)

    I1(z) = int_0^z erf(u) du = z*erf(z) + (exp(-z^2) - 1)/sqrt(pi)
    J1(z) = int_z^inf erfc(u) du = -z*erfc(z) + exp(-z^2)/sqrt(pi)

and the total must equal the heat removed through the wall::

    Q(t) = -2*cp_s*dT_s*sqrt(alpha_s*t/pi)/erf(lam)

``Q(t) == latent + sensible_solid + sensible_liquid`` to round-off; that
identity is an *independent* self-audit of the closed form (it exercises the
Stefan condition, both profiles and the erf integrals simultaneously) and is
checked in ``tests/test_stefan_two_phase_sweep.py``.

Sign convention: heat flowing **into** the domain is positive, so ``Q < 0``
(heat is extracted at the cold wall) and ``latent < 0`` (liquid freezes).

Limits / applicability
----------------------
* one-dimensional planar (no curvature), constant properties per phase;
* ``alpha_s != alpha_l`` is supported by the closed form, but the enthalpy
  lattice-Boltzmann solver used as the numerical counterpart diffuses a single
  sensible-enthalpy field, so the *LBM verification* runs equal properties
  (``alpha_s = alpha_l``); the unequal-property branch is used only as an extra
  exact-solution self-audit;
* a root always exists for ``Ste_s > 0``; as ``Ste_l -> inf`` it degenerates to
  ``lam -> 0`` -- the superheat then feeds the interface at the same rate the
  wall removes heat and the front stalls (``Ste_l = 200`` with ``Ste_s = 0.05``
  gives ``lam = 2.2e-4``).  The practical limit of this similarity family is
  therefore resolution, not existence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, erfc

__all__ = [
    "two_phase_residual",
    "two_phase_lambda",
    "TwoPhaseSimilarity",
    "erf_integral",
    "erfc_integral",
]


# ---------------------------------------------------------------------------
# erf integrals
# ---------------------------------------------------------------------------
def erf_integral(z: float) -> float:
    r"""``int_0^z erf(u) du = z*erf(z) + (exp(-z^2) - 1)/sqrt(pi)``."""
    return z * erf(z) + (math.exp(-z * z) - 1.0) / math.sqrt(math.pi)


def erfc_integral(z: float) -> float:
    r"""``int_z^inf erfc(u) du = -z*erfc(z) + exp(-z^2)/sqrt(pi)``."""
    return -z * erfc(z) + math.exp(-z * z) / math.sqrt(math.pi)


# ---------------------------------------------------------------------------
# similarity coefficient
# ---------------------------------------------------------------------------
def two_phase_residual(
    lam: float,
    stefan_solid: float,
    stefan_liquid: float = 0.0,
    alpha_solid: float = 1.0,
    alpha_liquid: float = 1.0,
) -> float:
    """Residual of the two-phase interface energy balance (zero at the root).

    ``lam*exp(lam^2)``-free form (see module docstring)::

        R(lam) = Ste_s*exp(-lam^2)/(sqrt(pi)*erf(lam))
                 - Ste_l*sqrt(ar)*exp(-lam^2/ar)/(sqrt(pi)*erfc(lam*sqrt(ar)))
                 - lam,        ar = alpha_l/alpha_s
    """
    if not (0.0 < lam < 30.0):
        # singular end points; callers bracket strictly inside
        raise ValueError("lam out of bracket")
    ar = alpha_liquid / alpha_solid
    lam_l = lam * math.sqrt(1.0 / ar)
    solid_term = stefan_solid * math.exp(-lam * lam) / (math.sqrt(math.pi) * erf(lam))
    liquid_term = stefan_liquid * math.sqrt(ar) * math.exp(-(lam * lam) / ar) / (
        math.sqrt(math.pi) * erfc(lam_l)
    )
    return solid_term - liquid_term - lam


def two_phase_lambda(
    stefan_solid: float,
    stefan_liquid: float = 0.0,
    alpha_solid: float = 1.0,
    alpha_liquid: float = 1.0,
) -> float:
    """Solve the two-phase Stefan relation for ``lam`` (brentq, rel. 1e-14).

    ``stefan_liquid = 0`` reproduces :func:`stefan_exact.planar_lambda`.
    """
    if stefan_solid <= 0.0:
        raise ValueError("stefan_solid must be positive")
    if stefan_liquid < 0.0:
        raise ValueError("stefan_liquid must be non-negative")
    if not (alpha_solid > 0.0 and alpha_liquid > 0.0):
        raise ValueError("diffusivities must be positive")

    def f(lam: float) -> float:
        return two_phase_residual(lam, stefan_solid, stefan_liquid, alpha_solid, alpha_liquid)

    lo, hi = 1e-10, 1.0
    f_lo = -1.0  # sign placeholder, filled below
    # walk the upper end outwards until the residual changes sign
    while hi < 29.0:
        f_lo, f_hi = f(lo), f(hi)
        if f_hi < 0.0:
            break
        hi *= 1.5
    else:  # pragma: no cover - defensive: the residual diverges at lam -> 0
        raise RuntimeError(
            f"two-phase Stefan bracket failed for Ste_s={stefan_solid}, Ste_l={stefan_liquid}"
        )
    if f_lo < 0.0:  # pragma: no cover - residual is +inf at lam->0 for Ste_s>0
        raise ValueError("bracket does not contain the root")
    return float(brentq(f, lo, hi, xtol=1e-15, rtol=1e-14, maxiter=200))


# ---------------------------------------------------------------------------
# similarity solution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TwoPhaseSimilarity:
    """Two-phase planar Stefan similarity solution (solid behind the wall).

    Parameters
    ----------
    stefan_solid, stefan_liquid :
        ``Ste_s = cp_s*(T_freeze-T_0)/L`` and ``Ste_l = cp_l*(T_i-T_freeze)/L``.
        ``Ste_l > 0`` is the superheated-melt (two-phase) case; ``0`` is the
        one-phase Neumann limit.
    alpha_solid, alpha_liquid :
        Thermal diffusivities; ``k = cp*alpha`` (``rho = 1``).
    """

    stefan_solid: float
    stefan_liquid: float = 0.0
    alpha_solid: float = 1.0 / 6.0
    alpha_liquid: float = 1.0 / 6.0
    t_freeze: float = 0.0
    cp_solid: float = 1.0
    cp_liquid: float = 1.0
    latent: float = 1.0
    _lam: float = field(default=None, repr=False, compare=False)

    # -- coefficients -------------------------------------------------------
    @property
    def lam(self) -> float:
        if self._lam is None:
            value = two_phase_lambda(
                self.stefan_solid,
                self.stefan_liquid,
                self.alpha_solid,
                self.alpha_liquid,
            )
            object.__setattr__(self, "_lam", value)
        return self._lam

    @property
    def lam_liquid(self) -> float:
        """``lam*sqrt(alpha_s/alpha_l)`` -- the liquid-side similarity argument."""
        return self.lam * math.sqrt(self.alpha_solid / self.alpha_liquid)

    @property
    def dT_solid(self) -> float:
        return self.stefan_solid * self.latent / self.cp_solid

    @property
    def dT_liquid(self) -> float:
        return self.stefan_liquid * self.latent / self.cp_liquid

    @property
    def wall_temperature(self) -> float:
        return self.t_freeze - self.dT_solid

    @property
    def initial_temperature(self) -> float:
        """Far-field / initial liquid temperature ``T_i >= T_freeze``."""
        return self.t_freeze + self.dT_liquid

    @property
    def dT_reference(self) -> float:
        """Temperature scale used to normalise profile errors."""
        return max(self.dT_solid, self.dT_liquid)

    @property
    def conductivity_solid(self) -> float:
        return self.cp_solid * self.alpha_solid

    @property
    def conductivity_liquid(self) -> float:
        return self.cp_liquid * self.alpha_liquid

    # -- kinematics ---------------------------------------------------------
    def front(self, t):
        """Front position ``2*lam*sqrt(alpha_s*t)`` (scalar or array)."""
        if np.ndim(t) == 0:
            t = float(t)
            return 0.0 if t <= 0.0 else 2.0 * self.lam * math.sqrt(self.alpha_solid * t)
        t_arr = np.maximum(np.asarray(t, dtype=float), 0.0)
        return 2.0 * self.lam * np.sqrt(self.alpha_solid * t_arr)

    def time_for_front(self, s: float) -> float:
        return (s / (2.0 * self.lam)) ** 2 / self.alpha_solid

    def front_velocity(self, t: float) -> float:
        if t <= 0.0:
            return float("inf")
        return self.lam * math.sqrt(self.alpha_solid / t)

    # -- fields -------------------------------------------------------------
    def solid_temperature(self, x, t: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if t <= 0.0:
            return np.full_like(x_arr, self.wall_temperature)
        eta = x_arr / (2.0 * math.sqrt(self.alpha_solid * t))
        return self.wall_temperature + self.dT_solid * erf(eta) / erf(self.lam)

    def liquid_temperature(self, x, t: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if t <= 0.0:
            return np.full_like(x_arr, self.initial_temperature)
        eta = x_arr / (2.0 * math.sqrt(self.alpha_liquid * t))
        return self.initial_temperature - self.dT_liquid * erfc(eta) / erfc(self.lam_liquid)

    def temperature(self, x, t: float) -> np.ndarray:
        """Piecewise temperature field (clamped to the physical branches)."""
        x_arr = np.asarray(x, dtype=float)
        s = self.front(t)
        return np.where(
            x_arr <= s,
            self.solid_temperature(x_arr, t),
            self.liquid_temperature(x_arr, t),
        )

    def interface_flux_solid(self, t: float) -> float:
        """``k_s*dT_s/dx`` at the front (positive, heat flowing towards +x)."""
        return (
            self.conductivity_solid
            * self.dT_solid
            * math.exp(-self.lam**2)
            / (math.sqrt(math.pi * self.alpha_solid * t) * erf(self.lam))
        )

    def interface_flux_liquid(self, t: float) -> float:
        """``k_l*dT_l/dx`` at the front (positive)."""
        return (
            self.conductivity_liquid
            * self.dT_liquid
            * math.exp(-self.lam_liquid**2)
            / (math.sqrt(math.pi * self.alpha_liquid * t) * erfc(self.lam_liquid))
        )

    def stefan_condition_residual(self, t: float) -> float:
        """``rho*L*ds/dt - (k_s dT_s/dx - k_l dT_l/dx)`` at the front (``rho = 1``)."""
        return (
            self.latent * self.front_velocity(t)
            - self.interface_flux_solid(t)
            + self.interface_flux_liquid(t)
        )

    # -- energy -------------------------------------------------------------
    def wall_heat_rate(self, t: float) -> float:
        """Heat rate through the wall into the domain (negative: extraction)."""
        if t <= 0.0:
            return float("-inf")
        return -self.conductivity_solid * self.dT_solid / (
            math.sqrt(math.pi * self.alpha_solid * t) * erf(self.lam)
        )

    def wall_heat_input(self, t: float) -> float:
        """Cumulative heat through the wall into the domain (negative)."""
        if t <= 0.0:
            return 0.0
        return (
            -2.0
            * self.cp_solid
            * self.dT_solid
            * math.sqrt(self.alpha_solid * t / math.pi)
            / erf(self.lam)
        )

    def energy_budget(self, t: float) -> dict[str, float]:
        """Analytic enthalpy decomposition relative to the all-liquid state."""
        s = self.front(t)
        root_s = math.sqrt(self.alpha_solid * t)
        root_l = math.sqrt(self.alpha_liquid * t)
        sensible_solid = self.cp_solid * (
            (self.wall_temperature - self.initial_temperature) * s
            + self.dT_solid * 2.0 * root_s * erf_integral(self.lam) / erf(self.lam)
        )
        sensible_liquid = (
            -self.cp_liquid * self.dT_liquid * 2.0 * root_l * erfc_integral(self.lam_liquid)
            / erfc(self.lam_liquid)
        )
        latent_change = -self.latent * s
        return {
            "time": float(t),
            "front": s,
            "latent": latent_change,
            "sensible_solid": sensible_solid,
            "sensible_liquid": sensible_liquid,
            "sensible": sensible_solid + sensible_liquid,
            "total": latent_change + sensible_solid + sensible_liquid,
            "wall_input": self.wall_heat_input(t),
            "wall_input_minus_total": self.wall_heat_input(t)
            - (latent_change + sensible_solid + sensible_liquid),
        }

    def energy_budget_window(self, t: float, x_lo: float, x_hi: float) -> dict[str, float]:
        """Analytic enthalpy decomposition over a finite window ``[x_lo, x_hi]``.

        This is the apples-to-apples reference for a *finite* control volume:
        a lattice run stores enthalpy on the nodes ``1..nx-2``, i.e. over
        ``[0.5, nx-1.5]``, and comparing against the semi-infinite budget would
        charge the solver for the wall half-cell it never owns.  All three
        pieces (latent, solid sensible, liquid sensible) are integrated
        analytically with the erf integrals above.
        """
        if x_hi <= x_lo:
            raise ValueError("empty window")
        s_front = self.front(t)
        root_s = math.sqrt(self.alpha_solid * t)
        root_l = math.sqrt(self.alpha_liquid * t)
        t_i = self.initial_temperature

        # latent: liquid fraction (1 - f_s) integrated over the window
        solid_lo, solid_hi = x_lo, min(s_front, x_hi)
        latent = -self.latent * max(0.0, solid_hi - solid_lo)

        # solid sensible: cp_s * int (T_s - T_i) dx
        sensible_solid = 0.0
        if solid_hi > solid_lo:
            eta_a = solid_lo / (2.0 * root_s) if solid_lo > 0 else 0.0
            eta_b = solid_hi / (2.0 * root_s)
            sensible_solid = self.cp_solid * (
                (self.wall_temperature - t_i) * (solid_hi - solid_lo)
                + self.dT_solid
                * 2.0
                * root_s
                * (erf_integral(eta_b) - erf_integral(eta_a))
                / erf(self.lam)
            )

        # liquid sensible: -cp_l * dT_l * int erfc in [max(x_lo, s), x_hi]
        liquid_lo = max(x_lo, s_front)
        sensible_liquid = 0.0
        if x_hi > liquid_lo:
            eta_lo = liquid_lo / (2.0 * root_l)
            eta_hi = x_hi / (2.0 * root_l)
            sensible_liquid = (
                -self.cp_liquid
                * self.dT_liquid
                * 2.0
                * root_l
                * (erfc_integral(eta_lo) - erfc_integral(eta_hi))
                / erfc(self.lam_liquid)
            )

        return {
            "time": float(t),
            "window": [float(x_lo), float(x_hi)],
            "front": s_front,
            "latent": latent,
            "sensible_solid": sensible_solid,
            "sensible_liquid": sensible_liquid,
            "sensible": sensible_solid + sensible_liquid,
            "total": latent + sensible_solid + sensible_liquid,
        }

    # -- self-audit ---------------------------------------------------------
    def pde_residual(self, t: float, n: int = 4001) -> dict[str, float]:
        """Max finite-difference residual of ``T'' + 2*eta*T' = 0`` per phase.

        The similarity profile is sampled in the Boltzmann variable and the
        residual is formed with second-order finite differences of the *sampled*
        profile -- not with the analytic derivatives, which would make the check
        an algebraic tautology.  ``T'' + 2*eta*T' = 0`` is exactly
        ``dT/dt = alpha*T_xx`` written in similarity variables, so a small
        residual means the closed form genuinely solves the heat equation.
        """
        out: dict[str, float] = {}
        for name, lam, kind in (
            ("solid", self.lam, "erf"),
            ("liquid", self.lam_liquid, "erfc"),
        ):
            lo, hi = (1e-6, lam) if kind == "erf" else (lam, lam + 6.0)
            eta = np.linspace(lo, hi, n)
            prof = erf(eta) / erf(lam) if kind == "erf" else erfc(eta) / erfc(lam)
            d1 = np.gradient(prof, eta, edge_order=2)
            d2 = np.gradient(d1, eta, edge_order=2)
            res = (d2 + 2.0 * eta * d1)[2:-2]  # one-sided edges are O(h)
            scale = float(np.max(np.abs(d1))) / max(lam, 1e-12)
            out[f"{name}_ode_residual_max"] = float(np.max(np.abs(res)))
            out[f"{name}_ode_residual_rel"] = float(np.max(np.abs(res)) / scale)
        return out
