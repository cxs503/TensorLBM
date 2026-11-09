"""Exact similarity solutions for the one-phase Stefan (moving-boundary) problem.

All quantities are in **lattice units** with ``rho = 1`` and
``cp_s = cp_l = cp``, so that the Stefan / Jacob number is
``Ste = Ja = cp*dT/L``.

Planar (Neumann) branch
-----------------------
Two complementary branches of the same one-phase similarity problem:

* ``side="solid"`` — **freezing**: a wall at ``T_wall < T_freeze`` drives a solid
  layer into liquid held at ``T_freeze``; conduction happens in the solid
  (``erf`` profile)::

      s(t) = 2*lam*sqrt(alpha*t)
      lam*exp(lam**2)*erf(lam) = Ste/sqrt(pi)

      T(x,t) = T_wall + dT*erf(eta)/erf(lam),   eta = x/(2 sqrt(alpha t)),  x <= s(t)
      T(x,t) = T_freeze,                                                   x >  s(t)

* ``side="liquid"`` — **melting**: a wall at ``T_wall > T_freeze`` melts a solid
  held at ``T_freeze``; conduction happens in the liquid (``erfc`` profile)::

      lam*exp(lam**2)*erfc(lam) = Ste/sqrt(pi)

      T(x,t) = T_wall - dT*erfc(eta)/erfc(lam),  x <= s(t)

with ``dT = Ste*L/cp`` and ``s(t) = 2*lam*sqrt(alpha*t)`` measured from the wall.

Both relations follow from the interface energy balance
``rho*L*ds/dt = k*|dT/dx|`` evaluated at ``eta = lam`` (derivation in
``docs/P155W1_N01_STEFAN_PHASE_CHANGE.md`` §2 and re-checked numerically by
``tests/test_stefan_phase_change_verification.py::test_planar_similarity_pde_residual``).

Cylindrical branch
------------------
A solid cylinder (``r < r_f``, isothermal at ``T_freeze`` — one-phase
idealisation) melts into a superheated liquid ``T_inf = T_freeze + Ja*L/cp``::

    r_f(t) = 2*lam*sqrt(alpha*t)
    exp(-lam**2)/E1(lam**2) = lam**2/Ja
    T(r,t) = T_inf - dT*E1(eta**2)/E1(lam**2),   eta = r/(2 sqrt(alpha t)),  r >= r_f

The front law is the same as for cylindrical solidification into an
*undercooled* melt with time reversed; undercooled liquid is metastable and
cannot be represented by a single-valued (equilibrium) enthalpy function, which
is why the cylindrical verification case is run on the melting branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, erfc, exp1

__all__ = [
    "NeumannSimilarity",
    "CylindricalSimilarity",
    "planar_lambda",
    "cylindrical_lambda",
    "front_from_phase_integral",
    "l2_profile_error",
]


# ---------------------------------------------------------------------------
# similarity coefficients
# ---------------------------------------------------------------------------
def planar_lambda(stefan_number: float, side: str = "solid") -> float:
    """Solve the one-phase planar Neumann relation for ``lam``.

    ``side="solid"``  -> ``lam*exp(lam^2)*erf(lam)  = Ste/sqrt(pi)`` (freezing)
    ``side="liquid"`` -> ``lam*exp(lam^2)*erfc(lam) = Ste/sqrt(pi)`` (melting)
    """
    if stefan_number <= 0.0:
        raise ValueError("stefan number must be positive")
    if side not in ("solid", "liquid"):
        raise ValueError(f"unknown side {side!r} (expected 'solid' or 'liquid')")
    target = stefan_number / math.sqrt(math.pi)

    def res(lam: float) -> float:
        # both branches are "wall at one end, front at 2*lam*sqrt(alpha t) at the
        # other, conduction in the phase in between", hence the same relation;
        # the erfc branch of the literature is the two-phase problem with a
        # semi-infinite superheated liquid, which is not the configuration here
        return lam * math.exp(lam * lam) * erf(lam) - target

    upper = 1.0
    while res(upper) < 0.0:
        upper *= 2.0
    return float(brentq(res, 1e-12, upper, xtol=1e-15, rtol=1e-14))


def cylindrical_lambda(jacob_number: float) -> float:
    """Solve ``exp(-lam^2)/E1(lam^2) = lam^2/Ja`` for the circular-front law."""
    if jacob_number <= 0.0:
        raise ValueError("jacob number must be positive")
    ja = float(jacob_number)

    def res(lam: float) -> float:
        z = lam * lam
        # e^{-z}/E1(z) with the large-z asymptotic z*(1 - 1/z + ...) to avoid the
        # exp1() underflow for z >~ 700 (the root is always at moderate z)
        ratio = z if z > 100.0 else math.exp(-z) / float(exp1(z))
        return ratio - z / ja

    # res(lam) > 0 for small lam and < 0 for large lam whenever Ja < 1;
    # Ja -> 1 pushes the root to lam -> infinity (no solution for Ja >= 1)
    hi = 0.5
    while res(hi) > 0.0:
        hi *= 2.0
        if hi > 1e3:
            raise ValueError(f"no cylindrical similarity solution for Ja={ja} (needs Ja < 1)")
    return float(brentq(res, 1e-9, hi, xtol=1e-15, rtol=1e-14))


# ---------------------------------------------------------------------------
# exact solutions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NeumannSimilarity:
    """One-phase planar Stefan similarity solution (see module docstring)."""

    stefan_number: float
    alpha: float
    side: str = "solid"
    t_freeze: float = 0.0
    cp: float = 1.0
    latent: float = 1.0

    @property
    def lam(self) -> float:
        return planar_lambda(self.stefan_number, self.side)

    @property
    def dT(self) -> float:
        """``|T_wall - T_freeze|``."""
        return self.stefan_number * self.latent / self.cp

    @property
    def boundary_temperature(self) -> float:
        return self.t_freeze - self.dT if self.side == "solid" else self.t_freeze + self.dT

    def front(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        return 2.0 * self.lam * math.sqrt(self.alpha * t)

    def time_for_front(self, s: float) -> float:
        return (s / (2.0 * self.lam)) ** 2 / self.alpha

    def front_velocity(self, t: float) -> float:
        if t <= 0.0:
            return float("inf")
        return self.lam * math.sqrt(self.alpha / t)

    def temperature(self, x, t: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if t <= 0.0:
            return np.full_like(x_arr, self.t_freeze)
        eta = x_arr / (2.0 * math.sqrt(self.alpha * t))
        lam = self.lam
        if self.side == "solid":
            # T = T_wall + dT*erf(eta)/erf(lam): T(0) = T_wall, T(s) = T_freeze
            prof = self.boundary_temperature + self.dT * erf(eta) / erf(lam)
        else:
            # T = T_freeze + dT*(erfc(eta) - erfc(lam))/erf(lam):
            # T(0) = T_freeze + dT*(1 - erfc(lam))/erf(lam) = T_wall, T(s) = T_freeze
            prof = self.t_freeze + (self.boundary_temperature - self.t_freeze) * (
                (erfc(eta) - erfc(lam)) / erf(lam)
            )
        return np.where(x_arr <= self.front(t), prof, self.t_freeze)

    def wall_heat_input(self, t: float) -> float:
        """Cumulative wall heat ``int_0^t |q_wall| dt'`` (per unit depth).

        ``|q_wall(t)| = k*dT/(sqrt(pi*alpha*t)*denom)`` with
        ``k = rho*cp*alpha = cp*alpha`` (``rho = 1``), hence

            ``Q(t) = 2*cp*dT*sqrt(alpha*t)/(sqrt(pi)*denom)``

        where ``denom`` is ``erf(lam)`` (freezing, heat leaves the domain) or
        ``erfc(lam)`` (melting, heat enters the domain).
        """
        if t <= 0.0:
            return 0.0
        lam = self.lam
        # |q_wall(t)| = k*dT/(sqrt(pi*alpha*t)*erf(lam));  Q(t) = int_0^t |q_wall| dt'
        return 2.0 * self.cp * self.dT * math.sqrt(self.alpha * t) / (
            math.sqrt(math.pi) * erf(lam)
        )

    def wall_flux(self, t: float) -> float:
        """Instantaneous wall heat flux magnitude ``|q_wall(t)|``."""
        if t <= 0.0:
            return float("inf")
        lam = self.lam
        return self.cp * self.dT * math.sqrt(self.alpha) / (math.sqrt(math.pi * t) * erf(lam))


@dataclass(frozen=True)
class CylindricalSimilarity:
    """One-phase cylindrical Stefan similarity solution with a circular front."""

    jacob_number: float
    alpha: float
    side: str = "undercooled"
    t_freeze: float = 0.0
    cp: float = 1.0
    latent: float = 1.0

    @property
    def lam(self) -> float:
        return cylindrical_lambda(self.jacob_number)

    @property
    def dT(self) -> float:
        """``|T_inf - T_freeze|`` (> 0)."""
        return self.jacob_number * self.latent / self.cp

    @property
    def far_field_temperature(self) -> float:
        """``T_freeze - dT`` (undercooled melt) or ``T_freeze + dT`` (superheated).

        A **growing** circular front requires an undercooled melt: the latent heat
        released at the interface is conducted outwards into liquid that is colder
        than ``T_freeze``.  The superheated branch describes a shrinking front
        (melting) and is provided only for completeness.
        """
        if self.side == "undercooled":
            return self.t_freeze - self.dT
        if self.side == "superheated":
            return self.t_freeze + self.dT
        raise ValueError(f"unknown side {self.side!r}")

    def front(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        return 2.0 * self.lam * math.sqrt(self.alpha * t)

    def time_for_front(self, radius: float) -> float:
        return (radius / (2.0 * self.lam)) ** 2 / self.alpha

    def front_velocity(self, t: float) -> float:
        if t <= 0.0:
            return float("inf")
        return self.lam * math.sqrt(self.alpha / t)

    def temperature(self, r, t: float) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        if t <= 0.0:
            return np.full_like(r_arr, self.far_field_temperature)
        eta = r_arr / (2.0 * math.sqrt(self.alpha * t))
        lam = self.lam
        e_lam = float(exp1(lam * lam))
        # T = T_inf + (T_freeze - T_inf)*E1(eta^2)/E1(lam^2): T(lam) = T_freeze,
        # T(inf) = T_inf
        prof = self.far_field_temperature + (self.t_freeze - self.far_field_temperature) * (
            exp1(eta * eta) / e_lam
        )
        return np.where(r_arr >= self.front(t), prof, self.t_freeze)

    def heat_input_per_length(self, t: float) -> float:
        """Cumulative heat absorbed at the front per unit depth.

        ``q(t) = 2*pi*r_f*k*dT*exp(-lam^2)/(E1(lam^2)*lam*sqrt(alpha*t))``;
        integrating with ``r_f = 2*lam*sqrt(alpha*t)`` gives

            ``Q(t) = 4*pi*cp*dT*alpha*t*exp(-lam^2)/E1(lam^2)``

        (``k = cp*alpha``, ``rho = 1``).
        """
        if t <= 0.0:
            return 0.0
        lam = self.lam
        z = lam * lam
        return 4.0 * math.pi * self.cp * self.dT * self.alpha * t * math.exp(-z) / float(exp1(z))


# ---------------------------------------------------------------------------
# error measures
# ---------------------------------------------------------------------------
def front_from_phase_integral(phase_indicator: np.ndarray, dx: float = 1.0) -> float:
    """Front position from the integral of a phase indicator (sub-cell accurate).

    Pass ``1 - f_l`` for a planar freezing case (solid behind the wall) or
    ``f_l`` for a melting case.
    """
    return float(np.sum(phase_indicator) * dx)


def l2_profile_error(
    T_num: np.ndarray,
    T_exact: np.ndarray,
    x: np.ndarray,
    front: float,
    band: float = 4.0,
    dT: float | None = None,
) -> dict[str, float]:
    """L2 / Linf error of a 1-D profile, excluding the smeared interface band."""
    mask = np.abs(np.asarray(x, dtype=float) - front) > band
    diff = np.asarray(T_num, dtype=float)[mask] - np.asarray(T_exact, dtype=float)[mask]
    scale = dT if dT is not None else max(float(np.max(np.abs(T_exact))), 1e-30)
    return {
        "l2_rel": float(np.sqrt(np.mean(diff**2)) / scale),
        "linf_rel": float(np.max(np.abs(diff)) / scale),
        "n_nodes_used": int(np.count_nonzero(mask)),
        "n_nodes_excluded": int(np.count_nonzero(~mask)),
    }