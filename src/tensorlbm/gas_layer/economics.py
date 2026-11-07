"""Compressor-work / net-power ledger for gas-layer drag reduction.

The engineering question behind an air-lubrication system is not "how much drag
does the gas layer remove" but "does the removed drag power exceed the
compressor power needed to supply the gas".  This module keeps that ledger in
explicit, testable form.

Definitions (per unit wetted hull area, SI)
------------------------------------------
* required gas volume flow  ``q_g = C_q U L``          [m^3/(s m)]
* wetted-area density of gas flow ``q_g'' = C_q U``    [m^3/(s m^2)]
* compressor specific work, isothermal, from ambient ``p_a``:
  ``w_c = (p_a/rho_a) ln(p_inj/p_a)``                  [J/kg]
  with ``p_inj = p_a + rho_w g_k T`` (injection at draft ``T``)
* compressor power density ``P_c'' = q_g'' p_a ln(p_inj/p_a) / eta_c`` [W/m^2]
* saved power density ``P_s'' = DR * tau_w * U``        [W/m^2]
* net ``dP_net'' = P_s'' - P_c''``                      [W/m^2]

``eta_c`` is the overall compressor efficiency (0.6 nominal, 0.5/0.7 in the
sensitivity study).  ``dP_net''`` is **strictly decreasing in the reciprocal of
``eta_c``**: a less efficient compressor can only shrink the positive domain.
That monotonicity is asserted by the tests -- it is the assertion that catches a
sign error in the ledger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .reference import skin_friction_ittc57

#: Physical constants / default operating point.
RHO_WATER = 1025.0  # kg/m^3
RHO_AIR = 1.225  # kg/m^3 at 15 C, 1 atm
G_ACCEL = 9.81  # m/s^2
P_ATM = 101325.0  # Pa
NU_WATER = 1.19e-6  # m^2/s at 15 C


@dataclass(frozen=True)
class EconomicAssumptions:
    """Operating-point assumptions for the ledger (all SI)."""

    plate_length: float = 100.0  # m, hull length over which gas is injected
    draft: float = 5.0  # m, injection depth below the free surface
    eta_c: float = 0.6  # overall compressor efficiency
    rho_water: float = RHO_WATER
    rho_air: float = RHO_AIR
    p_atm: float = P_ATM
    temperature: float = 288.15  # K
    nu_water: float = NU_WATER

    def validate(self) -> None:
        if self.plate_length <= 0 or self.draft < 0:
            raise ValueError("plate_length must be > 0 and draft >= 0")
        if not 0.0 < self.eta_c <= 1.0:
            raise ValueError("eta_c must be in (0, 1]")


def compressor_specific_work(
    p_atm: float = P_ATM,
    rho_water: float = RHO_WATER,
    draft: float = 5.0,
    g: float = G_ACCEL,
) -> float:
    """Isothermal compressor specific work per unit gas volume at ambient [J/m^3].

    ``w_c = p_a ln(p_inj/p_a)`` with ``p_inj = p_a + rho_w g draft``.  This is
    the reversible isothermal work; real machines enter through ``eta_c``.
    """
    p_inj = p_atm + rho_water * g * draft
    if p_inj <= p_atm:
        raise ValueError("injection pressure must exceed ambient")
    return p_atm * math.log(p_inj / p_atm)


def wall_shear_from_friction(U: float, length: float, nu: float = NU_WATER,
                             rho: float = RHO_WATER) -> float:
    """Mean wall shear stress from the ITTC-1957 line: ``tau_w = 0.5 rho U^2 Cf``."""
    re_l = U * length / nu
    return 0.5 * rho * U * U * skin_friction_ittc57(re_l)


def net_power_density(
    drag_reduction: float,
    U: float,
    tau_w: float,
    C_q: float,
    eta_c: float = 0.6,
    w_c: float | None = None,
) -> float:
    """Net power density [W/m^2] for one operating point.

    ``dP'' = DR tau_w U - C_q U w_c / eta_c``.  Positive means the gas layer pays
    for its own compressor.
    """
    if C_q < 0.0 or drag_reduction < 0.0:
        raise ValueError("C_q and drag_reduction must be non-negative")
    if not 0.0 < eta_c <= 1.0:
        raise ValueError("eta_c must be in (0, 1]")
    work = compressor_specific_work() if w_c is None else w_c
    saved = drag_reduction * tau_w * U
    spent = C_q * U * work / eta_c
    return saved - spent


def required_efficiency(
    drag_reduction: float,
    tau_w: float,
    C_q: float,
    w_c: float | None = None,
) -> float:
    """Break-even compressor efficiency for one operating point.

    ``eta_break_even = C_q w_c / (DR tau_w)``: for ``eta_c > eta_be`` the point
    has positive net benefit.  Returns ``inf`` when nothing is saved.
    """
    work = compressor_specific_work() if w_c is None else w_c
    saved = drag_reduction * tau_w
    if saved <= 0.0:
        return math.inf
    return C_q * work / saved


def positive_net_benefit_boundary(
    dr_model: Callable[[float], float],
    C_q_values: Sequence[float],
    U_values: Sequence[float],
    eta_c: float = 0.6,
    assumptions: EconomicAssumptions | None = None,
    w_c: float | None = None,
) -> dict[str, object]:
    """Map the positive-net-benefit (U, C_q) domain.

    Args:
        dr_model:   closure ``DR(C_q)`` from the lattice campaign (>= 0).
        C_q_values: dimensionless gas flow rates to test.
        U_values:   ship speeds [m/s] to test.
        eta_c:      compressor efficiency.
        assumptions: operating assumptions (plate length, draft, water props).
        w_c:        optional pre-computed compressor specific work.

    Returns:
        Dict with the grid, the net-power density grid [W/m^2] per (U, C_q), the
        smallest speed at which each ``C_q`` becomes profitable, and the
        break-even efficiency field (for the eta_c sensitivity).
    """
    a = assumptions or EconomicAssumptions()
    a.validate()
    if not 0.0 < eta_c <= 1.0:
        raise ValueError("eta_c must be in (0, 1]")
    work = compressor_specific_work(a.p_atm, a.rho_water, a.draft) if w_c is None else w_c
    dr = np.array([float(dr_model(cq)) for cq in C_q_values], dtype=float)
    dr = np.clip(dr, 0.0, None)
    U = np.asarray(U_values, dtype=float)
    tau = np.array([wall_shear_from_friction(float(u), a.plate_length, a.nu_water,
                                             a.rho_water) for u in U])
    net = np.zeros((U.size, dr.size), dtype=float)
    eta_be = np.zeros_like(net)
    for i, u in enumerate(U):
        for j, cq in enumerate(C_q_values):
            net[i, j] = dr[j] * tau[i] * u - cq * u * work / eta_c
            eta_be[i, j] = required_efficiency(dr[j], tau[i], cq, work)
    # smallest speed with positive net for each C_q (nan if never)
    min_speed = []
    for j in range(dr.size):
        pos = U[net[:, j] > 0.0]
        min_speed.append(float(pos.min()) if pos.size else float("nan"))
    return {
        "U_values": U.tolist(),
        "C_q_values": [float(c) for c in C_q_values],
        "drag_reduction": dr.tolist(),
        "tau_w": tau.tolist(),
        "net_power_density": net.tolist(),
        "min_profitable_speed": min_speed,
        "break_even_efficiency": eta_be.tolist(),
        "eta_c": eta_c,
        "compressor_specific_work": work,
        "assumptions": {
            "plate_length": a.plate_length,
            "draft": a.draft,
            "rho_water": a.rho_water,
            "nu_water": a.nu_water,
            "p_atm": a.p_atm,
        },
        "positive_area_fraction": float((net > 0.0).mean()),
    }


def efficiency_sensitivity(
    dr_model: Callable[[float], float],
    C_q_values: Sequence[float],
    U_values: Sequence[float],
    eta_values: Sequence[float] = (0.5, 0.6, 0.7),
    assumptions: EconomicAssumptions | None = None,
) -> dict[str, object]:
    """Sweep ``eta_c`` and report the positive-domain size for each value.

    The returned ``monotone`` flag is the *sign-error guard*: the profitable area
    must be non-decreasing in ``eta_c``.  A violation means the compressor power
    entered the ledger with the wrong sign or exponent.
    """
    sizes = []
    boundaries = {}
    for eta in eta_values:
        res = positive_net_benefit_boundary(dr_model, C_q_values, U_values,
                                            eta_c=float(eta), assumptions=assumptions)
        sizes.append(res["positive_area_fraction"])
        boundaries[float(eta)] = res
    ordered = sorted(zip([float(e) for e in eta_values], sizes))
    monotone = all(b >= a - 1e-12 for (_, a), (_, b) in zip(ordered, ordered[1:]))
    return {
        "eta_values": [float(e) for e in eta_values],
        "positive_area_fraction": sizes,
        "monotone_in_eta_c": bool(monotone),
        "boundaries": boundaries,
    }