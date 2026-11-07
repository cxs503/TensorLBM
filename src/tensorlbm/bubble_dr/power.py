"""Compressor ledger and net-power closure for ship-bottom gas lubrication.

Campaign node IN57-N02, criterion (4).

The question a ship operator asks is not "how much drag does the gas layer save"
but "does the saved power pay for the compressor".  Three models are kept apart
on purpose, so that the ledger can be *audited* rather than trusted:

``isothermal_work_closed_form``
    ``w = p_a ln(p_inj/p_a)`` per unit gas volume at ambient -- the reversible
    isothermal (multi-stage, perfect-intercooling) limit;
``isothermal_work_quadrature``
    the same number obtained by numerically integrating ``int V dp`` along the
    isothermal path ``pV = p_a V_a``; agreement with the closed form is the
    numerical check that the ledger's formula is implemented as stated;
``multistage_work``
    ``n`` stages with perfect intercooling, each ``p_a V_a ((r)^{(k-1)/k}-1)``
    with ``r = (p_inj/p_a)^{1/n}``.  A real machine is a *finite*-stage (or
    uncooled) device, so the gap between the n-stage number and the isothermal
    limit is the honest model error of assuming isothermal compression.

Criterion (4) grades ``|w_multistage(3) / w_isothermal - 1| <= 5 %`` at the
campaign operating point, and the net-power closure is reported as

    dP = DR * tau_w * U        (saved drag power per unit wetted area)
    P_c = C_q * U * w / eta_c  (compressor power per unit wetted area)
    eta_net = dP / (dP + P_c)  (>= 0; positive means the layer pays for itself)

Note that ``eta_net`` as defined in the campaign goal is *not* a thermodynamic
efficiency: it is the fraction of the gross (saved + spent) power that survives,
so it is positive whenever ``dP > 0``.  The engineering discriminator is
``dP_net = dP - P_c`` and the profitable domain is reported separately --
reporting only ``eta_net > 0`` would flatter the system, so both are given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RHO_WATER = 1025.0
RHO_AIR = 1.225
G_ACCEL = 9.81
P_ATM = 101325.0
NU_WATER = 1.19e-6
K_AIR = 1.4


@dataclass(frozen=True)
class OperatingPoint:
    """SI operating point of the air-lubrication system."""

    draft: float = 5.0          # injection depth below the waterline [m]
    plate_length: float = 100.0  # wetted length over which the gas is injected [m]
    eta_c: float = 0.6           # overall compressor efficiency
    p_atm: float = P_ATM
    rho_water: float = RHO_WATER
    g: float = G_ACCEL
    nu_water: float = NU_WATER

    @property
    def p_injection(self) -> float:
        return self.p_atm + self.rho_water * self.g * self.draft

    @property
    def pressure_ratio(self) -> float:
        return self.p_injection / self.p_atm


def isothermal_work_closed_form(op: OperatingPoint) -> float:
    """Reversible isothermal work per unit gas volume at ambient pressure [J/m^3]."""
    return op.p_atm * math.log(op.pressure_ratio)


def isothermal_work_quadrature(op: OperatingPoint, n: int = 20001) -> float:
    """``int_{p_a}^{p_inj} V dp`` with ``V = p_a V_a / p``, by Simpson's rule."""
    if n % 2 == 0:
        n += 1
    p = np.linspace(op.p_atm, op.p_injection, n)
    v = op.p_atm / p  # V per unit V_a
    return float(np.trapezoid(v, p))


def multistage_work(op: OperatingPoint, n_stages: int = 3, k: float = K_AIR) -> float:
    """``n``-stage compression with perfect intercooling, per unit gas volume.

    Per stage the shaft work is ``int V dp = k/(k-1) p_1 V_1 [(r)^{(k-1)/k} - 1]``
    (the ``k/(k-1)`` factor is the enthalpy/compression-work relation for an
    adiabatic stage -- dropping it makes a single stage *cheaper* than the
    isothermal limit, which is the sign error this function is guarded against by
    ``test_single_stage_exceeds_isothermal_limit``).  With ``n`` stages the limit
    ``n -> inf`` reproduces ``p_a ln(p_inj/p_a)`` exactly, because
    ``k/(k-1) * (k-1)/k = 1``.
    """
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    r = op.pressure_ratio ** (1.0 / n_stages)
    return float(
        n_stages * op.p_atm * (k / (k - 1.0)) * (r ** ((k - 1.0) / k) - 1.0)
    )


def friction_coefficient(re_length: float) -> float:
    """ITTC-1957 skin-friction line."""
    return 0.075 / (math.log10(re_length) - 2.0) ** 2


def wall_shear(U: float, op: OperatingPoint) -> float:
    """Mean wall shear ``tau_w = 0.5 rho U^2 Cf`` (ITTC-1957)."""
    return 0.5 * op.rho_water * U * U * friction_coefficient(U * op.plate_length / op.nu_water)


def saved_power_density(DR: float, U: float, tau_w: float) -> float:
    return DR * tau_w * U


def compressor_power_density(C_q: float, U: float, op: OperatingPoint, n_stages: int = 3) -> float:
    """Compressor power per unit wetted area [W/m^2] for a finite-stage machine."""
    return C_q * U * multistage_work(op, n_stages) / op.eta_c


def net_closure(
    DR: float,
    U: float,
    tau_w: float,
    C_q: float,
    op: OperatingPoint | None = None,
    n_stages: int = 3,
) -> dict[str, float]:
    """Saved / spent / net power densities and the two efficiency measures."""
    op = op or OperatingPoint()
    saved = saved_power_density(DR, U, tau_w)
    spent = compressor_power_density(C_q, U, op, n_stages)
    total = saved + spent
    return {
        "saved_power_density": saved,
        "compressor_power_density": spent,
        "net_power_density": saved - spent,
        "eta_net": saved / total if total > 0 else 0.0,
        "power_ratio": saved / spent if spent > 0 else float("inf"),
    }


def positive_domain(
    U_grid,
    Cq_grid,
    dr_of,
    op: OperatingPoint | None = None,
    n_stages: int = 3,
) -> dict[str, object]:
    """Map the region of positive net power over (U, C_q).

    ``dr_of(C_q)`` is the measured drag-reduction closure; it is evaluated at the
    measured abscissae only (no extrapolation is invented here).
    """
    op = op or OperatingPoint()
    U = np.asarray(list(U_grid), dtype=float)
    Cq = np.asarray(list(Cq_grid), dtype=float)
    net = np.zeros((U.size, Cq.size))
    eta = np.zeros_like(net)
    for i, u in enumerate(U):
        tw = wall_shear(u, op)
        for j, c in enumerate(Cq):
            res = net_closure(float(dr_of(c)), u, tw, c, op, n_stages)
            net[i, j] = res["net_power_density"]
            eta[i, j] = res["eta_net"]
    return {
        "U": U.tolist(),
        "C_q": Cq.tolist(),
        "net_power_density": net.tolist(),
        "eta_net": eta.tolist(),
        "positive_fraction": float((net > 0).mean()),
        "eta_net_positive_fraction": float((eta > 0).mean()),
        "positive_cells": [
            {"U": float(U[i]), "C_q": float(Cq[j]), "net_power_density": float(net[i, j])}
            for i in range(U.size)
            for j in range(Cq.size)
            if net[i, j] > 0
        ],
    }