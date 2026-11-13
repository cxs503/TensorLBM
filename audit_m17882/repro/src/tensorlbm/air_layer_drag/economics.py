"""Compressor / net-power ledger for air-layer drag reduction.

Economics of air lubrication, in SI units:

    P_saved = DR * R_1ph * V            [W]   (resistance times speed)
    P_comp  = Q_g_norm * p_gas / eta_c  [W]   (compression work / efficiency)
    dP_net  = P_saved - P_comp

The compressor is assumed to work isothermally from ambient ``p_atm`` to the
injection depth pressure ``p_atm + rho_w g h_sub`` (slow, submerged injection):

    P_ideal = Q_g * p_atm * ln(p_inj / p_atm)

``eta_c`` (overall compressor + transmission efficiency) divides the ideal work:
``P_comp = P_ideal / eta_c``.  The baseline ``eta_c = 0.6`` with a 0.5/0.7
sensitivity band follows the campaign specification.
"""

from __future__ import annotations

from dataclasses import dataclass

P_ATM = 101_325.0        # Pa
RHO_WATER = 1025.0       # kg/m^3
G = 9.80665              # m/s^2


@dataclass(frozen=True)
class CompressionCase:
    """Operating point for the compressor ledger."""

    speed: float = 10.0            # ship speed V [m/s]
    length: float = 100.0          # wetted length L [m]
    width: float = 20.0            # wetted width B [m]
    cf_1ph: float = 3.0e-3         # single-phase friction coefficient
    submergence: float = 5.0       # injection depth h [m]
    eta_c: float = 0.6             # compressor efficiency
    p_atm: float = P_ATM
    rho_water: float = RHO_WATER

    @property
    def wetted_area(self) -> float:
        return self.length * self.width

    @property
    def resistance_1ph(self) -> float:
        """R_1ph = 0.5 rho V^2 Cf A [N]."""
        q = 0.5 * self.rho_water * self.speed ** 2
        return q * self.cf_1ph * self.wetted_area

    @property
    def delivered_power_1ph(self) -> float:
        """Effective power R*V [W]."""
        return self.resistance_1ph * self.speed

    @property
    def injection_pressure(self) -> float:
        return self.p_atm + self.rho_water * G * self.submergence


def compressor_power(q_gas: float, case: CompressionCase, *, isothermal: bool = True) -> float:
    """Shaft power needed to deliver volumetric gas flow ``q_gas`` [m^3/s]."""
    if q_gas < 0.0:
        raise ValueError("q_gas must be non-negative")
    if case.eta_c <= 0.0 or case.eta_c > 1.0:
        raise ValueError("eta_c must lie in (0, 1]")
    p1 = case.p_atm
    p2 = case.injection_pressure
    if isothermal:
        ideal = q_gas * p1 * float(__import__("math").log(p2 / p1))
    else:  # hydraulic (incompressible) estimate p*dV
        ideal = q_gas * (p2 - p1)
    return ideal / case.eta_c


def net_power_gain(drag_reduction: float, gas_flow_ratio: float,
                   case: CompressionCase) -> dict:
    """Net power balance for one operating point.

    ``gas_flow_ratio`` = Q_g / (V * L * B) (dimensionless volumetric gas flux),
    so that ``q_gas = gas_flow_ratio * V * L * B``.
    """
    if not -1.0 < drag_reduction <= 1.0:
        raise ValueError("drag_reduction must lie in (-1, 1]")
    if gas_flow_ratio < 0.0:
        raise ValueError("gas_flow_ratio must be non-negative")
    q_gas = gas_flow_ratio * case.speed * case.wetted_area
    p_saved = drag_reduction * case.delivered_power_1ph
    p_comp = compressor_power(q_gas, case)
    return {
        "drag_reduction": float(drag_reduction),
        "gas_flow_ratio": float(gas_flow_ratio),
        "q_gas_m3_s": q_gas,
        "p_saved_W": p_saved,
        "p_comp_W": p_comp,
        "dP_net_W": p_saved - p_comp,
        "net_positive": bool(p_saved > p_comp),
        "eta_c": case.eta_c,
        "injection_pressure_Pa": case.injection_pressure,
    }


def positive_gain_boundary(drag_curve, case: CompressionCase, *, n: int = 400,
                           q_max: float = 0.05) -> dict | None:
    """Largest gas flow ratio with dP_net > 0 for a DR(q) curve (monotone DR).

    ``drag_curve`` is a callable q -> DR.  Returns the bracket where the net
    balance crosses zero (None when never positive).
    """
    last_pos = None
    for k in range(1, n + 1):
        q = q_max * k / n
        res = net_power_gain(drag_curve(q), q, case)
        if res["dP_net_W"] > 0.0:
            last_pos = q
        elif last_pos is not None:
            return {"q_max_positive": last_pos, "q_zero_cross": q,
                    "eta_c": case.eta_c}
    if last_pos is not None:
        return {"q_max_positive": last_pos, "q_zero_cross": None, "eta_c": case.eta_c}
    return None