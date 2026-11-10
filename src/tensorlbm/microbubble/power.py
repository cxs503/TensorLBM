"""Net-power ledger for microbubble drag reduction (IN62-N04).

The engineering question is whether the saved drag power pays for the air.

* saved power per unit wetted area, ``P_saved = DR * tau_w0 * U``
  (``tau_w0 = 0.5 rho U^2 Cf`` with ITTC-1957 for the clean hull);
* compressor power per unit wetted area,
  ``P_comp = q_g * w / eta_c`` where ``q_g`` is the **gas volume flux per unit
  wetted area** in m/s and ``w`` the specific compression work in J/m^3 of free
  air;
* ``P_net = P_saved - P_comp``.

``q_g`` is not a free knob: it is what the void-fraction transport solve says is
needed to hold the layer (wall capture plus entrainment out of the layer, in
steady state).  The transport solver returns the supply in units of ``u_tau``
(``supply_ratio = J_in / u_tau``, dimensionless), so

    q_g = supply_ratio * u_tau ,      u_tau = U sqrt(Cf/2) ,

and the dimensionless gas flow rate is ``C_q = q_g / U``.

Compression work is bounded below by the reversible isothermal value
``p_atm ln(p_inj/p_atm)`` and above by the ``n``-stage adiabatic value with
perfect intercooling; both are returned so the ledger is bracketed rather than
optimistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

RHO_WATER = 1025.0
NU_WATER = 1.19e-6
G_ACCEL = 9.81
P_ATM = 101325.0
K_AIR = 1.4


@dataclass(frozen=True)
class CompressorSpec:
    """Compressor / ship operating point."""

    draft: float = 5.0            # injection depth below the waterline [m]
    eta_c: float = 0.6            # overall compressor efficiency
    p_atm: float = P_ATM
    rho_water: float = RHO_WATER
    g: float = G_ACCEL
    nu_water: float = NU_WATER
    k: float = K_AIR
    n_stages: int = 3

    @property
    def p_injection(self) -> float:
        return self.p_atm + self.rho_water * self.g * self.draft

    @property
    def pressure_ratio(self) -> float:
        return self.p_injection / self.p_atm

    def with_eta(self, eta_c: float) -> "CompressorSpec":
        return replace(self, eta_c=float(eta_c))


def isothermal_work(op: CompressorSpec | None = None) -> float:
    """Reversible isothermal compression work per m^3 of free air [J/m^3]."""
    op = op or CompressorSpec()
    return op.p_atm * math.log(op.pressure_ratio)


def multistage_work(op: CompressorSpec | None = None, n_stages: int | None = None,
                    k: float | None = None) -> float:
    """``n``-stage adiabatic compression with perfect intercooling [J/m^3]."""
    op = op or CompressorSpec()
    n = n_stages or op.n_stages
    kk = op.k if k is None else k
    if n < 1:
        raise ValueError("n_stages must be >= 1")
    r = op.pressure_ratio ** (1.0 / n)
    return float(n * op.p_atm * (kk / (kk - 1.0)) * (r ** ((kk - 1.0) / kk) - 1.0))


def friction_coefficient_ittc(re_length: float) -> float:
    """ITTC-1957 skin-friction line."""
    return 0.075 / (math.log10(re_length) - 2.0) ** 2


def wall_shear(U: float, length: float, op: CompressorSpec | None = None) -> float:
    """Clean-hull mean wall shear ``tau_w0 = 0.5 rho U^2 Cf`` [Pa]."""
    op = op or CompressorSpec()
    cf = friction_coefficient_ittc(U * length / op.nu_water)
    return 0.5 * op.rho_water * U * U * cf


def gas_flow_rate(U: float, tau_w: float, supply_ratio: float,
                  op: CompressorSpec | None = None) -> dict:
    """Gas volume flux per unit wetted area implied by the transport solve."""
    op = op or CompressorSpec()
    u_tau = math.sqrt(tau_w / op.rho_water)
    q_g = supply_ratio * u_tau
    return {"u_tau": u_tau, "q_gas": q_g, "C_q": q_g / U}


def net_power(dr: float, U: float, tau_w: float, supply_ratio: float,
              op: CompressorSpec | None = None,
              recirculation: float = 0.0) -> dict:
    """Saved / spent / net power per unit wetted area [W/m^2].

    ``recirculation`` (0..1) is the fraction of the gas that is recovered
    (closed air cavity / recirculation duct), which reduces the make-up flow
    to ``(1 - recirculation) q_g``.  It is a design parameter, not a fit.
    """
    op = op or CompressorSpec()
    gfr = gas_flow_rate(U, tau_w, supply_ratio, op)
    gfr["q_makeup"] = gfr["q_gas"] * (1.0 - float(recirculation))
    gfr["recirculation"] = float(recirculation)
    saved = dr * tau_w * U
    spent_iso = gfr["q_makeup"] * isothermal_work(op) / op.eta_c
    spent_multi = gfr["q_makeup"] * multistage_work(op) / op.eta_c
    return {
        "saved_power_density": saved,
        "compressor_power_density": spent_multi,
        "compressor_power_density_isothermal": spent_iso,
        "net_power_density": saved - spent_multi,
        "net_power_density_isothermal": saved - spent_iso,
        "power_ratio": saved / spent_multi if spent_multi > 0 else float("inf"),
        **gfr,
    }


def ledger_scan(U: float, length: float, alpha_grid, dr_of, supply_of,
                eta_grid=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                op: CompressorSpec | None = None,
                recirculation: float = 0.0) -> dict:
    """Scan ``(alpha_w, eta_c)`` and report the profitability map.

    ``dr_of(alpha_w)`` and ``supply_of(alpha_w)`` are the closures measured by
    the campaign; nothing is calibrated here.
    """
    op = op or CompressorSpec()
    tau_w = wall_shear(U, length, op)
    alphas = np.asarray(list(alpha_grid), dtype=float)
    etas = np.asarray(list(eta_grid), dtype=float)
    net = np.zeros((alphas.size, etas.size))
    rows = []
    for i, a in enumerate(alphas):
        dr = float(dr_of(float(a)))
        sr = float(supply_of(float(a)))
        for j, e in enumerate(etas):
            res = net_power(dr, U, tau_w, sr, op.with_eta(float(e)),
                            recirculation=recirculation)
            net[i, j] = res["net_power_density"]
        rows.append({
            "alpha_w": float(a),
            "DR": dr,
            "supply_ratio": sr,
            "C_q": gas_flow_rate(U, tau_w, sr, op)["C_q"],
            "net_power_density_eta_0.6": float(
                net[i, list(etas).index(0.6)] if 0.6 in list(etas) else np.nan),
        })
    i_max = int(np.argmax(net.max(axis=1)))
    return {
        "U": U, "length": length,
        "tau_w0": tau_w,
        "alpha_w": alphas.tolist(),
        "eta_c": etas.tolist(),
        "net_power_density": net.tolist(),
        "rows": rows,
        "positive_fraction": float((net > 0).mean()),
        "best_alpha_w": float(alphas[i_max]),
        "best_net_power_density": float(net[i_max].max()),
        "isothermal_work": isothermal_work(op),
        "multistage_work": multistage_work(op),
    }

def profitability_threshold(U: float, tau_w: float, dr: float,
                            supply_ratio: float,
                            op: CompressorSpec | None = None) -> dict:
    """Design limits implied by the ledger.

    * ``C_q_max`` -- the largest gas flow rate (per unit wetted area) that the
      saved power can pay for, i.e. the break-even of the compressor budget;
    * ``recirculation_min`` -- the gas recovery fraction needed to make the
      measured point profitable;
    * ``entrainment_max`` -- the break-even of the *modelled* loss
      (``supply_ratio`` is proportional to the entrainment velocity), i.e. how
      leaky the bubble layer may be.
    """
    op = op or CompressorSpec()
    w = multistage_work(op)
    saved = dr * tau_w * U
    q_break = saved * op.eta_c / w if w > 0 else float("inf")
    gfr = gas_flow_rate(U, tau_w, supply_ratio, op)
    q_g = gfr["q_gas"]
    recirc = 1.0 - q_break / q_g if q_g > 0 else 0.0
    ratio = q_break / q_g if q_g > 0 else float("inf")
    return {
        "saved_power_density": saved,
        "q_gas": q_g,
        "C_q": gfr["C_q"],
        "q_break_even": q_break,
        "C_q_max": q_break / U,
        "flow_ratio": ratio,
        "recirculation_min": max(0.0, recirc),
        "entrainment_max_ratio": ratio if ratio < 1.0 else 1.0,
        "work_multistage": w,
    }
