"""Reference ice-resistance models used for the acceptance comparison (C5).

Two references are provided and reported side by side; the *literal* acceptance
comparison of node P66-W2-N04 is against the campaign-baseline Riska (1989)
implementation that lives in :mod:`tensorlbm.ship_hull_geometry` (baseline =
``main``), which is imported rather than copied so the comparison cannot drift.

The second reference (:func:`submersion_lindqvist`) is the dimensionally clean
submersion term of Lindqvist (1989), ``mu (rho_w - rho_i) g h B T``: it is given
because the root-cause analysis in the report shows that the campaign baseline's
submergence term is proportional to ``rho_w g h B L Cb psi`` - i.e. it scales with
the ship's displacement rather than with the submerged buoyancy of the ice - and
therefore cannot be used to judge a first-principles prediction of a
*broken-ice* (very close pack) field.

References
----------
- Riska, K. (1989). "A Model for Icebreaking Resistance." POAC 89.
- Lindqvist, G. (1989). "A Simplified Method for Calculation of Ship Resistance
  in Ice." POAC 89, Vol. 3, 845-859.
"""

from __future__ import annotations

import math
from typing import Dict

from ..ship_hull_geometry import riska_ice_resistance, riska_total_resistance


def riska_campaign_baseline(h_ice: float, v_ship: float, L_pp: float, B: float,
                            T: float, C_b: float, phi_deg: float, psi: float) -> Dict[str, float]:
    """Campaign-baseline Riska (1989) reference (imported from ``main``)."""
    R_ice = riska_ice_resistance(h_ice, v_ship, L_pp, B, phi_deg, C_b, psi)
    tot = riska_total_resistance(h_ice, v_ship, L_pp, B, T, C_b, phi_deg, psi)
    return {
        "R_ice": float(R_ice),
        "R_total": float(tot["R_total"]),
        "R_open_water": float(tot["R_open_water"]),
    }


def submersion_lindqvist(h_ice: float, B: float, T: float, mu: float = 0.15,
                         rho_ice: float = 917.0, rho_water: float = 1025.0,
                         psi: float = 1.0) -> float:
    """Lindqvist (1989) submersion term, ``mu (rho_w - rho_i) g h B T``."""
    return float(mu * (rho_water - rho_ice) * 9.81 * h_ice * B * T * psi)


def clearing_momentum(h_ice: float, B: float, U: float, concentration: float,
                      rho_ice: float = 917.0, k_deflection: float = 1.0) -> float:
    """Momentum cost of deflecting the swept ice corridor (first principles).

    In the ship frame the ice that is in the swept corridor ``h x B`` must be
    deflected out of it; the momentum flux that has to be re-directed per unit
    time is ``rho_i C h B U^2`` (per unit deflection efficiency ``k``).  This is
    the irreducible "clearing" part of the resistance of a stress-free broken-ice
    field: it does not depend on any empirical coefficient.
    """
    return float(k_deflection * rho_ice * concentration * h_ice * B * U ** 2)


def compare(R_chain: float, ref: Dict[str, float], key: str = "R_ice",
            tol: float = 0.15) -> Dict[str, float]:
    """Relative comparison of a chain prediction with a reference value."""
    target = ref[key]
    err = abs(R_chain - target) / abs(target) if target else float("inf")
    return {
        "R_chain": float(R_chain),
        "R_reference": float(target),
        "reference": key,
        "relative_error": float(err),
        "tolerance": float(tol),
        "pass": bool(err < tol),
    }