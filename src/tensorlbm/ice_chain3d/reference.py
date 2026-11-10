"""Reference models and coefficient-free estimates for the ice-resistance chain.

Provenance discipline
---------------------
Every number this module produces carries an explicit provenance tag, because
the validation of an ice-resistance chain is exactly where fabricated
"experimental data" would be fatal:

``FIRST_PRINCIPLES``
    Dimensional analysis without fitted coefficients:
    * clearing momentum ``rho_i C_i h B U^2`` -- the momentum flux that has to be
      given to the ice in the ship's path (mass flow ``rho_i C_i h B U`` times the
      ship speed);
    * submersion work estimate ``mu (rho_w - rho_i) g h B T C_i`` -- the friction
      of the buoyancy deficit of the submerged ice band.
``IN_REPO_BASELINE``
    ``tensorlbm.ship_hull_geometry.riska_ice_resistance``, the platform's
    "Riska (1989)" implementation used by the ship-performance platform.  Its
    level-ice crushing term is *not* a brash-ice law and the whole expression is
    finally multiplied by the ice concentration; the previous campaign node
    P66-W2-N04 measured a 258x discrepancy.  It is re-checked here and reported
    as an out-of-domain reference, not as an experimental anchor.
``WITHHELD``
    A Riska (1989) *model-test* curve for KVLCC2 in brash ice is not available in
    this repository and cannot be reconstructed from a citable source offline.
    Criterion C5 against experimental data is therefore reported as WITHHELD /
    honest FAIL; no number is invented.

References
----------
- Riska, K. (1989). Ice resistance model tests (cited by the platform's
  ``ship_hull_geometry`` and by the campaign node specification).
- Lindqvist, G. (1989). A straightforward method for calculation of ice
  resistance of ships. POAC'89.
- ITTC 1957 model-ship correlation line (friction reference for the water phase).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

RHO_ICE = 917.0
RHO_W = 1025.0
G = 9.81
SIGMA_C = 1.5e6          # ice compressive strength [Pa] (Baltic brash ice)


# ---------------------------------------------------------------------------
# coefficient-free estimates
# ---------------------------------------------------------------------------
def clearing_momentum(B: float, h: float, U: float, C_i: float,
                      rho_ice: float = RHO_ICE) -> float:
    """Clearing momentum flux ``rho_i C_i h B U^2`` [N] (FIRST_PRINCIPLES)."""
    return rho_ice * C_i * h * B * U ** 2


def submersion_estimate(B: float, T: float, h: float, C_i: float,
                        mu: float = 0.15) -> float:
    """Submersion/friction estimate ``mu (rho_w - rho_i) g h B T C_i`` [N].

    FIRST_PRINCIPLES in the Lindqvist sense: the buoyancy deficit of the ice band
    that is dragged along the hull, multiplied by the ice-hull friction.
    """
    return mu * (RHO_W - RHO_ICE) * G * h * B * T * C_i


def crushing_per_area(contact_width: float = 0.2) -> float:
    """Ice crushing force scale ``sigma_c h d`` for a contact patch [N]."""
    return SIGMA_C * 0.15 * contact_width


# ---------------------------------------------------------------------------
# in-repo baseline (out of domain for brash ice)
# ---------------------------------------------------------------------------
def in_repo_riska(L_pp: float = 230.0, B: float = 32.29, T: float = 14.5,
                  h: float = 0.15, U: float = 1.5, phi_deg: float = 20.0,
                  C_b: float = 0.8098, psi: float = 0.6) -> Dict[str, float]:
    """Evaluate the platform's ``riska_ice_resistance`` and expose its parts."""
    from ..ship_hull_geometry import riska_ice_resistance

    total = riska_ice_resistance(h, U, L_pp, B, phi_deg, C_b, psi)
    phi = math.radians(phi_deg)
    R_c = 0.5 * SIGMA_C * h ** 2 * (B / max(math.tan(phi), 0.1))
    R_f = 0.15 * R_c * (1.0 + U / 5.0)
    R_s = 0.15 * RHO_W * G * h * B * L_pp * C_b * psi
    return {
        "provenance": "IN_REPO_BASELINE",
        "R_total_N": float(total),
        "R_c_N": float(R_c),
        "R_f_N": float(R_f),
        "R_s_N": float(R_s),
        "psi": psi,
        "note": ("level-ice crushing term scaled by the ice concentration; "
                 "not a brash-ice law"),
    }


# ---------------------------------------------------------------------------
# water-phase friction reference
# ---------------------------------------------------------------------------
def ittc57_cf(Re: float) -> float:
    """ITTC-1957 model-ship correlation line."""
    return 0.075 / (math.log10(max(Re, 1e3)) - 2.0) ** 2


def ittc_friction_reference(U: float = 1.5, S: float = 11500.0,
                            nu: float = 1.0e-6, L: float = 230.0,
                            form_factor: float = 1.16) -> Dict[str, float]:
    """Reference friction resistance of KVLCC2 from ITTC-57 + form factor.

    ``R_f = 0.5 rho U^2 S (1+k) C_f(Re)`` with the wetted surface ``S`` from the
    hull quadrature (``hull.wetted_area()``-like), ``Re = U L / nu`` and a
    published KVLCC2 form factor ``1+k ~ 1.16``.
    """
    Re = U * L / nu
    Cf = ittc57_cf(Re)
    R_f = 0.5 * RHO_W * U ** 2 * S * Cf
    return {"Re": Re, "Cf": Cf, "R_friction_N": R_f,
            "R_friction_with_form_factor_N": R_f * form_factor,
            "form_factor": form_factor, "wetted_area_m2": S,
            "provenance": "ITTC57+form_factor"}


def compare(pred: float, ref: float) -> Dict[str, float]:
    """Relative comparison of a prediction with a reference value."""
    if ref == 0.0:
        return {"pred": pred, "ref": ref, "rel_error": float("inf"), "ratio": float("inf")}
    return {"pred": float(pred), "ref": float(ref),
            "rel_error": abs(pred - ref) / abs(ref),
            "ratio": float(pred / ref)}


# ---------------------------------------------------------------------------
# GCI (Roache) helpers
# ---------------------------------------------------------------------------
def gci(f_fine: float, f_mid: float, f_coarse: float, r: float = 1.5) -> Dict[str, float]:
    """Grid convergence index on a three-grid triplet (Roache 1994).

    ``p`` from the standard three-grid expression, ``GCI_fine`` from the
    Richardson-extrapolated value; when the observed order is negative or
    non-physical the function reports ``p_apparent`` but flags ``valid = False``.
    """
    f1, f2, f3 = float(f_fine), float(f_mid), float(f_coarse)
    eps12 = f2 - f1
    eps23 = f3 - f2
    out: Dict[str, float] = {"f_fine": f1, "f_mid": f2, "f_coarse": f3, "r": r}
    if eps12 == 0.0 or eps23 == 0.0:
        out.update({"p_apparent": None, "valid": False, "GCI_fine": None,
                    "f_extrapolated": None})
        return out
    q = eps23 / eps12
    if q > 0 and abs(math.log(abs(q))) > 1e-12:
        p = abs(math.log(abs(q))) / math.log(r)
        sign = 1.0 if eps12 < 0 else -1.0
        # Richardson extrapolation with the observed order
        denom = r ** p - 1.0
        f_ex = f1 + (f1 - f2) / (r ** p - 1.0) if abs(denom) > 1e-12 else float("nan")
        gci_fine = abs((f2 - f1) / max(f1, 1e-30)) / max(denom, 1e-30)
        out.update({"p_apparent": p, "valid": bool(p > 0.0 and np.isfinite(f_ex)),
                    "GCI_fine": gci_fine, "f_extrapolated": f_ex,
                    "eps_mid_fine": eps12, "eps_coarse_mid": eps23})
    else:
        out.update({"p_apparent": None, "valid": False, "GCI_fine": None,
                    "f_extrapolated": None})
    return out