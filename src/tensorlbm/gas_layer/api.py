"""Design-point API for the gas-layer drag-reduction closure.

This is the *consumer* end of the campaign: given a ship operating point (speed,
dimensionless gas flow, compressor efficiency) it returns the engineering
numbers an air-lubrication design needs, using the drag-reduction closure
measured by :mod:`tensorlbm.gas_layer.campaign`.

The closure is loaded from the campaign artifact when it is present
(``artifacts/gas_layer_in39_n06/campaign.json``), otherwise the shipped fallback
closure is used and the response says so.  Nothing is silently extrapolated:
``C_q`` outside the measured range is flagged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .economics import (
    EconomicAssumptions,
    compressor_specific_work,
    net_power_density,
    wall_shear_from_friction,
)

#: Measured closure fallback (C_q, DR) from the IN39-N06 campaign.  Used only
#: when the artifact is unavailable; every response reports which one was used.
FALLBACK_CLOSURE: tuple[list[float], list[float]] = (
    [0.0, 0.013, 0.033, 0.078, 0.195, 0.521],
    [0.0, 0.30, 0.30, 0.30, 0.30, 0.30],
)

DEFAULT_ARTIFACT = Path("artifacts/gas_layer_in39_n06/campaign.json")


def load_closure(
    path: str | Path | None = None, extrapolation: str = "plateau"
) -> dict[str, Any]:
    """Load the measured ``DR(C_q)`` closure from the campaign artifact.

    Below the smallest measured ``C_q`` the closure is not measured, so the
    caller must choose an extrapolation:

    * ``"plateau"`` (default) -- the gas-retaining wall keeps the layer, so DR
      holds the lowest measured value; this is the physically supported branch
      for a non-wetting plate,
    * ``"linear"`` -- DR falls linearly to the no-gas limit ``DR(0) = 0``
      (pessimistic lower bound).
    """
    if extrapolation not in {"plateau", "linear"}:
        raise ValueError("extrapolation must be 'plateau' or 'linear'")
    target = Path(path) if path is not None else DEFAULT_ARTIFACT
    if target.exists():
        data = json.loads(target.read_text())
        closure = data.get("economics", {}).get("dr_closure", {})
        cq = closure.get("C_q")
        dr = closure.get("DR")
        if cq and dr and len(cq) == len(dr):
            pairs = [(float(c), float(d)) for c, d in zip(cq, dr, strict=True)]
            cq_out, dr_out = _apply_extrapolation(pairs, extrapolation)
            return {
                "C_q": cq_out,
                "DR": dr_out,
                "source": str(target),
                "measured": True,
                "extrapolation": extrapolation,
                "measured_C_q_range": [
                    min(c for c, _ in pairs if c > 0.0),
                    max(c for c, _ in pairs),
                ],
            }
    base = list(zip(FALLBACK_CLOSURE[0], FALLBACK_CLOSURE[1], strict=True))
    cq_out, dr_out = _apply_extrapolation(base, extrapolation)
    return {
        "C_q": cq_out,
        "DR": dr_out,
        "source": "fallback",
        "measured": False,
        "extrapolation": extrapolation,
        "measured_C_q_range": [min(c for c, _ in base), max(c for c, _ in base)],
    }


def _apply_extrapolation(
    pairs: list[tuple[float, float]], mode: str
) -> tuple[list[float], list[float]]:
    """Rewrite the closure table for the requested sub-range extrapolation."""
    positive = sorted(p for p in pairs if p[0] > 0.0)
    if not positive:
        return [0.0], [0.0]
    c_min, dr_min = positive[0]
    if mode == "plateau":
        table = [(c_min * 1e-3, dr_min)] + positive
    else:
        table = [(0.0, 0.0)] + positive
    return [c for c, _ in table], [d for _, d in table]


def _interp(closure: dict[str, Any], C_q: float) -> float:
    cq = np.asarray(closure["C_q"], dtype=float)
    dr = np.asarray(closure["DR"], dtype=float)
    return float(np.interp(float(C_q), cq, dr))


def drag_reduction(closure: dict[str, Any], C_q: float) -> float:
    """Drag reduction at a given dimensionless gas flow (linear interpolation)."""
    if C_q < 0.0:
        raise ValueError("C_q must be non-negative")
    return _interp(closure, C_q)


def design_point(
    U: float,
    C_q: float,
    eta_c: float = 0.6,
    closure: dict[str, Any] | None = None,
    assumptions: EconomicAssumptions | None = None,
) -> dict[str, Any]:
    """Evaluate one (speed, gas flow, compressor efficiency) design point.

    Returns drag reduction, saved power density, compressor power density, net
    power density [W/m^2] and the flags a designer needs: whether the point is
    inside the measured ``C_q`` range and whether the net benefit is positive.
    """
    a = assumptions or EconomicAssumptions()
    a.validate()
    if U <= 0.0:
        raise ValueError("U must be positive")
    cl = closure or load_closure()
    dr = drag_reduction(cl, C_q)
    tau = wall_shear_from_friction(U, a.plate_length, a.nu_water, a.rho_water)
    work = compressor_specific_work(a.p_atm, a.rho_water, a.draft)
    net = net_power_density(dr, U, tau, C_q, eta_c=eta_c, w_c=work)
    saved = dr * tau * U
    spent = C_q * U * work / eta_c
    return {
        "U": float(U),
        "C_q": float(C_q),
        "eta_c": float(eta_c),
        "drag_reduction": dr,
        "tau_w_1ph": tau,
        "power_saved_density": saved,
        "power_compressor_density": spent,
        "net_power_density": net,
        "profitable": bool(net > 0.0),
        "inside_measured_C_q_range": bool(
            cl["measured_C_q_range"][0] <= C_q <= cl["measured_C_q_range"][1]
        ),
        "closure_extrapolation": cl.get("extrapolation", "linear"),
        "closure_source": cl["source"],
        "closure_measured": cl["measured"],
        "compressor_specific_work": work,
    }


def profit_boundary(
    U_values: Iterable[float],
    C_q_values: Iterable[float],
    eta_c: float = 0.6,
    closure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Smallest speed at which each ``C_q`` becomes profitable."""
    cl = closure or load_closure()
    U = [float(u) for u in U_values]
    Cq = [float(c) for c in C_q_values]
    out = []
    for c in Cq:
        row = []
        best = None
        for u in U:
            dp = design_point(u, c, eta_c=eta_c, closure=cl)
            row.append(dp["net_power_density"])
            if dp["profitable"] and best is None:
                best = u
        out.append({"C_q": c, "min_profitable_speed": best, "net_density": row})
    return {"U_values": U, "eta_c": float(eta_c), "rows": out,
            "closure_source": cl["source"]}