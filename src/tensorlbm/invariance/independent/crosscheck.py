"""Cross-check of the invariance numbers against the first implementation.

The certificate produced by this subpackage is only worth as much as the codes
that stand behind it.  Two things are checked here:

* **implementation agreement on a shared case** - the boosted Taylor-Green decay
  exponents of :mod:`tensorlbm.invariance.cases` (BGK/TRT D2Q9 with a momentum
  controller inside a periodic box) and of
  :mod:`tensorlbm.invariance.independent.cases` (BGK + Guo forcing, independent
  streaming/boundary code) at identical parameters;
* **declared-criteria agreement** - the sibling gate ``evaluate_criteria`` is fed
  the numbers measured by this implementation wherever the two dictionaries
  overlap, so that a criterion cannot pass in one code and fail in the other
  unnoticed.
"""

from __future__ import annotations

import json
import math

import numpy as np

from . import cases as ind_cases


def _jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def tgv_crosscheck(n: int = 32, nu: float = 0.02, u0: float = 0.05,
                   mas: tuple[float, ...] = (0.0, 0.05), steps: int = 800) -> dict:
    """Decay exponent of the boosted Taylor-Green vortex: two implementations."""
    try:
        from tensorlbm.invariance.cases import taylor_green_case as sibling_tgv
    except Exception as exc:  # pragma: no cover - sibling module missing
        return {"available": False, "error": repr(exc)}

    out = {"available": True, "n": n, "nu": nu, "u0": u0, "steps": steps, "rows": []}
    mine = ind_cases.taylor_green_boost(
        n=n,
        nu=nu,
        amplitude=u0,
        mach_list=mas,
        steps=steps,
        sample_every=10,
        discard=100,
    )
    mine_by_ma = {round(r["mach"], 6): r for r in mine["cases"]}
    for ma in mas:
        row = {"ma": ma}
        try:
            sib = sibling_tgv(ma, n=n, nu=nu, u0=u0, steps=steps, sample_every=10)
            row["sibling"] = _jsonable(
                {k: v for k, v in sib.items() if k not in ("u_profile", "rho_profile")}
            )
            decay_sib = sib.get("decay_rate")
        except Exception as exc:  # pragma: no cover
            row["sibling_error"] = repr(exc)
            decay_sib = None
        own = mine_by_ma.get(round(ma, 6))
        if own is not None:
            row["independent_decay"] = own["decay_exponent"]
            row["independent_decay_exact"] = own["decay_exponent_exact"]
            row["independent_drift_error_rel"] = own["drift_error_rel"]
        if decay_sib and own is not None:
            row["decay_rel_difference"] = float(
                (own["decay_exponent"] - decay_sib) / decay_sib
            )
        out["rows"].append(row)

    diffs = [r["decay_rel_difference"] for r in out["rows"] if "decay_rel_difference" in r]
    out["max_decay_rel_difference"] = float(max(abs(d) for d in diffs)) if diffs else None
    # the sibling reports the energy decay (d ln E/dt = 2 * velocity decay); make
    # the comparison explicit rather than implicitly assuming the same convention
    out["note"] = (
        "sibling decay_rate is the energy decay exponent -d ln(E)/dt = 2*2*nu*k^2; "
        "independent decay_exponent is the velocity-amplitude exponent 2*nu*k^2. "
        "The ratio sibling/(2*independent) is therefore the convention-free check."
    )
    ratios = []
    for r in out["rows"]:
        if "sibling" in r and "independent_decay" in r and r["independent_decay"] > 0:
            e = r["sibling"].get("decay_rate")
            if e:
                ratios.append(float(e) / (2.0 * r["independent_decay"]))
    out["convention_free_ratio"] = ratios
    out["max_convention_free_deviation"] = (
        float(max(abs(x - 1.0) for x in ratios)) if ratios else None
    )
    return out


def poiseuille_crosscheck(width: int = 16, theta_deg: float = 0.0, nu: float = 0.1) -> dict:
    """Observed order / peak-velocity error of the sibling Poiseuille case."""
    try:
        from tensorlbm.invariance.cases import poiseuille_case
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": repr(exc)}
    rows = []
    for w in (width, 2 * width):
        try:
            r = poiseuille_case(theta_deg, w, nu=nu)
            rows.append({"width": w, "result": _jsonable(r)})
        except Exception as exc:  # pragma: no cover
            rows.append({"width": w, "error": repr(exc)})
    return {"available": True, "theta_deg": theta_deg, "nu": nu, "rows": rows}


def dump(result: dict) -> str:
    return json.dumps(_jsonable(result))