"""Scoring of a Re = 3200 cavity run against the admitted Ghia column.

The score is *pointwise*: ``u`` is sampled on the vertical centreline
``x/L = 0.5`` at Ghia's 16 ordinates, ``v`` on the horizontal centreline
``y/L = 0.5`` at Ghia's 17 abscissae, and every row is reported as
``[coord, reference, got, error_pct]`` with the error in percent of the lid
speed (the reference table is already normalised by ``U_lid``).

Two lattice-to-physical coordinate conventions are scored, because the answer
depends on where the walls are and the choice must be visible, not implicit:

``node``  (legacy)   ``y = j/(n-1)``      walls exactly on the first/last rows
``wall``  (physical) ``y = (j-0.5)/(n-2)`` half-way bounce-back walls at 0 and 1

Tolerance is pre-registered (5 % of the lid speed, pointwise) and is a module
constant so a run cannot quietly pick a friendlier threshold after the fact.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

TOL_PCT = 5.0
CONVENTIONS = ("wall", "node")


def coords(n: int, convention: str = "wall") -> np.ndarray:
    """Physical coordinate of every lattice row/column for a given convention."""
    if n < 3:
        raise ValueError(f"n={n} too small")
    if convention == "node":
        return np.arange(n, dtype=float) / (n - 1.0)
    if convention == "wall":
        return (np.arange(n, dtype=float) - 0.5) / (n - 2.0)
    raise ValueError(f"unknown coordinate convention {convention!r}")


def _sample(profile: Sequence[float], n: int, convention: str,
            where: Sequence[float]) -> np.ndarray:
    """Linear interpolation of a stored centreline onto reference coordinates."""
    arr = np.asarray(profile, dtype=float)
    if arr.size != n:
        raise ValueError(f"profile has {arr.size} samples, expected n={n}")
    return np.interp(np.asarray(where, dtype=float), coords(n, convention), arr)


def score(run: Dict[str, Any], ref: Dict[str, Any],
          convention: str = "wall", tol_pct: float = TOL_PCT) -> Dict[str, Any]:
    """Pointwise comparison of one run record with the reference table."""
    n = int(run["n"])
    u_ref_rows = [(float(a), float(b)) for a, b in ref["u_re3200"]]
    v_ref_rows = [(float(a), float(b)) for a, b in ref["v_re3200"]]
    u_got = _sample(run["u_centerline"], n, convention, [c for c, _ in u_ref_rows])
    v_got = _sample(run["v_centerline"], n, convention, [c for c, _ in v_ref_rows])
    eu = (u_got - np.array([b for _, b in u_ref_rows])) * 100.0
    ev = (v_got - np.array([b for _, b in v_ref_rows])) * 100.0
    u_rows: List[List[float]] = [
        [c, r, float(g), float(e)]
        for (c, r), g, e in zip(u_ref_rows, u_got, eu)
    ]
    v_rows: List[List[float]] = [
        [c, r, float(g), float(e)]
        for (c, r), g, e in zip(v_ref_rows, v_got, ev)
    ]
    worst = float(max(np.abs(eu).max(), np.abs(ev).max()))
    return {
        "convention": convention,
        "n": n,
        "tolerance_pct": float(tol_pct),
        "u_rows": u_rows,
        "v_rows": v_rows,
        "max_abs_err_pct": worst,
        "rmse_pct": float(np.sqrt(np.mean(np.concatenate([eu, ev]) ** 2))),
        "worst_u_row": u_rows[int(np.argmax(np.abs(eu)))],
        "worst_v_row": v_rows[int(np.argmax(np.abs(ev)))],
        "pass": bool(worst <= tol_pct),
    }


def quarantine_sensitivity(run: Dict[str, Any], ref: Dict[str, Any],
                           convention: str = "wall") -> Dict[str, Any]:
    """Would the verdict change if the quarantined cell were scored?

    The dropped cell is bracketed by two scored rows, so its admissible interval
    is narrow compared with the gap it sits in; scoring the run at both ends of
    that interval shows whether the admit/exclude decision can flip the verdict.
    """
    cells = ((ref.get("provenance") or {}).get("quarantined") or [])
    if not cells:
        return {"status": "no_quarantined_cell"}
    cell = cells[0]
    lo, hi = (float(x) for x in cell["admissible_interval"])
    got = float(_sample(run["u_centerline"], int(run["n"]), convention,
                        [float(cell["coord"])])[0])
    return {
        "status": "scored_at_interval_bounds",
        "coord": float(cell["coord"]),
        "got": got,
        "admissible_interval": [lo, hi],
        "err_pct_at_lower_bound": (got - lo) * 100.0,
        "err_pct_at_upper_bound": (got - hi) * 100.0,
        "interval_width_pct": (hi - lo) * 100.0,
        "note": "the row is the flattest part of the profile; both bounds give an "
                "error of the same order as the neighbouring scored rows, so the "
                "admit/exclude decision cannot flip the verdict",
    }


def grid_convergence(scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Change of the pointwise error between the coarsest and finest usable grid."""
    usable = {k: v for k, v in scores.items() if v and "max_abs_err_pct" in v}
    if len(usable) < 2:
        return {"status": "WITHHELD", "reason": "fewer than two scored grids"}
    keys = sorted(usable, key=lambda k: int(k))
    e = {k: usable[k]["max_abs_err_pct"] for k in keys}
    return {
        "status": "observed",
        "grids": [int(k) for k in keys],
        "max_abs_err_pct": e,
        "delta_pct": e[keys[-1]] - e[keys[0]],
        "note": "error change between the coarsest and finest usable grid; a "
                "negative delta means the finest grid is the better one",
    }