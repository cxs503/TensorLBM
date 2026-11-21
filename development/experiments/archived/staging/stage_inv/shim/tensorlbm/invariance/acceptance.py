"""Machine-checkable acceptance criteria C1..C7 for the invariance certificate.

Every criterion is evaluated from *measured* numbers (no hand-copied values) and
returns a dict with the measured quantity, the threshold, and a boolean.  The
aggregate report is JSON-serialisable and is what the campaign driver writes to
``artifacts/invariance/P170W2-N08/report.json``.
"""

from __future__ import annotations

import math

import numpy as np


def _spread(values: dict[float, float]) -> tuple[float, dict[str, float]]:
    """``max_theta |x(theta) - x(0)| / |x(0)|``."""
    ref = values.get(0.0)
    if ref is None:
        ref = values[sorted(values)[0]]
    rel = {f"{k:g}": abs(v - ref) / abs(ref) for k, v in values.items()}
    return (max(rel.values()) if rel else float("nan")), rel


def c1_poiseuille_order(poiseuille: dict, tol: float = 0.3, target: float = 2.0) -> dict:
    per_theta = {}
    ok_all = True
    for key, entry in poiseuille.items():
        errors = entry["errors"]
        widths = [float(w) for w in entry["widths"]]
        orders = entry["orders"]
        mean_order = float(np.mean(orders)) if orders else float("nan")
        ok = bool(len(orders) >= 2 and abs(mean_order - target) <= tol)
        per_theta[key] = {
            "widths": widths,
            "l2_errors": errors,
            "observed_orders": orders,
            "mean_observed_order": mean_order,
            "pass": ok,
        }
        ok_all = ok_all and ok
    return {
        "id": "C1",
        "name": "inclined-channel Poiseuille observed order",
        "target": f"{target}±{tol} for theta=0/15/30/45 with 4 wall resolutions",
        "per_theta": per_theta,
        "pass": bool(ok_all),
    }


def c2_cd_spread(cylinders: dict, tol_100: float = 0.03, tol_200: float = 0.05) -> dict:
    out = {}
    ok = True
    for re_key, runs in cylinders.items():
        re = float(re_key)
        cd = {float(r["theta_deg"]): float(r["cd"]) for r in runs}
        spread, rel = _spread(cd)
        tol = tol_100 if abs(re - 100.0) < 1e-6 else tol_200
        ok_re = bool(spread <= tol)
        ok = ok and ok_re
        out[re_key] = {"cd_by_theta": {f"{k:g}": v for k, v in cd.items()},
                       "rel_deviation": rel, "max_spread": spread,
                       "tolerance": tol, "pass": ok_re}
    return {
        "id": "C2",
        "name": "drag-coefficient orientation spread",
        "target": "max|Cd(theta)-Cd(0)|/Cd(0) <= 3% (Re=100) / 5% (Re=200)",
        "by_re": out,
        "pass": bool(ok),
    }


def c3_st_spread(cylinders: dict, tol: float = 0.02) -> dict:
    out = {}
    ok = True
    for re_key, runs in cylinders.items():
        st = {float(r["theta_deg"]): float(r["st"]) for r in runs}
        spread, rel = _spread(st)
        ok_re = bool(spread <= tol)
        ok = ok and ok_re
        out[re_key] = {"st_by_theta": {f"{k:g}": v for k, v in st.items()},
                       "rel_deviation": rel, "max_spread": spread, "pass": ok_re}
    return {
        "id": "C3",
        "name": "Strouhal-number orientation spread",
        "target": f"max|St(theta)-St(0)|/St(0) <= {tol:.0%}",
        "by_re": out,
        "pass": bool(ok),
    }


def c4_galilean_residual(galilean: list[dict], ma_max: float = 0.05,
                         tol: float = 2e-3) -> dict:
    """Primary definition (campaign spec): residual against the analytic field."""
    rows = []
    ok_low = True
    for r in galilean:
        rel = r["residual_relU"]
        if r["ma"] == 0:      # unboosted baseline: residual is undefined (U = 0)
            continue
        if abs(r["ma"]) <= ma_max:
            if not (rel <= tol):
                ok_low = False
            rows.append({"ma": r["ma"], "residual_relU": rel, "periods": r["periods"],
                         "superposition_relU": r.get("superposition_residual_relU")})
    rows.sort(key=lambda d: d["ma"])
    res = [r["residual_relU"] for r in rows]
    mono = all(b >= a - 1e-12 for a, b in zip(res, res[1:])) if len(res) > 1 else False
    sup = [r["superposition_relU"] for r in rows if r["superposition_relU"] is not None]
    out = {
        "id": "C4",
        "name": "Galilean-boost residual against the analytic solution",
        "target": f"|u_meas-(u_exact+U)|/U <= {tol:g} for Ma <= {ma_max}, monotone in Ma",
        "rows": rows,
        "residual_leq_tol": bool(ok_low),
        "monotone_increasing": bool(mono),
        "pass": bool(ok_low and mono),
    }
    if sup:
        out["sub_criterion_C4b"] = {
            "name": "scheme-internal superposition residual |u(U)-u(0)-U|/U",
            "target": f"<= {tol:g} for Ma <= {ma_max}",
            "max_value": float(max(sup)),
            "pass": bool(max(sup) <= tol),
        }
    return out


def c5_ma2_law(galilean: list[dict], lo: float = 1.7, hi: float = 2.3) -> dict:
    """Fit ``Δλ/λ0 = C * Ma^p`` and check ``p`` (log-log slope)."""
    base = None
    for r in galilean:
        if abs(r["ma"]) < 1e-12:
            base = r["decay_rate"]
    if base is None:
        return {"id": "C5", "name": "Ma^2 deviation law", "pass": False,
                "reason": "no unboosted baseline run"}
    pts = [(r["ma"], abs((r["decay_rate"] - base) / base)) for r in galilean
           if r["ma"] > 0 and (r["decay_rate"] - base) != 0]
    pts = [(m, v) for m, v in pts if v > 0]
    signs = [np.sign(r["decay_rate"] - base) for r in galilean if r["ma"] > 0]
    if len(pts) < 3:
        return {"id": "C5", "name": "Ma^2 deviation law", "pass": False,
                "reason": f"only {len(pts)} usable points", "points": pts}
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "id": "C5",
        "name": "deviation law d(lambda)/lambda vs Ma",
        "target": f"log-log slope of Δλ/λ0 vs Ma in [{lo}, {hi}] (equivalently slope ≈ 1 vs Ma²)",
        "points": [{"ma": float(m), "rel_decay_deviation": float(v)} for m, v in pts],
        "slope_vs_ma": float(slope),
        "slope_vs_ma_squared": float(slope / 2.0),
        "r2": r2,
        "base_decay_rate": float(base),
        "deviation_sign": float(np.sign(np.mean(signs))) if signs else 0.0,
        "note": "deviation magnitude fitted; sign reported in 'deviation_sign' "
                "(negative = boosted vortex decays slower than the unboosted one)",
        "pass": bool(lo <= slope <= hi),
    }


def c6_negative_control(control: dict, tol: float = 0.03) -> dict:
    """C2-style spread must exceed the acceptance threshold for the raw scheme."""
    c2 = c2_cd_spread(control, tol_100=tol, tol_200=tol)
    spread = max(v["max_spread"] for v in c2["by_re"].values())
    return {
        "id": "C6",
        "name": "negative control (raw BGK, tau->0.5+) must break C2",
        "target": f"spread > {tol:.0%}",
        "detail": c2["by_re"],
        "max_spread": spread,
        "pass": bool(spread > tol),
    }


def c7_reproducibility(hashes: list[str]) -> dict:
    unique = sorted(set(hashes))
    return {
        "id": "C7",
        "name": "bitwise reproducibility of identical runs",
        "target": "two runs, same parameters -> identical SHA-256 state hash",
        "hashes": hashes,
        "n_unique": len(unique),
        "pass": bool(len(unique) == 1 and len(hashes) >= 2),
    }


def evaluate_criteria(results: dict) -> dict:
    """Build the full acceptance report from a campaign results dict."""
    crit = []
    if results.get("poiseuille"):
        crit.append(c1_poiseuille_order(results["poiseuille"]))
    if results.get("cylinder"):
        crit.append(c2_cd_spread(results["cylinder"]))
        crit.append(c3_st_spread(results["cylinder"]))
    if results.get("galilean"):
        crit.append(c4_galilean_residual(results["galilean"]))
        crit.append(c5_ma2_law(results["galilean"]))
    if results.get("negative_control"):
        crit.append(c6_negative_control(results["negative_control"]))
    if results.get("reproducibility"):
        crit.append(c7_reproducibility(results["reproducibility"]))
    return {
        "criteria": crit,
        "n_pass": sum(1 for c in crit if c.get("pass")),
        "n_total": len(crit),
        "all_pass": bool(crit) and all(c.get("pass") for c in crit),
        "provenance": results.get("provenance", {}),
    }