"""Acceptance gates for the perforated-plate oscillatory-flow verification face.

Machine-checkable criteria (campaign node P333W2-N06):

===== =================================================================
tag   criterion
===== =================================================================
A     quasi-steady limit (KC >= 15): C_d within +-10% of the sharp-edged
      perforated-plate closed form ``C_d = C_c/(1-beta C_c)``
B     phase monotonicity: the dp-Q phase lag decreases monotonically with KC
      (8 KC values, Spearman rho = -1) and spans from > 60 deg to < 25 deg
C     C_d(beta) monotonically increasing (5 beta values, Spearman rho = +1)
      with physical magnitude 0.4 ... 0.9
D     grid independence: GCI(C_d) < 5% over three resolutions (r = 2)
E     negative controls: (1) no plate -> dp vs analytic channel budget < 3%
      and velocity profile < 3%; (2) Stokes-layer wall shear < 5%
F     energy budget: cycle dissipation from the two independent paths
      (momentum exchange vs pressure stations) closes within 3%
G     pytest suite with >= 10 cases (checked in the test suite itself)
===== =================================================================
"""

from __future__ import annotations

import math

import numpy as np

THRESHOLDS = {
    "A_quasi_steady_rel_error": 0.10,
    "A_min_kc": 15.0,
    "B_spearman_target": -1.0,
    "B_spearman_tol": 0.05,      # |rho + 1| <= tol counts as monotone
    "B_lag_low_kc_min_deg": 60.0,
    "B_lag_high_kc_max_deg": 25.0,
    "C_spearman_target": 1.0,
    "C_spearman_tol": 0.05,
    "C_cd_min": 0.4,
    "C_cd_max": 0.9,
    "D_gci_max": 0.05,
    "D_ratio": 2.0,
    "D_safety_factor": 1.25,
    "E_nc1_dp": 0.03,
    "E_nc1_profile": 0.03,
    "E_nc2_tau": 0.05,
    "F_closure": 0.03,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def spearman(x, y) -> float:
    """Spearman rank correlation (no SciPy dependency)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(v.size, dtype=float)
        # average ties
        sv = v[order]
        i = 0
        while i < sv.size:
            j = i
            while j + 1 < sv.size and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# A: quasi-steady limit
# ---------------------------------------------------------------------------

def gate_A_quasi_steady(cases: list[dict], thr: dict | None = None) -> dict:
    thr = thr or THRESHOLDS
    rows = []
    for c in cases:
        if not _finite(c.get("C_d")):
            continue
        if c.get("KC", 0.0) < thr["A_min_kc"]:
            continue
        cd_ref = c.get("C_d_reference", float("nan"))
        err = abs(c["C_d"] - cd_ref) / cd_ref if _finite(cd_ref) and cd_ref > 0 else float("nan")
        rows.append({"id": c["id"], "KC": c["KC"], "beta": c["beta"],
                     "C_d": c["C_d"], "C_d_reference": cd_ref, "rel_error": err,
                     "r_squared": c.get("r_squared")})
    if not rows:
        return {"passed": False, "reason": "no case with KC >= 15", "rows": []}
    worst = max(rows, key=lambda r: r["rel_error"] if _finite(r["rel_error"]) else 1e9)
    passed = all(_finite(r["rel_error"]) and r["rel_error"] < thr["A_quasi_steady_rel_error"]
                 for r in rows)
    return {"passed": passed, "threshold": thr["A_quasi_steady_rel_error"],
            "worst": worst, "rows": rows}


# ---------------------------------------------------------------------------
# B: phase monotonicity
# ---------------------------------------------------------------------------

def gate_B_phase_monotonic(kc_cases: list[dict], thr: dict | None = None) -> dict:
    thr = thr or THRESHOLDS
    rows = [c for c in kc_cases if _finite(c.get("phase_lag_deg"))]
    if len(rows) < 8:
        return {"passed": False, "reason": f"only {len(rows)} valid KC cases (<8)",
                "rows": rows and [{k: c[k] for k in ('id', 'KC', 'phase_lag_deg')} for c in rows]}
    rows = sorted(rows, key=lambda c: c["KC"])
    kc = [c["KC"] for c in rows]
    lag = [c["phase_lag_deg"] for c in rows]
    rho = spearman(kc, lag)
    mono = all(lag[i] > lag[i + 1] for i in range(len(lag) - 1))
    passed = (
        abs(rho - thr["B_spearman_target"]) <= thr["B_spearman_tol"]
        and mono
        and lag[0] >= thr["B_lag_low_kc_min_deg"]
        and lag[-1] <= thr["B_lag_high_kc_max_deg"]
    )
    return {
        "passed": bool(passed),
        "spearman_rho": rho,
        "strict_spearman_minus_one": bool(abs(rho + 1.0) < 1e-12),
        "monotone_decreasing": bool(mono),
        "lag_low_kc_deg": lag[0],
        "lag_high_kc_deg": lag[-1],
        "n_points": len(rows),
        "rows": [{"id": c["id"], "KC": c["KC"], "phase_lag_deg": c["phase_lag_deg"],
                  "C_d": c["C_d"], "C_m": c["C_m"]} for c in rows],
    }


# ---------------------------------------------------------------------------
# C: C_d vs beta
# ---------------------------------------------------------------------------

def gate_C_cd_vs_beta(beta_cases: list[dict], thr: dict | None = None) -> dict:
    thr = thr or THRESHOLDS
    rows = [c for c in beta_cases if _finite(c.get("C_d"))]
    if len(rows) < 5:
        return {"passed": False, "reason": f"only {len(rows)} valid beta cases (<5)"}
    rows = sorted(rows, key=lambda c: c["beta"])
    beta = [c["beta"] for c in rows]
    cd = [c["C_d"] for c in rows]
    rho = spearman(beta, cd)
    in_range = all(thr["C_cd_min"] <= v <= thr["C_cd_max"] for v in cd)
    mono = all(cd[i] < cd[i + 1] for i in range(len(cd) - 1))
    ref = [c.get("C_d_reference", float("nan")) for c in rows]
    rel = [abs(cd[i] - ref[i]) / ref[i] if _finite(ref[i]) else float("nan")
           for i in range(len(rows))]
    passed = (abs(rho - thr["C_spearman_target"]) <= thr["C_spearman_tol"]
              and in_range and mono)
    return {
        "passed": bool(passed),
        "spearman_rho": rho,
        "monotone_increasing": bool(mono),
        "all_in_physical_range": bool(in_range),
        "rows": [{"id": c["id"], "beta": c["beta"], "C_d": c["C_d"],
                  "C_d_reference": ref[i], "rel_error": rel[i],
                  "KC": c["KC"], "r_squared": c.get("r_squared")} for i, c in enumerate(rows)],
    }


# ---------------------------------------------------------------------------
# D: grid convergence (GCI, Roache)
# ---------------------------------------------------------------------------

def gate_D_gci(gci_cases: list[dict], thr: dict | None = None) -> dict:
    thr = thr or THRESHOLDS
    rows = sorted([c for c in gci_cases if _finite(c.get("C_d"))],
                  key=lambda c: c["C_d"] * 0 + c.get("level", 0))
    rows = sorted(rows, key=lambda c: (c["level"] if "level" in c else 0))
    if len(rows) < 3:
        return {"passed": False, "reason": f"only {len(rows)} GCI levels (<3)"}
    f3 = rows[0]["C_d"]   # coarse
    f2 = rows[1]["C_d"]
    f1 = rows[2]["C_d"]   # fine
    r = thr["D_ratio"]
    fs = thr["D_safety_factor"]

    e21 = (f2 - f1) / f1 if f1 else float("nan")
    e32 = (f3 - f2) / f2 if f2 else float("nan")
    p = float("nan")
    p_note = ""
    if _finite(e21) and _finite(e32) and abs(e21) > 0 and e32 / e21 > 0:
        p = abs(math.log(abs(e32 / e21)) / math.log(r))
        if not (0.5 <= p <= 4.0):
            p_note = f"apparent order {p:.2f} out of range -> fallback p=2"
            p = 2.0
    else:
        p_note = "sign change or zero difference -> fallback p=2"
        p = 2.0

    gci_fine = fs * abs(e21) / (r ** p - 1.0)
    c_ext = f1 + (f1 - f2) / (r ** p - 1.0)
    return {
        "passed": bool(gci_fine < thr["D_gci_max"]),
        "C_d_coarse": f3, "C_d_medium": f2, "C_d_fine": f1,
        "apparent_order": p, "order_note": p_note,
        "e21": e21, "e32": e32,
        "GCI_fine": gci_fine, "threshold": thr["D_gci_max"],
        "C_d_richardson": c_ext,
        "levels": [{"id": c["id"], "ny": c["ny"], "hole_size": c["hole_size"],
                    "C_d": c["C_d"], "KC": c["KC"], "Re_h": c["Re_h"],
                    "beta": c["beta"], "steps": c.get("steps")} for c in rows],
        "ratio": r, "safety_factor": fs,
    }


# ---------------------------------------------------------------------------
# E: negative controls
# ---------------------------------------------------------------------------

def gate_E_negative_controls(nc_cases: list[dict], thr: dict | None = None) -> dict:
    thr = thr or THRESHOLDS
    rows = [c for c in nc_cases if isinstance(c.get("nc"), dict)]
    if not rows:
        return {"passed": False, "reason": "no negative-control case"}
    nc = rows[0]["nc"]
    nc1 = (_finite(nc.get("dp_rel_error")) and nc["dp_rel_error"] < thr["E_nc1_dp"]
           and _finite(nc.get("profile_rel_error_l2"))
           and nc["profile_rel_error_l2"] < thr["E_nc1_profile"])
    nc2 = (_finite(nc.get("tau_rel_error")) and nc["tau_rel_error"] < thr["E_nc2_tau"]
           and _finite(nc.get("tau_phase_error_deg"))
           and nc["tau_phase_error_deg"] < 10.0)
    return {
        "passed": bool(nc1 and nc2),
        "nc1": {"passed": bool(nc1),
                "dp_rel_error": nc.get("dp_rel_error"),
                "profile_rel_error_l2": nc.get("profile_rel_error_l2"),
                "threshold": thr["E_nc1_dp"]},
        "nc2": {"passed": bool(nc2),
                "tau_rel_error": nc.get("tau_rel_error"),
                "tau_phase_error_deg": nc.get("tau_phase_error_deg"),
                "threshold": thr["E_nc2_tau"]},
        "detail": {k: v for k, v in nc.items()
                   if k not in ("y", "profile_amp", "profile_amp_ref")},
    }


# ---------------------------------------------------------------------------
# F: energy budget
# ---------------------------------------------------------------------------

def gate_F_energy_closure(cases: list[dict], thr: dict | None = None,
                          min_kc: float = 1.0) -> dict:
    thr = thr or THRESHOLDS
    rows = []
    for c in cases:
        if c.get("kind") not in ("kc", "beta", "gci"):
            continue
        if not (_finite(c.get("phi")) and _finite(c.get("phi_stations"))):
            continue
        if abs(c["phi"]) <= 0 or c.get("KC", 0.0) < min_kc:
            continue
        err = abs(c["phi"] - c["phi_stations"]) / abs(c["phi"])
        rows.append({"id": c["id"], "KC": c["KC"], "phi_force": c["phi"],
                     "phi_stations": c["phi_stations"], "rel_error": err})
    if not rows:
        return {"passed": False, "reason": "no usable case"}
    worst = max(rows, key=lambda r: r["rel_error"])
    return {
        "passed": bool(worst["rel_error"] < thr["F_closure"]),
        "threshold": thr["F_closure"],
        "max_rel_error": worst["rel_error"],
        "median_rel_error": float(np.median([r["rel_error"] for r in rows])),
        "worst": worst,
        "n_cases": len(rows),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def is_valid(case: dict) -> bool:
    """A case is usable if the run produced a finite C_d at a positive KC."""
    return (_finite(case.get("C_d")) and _finite(case.get("KC"))
            and case["KC"] > 0 and abs(case["C_d"]) > 0)


def evaluate_all(cases: list[dict], thr: dict | None = None) -> dict:
    thr = thr or THRESHOLDS
    excluded = [c["id"] for c in cases if not is_valid(c)]
    cases = [c for c in cases if is_valid(c) or c.get("kind") == "nc"]
    by_kind: dict[str, list[dict]] = {}
    for c in cases:
        by_kind.setdefault(c.get("kind", "?"), []).append(c)
    gates = {
        "A_quasi_steady": gate_A_quasi_steady(cases, thr),
        "B_phase_monotonic": gate_B_phase_monotonic(by_kind.get("kc", []), thr),
        "C_cd_vs_beta": gate_C_cd_vs_beta(by_kind.get("beta", []), thr),
        "D_gci": gate_D_gci(by_kind.get("gci", []), thr),
        "E_negative_controls": gate_E_negative_controls(by_kind.get("nc", []), thr),
        "F_energy_closure": gate_F_energy_closure(cases, thr),
    }
    all_pass = all(g.get("passed", False) for g in gates.values())
    return {"gates": gates, "all_passed": bool(all_pass),
            "n_cases": len(cases), "excluded_cases": excluded,
            "thresholds": thr}