"""Acceptance verdicts for IN101-N01, computed from campaign records.

Every criterion is a function of the records only -- no hand-entered numbers --
and returns a dict with ``value``, ``target``, ``pass`` and the numbers used, so
that the report JSON can quote them verbatim.

(a) ``smooth_cf``      Cf of the flat-underside channel against the log law and
                       Dean's correlation over ``Re_h = 1e4..1e5``, rel. error
                       <= 5 %, plus the grid-convergence index (GCI <= 5 %).
(b) ``form_scaling``   exponent ``n`` of ``C_d^form`` vs ``(S_fr/S_plan)``
                       against the analytic value 1, error <= 15 %.
(c) ``momentum_budget`` periodic-channel momentum balance residual <= 3 % and
                       ``C_d^skin + C_d^form = C_d^total`` closure <= 2 %.
(d) ``observed_band``  swept ``C_d`` inside the drift-inversion band
                       0.004 <= C_d <= 0.02.
(e) ``negative_control`` keels off -> ``C_d -> Cf,0`` (<= 5 %) and
                       ``C_d^form <= 1e-3``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from ..underice_drag import OBSERVED_CD_BAND, cf_log_law, dean_cf

CD_FORM_FLOOR = 1.0e-3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def gci(fine: float, medium: float, coarse: float, ratio: float = 4.0 / 3.0) -> dict[str, Any]:
    """Roache grid-convergence index (three grids, observed order).

    ``fine``/``medium``/``coarse`` are the same functional on successively
    coarser grids with a constant refinement ratio ``ratio``.  Returns the
    observed order ``p``, the Richardson estimate and the *fine-grid* GCI,
    ``GCI = Fs |(f2 - f1)/f1| / (r^p - 1)`` with the usual safety factor
    ``Fs = 1.25``.
    """
    f1, f2, f3 = float(fine), float(medium), float(coarse)
    if not all(math.isfinite(v) and v > 0 for v in (f1, f2, f3)):
        return {"ok": False, "reason": "non-positive or non-finite values",
                "values": [f1, f2, f3]}
    r21 = f2 - f1
    r32 = f3 - f2
    out: dict[str, Any] = {"values": [f1, f2, f3], "ratio": ratio}
    if abs(r32) < 1e-30 or abs(r21) < 1e-30:
        out.update(ok=False, reason="identical or non-monotone triplet")
        return out
    p = abs(math.log(abs(r32 / r21)) / math.log(ratio))
    if not (0.1 < p < 6.0):
        out.update(ok=False, reason=f"observed order {p:.3f} outside [0.1, 6]", p=p)
        return out
    f_exact = f1 + (f1 - f2) / (ratio**p - 1.0)
    gci_fine = 1.25 * abs((f2 - f1) / f1) / (ratio**p - 1.0)
    out.update(ok=True, p=p, f_exact=f_exact, gci_fine=gci_fine,
               rel_change_medium=abs((f2 - f1) / f1))
    return out


def fit_power(x: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    """Least-squares power law ``y = A x^n`` (log-log), with ``r2``."""
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    good = (xa > 0) & (ya > 0) & np.isfinite(xa) & np.isfinite(ya)
    if good.sum() < 2:
        return {"ok": False, "reason": "fewer than two usable points",
                "n_points": int(good.sum())}
    lx, ly = np.log(xa[good]), np.log(ya[good])
    n, a = np.polyfit(lx, ly, 1)
    resid = ly - (a + n * lx)
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"ok": True, "exponent": float(n), "logA": float(a), "A": float(math.exp(a)),
            "r2": r2, "n_points": int(good.sum()),
            "x": xa[good].tolist(), "y": ya[good].tolist()}


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------


def criterion_a_smooth(records: dict[str, dict[str, Any]],
                       ladder_ids: list[str], turbulent_id: str | None = None) -> dict[str, Any]:
    """(a) smooth-wall Cf against the log law / Dean correlation + GCI."""
    ladder = [records[i] for i in ladder_ids if i in records]
    out: dict[str, Any] = {"criterion": "a_smooth_cf", "target_rel_error": 0.05,
                           "target_gci": 0.05, "cases": [r["case_id"] for r in ladder]}
    if len(ladder) < 3:
        out.update(passed=False, reason="need three ladder levels")
        return out
    ladder = sorted(ladder, key=lambda r: r["grid"]["nz"])
    cf_vals = [r["cf"] for r in ladder]
    g = gci(cf_vals[-1], cf_vals[-2], cf_vals[0])
    fine = ladder[-1]
    re_h = float(fine["re_h"])
    cf_log = cf_log_law(float(fine["re_tau"]))
    cf_dean = dean_cf(re_h)
    err_log = abs(fine["cf"] / cf_log - 1.0)
    err_dean = abs(fine["cf"] / cf_dean - 1.0)
    # exact laminar (plane Poiseuille) reference of the same forced channel:
    # U_b = g h^2/(3 nu) and Cf = 2 tau/(rho U_b^2) give Cf = 18 / Re_tau^2,
    # independent of nu.  This is the anchor that certifies the *solver*; the
    # log law / Dean correlation certify the *regime*, which is a different
    # question (see the turbulent attempt).
    cf_poiseuille = 18.0 / float(fine["re_tau"]) ** 2
    out.update(
        finest_case=fine["case_id"], cf_fine=float(fine["cf"]), re_h=re_h,
        re_tau=float(fine["re_tau"]), cf_log_law=float(cf_log), cf_dean=float(cf_dean),
        rel_error_log_law=float(err_log), rel_error_dean=float(err_dean),
        cf_poiseuille_exact=float(cf_poiseuille),
        rel_error_poiseuille=float(abs(fine["cf"] / cf_poiseuille - 1.0)),
        u_plus_bulk=float(fine["u_plus_bulk"]), gci=g,
        stationarity_drift=float(fine.get("stationarity_drift", float("nan"))),
    )
    out["cf_target_met"] = bool(min(err_log, err_dean) <= 0.05)
    out["gci_target_met"] = bool(g.get("ok") and g.get("gci_fine", 1.0) <= 0.05)
    if turbulent_id and turbulent_id in records:
        t = records[turbulent_id]
        out["turbulent_attempt"] = {
            "case_id": t["case_id"], "cf": float(t["cf"]), "re_h": float(t["re_h"]),
            "u_plus_bulk": float(t["u_plus_bulk"]),
            "rel_error_log_law": float(abs(t["cf"] / cf_log_law(float(t["re_tau"])) - 1.0)),
            "stationarity_drift": float(t.get("stationarity_drift", float("nan"))),
            "log_fit_kappa": (t.get("log_fit") or {}).get("kappa"),
            "note": "log-law seed + 3% noise, the turbulent-branch attempt",
        }
        out["turbulent_branch_reached"] = bool(
            abs((t.get("log_fit") or {}).get("kappa", 0.0) or 0.0 - 0.41) < 0.08)
    # the pass verdict requires the *targeted* regime: both the correlation and
    # the convergence index must be met
    out["passed"] = bool(out["cf_target_met"] and out["gci_target_met"])
    return out


def criterion_b_form_scaling(records: dict[str, dict[str, Any]],
                             keel_ids: list[str],
                             reference_cd: float | None = None) -> dict[str, Any]:
    """(b) form-drag scaling exponent vs the analytic frontal-area estimate.

    Two fits are reported.  ``fit`` uses ``C_d^form`` from the wall-pressure
    integral, which is the quantity the criterion asks about.  ``fit_excess``
    uses the *excess* total drag over the smooth reference at the same
    operating point, ``C_d(keels) - C_d(flat)``, which is the part of the drag
    that the momentum balance can actually certify: it is immune to the
    cancellation error of the surface integral (see criterion (c)).
    """
    keels = [records[i] for i in keel_ids if i in records and records[i]["family"] == "keel"]
    out: dict[str, Any] = {"criterion": "b_form_scaling", "target_rel_error": 0.15,
                           "cases": [r["case_id"] for r in keels]}
    if len(keels) < 3:
        out.update(passed=False, reason="need three keel cases")
        return out
    # normalised form drag: C_d^form / (S_fr / S_plan); the analytic argument
    # says this is 0.5 C_D,keel, i.e. *constant* along the sweep, so the
    # exponent of C_d^form against (S_fr/S_plan) must be 1.
    rows = []
    for r in keels:
        s = r["S_fr_over_plan"]
        if s <= 0:
            continue
        rows.append(dict(case_id=r["case_id"], S_fr_over_plan=float(s),
                         cd_form=float(r["cd_form"]), cd_total=float(r["cd_total"]),
                         cd_form_normalised=float(r["cd_form"]) / s,
                         h_keel=float(r["tags"]["h_keel"]),
                         lam=float(r["tags"]["lam"]),
                         geometry_ratio=float(r["geometry"]["S_fr_x_over_plan"]),
                         analytic=float(r["analytic_S_fr_over_plan"])))
    fit = fit_power([row["S_fr_over_plan"] for row in rows], [row["cd_form"] for row in rows])
    out["rows"] = rows
    out["fit"] = fit
    if not fit.get("ok"):
        out.update(passed=False, reason="power-law fit failed")
        return out
    n = fit["exponent"]
    out["exponent_error"] = float(abs(n - 1.0))
    if reference_cd is not None:
        excess = fit_power([row["S_fr_over_plan"] for row in rows],
                           [row["cd_total"] - reference_cd for row in rows])
        out["fit_excess"] = excess
        out["reference_cd"] = float(reference_cd)
        if excess.get("ok"):
            out["exponent_excess"] = float(excess["exponent"])
            out["exponent_excess_error"] = float(abs(excess["exponent"] - 1.0))
        for row in rows:
            row["cd_excess"] = row["cd_total"] - float(reference_cd)
    # geometry audit: the *measured* frontal-area ratio must be within 15 % of
    # the analytic 0.5 lambda h_keel / h_eff, otherwise the fit tests nothing
    geo_errs = [abs(row["S_fr_over_plan"] / row["analytic"] - 1.0)
                for row in rows if row["analytic"] > 0]
    out["geometry_rel_error_max"] = float(max(geo_errs)) if geo_errs else float("nan")
    # The machine criterion is written on the *pressure-integral* C_d^form.  That
    # quantity is reported, but criterion (c) shows its absolute calibration
    # fails (form > total drag), so the exponent alone is not evidence of
    # physics: the *certified* statement is the exponent of the excess drag over
    # the smooth reference, which the momentum balance does support.
    out["passed"] = bool(out["exponent_error"] <= 0.15)
    out["caveat"] = ("C_d^form comes from the wall-pressure integral; criterion (c) "
                     "shows that integral is not calibrated (form > total drag), so "
                     "the exponent pass is conditional on that route.  The certified "
                     "measure is the excess drag over the smooth reference.")
    if "exponent_excess" in out:
        out["passed_certified"] = bool(out["exponent_excess_error"] <= 0.15)
    return out


def criterion_c_budget(records: dict[str, dict[str, Any]],
                       case_ids: list[str]) -> dict[str, Any]:
    """(c) momentum balance residual and skin+form closure."""
    out: dict[str, Any] = {"criterion": "c_momentum_budget",
                           "target_momentum_residual": 0.03, "target_closure": 0.02,
                           "rows": []}
    ok_rows = []
    for cid in case_ids:
        r = records.get(cid)
        if not r:
            continue
        row = dict(case_id=cid,
                   momentum_balance_residual=float(r.get("momentum_balance_residual", float("nan"))),
                   closure_residual=float(r.get("closure_residual", float("nan"))),
                   tau_total=float(r["tau_total"]), tau_form=float(r["tau_form"]),
                   tau_skin=float(r.get("tau_skin", float("nan"))),
                   tau_me=float(r.get("tau_me", float("nan"))),
                   tau_flux=float(r.get("tau_flux", float("nan"))))
        out["rows"].append(row)
        ok_rows.append(row)
    if not ok_rows:
        out.update(passed=False, reason="no records")
        return out
    mom = np.array([abs(row["momentum_balance_residual"]) for row in ok_rows])
    clo = np.array([abs(row["closure_residual"]) for row in ok_rows])
    out["momentum_residual_max"] = float(np.nanmax(mom)) if np.any(np.isfinite(mom)) else float("nan")
    out["closure_residual_max"] = float(np.nanmax(clo)) if np.any(np.isfinite(clo)) else float("nan")
    out["closure_residual_median"] = float(np.nanmedian(clo)) if np.any(np.isfinite(clo)) else float("nan")
    out["momentum_target_met"] = bool(out["momentum_residual_max"] <= 0.03)
    out["closure_target_met"] = bool(out["closure_residual_max"] <= 0.02)
    out["passed"] = bool(out["momentum_target_met"] and out["closure_target_met"])
    return out


def criterion_d_band(records: dict[str, dict[str, Any]],
                     keel_ids: list[str]) -> dict[str, Any]:
    """(d) do the swept drag coefficients land in the observed band?"""
    lo, hi = OBSERVED_CD_BAND
    out: dict[str, Any] = {"criterion": "d_observed_band", "band": [lo, hi], "rows": []}
    for cid in keel_ids:
        r = records.get(cid)
        if not r:
            continue
        out["rows"].append(dict(case_id=cid, cd_total=float(r["cd_total"]),
                                cd_skin=float(r["cd_skin"]), cd_form=float(r["cd_form"]),
                                re_h=float(r["re_h"]),
                                in_band=bool(lo <= float(r["cd_total"]) <= hi)))
    if not out["rows"]:
        out.update(passed=False, reason="no keel records")
        return out
    cd = np.array([row["cd_total"] for row in out["rows"]])
    out.update(cd_min=float(cd.min()), cd_max=float(cd.max()), cd_median=float(np.median(cd)),
               n_in_band=int(sum(row["in_band"] for row in out["rows"])),
               n_cases=len(out["rows"]))
    out["passed"] = bool(out["n_in_band"] == out["n_cases"])
    return out


def criterion_e_negative_control(smooth: dict[str, Any], keel_cases: list[dict[str, Any]]) -> dict[str, Any]:
    """(e) keels off -> ``C_d = Cf,0`` and a vanishing form component."""
    out: dict[str, Any] = {"criterion": "e_negative_control", "target_cd_error": 0.05,
                           "target_cd_form": CD_FORM_FLOOR}
    cd = float(smooth["cd_total"])
    cf0 = 0.5 * float(smooth["cf"])
    out.update(smooth_case=smooth["case_id"], cf0_over_2=cf0, cd_smooth=cd,
               cd_form_smooth=float(smooth["cd_form"]),
               rel_error_cd_vs_cf0=float(abs(cd / cf0 - 1.0)) if cf0 > 0 else float("nan"),
               keel_form_min=float(min((k["cd_form"] for k in keel_cases), default=float("nan"))))
    out["cd_matches_cf0"] = bool(cf0 > 0 and abs(cd / cf0 - 1.0) <= 0.05)
    out["form_vanishes"] = bool(float(smooth["cd_form"]) <= CD_FORM_FLOOR)
    out["passed"] = bool(out["cd_matches_cf0"] and out["form_vanishes"])
    return out


def evaluate(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Run every criterion on a campaign's records and summarise the verdicts."""
    smooth_ids = sorted(
        (cid for cid, r in records.items() if r["family"] == "smooth"),
        key=lambda cid: records[cid]["grid"]["nz"])
    keel_ids = sorted(cid for cid, r in records.items() if r["family"] == "keel")
    ref_ids = sorted(cid for cid, r in records.items() if r["family"] == "smooth_ref")
    turb_ids = [cid for cid, r in records.items() if r["family"] == "smooth_transient"]
    a = criterion_a_smooth(records, smooth_ids, turb_ids[0] if turb_ids else None)
    ref_cd = (float(records[ref_ids[0]]["cd_total"]) if ref_ids else None)
    b = criterion_b_form_scaling(records, keel_ids, reference_cd=ref_cd)
    c = criterion_c_budget(records, smooth_ids + keel_ids)
    d = criterion_d_band(records, keel_ids)
    control = (records[ref_ids[0]] if ref_ids else
               (records[smooth_ids[-1]] if smooth_ids else None))
    e = (criterion_e_negative_control(control, [records[i] for i in keel_ids])
         if control is not None else {"criterion": "e_negative_control", "passed": False,
                                      "reason": "missing smooth reference"})
    verdicts = {k: bool(v.get("passed")) for k, v in
                (("a", a), ("b", b), ("c", c), ("d", d), ("e", e))}
    return {"criteria": {"a_smooth_cf": a, "b_form_scaling": b, "c_momentum_budget": c,
                         "d_observed_band": d, "e_negative_control": e},
            "verdicts": verdicts,
            "n_passed": int(sum(verdicts.values())), "n_criteria": len(verdicts),
            "all_passed": bool(all(verdicts.values()))}