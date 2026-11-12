"""Grid-refinement extrapolation with *uncertainty* and *model-form* accountability.

Why this module exists
----------------------
The repository has a dozen hand-rolled Richardson/GCI helpers (``compressible/
convergence.py``, ``compressible_shock/resolvability.py``, ``airfoil_polar.gci_analysis``,
``convergence_mms.richardson_order_nested`` ...).  All of them return a **point
estimate** of the h -> 0 limit and nothing else.  Several case-pool verdicts
consume such an estimate as if it were exact, e.g. TensorLBM's Rayleigh-Benard
onset case (``CASE-LBM-RB-ONSET-RA1708``) fails its 2 % gate because a three
grid ``Ra_c = Ra_inf + C/N`` fit returns ``Ra_inf = 1755.3`` (a value outside the
range of every measured point) with no stated uncertainty at all.

A point estimate is only as good as the convergence *form* assumed.  With N
grids and one fitted slope there is frequently more than one admissible form,
and the disagreement between them is *model-form uncertainty* -- it usually
dominates the statistical (measurement) uncertainty and it is invisible in any
single-form fit.  Reporting ``2.79 % FAIL`` from a single form therefore claims
a precision the data do not support.

What this module provides
-------------------------
``RefinementPoint``    one (grid measure ``h``, measured value ``q``, 1-sigma
                       ``sigma``) triple.  ``h`` is the small quantity that goes
                       to zero (``1/N``, ``1/sqrt(N_cells)``, ``dx`` ...).
``fit_power_law``      weighted least squares for ``q(h) = q0 + C h^p`` for a
                       *fixed* order ``p`` -> ``q0``, ``C``, ``se_q0``, ``chi2_red``.
``fit_free_order``     same with ``p`` fitted (1-D scan + refinement), i.e. the
                       observed order of accuracy.
``fit_saturated``      the degenerate ``q(h) = q0`` form (no refinement trend).
``identify_forms``     runs the candidate forms, marks each *licensed* /
                       *rejected* by a reduced-chi2 consistency test, and reports
                       the **model-form band** (spread of ``q0`` across the
                       licensed forms) plus AIC weights.
``extrapolation_verdict`` turns that into ``pass`` / ``fail`` /
                       ``undecided_form`` / ``form_falsified`` against a
                       reference value and a relative tolerance, using the band
                       widened by ``k`` standard errors.
``still_converging``   is the finest grid pair separated by more than noise?
                       A ``False`` answer means the ladder has saturated, so no
                       extrapolation is admissible no matter which form fits.

Discipline notes
----------------
* If *any* point carries ``sigma > 0`` the fit is weighted by ``1/sigma^2``; if
  no point does, the fit is unweighted and the residual-based standard error is
  used, and every form is flagged ``weighted=False`` (the chi2 test is then
  vacuous -- the caller must not quote a form-selection verdict as if it were
  tested).
* A form is rejected, never "tuned away": the consistency test is
  ``chi2_red <= 1 + 2 sqrt(2/dof)`` (a ~2 sigma upper bound on the chi-square
  fluctuation at ``dof`` degrees of freedom).
* No domain vocabulary here: h/q/sigma are generic.

NumPy only; deterministic; no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "RefinementPoint",
    "fit_power_law",
    "fit_free_order",
    "fit_saturated",
    "identify_forms",
    "extrapolation_verdict",
    "still_converging",
    "gci_triplet",
]


@dataclass(frozen=True)
class RefinementPoint:
    """One measured value ``q`` at grid measure ``h`` with 1-sigma ``sigma``."""

    h: float
    q: float
    sigma: float = 0.0
    label: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _prepare(points: Iterable[RefinementPoint | Sequence[float] | dict]):
    pts: list[RefinementPoint] = []
    for p in points:
        if isinstance(p, RefinementPoint):
            pts.append(p)
        elif isinstance(p, dict):
            pts.append(RefinementPoint(float(p["h"]), float(p["q"]),
                                       float(p.get("sigma", 0.0) or 0.0),
                                       str(p.get("label", ""))))
        else:
            seq = list(p)
            pts.append(RefinementPoint(float(seq[0]), float(seq[1]),
                                       float(seq[2]) if len(seq) > 2 else 0.0))
    pts.sort(key=lambda r: -r.h)          # coarse -> fine
    if len({round(r.h, 12) for r in pts}) != len(pts):
        raise ValueError("duplicate h values in refinement ladder")
    if any(r.h <= 0 for r in pts):
        raise ValueError("h must be positive (h -> 0 is the extrapolation limit)")
    return pts


def _weights(pts: Sequence[RefinementPoint]) -> tuple[np.ndarray, bool]:
    if all(r.sigma > 0 for r in pts):
        return np.array([1.0 / r.sigma ** 2 for r in pts]), True
    return np.ones(len(pts)), False


def _wls(pts, power: float, weights: np.ndarray, weighted: bool) -> dict:
    h = np.array([r.h for r in pts], dtype=float)
    q = np.array([r.q for r in pts], dtype=float)
    X = np.column_stack([np.ones_like(h), h ** power])
    XtW = X.T * weights
    A = XtW @ X
    b = XtW @ q
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:            # singular (e.g. single h)
        return {"ok": False, "reason": "singular normal matrix"}
    resid = q - X @ coef
    ssr = float(np.sum(weights * resid ** 2))
    dof = len(pts) - 2
    out = {
        "ok": True,
        "form": f"q(h) = q0 + C*h^{power:g}",
        "power": float(power),
        "q0": float(coef[0]),
        "C": float(coef[1]),
        "residuals": [float(v) for v in resid],
        "ssr": ssr,
        "dof": int(dof),
        "weighted": bool(weighted),
    }
    if dof > 0:
        cov = np.linalg.inv(A)
        if weighted:
            se0 = math.sqrt(max(cov[0, 0], 0.0))
            chi2_red = ssr / dof
        else:
            s2 = ssr / dof
            se0 = math.sqrt(max(cov[0, 0] * s2, 0.0))
            chi2_red = None
        out["se_q0"] = float(se0)
        out["chi2_red"] = float(chi2_red) if chi2_red is not None else None
    else:
        out["se_q0"] = float("nan")
        out["chi2_red"] = None
    if weighted and out["chi2_red"] is not None:
        out["chi2"] = float(ssr)
        out["aic"] = float(ssr + 2 * 2)
    else:
        out["aic"] = float(len(pts) * math.log(max(ssr / len(pts), 1e-300)) + 2 * 2)
    return out


def _licence(fit: dict) -> dict:
    """Mark a fit licensed/rejected by the reduced-chi2 consistency test."""
    if not fit.get("ok"):
        return {"licensed": False, "why": fit.get("reason", "fit failed")}
    chi2_red, dof = fit.get("chi2_red"), fit.get("dof", 0)
    if chi2_red is None:
        return {"licensed": dof >= 1,
                "why": "no point uncertainties -> chi2 test vacuous"}
    thresh = 1.0 + 2.0 * math.sqrt(2.0 / max(dof, 1))
    return {"licensed": bool(chi2_red <= thresh),
            "why": f"chi2_red={chi2_red:.3g} vs threshold {thresh:.3g} (dof={dof})"}


# ---------------------------------------------------------------------------
# forms
# ---------------------------------------------------------------------------
def fit_power_law(points, power: float = 1.0) -> dict:
    """Weighted LS of ``q(h) = q0 + C h^p`` at a fixed order ``p``."""
    pts = _prepare(points)
    w, weighted = _weights(pts)
    fit = _wls(pts, float(power), w, weighted)
    if fit.get("ok"):
        fit.update(_licence(fit))
        fit["n"] = len(pts)
    return fit


def fit_free_order(points, p_bounds: tuple[float, float] = (0.25, 4.0),
                   n_scan: int = 76) -> dict:
    """``q(h) = q0 + C h^p`` with the observed order ``p`` fitted by 1-D scan.

    The scan minimises the (weighted) residual sum of squares in ``p``; the
    reported ``p`` is the argmin and ``p_interval`` the set of ``p`` inside the
    scan whose SSR is within ``1 + 1/N`` of the minimum (a coarse but honest
    identifiability interval -- flat SSR in ``p`` means ``p`` is not pinned).
    """
    pts = _prepare(points)
    if len(pts) < 3:
        return {"ok": False, "reason": "free order needs >= 3 grids"}
    w, weighted = _weights(pts)
    ps = np.linspace(p_bounds[0], p_bounds[1], int(n_scan))
    ssrs = []
    for p in ps:
        f = _wls(pts, float(p), w, weighted)
        ssrs.append(f["ssr"] if f.get("ok") else np.inf)
    ssrs = np.asarray(ssrs)
    i = int(np.argmin(ssrs))
    smin = float(ssrs[i])
    tol = smin * (1.0 + 1.0 / len(pts)) if smin > 0 else 1e-12
    inside = ps[ssrs <= tol]
    best = _wls(pts, float(ps[i]), w, weighted)
    if best.get("ok"):
        best.update(_licence(best))
        best["p_interval"] = [float(inside.min()), float(inside.max())]
        best["n"] = len(pts)
        best["form"] = "q(h) = q0 + C*h^p (p fitted)"
    return best


def fit_saturated(points) -> dict:
    """Degenerate form ``q(h) = q0``: no refinement trend at all."""
    pts = _prepare(points)
    w, weighted = _weights(pts)
    q = np.array([r.q for r in pts], dtype=float)
    sw = float(np.sum(w))
    q0 = float(np.sum(w * q) / sw)
    resid = q - q0
    ssr = float(np.sum(w * resid ** 2))
    dof = len(pts) - 1
    out = {"ok": True, "form": "q(h) = q0 (saturated)",
           "power": 0.0, "q0": q0, "C": 0.0,
           "residuals": [float(v) for v in resid], "ssr": ssr, "dof": int(dof),
           "weighted": bool(weighted), "n": len(pts)}
    if dof > 0:
        out["se_q0"] = float(math.sqrt(1.0 / sw))
        out["chi2_red"] = float(ssr / dof) if weighted else None
        out["aic"] = (float(ssr + 2 * 1) if weighted
                      else float(len(pts) * math.log(max(ssr / len(pts), 1e-300)) + 2))
    else:
        out["se_q0"] = float("nan")
        out["chi2_red"] = None
        out["aic"] = None
    out.update(_licence(out))
    return out


# ---------------------------------------------------------------------------
# identifiability
# ---------------------------------------------------------------------------
def identify_forms(points, orders: Sequence[float] = (1.0, 2.0),
                   free_order: bool = True, saturated: bool = True) -> dict:
    """Fit every candidate form and report the model-form band of ``q0``.

    Returns ``{forms:[...], licensed:[...], q0_band, band_rel, best_by_aic, ...}``.
    ``q0_band`` is the spread of the extrapolated limit across the *licensed*
    forms: it is the honest precision of the extrapolation, and it is what a
    verdict has to be built from.
    """
    pts = _prepare(points)
    forms: list[dict] = []
    for p in orders:
        f = fit_power_law(pts, float(p))
        f["name"] = f"h^{p:g}"
        forms.append(f)
    if free_order and len(pts) >= 3:
        f = fit_free_order(pts)
        f["name"] = "h^p"
        forms.append(f)
    if saturated:
        f = fit_saturated(pts)
        f["name"] = "saturated"
        forms.append(f)
    usable = [f for f in forms if f.get("ok") and f.get("se_q0") == f.get("se_q0")]
    licensed = [f for f in usable if f.get("licensed")]
    band_src = licensed if licensed else usable
    q0s = [f["q0"] for f in band_src]
    out = {
        "n": len(pts),
        "h": [r.h for r in pts],
        "q": [r.q for r in pts],
        "sigma": [r.sigma for r in pts],
        "forms": forms,
        "licensed": [f["name"] for f in licensed],
        "rejected": [f["name"] for f in usable if not f.get("licensed")],
        "weighted": bool(any(f.get("weighted") for f in usable)),
        "q0_band": [float(min(q0s)), float(max(q0s))] if q0s else [float("nan")] * 2,
    }
    if q0s:
        c = 0.5 * (min(q0s) + max(q0s))
        out["band_rel"] = float((max(q0s) - min(q0s)) / abs(c)) if c else float("nan")
    else:
        out["band_rel"] = float("nan")
    cand = [f for f in usable if f.get("aic") is not None]
    out["best_by_aic"] = min(cand, key=lambda f: f["aic"])["name"] if cand else None
    return out


def extrapolation_verdict(res: dict, ref: float | None = None,
                          tol_rel: float = 0.02, k: float = 2.0) -> dict:
    """Classify an extrapolation against ``ref`` using the licensed band.

    * ``form_falsified``  no candidate form is consistent with the data
      (every form rejected) -> the ladder must not be extrapolated at all.
    * ``undecided_form``  the band (widened by ``k`` standard errors each side)
      straddles a tolerance boundary, or the band alone exceeds the tolerance:
      the data cannot decide, and any pass/fail claim would be a form choice.
    * ``pass`` / ``fail`` the whole band lies inside / outside the tolerance
      interval.
    """
    lo_b, hi_b = res.get("q0_band", [float("nan")] * 2)
    licensed = [f for f in res.get("forms", [])
                if f.get("ok") and f.get("licensed") and f.get("name") in res.get("licensed", [])]
    out = {"verdict": "undecided_form", "ref": ref, "tol_rel": tol_rel, "k": k,
           "band_raw": [lo_b, hi_b], "n_licensed": len(licensed)}
    if not licensed:
        out["verdict"] = "form_falsified"
        out["why"] = "no candidate convergence form passes the consistency test"
        return out
    lo = min(f["q0"] - k * f["se_q0"] for f in licensed)
    hi = max(f["q0"] + k * f["se_q0"] for f in licensed)
    out["band_k_sigma"] = [float(lo), float(hi)]
    if ref is None:
        return out
    lo_ref, hi_ref = ref * (1.0 - tol_rel), ref * (1.0 + tol_rel)
    out["tol_interval"] = [float(lo_ref), float(hi_ref)]
    out["band_vs_tol"] = float((hi - lo) / (hi_ref - lo_ref)) if hi_ref > lo_ref else float("nan")
    if lo_ref <= lo and hi <= hi_ref:
        out["verdict"] = "pass"
        out["why"] = "whole licensed band inside the tolerance interval"
    elif hi < lo_ref or lo > hi_ref:
        out["verdict"] = "fail"
        out["why"] = "whole licensed band outside the tolerance interval"
    else:
        out["verdict"] = "undecided_form"
        out["why"] = "licensed band straddles the tolerance boundary"
    return out


def still_converging(points, k: float = 2.0) -> dict:
    """Do the two finest grids differ by more than ``k`` combined standard errors?

    ``False`` => the ladder has saturated within its own noise; extrapolating it
    (in either direction) would be extrapolating noise.
    """
    pts = _prepare(points)
    if len(pts) < 2:
        return {"ok": False, "reason": "need >= 2 grids"}
    a, b = pts[-2], pts[-1]
    d = b.q - a.q
    s = math.hypot(a.sigma, b.sigma)
    out = {"delta": float(d), "combined_sigma": float(s),
           "z": float(d / s) if s > 0 else float("inf"),
           "k": k}
    out["still_converging"] = bool(abs(d) > k * s) if s > 0 else bool(abs(d) > 0)
    out["saturated_within_noise"] = not out["still_converging"]
    return out


def gci_triplet(f_coarse: float, f_medium: float, f_fine: float,
                r: float = 2.0, safety: float = 1.25) -> dict:
    """Roache GCI with the observed order from a triplet (point estimates only).

    Included for continuity with the ad-hoc helpers elsewhere in the package;
    unlike them this returns the observed order, the extrapolated value, the
    *estimated* discretisation uncertainty and the safety factor, and it flags
    the non-monotone case instead of silently dividing by zero.
    """
    r = float(r)
    d1 = f_medium - f_coarse
    d2 = f_fine - f_medium
    out = {"r": r, "d1": float(d1), "d2": float(d2)}
    # Monotone approach requires successive differences of the *same* sign
    # (both steps move toward the limit); an alternating sign is oscillatory
    # convergence and a zero difference means the ladder has saturated.
    if d1 == 0 or d2 == 0 or d1 * d2 < 0:
        out.update({"ok": False,
                    "reason": ("zero difference: ladder saturated" if (d1 == 0 or d2 == 0)
                               else "oscillatory (alternating sign) convergence"),
                    "p_obs": None, "extrapolated": None, "gci_fine": None})
        return out
    ratio = d1 / d2
    p = math.log(abs(ratio)) / math.log(r)
    ext = f_fine + (f_fine - f_medium) / (r ** p - 1.0)
    out.update({"ok": True, "p_obs": float(p), "extrapolated": float(ext),
                "gci_fine": float(safety * abs((f_fine - ext) / ext)) if ext else None,
                "safety": float(safety),
                "p_obs_note": ("p_obs > 4: the ladder is numerically saturated, the "
                               "extrapolated value is not distinguishable from the "
                               "finest point" if p > 4.0 else "p_obs in the usual band")})
    return out