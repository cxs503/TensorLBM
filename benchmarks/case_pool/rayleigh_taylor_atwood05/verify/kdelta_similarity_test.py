"""CASE-RT-INSTABILITY-ATWOOD — is the growth-rate deficit caused by the
interface thickness k*xi, or by the Allen-Cahn relaxation rate breaking the
similarity of the ladder?

Background
----------
``verify/kdelta_convergence.py`` runs a similarity-preserving resolution ladder
(lam = 64/96/128) in which every dimensionless group of the linear RT problem is
pinned *except* the interface thickness in wavelength units,

    k*xi = 2 pi ac_width / lam ,        ac_width = 2 cells  ->  k*xi ~ 1/lam

and reports that the growth-rate deficit shrinks with lam:

    err_W = -11.52 % (lam 64) -> -9.45 % (lam 96) -> -8.55 % (lam 128)   (vs sigma_inv)
    linear fit  err(k*xi) = e0 + m k*xi  ->  e0 = -5.53 %, R^2 = 0.9988

with the conclusion "about half of the accepted -10.14 % is finite-interface-
thickness *model form*, and it extrapolates away as k*xi -> 0".

That conclusion rests on one unexamined assumption: that ``k*xi`` is the knob
that changed along the ladder.  It is not the only one.  The Allen-Cahn
relaxation rate in *lattice* units is ``~ M * beta`` with ``M = ac_mobility``
held **fixed** (= 20) across the ladder, while the physical rate to be resolved
(``sigma ~ 1/lam`` in lattice units) *halves* per refinement.  So the ladder also
changes

    Pi = (AC relaxation rate) / (linear growth rate)  ~  lam ,

i.e. it makes the interface progressively stiffer relative to the instability.
``KD_l096_m13`` and ``KD_l128_m10`` (already pre-declared in
``run_case.SPECS_KDELTA``, never run before this study) repeat the two finest
rungs with ``M ~ 1/lam`` so that ``Pi`` is pinned too.  They are the discriminator
and they cost two runs.

Pre-declared hypotheses and their falsifiers (written before the runs returned)
-----------------------------------------------------------------------------
H_thickness : the deficit is a function of ``k*xi`` alone; the M-rescaled arms
              land on the base ladder, ``|err_M(lam) - err_base(lam)|`` small
              compared with the ladder step, and ``err`` stays monotone in lam.
              Falsifier: ``|err_M - err_base| >= 0.5 * |err_base(lam) - err_base(64)|``.
H_mobility  : the lam-trend is the Allen-Cahn relaxation-rate dissimilarity; the
              M-rescaled arms move *back towards the lam = 64 value*, i.e.
              ``|err_M(lam) - err_base(64)| < |err_base(lam) - err_base(64)|``
              by a margin >= 0.5 pp.  Falsifier: no such regression towards the
              coarse-rung value.
Extrapolation stability: the k*xi -> 0 intercept ``e0`` is reported together with
              (a) its leave-one-out spread over the ladder and (b) the intercepts
              of the *quadratic* and *proportional* fit forms, so the sharp-
              interface limit carries a model-form error bar instead of a single
              number.

Nothing here is tuned after the fact: the fit forms, the LOO protocol and the
two falsifiers above are fixed in this file; the numbers come from
``artifacts/runs_kdelta/*/case.json`` + ``instant_rate.csv`` (same reader as
``kdelta_convergence.py``, so estimator definitions cannot drift).

Usage::

    python3 verify/kdelta_similarity_test.py          # table + json + figure
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import kdelta_convergence as kd

BENCH_DIR = Path(__file__).resolve().parents[1]
RUNS = BENCH_DIR / "artifacts" / "runs_kdelta"
ART = BENCH_DIR / "artifacts" / "kdelta"
FIGS = BENCH_DIR / "figures"

#: a paired-arm difference is "small" (H_thickness supported) if it is below
#: this fraction of the corresponding base-ladder step
PAIRED_FRACTION = 0.5
#: and H_mobility needs at least this regression towards the coarse rung
REGRESSION_PP = 0.5

#: base ladder rungs (fixed M) and the M-rescaled duplicates (M ~ 1/lam)
BASE = ("KD_l064", "KD_l096", "KD_l128", "KD_l192", "KD_l256")
RESCALED = ("KD_l096_m13", "KD_l128_m10")


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def fit_form(pts: list[tuple[float, float]], form: str) -> dict:
    """Least-squares fit of ``err`` against ``x = k*xi`` for a given model form.

    ``linear``       err = e0 + m x
    ``quadratic``    err = e0 + q x^2
    ``proportional`` err = c x              (no intercept: pure model form)
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = len(pts)
    if n < 2:
        return {"form": form, "n": n, "note": "insufficient points (need >= 2)"}
    if form == "linear":
        e0, m, r2 = kd.ls(xs, ys)
        pred = [e0 + m * x for x in xs]
        params = {"e0_pct": e0, "slope_pct_per_kxi": m}
    elif form == "quadratic":
        # err = e0 + q x^2  ->  linear LS on u = x^2
        us = [x * x for x in xs]
        e0, q, r2 = kd.ls(us, ys)
        pred = [e0 + q * u for u in us]
        params = {"e0_pct": e0, "slope_pct_per_kxi2": q}
    elif form == "proportional":
        den = sum(x * x for x in xs)
        c = sum(x * y for x, y in zip(xs, ys)) / den if den else float("nan")
        pred = [c * x for x in xs]
        my = sum(ys) / n
        ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
        ss_tot = sum((y - my) ** 2 for y in ys)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        params = {"c_pct_per_kxi": c, "e0_pct": 0.0}
    else:  # pragma: no cover - guards a typo'd form name
        raise ValueError(form)
    rms = math.sqrt(sum((y - p) ** 2 for y, p in zip(ys, pred)) / n)
    out = {"form": form, "n": n, "r2": r2, "rms_pct": rms, **params}
    # AIC with Gaussian residuals, k = number of free parameters
    k = {"linear": 2, "quadratic": 2, "proportional": 1}[form]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    if ss_res > 0 and n > k:
        out["aic"] = n * math.log(ss_res / n) + 2 * k
    else:
        out["aic"] = float("nan")
    return out


def leave_one_out(pts: list[tuple[float, float]], form: str = "linear") -> dict:
    """Spread of the k*xi -> 0 intercept when each rung is dropped in turn."""
    e0s = {}
    for i, p in enumerate(pts):
        sub = [q for j, q in enumerate(pts) if j != i]
        f = fit_form(sub, form)
        if "e0_pct" in f:
            e0s[p[0]] = f["e0_pct"]
    vals = [v for v in e0s.values() if math.isfinite(v)]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "e0_min_pct": min(vals), "e0_max_pct": max(vals),
            "e0_spread_pct": max(vals) - min(vals), "per_dropped_rung": e0s,
            "e0_full_pct": fit_form(pts, form).get("e0_pct")}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_dirs() -> dict[str, dict]:
    out = {}
    for d in sorted(RUNS.glob("*")):
        if (d / "case.json").is_file():
            rec = kd.arm_record(d)
            out[rec["name"]] = rec
    return out


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    arms = run_dirs()

    base = [arms[n] for n in BASE if n in arms]
    base.sort(key=lambda r: r["lam"])
    rescaled = [arms[n] for n in RESCALED if n in arms]
    rescaled.sort(key=lambda r: r["lam"])

    result: dict = {
        "arms_present": sorted(arms),
        "base_ladder": [{"name": r["name"], "lam": r["lam"], "kxi": r["kxi"],
                         "zeta": r["zeta"], "ma_peak": r["ma_peak"],
                         "ac_mobility": r["ac_mobility"],
                         "err_W_pct": r["err_W_vs_inv_pct"],
                         "err_P_pct": r["err_P_vs_inv_pct"],
                         "err_W_vs_visc_pct": r["err_W_vs_visc_pct"]} for r in base],
        "rescaled_ladder": [{"name": r["name"], "lam": r["lam"], "kxi": r["kxi"],
                             "ac_mobility": r["ac_mobility"],
                             "err_W_pct": r["err_W_vs_inv_pct"],
                             "err_P_pct": r["err_P_vs_inv_pct"]} for r in rescaled],
    }

    # --- fit forms on the base ladder, both estimators -----------------------
    result["fits"] = {}
    for tag, key in (("W", "err_W_vs_inv_pct"), ("P", "err_P_vs_inv_pct")):
        pts = [(r["kxi"], r[key]) for r in base if math.isfinite(r[key])]
        if len(pts) < 3:
            result["fits"][tag] = {"n": len(pts), "note": "insufficient points"}
            continue
        result["fits"][tag] = {
            "points": pts,
            "linear": fit_form(pts, "linear"),
            "quadratic": fit_form(pts, "quadratic"),
            "proportional": fit_form(pts, "proportional"),
            "loo_linear": leave_one_out(pts, "linear"),
        }

    # --- the discriminator: paired M-rescaled vs base arm at the same lam ----
    paired = []
    for r in rescaled:
        b = next((x for x in base if x["lam"] == r["lam"]), None)
        if b is None:
            continue
        coarsest = next((x for x in base if x["lam"] == 64), None)
        dW = r["err_W_vs_inv_pct"] - b["err_W_vs_inv_pct"]
        step = b["err_W_vs_inv_pct"] - coarsest["err_W_vs_inv_pct"] if coarsest else float("nan")
        regress = (abs(r["err_W_vs_inv_pct"] - coarsest["err_W_vs_inv_pct"])
                   - abs(b["err_W_vs_inv_pct"] - coarsest["err_W_vs_inv_pct"])
                   if coarsest else float("nan"))
        paired.append({
            "lam": r["lam"],
            "base": b["name"], "rescaled": r["name"],
            "ac_mobility_base": b["ac_mobility"], "ac_mobility_rescaled": r["ac_mobility"],
            "err_W_base_pct": b["err_W_vs_inv_pct"], "err_W_rescaled_pct": r["err_W_vs_inv_pct"],
            "delta_W_pp": dW,
            "err_P_base_pct": b["err_P_vs_inv_pct"], "err_P_rescaled_pct": r["err_P_vs_inv_pct"],
            "delta_P_pp": r["err_P_vs_inv_pct"] - b["err_P_vs_inv_pct"],
            "base_ladder_step_vs_lam64_pp": step,
            "regression_toward_lam64_pp": regress,
            "H_thickness_supported": bool(abs(dW) < PAIRED_FRACTION * abs(step)),
            "H_mobility_supported": bool(regress > REGRESSION_PP),
        })
    result["paired_M_test"] = paired
    result["pre_registered"] = {"paired_fraction": PAIRED_FRACTION,
                                "regression_pp": REGRESSION_PP}

    (ART / "similarity_test.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # --- markdown table ------------------------------------------------------
    lines = ["# kdelta ladder: is k*xi the knob, or the Allen-Cahn mobility?",
             "",
             "Base ladder (`M = 20` fixed), similarity-preserving in zeta, Ma and",
             "`a0/lam`; only `k*xi = 2 pi ac_width / lam` and the AC relaxation-rate",
             "ratio `Pi` change.", "",
             "| arm | lam | k*xi | zeta | Ma_peak | M | err_W vs inv | err_P vs inv |",
             "|---|---|---|---|---|---|---|---|"]
    for r in base:
        lines.append(f"| {r['name']} | {r['lam']} | {r['kxi']:.4f} | {r['zeta']:.4e} | "
                     f"{r['ma_peak']:.3f} | {r['ac_mobility']:.1f} | "
                     f"{r['err_W_vs_inv_pct']:+.2f}% | {r['err_P_vs_inv_pct']:+.2f}% |")
    if len(base) >= 3:
        lines += ["", "| estimator | form | n | e0 (k*xi -> 0) | slope | R^2 | rms | AIC |",
                  "|---|---|---|---|---|---|---|---|"]
        for tag in ("W", "P"):
            f = result["fits"].get(tag, {})
            if "linear" not in f:
                continue
            for form in ("linear", "quadratic", "proportional"):
                ff = f[form]
                slope = ff.get("slope_pct_per_kxi", ff.get("slope_pct_per_kxi2",
                               ff.get("c_pct_per_kxi", float("nan"))))
                lines.append(f"| {tag} | {form} | {ff['n']} | {ff.get('e0_pct', float('nan')):+.2f}% | "
                             f"{slope:+.4g} | {ff['r2']:.4f} | {ff['rms_pct']:.3f} pp | "
                             f"{ff['aic']:.1f} |")
            loo = f["loo_linear"]
            lines.append(f"| {tag} | **LOO linear** | {loo.get('n', 0)} | "
                         f"[{loo.get('e0_min_pct', float('nan')):+.2f}, "
                         f"{loo.get('e0_max_pct', float('nan')):+.2f}]% "
                         f"(full {loo.get('e0_full_pct', float('nan')):+.2f}%) | — | — | — | — |")
    if paired:
        lines += ["", "## Paired discriminator (same lam, M rescaled to pin Pi)", "",
                  "| lam | base (M) | rescaled (M) | err_W base | err_W rescaled | delta | "
                  "base step vs lam=64 | regression toward lam=64 | H_thickness | H_mobility |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for p in paired:
            lines.append(f"| {p['lam']} | {p['ac_mobility_base']:.2f} | "
                         f"{p['ac_mobility_rescaled']:.2f} | {p['err_W_base_pct']:+.2f}% | "
                         f"{p['err_W_rescaled_pct']:+.2f}% | {p['delta_W_pp']:+.2f} pp | "
                         f"{p['base_ladder_step_vs_lam64_pp']:+.2f} pp | "
                         f"{p['regression_toward_lam64_pp']:+.2f} pp | "
                         f"{'yes' if p['H_thickness_supported'] else 'no'} | "
                         f"{'yes' if p['H_mobility_supported'] else 'no'} |")
    (ART / "similarity_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())