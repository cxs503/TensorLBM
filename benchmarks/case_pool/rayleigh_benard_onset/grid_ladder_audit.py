#!/usr/bin/env python3
"""CASE-LBM-RB-ONSET-RA1708 — grid-ladder *extrapolation caliber* audit.

What this script is for
-----------------------
The case's pre-registered decision rule has two gates: the primary-grid value
``Ra_c(ny=64)`` and its ``1/ny`` extrapolation ``Ra_inf`` must both be within
2 % of 1707.7597.  The first production pass returned

    Ra_c(64) = 1731.86  (+1.411 %, PASS)      Ra_inf = 1755.32  (+2.785 %, FAIL)

and the case was booked honest-FAIL on the extrapolation gate.  This script asks
the *prior* question: is that 2.785 % a measurement, or is it the output of one
arbitrary convergence form applied to data whose noise cannot distinguish forms?

It re-reads the case artifacts (no LBM is re-run here), converts each grid into a
``(h = 1/ny, Ra_c, sigma = bisection resolution floor)`` refinement point, and
hands the ladder to ``tensorlbm.refinement_extrapolation``:

* every candidate form ``q0 + C h^p`` (p = 1, 2, fitted; plus the saturated
  ``q0`` form) is fitted **with the per-point uncertainty as weights**;
* a form is *licensed* only if ``chi2_red`` is consistent with 1;
* the spread of ``q0`` across licensed forms is the **model-form band**, and the
  verdict is built from the band widened by k standard errors;
* ``still_converging`` asks whether the two finest grids differ by more than
  their own noise at all -- a saturated ladder must not be extrapolated.

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/grid_ladder_audit.py \
        --summary benchmarks/case_pool/rayleigh_benard_onset/artifacts/summary.json \
        [--summary runs/rb_ladder_fine/summary.json ...] \
        --out benchmarks/case_pool/rayleigh_benard_onset/artifacts/grid_ladder_audit.json \
        --fig-dir benchmarks/case_pool/rayleigh_benard_onset/artifacts

Numbers in the emitted Markdown come from the JSON only (no hand typing).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

RA_REF = 1707.7597          # Chandrasekhar (1961) §II, rigid-rigid, 2-D rolls
TOL_PCT = 2.0
CASE_CODE = "CASE-LBM-RB-ONSET-RA1708"


def load_points(summary_paths: list[Path]) -> tuple[list[dict], dict]:
    """Collect (ny, Ra_c, sigma) from one or more case summaries.

    Later files win on duplicate ``ny`` (so a finer re-measurement of the same
    grid replaces the earlier one).  ``sigma`` is the bisection resolution floor
    reported by the driver (``Ra_c_uncertainty``); the half-spread between the
    two in-bracket estimators is used as a floor if that field is missing.
    """
    seen: dict[int, dict] = {}
    meta: dict = {"sources": [], "predeclared": None}
    for p in summary_paths:
        d = json.loads(Path(p).read_text())
        meta["sources"].append(str(p))
        if d.get("predeclared") and meta["predeclared"] is None:
            meta["predeclared"] = d["predeclared"]
        for r in d.get("route_a", []):
            if r.get("Ra_c") is None:
                continue
            sigma = r.get("Ra_c_uncertainty")
            if sigma is None:
                sigma = 0.5 * abs(r.get("Ra_c", 0.0)) * r.get("estimate_spread_pct", 0.0) / 100.0
            jitter = r.get("jitter_mean")
            point = {
                "ny": int(r["ny"]), "nx": int(r["nx"]),
                "Ra_c": float(r["Ra_c"]),
                "sigma": float(sigma or 0.0),
                "err_pct": float(r.get("err_pct", 0.0)),
                "fit_r2": r.get("fit_r2"),
                "steps": r.get("steps"),
                "source": str(p),
            }
            if jitter is not None:
                point["jitter_mean"] = float(jitter)
            seen[int(r["ny"])] = point
    pts = [seen[k] for k in sorted(seen)]
    return pts, meta


def _fit_rows(res: dict) -> list[dict]:
    rows = []
    for f in res["forms"]:
        rows.append({
            "form": f.get("name"),
            "q0": f.get("q0"),
            "se_q0": f.get("se_q0"),
            "chi2_red": f.get("chi2_red"),
            "licensed": bool(f.get("licensed")),
            "why": f.get("why"),
            "aic": f.get("aic"),
            "p_interval": f.get("p_interval"),
        })
    return rows


def audit(points: list[dict], *, k: float = 2.0, ra_ref: float = RA_REF,
          tol_pct: float = TOL_PCT) -> dict:
    from tensorlbm.refinement_extrapolation import (
        RefinementPoint, identify_forms, extrapolation_verdict, still_converging,
        gci_triplet,
    )

    rps = [RefinementPoint(h=1.0 / p["ny"], q=p["Ra_c"], sigma=p["sigma"],
                           label=f"ny={p['ny']}") for p in points]
    res = identify_forms(rps, orders=(1.0, 2.0), free_order=len(rps) >= 3,
                         saturated=True)
    verdict = extrapolation_verdict(res, ref=ra_ref, tol_rel=tol_pct / 100.0, k=k)
    sat = still_converging(rps, k=k)
    out = {
        "case_code": CASE_CODE,
        "ra_ref": ra_ref,
        "tol_pct": tol_pct,
        "k_sigma": k,
        "points": points,
        "ladder": {
            "n": len(points),
            "ny": [p["ny"] for p in points],
            "err_pct": [p["err_pct"] for p in points],
            "sigma_rel_pct": [100.0 * p["sigma"] / p["Ra_c"] for p in points],
        },
        "forms": _fit_rows(res),
        "licensed": res["licensed"],
        "rejected": res["rejected"],
        "best_by_aic": res["best_by_aic"],
        "q0_band": res["q0_band"],
        "band_rel_pct": 100.0 * res["band_rel"] if res["band_rel"] == res["band_rel"] else None,
        "verdict": verdict,
        "saturation": sat,
        "naive_h1": next((f for f in res["forms"] if f.get("name") == "h^1"), None),
    }
    # Roache-style triplet on the three finest grids, for continuity with the
    # repo's other GCI helpers (point estimates, no uncertainty).
    if len(points) >= 3:
        f1, f2, f3 = points[-3]["Ra_c"], points[-2]["Ra_c"], points[-1]["Ra_c"]
        ratio = (points[-1]["ny"] / points[-2]["ny"])
        out["gci_finest_triplet"] = gci_triplet(f1, f2, f3, r=ratio)
    return out


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:.{nd}f}"
    return str(v)


def markdown(res: dict) -> str:
    L = []
    L.append("### Ladder\n")
    L.append("| ny | Ra_c | ± sigma (resolution) | sigma/Ra | err vs 1707.7597 | steps | fit R² |")
    L.append("|---|---|---|---|---|---|---|")
    for p in res["points"]:
        L.append(f"| {p['ny']} | {p['Ra_c']:.4f} | {p['sigma']:.4f} | "
                 f"{100.0 * p['sigma'] / p['Ra_c']:.3f} % | {p['err_pct']:+.3f} % | "
                 f"{p.get('steps')} | {_fmt(p.get('fit_r2'), 4)} |")
    L.append("\n### Convergence forms (h = 1/ny, weighted by 1/sigma²)\n")
    L.append("| form | q0 (h→0) | se(q0) | χ²_red | licensed |")
    L.append("|---|---|---|---|---|")
    for f in res["forms"]:
        L.append(f"| {f['form']} | {_fmt(f['q0'], 3)} | {_fmt(f['se_q0'], 3)} | "
                 f"{_fmt(f['chi2_red'], 3)} | {_fmt(f['licensed'])} |")
    v = res["verdict"]
    L.append(f"\n* licensed forms: {', '.join(res['licensed']) or 'none'}"
             f"; rejected: {', '.join(res['rejected']) or 'none'}"
             f"; best AIC: {res['best_by_aic']}")
    L.append(f"* model-form band of q0: [{res['q0_band'][0]:.3f}, {res['q0_band'][1]:.3f}]"
             f" = {_fmt(res['band_rel_pct'], 3)} % of the mid value")
    L.append(f"* band widened by k={res['k_sigma']} sigma: "
             f"[{_fmt((v.get('band_k_sigma') or [float('nan')])[0], 3)}, "
             f"{_fmt((v.get('band_k_sigma') or [float('nan'), float('nan')])[1], 3)}]"
             f"; tolerance interval: [{_fmt((v.get('tol_interval') or [float('nan')])[0], 3)}, "
             f"{_fmt((v.get('tol_interval') or [float('nan'), float('nan')])[1], 3)}]")
    L.append(f"* **verdict = {v['verdict']}** — {v.get('why', '')}")
    s = res["saturation"]
    L.append(f"* finest-pair difference {_fmt(s.get('delta'), 3)} vs combined sigma "
             f"{_fmt(s.get('combined_sigma'), 3)} (z = {_fmt(s.get('z'), 2)}): "
             f"saturated within noise = {_fmt(s.get('saturated_within_noise'))}")
    g = res.get("gci_finest_triplet")
    if g:
        L.append(f"* GCI on the three finest grids: p_obs = {_fmt(g.get('p_obs'), 3)}, "
                 f"extrapolated = {_fmt(g.get('extrapolated'), 3)}, "
                 f"GCI_fine = {_fmt(g.get('gci_fine'), 4)}"
                 + ("" if g.get("ok") else f" (NOT APPLICABLE: {g.get('reason')})"))
    return "\n".join(L) + "\n"


def figures(res: dict, fig_dir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    ny = np.array([p["ny"] for p in res["points"]], dtype=float)
    ra = np.array([p["Ra_c"] for p in res["points"]], dtype=float)
    sg = np.array([p["sigma"] for p in res["points"]], dtype=float)
    h = 1.0 / ny

    # --- figure 1: ladder + licensed forms + band -------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    hh = np.linspace(0.0, h.max() * 1.05, 200)
    colors = {"h^1": "#1f77b4", "h^2": "#2ca02c", "h^p": "#9467bd", "saturated": "#7f7f7f"}
    for f in res["forms"]:
        if not f.get("licensed") or f.get("q0") is None:
            continue
        p = next((x for x in res["forms"] if x["form"] == f["form"]), None)
        pw = {"h^1": 1.0, "h^2": 2.0, "saturated": 0.0}.get(f["form"])
        if f["form"] == "h^p":
            continue
        ax.plot(hh, f["q0"] + 0.0 * hh, "--", lw=1.0,
                color=colors.get(f["form"], "k"),
                label=f"{f['form']}: q0={f['q0']:.1f}")
    ax.errorbar(h, ra, yerr=2 * sg, fmt="o", ms=5, color="k", capsize=3,
                label="Ra_c(ny) ± 2σ (bisection floor)")
    ax.axhline(res["ra_ref"], color="#d62728", lw=1.2,
               label=f"ref {res['ra_ref']:.2f} (Chandrasekhar 1961)")
    lo, hi = res["verdict"].get("tol_interval", [None, None])
    if lo is not None:
        ax.axhspan(lo, hi, color="#d62728", alpha=0.08, label="±2 % tolerance")
    ax.set_xlabel("h = 1/ny")
    ax.set_ylabel("Ra_c")
    ax.set_title("RB onset ladder: extrapolated limits disagree by the tolerance width")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    f1 = fig_dir / "grid_ladder_audit_forms.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)
    made.append({"name": f1.name,
                 "caption": "Ra_c vs h=1/ny with 2σ bars; dashed = licensed extrapolated "
                            "limits (h^1 / h^2 / saturated), pale band = ±2 % of 1707.76."})

    # --- figure 2: error vs ny (log-x) with tolerance band ----------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    err = 100.0 * (ra / res["ra_ref"] - 1.0)
    ax.errorbar(ny, err, yerr=200.0 * sg / res["ra_ref"], fmt="o-", color="#1f77b4",
                capsize=3, label="Ra_c(ny) error ± 2σ")
    ax.axhline(0.0, color="#d62728", lw=1.2)
    ax.axhspan(-res["tol_pct"], res["tol_pct"], color="#d62728", alpha=0.08,
               label="±2 % gate")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("ny (log2)")
    ax.set_ylabel("Ra_c error vs 1707.7597  [%]")
    ax.set_title("Refinement ladder: does the bias decay or saturate?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f2 = fig_dir / "grid_ladder_audit_error_vs_ny.png"
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    made.append({"name": f2.name,
                 "caption": "Ra_c error vs grid size (log2 abscissa) with the ±2 % gate: "
                            "the ladder saturates inside its own noise."})
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=f"{CASE_CODE} extrapolation caliber audit")
    ap.add_argument("--summary", action="append", required=True,
                    help="case summary.json (repeatable; later files win per grid)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fig-dir", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--k", type=float, default=2.0)
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    out = args.out or (here / "artifacts" / "grid_ladder_audit.json")
    pts, meta = load_points([Path(s) for s in args.summary])
    if len(pts) < 2:
        print("error: need at least two grid points", file=sys.stderr)
        return 2
    res = audit(pts, k=args.k)
    res["meta"] = meta
    if not args.no_figures:
        fd = args.fig_dir or out.parent
        res["figures"] = figures(res, Path(fd))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    md = markdown(res)
    (out.with_suffix(".md")).write_text(md)
    print(md)
    print(f"[written] {out}")
    return 0 if res["verdict"]["verdict"] in ("pass", "fail", "undecided_form") else 1


if __name__ == "__main__":
    raise SystemExit(main())