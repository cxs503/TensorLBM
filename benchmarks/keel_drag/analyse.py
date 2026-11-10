#!/usr/bin/env python
"""Analyse the keel-drag campaign: criteria (a)-(e), field-scale C_d, figures.

    python benchmarks/keel_drag/analyse.py --results benchmarks/keel_drag/campaign

Reads only the stored JSON records (``results/cases/*.json``) and the pure
functions of :mod:`tensorlbm.keel_drag` -- no hand-entered number reaches the
verdict block. Writes:

* ``campaign/cases.json``   merged record table
* ``campaign/verdicts.json`` acceptance verdicts + field-scale prediction
* ``campaign/figures/*.png`` paper figures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorlbm import keel_drag as kd  # noqa: E402
from tensorlbm.underice_drag import LEVEL_ICE_TO_HALF_CLEARANCE  # noqa: E402

#: Full-scale operating point of the field prediction (declared, not fitted):
#: 200 m ice field / hull length, 5 m depth of the under-ice mixed layer and a
#: 0.1 m/s ice-water relative velocity give Re = 5.6e6 -- the low end of the
#: observational drift regime, where the ITTC-57 line is still usable.
FULL_SCALE = {"re_full": 5.6e6, "h_eff_m": 30.0, "u_rel_m_s": 0.1, "L_m": 100.0}

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def load_records(results: Path) -> tuple[list[dict], list[dict]]:
    ok, bad = [], []
    for path in sorted((results / "cases").glob("*.json")):
        rec = json.loads(path.read_text())
        (ok if rec.get("status") == "ok" else bad).append(rec)
    return ok, bad


def pick_model(records: list[dict]) -> dict:
    """Representative ridge drag coefficient ``C_D`` from the campaign itself.

    Uses the keel cases whose aspect ratio sits in the observed band
    ``b_k/h_k = 3-6`` (Wadhams 2000) and whose height is at least two cells, so
    that the voxelisation is not the dominant error. The median is taken, and the
    spread over those cases is reported instead of a single number.
    """
    rows = []
    for r in records:
        g = r["geometry"]
        if g["h_k"] < 2.0 or not g["S_fr_x_over_plan"] > 0.0:
            continue
        aspect = g["b_k"] / g["h_k"]
        if not (2.5 <= aspect <= 6.5):
            continue
        cd = float(r["C_D_keel"])
        if math.isfinite(cd) and cd > 0.0:
            rows.append((r["case_id"], aspect, cd))
    vals = [c for _, _, c in rows]
    return {"c_d_keel_median": float(np.median(vals)) if vals else float("nan"),
            "c_d_keel_min": float(min(vals)) if vals else float("nan"),
            "c_d_keel_max": float(max(vals)) if vals else float("nan"),
            "n": len(vals),
            "cases": [{"case_id": cid, "aspect": a, "C_D_keel": c} for cid, a, c in rows]}


def field_prediction(c_d_keel: float) -> dict:
    out = {"full_scale": FULL_SCALE, "sensitivity": []}
    base = kd.predict_field_drag(c_d_keel=c_d_keel, h_eff_m=FULL_SCALE["h_eff_m"],
                                 re_full=FULL_SCALE["re_full"])
    out["base"] = base
    for density in (0.10, 0.20, 0.30):
        for wr in (3.0, 4.0, 6.0):
            p = kd.predict_field_drag(c_d_keel=c_d_keel, h_eff_m=FULL_SCALE["h_eff_m"],
                                      re_full=FULL_SCALE["re_full"], density=density,
                                      width_ratio=wr)
            out["sensitivity"].append({"density": density, "width_ratio": wr,
                                       "cd_total": p["cd_total"], "cd_form": p["cd_form"],
                                       "cd_skin": p["cd_skin"],
                                       "in_observed_band": p["in_observed_band"]})
    lo, hi = kd.OBSERVED_CD_BAND_KEEL
    vals = [s["cd_total"] for s in out["sensitivity"]]
    out["band"] = [lo, hi]
    out["band_reachable_fraction"] = float(np.mean([lo <= v <= hi for v in vals]))
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_decomposition(records: list[dict], out: Path) -> str:
    depth = sorted([r for r in records if r["family"] == "depth"],
                   key=lambda r: r["geometry"]["h_k"])
    if not depth:
        return ""
    h = [r["geometry"]["h_k"] for r in depth]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ax[0].plot(h, [r["cd_total_balance"] for r in depth], "k-o", label=r"$C_d$ total (balance)")
    ax[0].plot(h, [r["cd_skin"] for r in depth], "b-s", label=r"$C_d$ skin")
    ax[0].plot(h, [r["cd_form"] for r in depth], "r-^", label=r"$C_d$ form")
    ax[0].plot(h, [r["cd_skin"] + r["cd_form"] for r in depth], "k:", lw=1,
               label="skin + form")
    ax[0].set_xlabel(r"keel depth $h_k$ [cells]")
    ax[0].set_ylabel(r"$C_d$ ($\tau=\rho C_d U^2$)")
    ax[0].set_title("drag decomposition vs keel depth")
    ax[0].legend(fontsize=7)
    ax[0].set_ylim(bottom=0)
    ax[1].plot(h, [r["C_D_keel"] for r in depth], "r-^")
    ax[1].set_xlabel(r"keel depth $h_k$ [cells]")
    ax[1].set_ylabel(r"$C_D$ (frontal-area referenced)")
    ax[1].set_title(r"ridge pressure-drag coefficient")
    for a, r in ((ax[0], None), (ax[1], None)):
        a.set_axisbelow(True)
    p = out / "fig1_decomposition_vs_depth.png"
    fig.savefig(p)
    plt.close(fig)
    return str(p)


def fig_skin_vs_re(records: list[dict], crit_c: dict, out: Path) -> str:
    rows = [r for r in records if r["family"] in ("reynolds", "smooth")]
    if not rows:
        return ""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for fam, style in (("reynolds", "o-"), ("smooth", "s--")):
        sub = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["re_h_converged"])
        if not sub:
            continue
        ax.loglog([r["re_h_converged"] for r in sub], [r["cd_skin"] for r in sub], style,
                  label=("keel array" if fam == "reynolds" else "level ice"))
    ll = crit_c.get("loglog", {})
    ax.set_xlabel(r"$Re_h = U_b h_{eff}/\nu$")
    ax.set_ylabel(r"$C_d^{skin}$")
    if math.isfinite(ll.get("slope", float("nan"))):
        ax.set_title("skin drag vs Reynolds number\n"
                     rf"log-log slope ${ll['slope']:+.2f}$, "
                     rf"$r_{{log}}$ = {ll.get('r', float('nan')):+.2f}", fontsize=8)
    ax.legend(fontsize=7)
    p = out / "fig2_skin_vs_re.png"
    fig.savefig(p)
    plt.close(fig)
    return str(p)


def fig_ittc78(records: list[dict], crit_d: dict, out: Path) -> str:
    rows = crit_d.get("rows", [])
    if not rows:
        return ""
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    u = np.logspace(-2.2, 0.9, 200)
    ax.loglog(u, kd.ittc78_allowance(u), "k-", label="ITTC-1978 allowance")
    ax.axvspan(1e-6, 1e-4, color="tab:green", alpha=0.15,
               label="ITTC-78 calibration window")
    for key, mk in zip(kd.ITTC78_CONVENTIONS, ("o", "s", "^", "D")):
        xs = [r["conventions"][key]["ks_over_L"] for r in rows]
        ys = [abs(r["dCf_measured"]) for r in rows]
        ax.loglog(xs, ys, mk, ms=4, label=f"LBM, {key}")
    ax.set_xlabel(r"$k_s/L$")
    ax.set_ylabel(r"$\Delta C_F$")
    ax.set_title("measured roughness increment vs ITTC-78", fontsize=8)
    ax.legend(fontsize=6)
    p = out / "fig3_ittc78.png"
    fig.savefig(p)
    plt.close(fig)
    return str(p)


def fig_geometry_family(records: list[dict], out: Path) -> str:
    fams = {
        "width": sorted([r for r in records if r["family"] == "width"],
                        key=lambda r: r["geometry"]["b_k"]),
        "spacing": sorted([r for r in records if r["family"] == "spacing"],
                          key=lambda r: r["geometry"]["s_k"]),
        "streamwise": [r for r in records if r["family"] == "streamwise"],
    }
    if not any(fams.values()):
        return ""
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1))
    for r in fams["width"]:
        ax[0].plot(r["geometry"]["b_k"] / r["geometry"]["h_k"], r["cd_form"], "o-")
    ax[0].set_xlabel("aspect ratio $b_k/h_k$ (width sweep)")
    if fams["width"]:
        ax[0].set_title(f"form drag vs aspect ratio ($h_k$ = "
                        f"{fams['width'][0]['geometry']['h_k']:g} cells)", fontsize=8)
    ax[0].set_ylabel(r"$C_d^{form}$")
    ax[0].set_title("form drag vs ridge aspect ratio")
    if fams["spacing"]:
        s = [r["geometry"]["s_k"] for r in fams["spacing"]]
        ax[1].plot(s, [r["cd_form"] for r in fams["spacing"]], "s-", label=r"$C_d^{form}$")
        ax[1].plot(s, [r["cd_skin"] for r in fams["spacing"]], "o-", label=r"$C_d^{skin}$")
        ax[1].set_xscale("log")
        ax[1].set_xlabel(r"ridge spacing $s_k$ [cells]")
        ax[1].set_ylabel(r"$C_d$")
        ax[1].set_title("dependence on the ridge pitch")
        ax[1].legend(fontsize=7)
    p = out / "fig4_geometry_family.png"
    fig.savefig(p)
    plt.close(fig)
    return str(p)


def fig_field_band(field: dict, out: Path) -> str:
    sens = field.get("sensitivity", [])
    if not sens:
        return ""
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    lo, hi = field["band"]
    ax.axhspan(lo, hi, color="tab:orange", alpha=0.2, label="drift-inversion band")
    for wr, mk in zip(sorted({s["width_ratio"] for s in sens}), ("o", "s", "^")):
        sub = [s for s in sens if s["width_ratio"] == wr]
        ax.plot([s["density"] for s in sub], [s["cd_total"] for s in sub], mk + "-",
                label=rf"$b_k/h_k$ = {wr:g}")
    ax.set_yscale("log")
    ax.set_xlabel(r"areal keel density $\lambda$")
    ax.set_ylabel(r"field $C_d$ (full scale)")
    ax.set_title("keel-population drag vs the observed band", fontsize=8)
    ax.legend(fontsize=7)
    p = out / "fig5_field_cd_band.png"
    fig.savefig(p)
    plt.close(fig)
    return str(p)


def fig_profiles(records: list[dict], out: Path) -> str:
    sub = [r for r in records if r["family"] in ("smooth", "depth")][:6]
    if not sub:
        return ""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for r in sub:
        prof = r.get("profile") or {}
        yp, up = prof.get("y_plus"), prof.get("u_plus")
        if not yp or not up:
            continue
        ax.semilogx(yp, up, lw=1, label=f"h_k={r['geometry']['h_k']:g}")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$U^+$")
    ax.set_title("mean velocity profile (laminar core, log scaling)", fontsize=8)
    ax.legend(fontsize=6)
    p = out / "fig6_profiles.png"
    fig.savefig(p)
    plt.close(fig)
    return str(p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(_ROOT / "benchmarks" / "keel_drag" / "campaign"))
    args = ap.parse_args(argv)
    results = Path(args.results)
    figdir = results / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    records, failed = load_records(results)
    if not records:
        print("[analyse] no successful records found", flush=True)
        return 1
    (results / "cases.json").write_text(json.dumps(
        {"records": records, "failed": failed}, indent=1))

    smooth = [r for r in records if r["geometry"]["h_k"] == 0.0]
    model = pick_model(records)
    field = field_prediction(model["c_d_keel_median"])
    verdict = kd.evaluate(records, smooth, field_drag=field)
    verdict["model"] = model
    verdict["n_records"] = len(records)
    verdict["n_failed"] = len(failed)
    verdict["failed_cases"] = [{"case_id": r.get("case_id"), "error": r.get("error")}
                               for r in failed]
    r0 = records[0]
    nz = len(r0["profile"]["k"]) if r0.get("profile", {}).get("k") else None
    verdict["operating_point"] = {
        "grid": [r0["geometry"].get("nx"), r0["geometry"].get("ny") or 32, nz],
        "re_tau": r0["re_tau"],
        "re_tau_range": [min(r["re_tau"] for r in records),
                         max(r["re_tau"] for r in records)],
        "u_tau": r0["u_tau_target"],
        "h_eff": r0["h_eff"],
        "nu": r0["nu"], "tau": r0["tau"],
        "steps": sorted({r["steps"] for r in records}),
        "level_ice_to_half_clearance": LEVEL_ICE_TO_HALF_CLEARANCE,
        "max_mach": max(r["max_mach"] for r in records),
    }

    figs = []
    for name, fn in (("fig1_decomposition_vs_depth", fig_decomposition),
                     ("fig2_skin_vs_re", lambda r, o: fig_skin_vs_re(r, verdict["criteria"]["c_skin_vs_re"], o)),
                     ("fig3_ittc78", lambda r, o: fig_ittc78(r, verdict["criteria"]["d_ittc78"], o)),
                     ("fig4_geometry_family", fig_geometry_family),
                     ("fig5_field_cd_band", lambda r, o: fig_field_band(field, o)),
                     ("fig6_profiles", fig_profiles)):
        p = fn(records, figdir)
        if p:
            figs.append({"name": name, "path": p})
    verdict["figures"] = figs
    (results / "verdicts.json").write_text(json.dumps(verdict, indent=1, default=str))

    print(json.dumps({"n_records": len(records), "n_failed": len(failed),
                      "passed": verdict["passed"], "failed": verdict["failed"],
                      "all_pass": verdict["all_pass"],
                      "c_d_keel": model["c_d_keel_median"],
                      "field_cd_total": field["base"]["cd_total"],
                      "figures": [f["path"] for f in figs]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())