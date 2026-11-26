#!/usr/bin/env python3
"""Analysis + figures of ``XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETON``.

Reads ``results/results.json`` (written by ``run_all.py``) and emits
``results/tables.md``, ``results/summary.json`` and ``figures/*.png``.
Every number in the tables comes from the JSON (nothing hand-typed).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

OWNER_PCT, TARGET_PCT, GEOM_ABS = 15.0, 3.0, 0.01
KPI_KEYS = ("eps_inf", "eps_l2_rel", "q_err", "u_peak_err", "shape_err")
CODES = (("tensorlbm", "TensorLBM"), ("lettuce", "OSS-lettuce"))


def fmt(x, nd=2):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}e}" if (x != 0 and abs(x) < 1e-2) else f"{x:.{nd}f}"


def pct(x):
    return float("nan") if x is None else 100.0 * float(x)


def mark(v, accept=OWNER_PCT, target=TARGET_PCT):
    if v is None or not np.isfinite(v):
        return "n/a"
    return "OK<3%" if v < target else ("OK<15%" if v < accept else "FAIL")


def slope(h, e):
    h, e = np.asarray(h, float), np.asarray(e, float)
    ok = np.isfinite(e) & (e > 0)
    return float(np.polyfit(np.log(h[ok]), np.log(e[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")


def main():
    data = json.loads((RES / "results.json").read_text())
    table = {r["label"]: r for r in data["table"]}
    order = [r["label"] for r in data["table"]]
    runs = {(r["label"], r["code"]): r for r in data["runs"] if r.get("status") == "DONE"}
    meta = data["meta"]
    L = ["# XVAL baseline-skeleton tables (generated)", "",
         f"git {meta['git_rev']} | OSS = lettuce {meta['oss_version']} @ `{meta['oss_checkout']}` | "
         f"steady tol {meta['steady_tol']:.0e} / {meta['steady_window']} steps | owner band {OWNER_PCT}%",
         ""]

    L += ["## Reference cross-check (closed-form parabola vs N-layer linear solve)", "",
          "| res | max rel dev | Q closed form | Q linear solve | Q continuum |", "|---|---|---|---|---|"]
    for res, d in meta["reference_cross_check"].items():
        L.append(f"| {res} | {d['max_rel_dev_closed_vs_linear_solve']:.2e} | {d['q_closed_form']:.6e} | "
                 f"{d['q_linear_solve']:.6e} | {d['q_continuum']:.6e} |")
    L.append("")

    L += ["## Table 1 -- arms and dimensionless numbers", "",
          "| arm | family | res | H_lu | nu_lu | tau | a_lu | u_peak_lu | Ma_peak | Re_H | nu/(H c_s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab in order:
        d = table[lab]["dimensionless"]
        L.append(f"| {lab} | {table[lab]['family']} | {table[lab]['resolution']} | {d['H_lu']:.0f} | "
                 f"{d['nu_lu']:.6f} | {d['tau']:.6f} | {d['a_lu']:.4e} | {d['u_peak_lu']:.4e} | "
                 f"{d['Ma_peak']:.3e} | {d['Re_H']:.4f} | {d['Kn_lattice']:.4f} |")
    L.append("")

    L += ["## Table 2 -- error vs exact solution (fluid rows, lattice units)", "",
          "| arm | code | steps | conv | " + " | ".join(k + " %" for k in KPI_KEYS)
          + " | \\|mean H_eff-H\\| | max-node dev | verdict |",
          "|" + "---|" * (6 + len(KPI_KEYS))]
    summary = {"arms": {}, "fail_arms": [], "max_err_pct": 0.0, "max_cross_pct": 0.0}
    for lab in order:
        for key, code in CODES:
            r = table[lab].get(key)
            if r is None:
                L.append(f"| {lab} | {code} | - | - |" + " n/a |" * len(KPI_KEYS) + " - | WITHHELD |")
                continue
            vals = [pct(r[k]) for k in KPI_KEYS]
            worst = max(v for v in vals if np.isfinite(v))
            hdev = abs(r["h_eff_mean"] - table[lab]["dimensionless"]["H_lu"])
            geom_ok = hdev < GEOM_ABS
            verdict = mark(worst)
            if not geom_ok or not r["converged"]:
                verdict = "FAIL"
            summary["arms"].setdefault(lab, {})[code] = dict(
                worst_kpi_pct=worst, verdict=verdict,
                h_eff_mean_dev=hdev, h_eff_max_dev_node=r["h_eff_max_dev"],
                steps=r["steps"], converged=r["converged"])
            if code == "TensorLBM" or lab.startswith("D"):
                summary["max_err_pct"] = max(summary["max_err_pct"], worst if np.isfinite(worst) else 0.0)
            if verdict == "FAIL":
                summary["fail_arms"].append(f"{lab}/{code}")
            L.append(f"| {lab} | {code} | {r['steps']} | {int(r['converged'])} | "
                     + " | ".join(f"{mark(v)} {fmt(v)}" for v in vals)
                     + f" | {hdev:.2e} | {r['h_eff_max_dev']:.2e} | {verdict} |")
    L.append("")

    L += ["## Table 3 -- OSS run vs TensorLBM run at identical lattice numbers", "",
          "| arm | max-norm diff % | L2 diff % | peak_TL/peak_OSS | verdict |", "|---|---|---|---|---|"]
    for lab in order:
        cc = table[lab].get("cross_code")
        if cc is None:
            L.append(f"| {lab} | n/a | n/a | n/a | single-code arm |")
            continue
        v = pct(cc["tv_rel"])
        summary["max_cross_pct"] = max(summary["max_cross_pct"], v)
        L.append(f"| {lab} | {v:.3f} {mark(v)} | {pct(cc['l2_rel']):.3f} | {cc['peak_ratio']:.6f} | {mark(v)} |")
    L.append("")

    L += ["## Table 4 -- resolution trend of eps_inf (log-log slope vs H)", "",
          "| family | code | H values | eps_inf % | slope |", "|---|---|---|---|---|"]
    conv = {}
    for fam in ("verbatim", "refine"):
        labs = [l for l in order if table[l]["family"] == fam]
        for key, code in CODES:
            hs, es = [], []
            for lab in labs:
                r = table[lab].get(key)
                if r:
                    hs.append(table[lab]["dimensionless"]["H_lu"])
                    es.append(pct(r["eps_inf"]))
            if len(hs) < 2:
                continue
            s = slope(hs, es)
            conv[f"{fam}_{key}"] = dict(H=hs, eps_pct=es, slope=s)
            L.append(f"| {fam} | {code} | {', '.join(f'{h:.0f}' for h in hs)} | "
                     f"{', '.join(f'{e:.3f}' for e in es)} | {s:+.2f} |")
    L.append("")
    summary["resolution_slopes"] = {k: v["slope"] for k, v in conv.items()}

    L += ["## Table 5 -- force (Mach) sensitivity and dtype", "",
          "| arm | force x | u_peak_lu | TensorLBM eps_inf % | OSS eps_inf % | cross-code diff % |", "|---|---|---|---|---|---|"]
    for lab in order:
        if table[lab]["family"] not in ("forcing", "dtype", "verbatim"):
            continue
        d = table[lab]["dimensionless"]
        tl = table[lab].get("tensorlbm")
        os_ = table[lab].get("lettuce")
        cc = table[lab].get("cross_code")
        L.append(f"| {lab} | {table[lab]['force_scale']:.0e} | {d['u_peak_lu']:.4e} | "
                 f"{fmt(pct(tl['eps_inf'])) if tl else 'n/a'} | {fmt(pct(os_['eps_inf'])) if os_ else 'n/a'} | "
                 + (f"{pct(cc['tv_rel']):.3f}" if cc else "n/a") + " |")
    L.append("")
    (RES / "tables.md").write_text("\n".join(L))
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(L[-40:]))

    # ----------------------------- figures -------------------------------- #
    plt.rcParams.update({"figure.dpi": 130, "font.size": 8})

    # fig 1: profiles at res 16 and 48 + error
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.5))
    for j, lab in enumerate(["V16", "V48"]):
        r = table[lab]
        _tl0 = runs.get((lab, "tensorlbm")) or runs.get((lab, "lettuce"))
        y = np.asarray(_tl0["y"])
        tl_r = runs.get((lab, "tensorlbm"))
        os_r = runs.get((lab, "lettuce"))
        ue = None
        if tl_r:
            ue = np.asarray(tl_r["u_profile"])
        ax[0, j].set_title(f"profile, {lab} (res={r['resolution']})")
        if os_r is not None:
            ax[0, j].plot(y, np.asarray(os_r["u_profile"]), "o-", ms=3, label="OSS lettuce")
        if tl_r is not None:
            ax[0, j].plot(y, np.asarray(tl_r["u_profile"]), "s--", ms=3, label="TensorLBM")
        if tl_r is not None and os_r is not None:
            from tensorlbm.verification.xval.poiseuille_skeleton import analytic_profile
            p = tl_r["params"]
            y_ex, _ = analytic_profile(y, p["height_lu"], p["acceleration_lu"], p["viscosity_lu"]), None
            ax[0, j].plot(y, y_ex, "k-", lw=1, label="exact")
            rel = 100.0 * (np.asarray(tl_r["u_profile"]) - y_ex) / np.max(np.abs(y_ex))
            ax[1, j].plot(y, rel, "s--", ms=3, label="TensorLBM")
            rel_o = 100.0 * (np.asarray(os_r["u_profile"]) - y_ex) / np.max(np.abs(y_ex))
            ax[1, j].plot(y, rel_o, "o-", ms=3, label="OSS lettuce")
            ax[1, j].set_xlabel("y [lu]")
            ax[1, j].set_ylabel("(u - u_exact)/u_peak [%]")
            ax[1, j].axhline(0, color="k", lw=0.5)
        ax[0, j].set_xlabel("y [lu]")
        ax[0, j].set_ylabel("u_x [lu]")
        ax[0, j].legend(fontsize=7)
        ax[1, j].legend(fontsize=7)
    fig.suptitle("Single-phase Poiseuille: OSS lettuce vs TensorLBM vs exact (identical lattice numbers)")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_profiles.png")
    plt.close(fig)

    # fig 2: convergence log-log
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for k, v in conv.items():
        style = "-s" if k.endswith("tensorlbm") else "--o"
        ax.loglog(v["H"], v["eps_pct"], style, ms=4, label=f"{k} (slope {v['slope']:+.2f})")
    ax.set_xlabel("H [lattice units]")
    ax.set_ylabel(r"$\varepsilon_\infty$ [%]")
    ax.set_title("resolution trend (verbatim = OSS Re/Ma rule, refine = fixed nu)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_convergence.png")
    plt.close(fig)

    # fig 3: force sensitivity
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    labs = [l for l in order if table[l]["family"] == "forcing"]
    for key, code in CODES:
        xs = [table[l]["force_scale"] for l in labs if table[l].get(key)]
        ys = [pct(table[l][key]["eps_inf"]) for l in labs if table[l].get(key)]
        ax.loglog(xs, ys, "-s" if key == "tensorlbm" else "--o", ms=4, label=f"{code} vs exact")
    xs = [table[l]["force_scale"] for l in labs if table[l].get("cross_code")]
    ys = [pct(table[l]["cross_code"]["tv_rel"]) for l in labs if table[l].get("cross_code")]
    ax.loglog(xs, ys, "-^", ms=5, color="crimson", label="OSS vs TensorLBM")
    ax.set_xlabel("lattice force multiplier")
    ax.set_ylabel("error [%]")
    ax.set_title("forcing / Mach sensitivity at res=16")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_force_sensitivity.png")
    plt.close(fig)

    # fig 4: field + difference maps (largest common resolution)
    lab = "V32" if "V32" in table else "V16"
    tl_r, os_r = runs.get((lab, "tensorlbm")), runs.get((lab, "lettuce"))
    if tl_r and os_r:
        ut = np.asarray(tl_r["u_field"])
        uo = np.asarray(os_r["u_field"])
        scale = np.max(np.abs(ut))
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
        for a, f, t in ((ax[0], ut, "TensorLBM u_x"), (ax[1], uo, "OSS lettuce u_x"),
                        (ax[2], (ut - uo) / scale * 100.0, "(TL - OSS)/u_peak [%]")):
            im = a.imshow(f.T, origin="lower", cmap="viridis" if "u_x" in t else "coolwarm")
            a.set_title(f"{t} ({lab})")
            a.set_xlabel("x [lu]")
            a.set_ylabel("y [lu]")
            fig.colorbar(im, ax=a, fraction=0.046)
        fig.tight_layout()
        fig.savefig(FIG / "fig4_fields.png")
        plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()