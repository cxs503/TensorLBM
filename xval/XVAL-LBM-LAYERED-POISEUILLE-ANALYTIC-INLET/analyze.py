"""Reduce the arm JSONs of XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET.

Writes table.csv, summary.json and the report figures.  Reads only the per-arm
JSONs written by run_case.run_arm (no re-simulation, no hand-typed numbers).
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ART = os.path.join(ROOT, "artifacts", "xval_layered_poiseuille_analytic_inlet")
FIG = os.path.join(ART, "figures")

KPI = ["eps_inf", "eps_l2_rel", "q_total_err", "u_peak_err", "y_peak_err_abs"]


def load():
    recs = []
    for p in sorted(glob.glob(os.path.join(ART, "arm_*.json"))):
        with open(p) as fh:
            recs.append(json.load(fh))
    return recs


def table(recs):
    rows = []
    for r in recs:
        n, q, a = r["numerics"], r["geometry"]["check"], r["analytic"]
        rows.append({
            "arm": r["arm"], "family": r["family"], "H": r["geometry"]["height"],
            "M": round(r["dimensionless"]["viscosity_ratio"], 3),
            "iface": r["dimensionless"]["interface_frac"][0] if r["dimensionless"]["interface_frac"] else 0.0,
            "steps": r["run"]["steps"], "converged": int(r["run"]["converged"]),
            "eps_inf_pct": 100 * n["eps_inf"], "eps_l2_pct": 100 * n["eps_l2_rel"],
            "q_total_pct": 100 * n["q_total_err"], "u_peak_pct": 100 * n["u_peak_err"],
            "y_peak_err": n["y_peak_err_abs"], "t_conv_1e3": r["run"]["t_conv_1e3"],
            "h_eff_dev": (q["dev"] if q else None),
            "wall_s": round(r["run"]["wall_seconds"], 1),
        })
    return rows


def write_csv(rows, path):
    keys = list(rows[0])
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")


def fig_profiles(recs):
    layered = [r for r in recs if len(r["config"]["nus"]) > 1]
    single = [r for r in recs if len(r["config"]["nus"]) == 1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, group, title in ((axes[0], single, "single layer (1-D Poiseuille)"),
                             (axes[1], layered, "layered (viscosity ratio M)")):
        for r in group:
            y = np.array(r["profile"]["y_wall"]); H = r["geometry"]["height"]
            u = np.array(r["profile"]["u_num"]); ue = np.array(r["profile"]["u_exact"])
            lab = f"{r['arm']}: eps_inf={100*r['numerics']['eps_inf']:.3f}%"
            ax.plot(u / ue.max(), y / H, "o", ms=3, label=lab)
            ax.plot(ue / ue.max(), y / H, "-", lw=1.0, color="k", alpha=0.5)
        ax.set_xlabel("$u_x/u_{x,peak}$"); ax.set_ylabel("$y/H$")
        ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    axes[0].text(0.03, 0.03, "symbols = LBM, lines = exact", transform=axes[0].transAxes, fontsize=7)
    fig.tight_layout(); p = os.path.join(FIG, "profiles.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_convergence(recs, rows):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for M, mk in ((1.0, "o"), (3.0, "s"), (5.0, "^")):
        sel = [r for r in recs if abs(r["dimensionless"]["viscosity_ratio"] - M) < 1e-9]
        sel = sorted(sel, key=lambda r: r["geometry"]["height"])
        if len(sel) < 2:
            continue
        H = [r["geometry"]["height"] for r in sel]
        ax[0].loglog(H, [r["numerics"]["eps_inf"] for r in sel], mk + "-",
                     label=f"M={M:g}  eps_inf")
        ax[1].loglog(H, [r["numerics"]["q_total_err"] for r in sel], mk + "-",
                     label=f"M={M:g}  Q err")
    for a, t in zip(ax, ("profile error $\\epsilon_\\infty$", "flow-rate error")):
        a.set_xlabel("H (fluid rows)"); a.set_title(t); a.grid(alpha=0.3, which="both")
        a.axhline(0.15, color="r", ls="--", lw=1)
        a.axhline(0.03, color="orange", ls=":", lw=1)
        a.legend(fontsize=7)
    fig.tight_layout(); p = os.path.join(FIG, "convergence.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_init_effect(recs):
    pairs = [("S32", "H32")]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for lab, col in zip(pairs[0], ("C0", "C3")):
        r = [x for x in recs if x["arm"] == lab]
        if not r:
            continue
        r = r[0]
        st = [t["step"] for t in r["trace"]]
        ep = [t["eps_inf"] for t in r["trace"]]
        ax.semilogy(st, ep, col, label=f"{lab} ({r['config']['init_mode']} init)")
    ax.set_xlabel("time step"); ax.set_ylabel("$\\epsilon_\\infty$"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8); ax.set_title("value of the analytic initial condition")
    fig.tight_layout(); p = os.path.join(FIG, "init_effect.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main():
    os.makedirs(FIG, exist_ok=True)
    recs = load()
    if not recs:
        print("no arms found", file=sys.stderr); return 2
    rows = table(recs)
    write_csv(rows, os.path.join(ART, "table.csv"))
    figs = [fig_profiles(recs), fig_convergence(recs, rows), fig_init_effect(recs)]
    thr = json.load(open(os.path.join(HERE, "predeclared.json")))["predeclared_thresholds"]
    ok_pct = thr["owner_acceptance_pct"]
    verdict = all(max(r["eps_inf_pct"], r["eps_l2_pct"], r["q_total_pct"], r["u_peak_pct"]) < ok_pct
                  for r in rows)
    geom_ok = all(r["h_eff_dev"] is None or abs(r["h_eff_dev"]) < thr["geometry_check_abs"] for r in rows)
    summary = {
        "anchor": "XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET",
        "arms": rows, "figures": figs,
        "thresholds": thr,
        "worst": {
            "eps_inf_pct": max(r["eps_inf_pct"] for r in rows),
            "eps_l2_pct": max(r["eps_l2_pct"] for r in rows),
            "q_total_pct": max(r["q_total_pct"] for r in rows),
            "u_peak_pct": max(r["u_peak_pct"] for r in rows),
            "y_peak_err": max(r["y_peak_err"] for r in rows),
        },
        "verdict_all_under_15pct": bool(verdict),
        "verdict_geometry_ok": bool(geom_ok),
        "pass": bool(verdict and geom_ok),
    }
    with open(os.path.join(ART, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: summary[k] for k in ("worst", "verdict_all_under_15pct",
                                              "verdict_geometry_ok", "pass")}, indent=1))
    for r in rows:
        print(f"{r['arm']:>10} H={r['H']:>3} M={r['M']:>4} eps_inf={r['eps_inf_pct']:7.3f}% "
              f"eps_l2={r['eps_l2_pct']:7.3f}% Q={r['q_total_pct']:7.3f}% "
              f"uP={r['u_peak_pct']:7.3f}% yPk={r['y_peak_err']:.2f} steps={r['steps']:.0f} "
              f"tconv={r['t_conv_1e3']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
