#!/usr/bin/env python3
"""Independent reference evaluation for 2D dam-break front position.

References (loaded from CSV files, NOT from TensorLBM source code):
  1. Martin & Moyce (1952) — experimental surge-front position, alpha=2 case
  2. Koshizuka & Oka (1996) — experimental front position

Normalisation
-------------
TensorLBM code outputs:
    t* = step * sqrt(2g / L),   X* = x_front / L      (L = dam_width)

Koshizuka & Oka (L = column width):
    T_KO = t*  (same),   X_KO = x_front / L = X*

Martin & Moyce (alpha=2, a = L/2 half-width):
    T_MM = t*  (same, since sqrt(g/a)=sqrt(2g/L))
    Z_MM = 1 + (x_front - L) / a = 2*X* - 1
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from tensorlbm import DamBreakConfig, run_dam_break

HERE = Path(__file__).resolve().parent
REF_DIR = HERE / "reference_data"
OUT_DIR = HERE / "results_dam_break"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(name, tc, xc):
    rows = []
    with (REF_DIR / name).open() as f:
        for r in csv.DictReader(f):
            rows.append((float(r[tc]), float(r[xc])))
    return rows


def interp_at(tt, ts, xs):
    if tt < ts[0] or tt > ts[-1]:
        return None
    for i in range(len(ts) - 1):
        if ts[i] <= tt <= ts[i + 1]:
            dt = ts[i + 1] - ts[i]
            if dt < 1e-15:
                return xs[i]
            w = (tt - ts[i]) / dt
            return xs[i] + w * (xs[i + 1] - xs[i])
    return None


def evaluate():
    mm_ref = load_csv("martin_moyce_1952_front.csv", "T", "Z")
    ko_ref = load_csv("koshizuka_oka_1996_front.csv", "T", "X")
    check_Ts = [1.0, 2.0]
    levels = [("coarse_L12", 12, 60, 30, 300), ("fine_L24", 24, 96, 48, 400)]

    cases, error_table = {}, []
    for ln, L, nx, ny, ns in levels:
        print(f"Running {ln} L={L} {nx}x{ny} steps={ns}")
        cfg = DamBreakConfig(
            nx=nx, ny=ny, dam_width=L, model="cg",
            rho_heavy=0.8, rho_light=0.4, G=0.9, tau=1.0,
            g=5e-4, n_steps=ns,
            output_interval=max(15, ns // 20),
            output_root=str(OUT_DIR), run_name=f"dam_cg_{ln}",
            device="cpu", overwrite=True,
        )
        t0 = time.perf_counter()
        rd = run_dam_break(cfg)
        el = time.perf_counter() - t0
        ts, xs = [], []
        with (rd / "front_position.csv").open() as f:
            for r in csv.DictReader(f):
                ts.append(float(r["t_star"]))
                xs.append(float(r["X_star"]))
        cases[ln] = {"level": ln, "L": L, "nx": nx, "ny": ny,
                     "n_steps": ns, "elapsed_s": round(el, 1),
                     "t_series": ts, "x_series": xs}
        for T in check_Ts:
            xc = interp_at(T, ts, xs)
            if xc is None:
                continue
            ko_r = interp_at(T, [r[0] for r in ko_ref], [r[1] for r in ko_ref])
            ko_e = abs(xc - ko_r) / ko_r * 100 if ko_r else None
            mm_z = 2.0 * xc - 1.0
            mm_r = interp_at(T, [r[0] for r in mm_ref], [r[1] for r in mm_ref])
            mm_e = abs(mm_z - mm_r) / mm_r * 100 if mm_r else None
            error_table.append({
                "level": ln, "T": T, "X_code": round(xc, 4),
                "KO_X_ref": round(ko_r, 4) if ko_r else None,
                "KO_err_pct": round(ko_e, 2) if ko_e is not None else None,
                "MM_Z_sim": round(mm_z, 4),
                "MM_Z_ref": round(mm_r, 4) if mm_r else None,
                "MM_err_pct": round(mm_e, 2) if mm_e is not None else None,
            })

    conv = []
    for T in check_Ts:
        c = [r for r in error_table if r["level"] == "coarse_L12" and r["T"] == T]
        f = [r for r in error_table if r["level"] == "fine_L24" and r["T"] == T]
        if c and f:
            conv.append({
                "T": T,
                "KO_err_coarse": c[0]["KO_err_pct"], "KO_err_fine": f[0]["KO_err_pct"],
                "KO_convergence": "improved" if (f[0]["KO_err_pct"] or 999) < (c[0]["KO_err_pct"] or 999) else "worsened",
                "MM_err_coarse": c[0]["MM_err_pct"], "MM_err_fine": f[0]["MM_err_pct"],
                "MM_convergence": "improved" if (f[0]["MM_err_pct"] or 999) < (c[0]["MM_err_pct"] or 999) else "worsened",
            })

    results = {
        "benchmark": "dam_break_front_position", "model": "cg",
        "references": {
            "martin_moyce_1952": {"citation": "Martin & Moyce (1952) Phil Trans R Soc Lond A 244:312-324", "figure": "Fig.5 alpha=2", "normalization": "Z=2*X*-1", "n_points": len(mm_ref)},
            "koshizuka_oka_1996": {"citation": "Koshizuka & Oka (1996) Nucl Sci Eng 123:421-434", "figure": "Fig.4", "normalization": "X=X*", "n_points": len(ko_ref)},
        },
        "check_points": check_Ts, "cases": cases,
        "error_table": error_table, "mesh_convergence": conv,
    }

    # plots
    mm_T = [r[0] for r in mm_ref]; mm_Z = [r[1] for r in mm_ref]
    ko_T = [r[0] for r in ko_ref]; ko_X = [r[1] for r in ko_ref]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    ax = axes[0]
    ax.plot(ko_T, ko_X, "ks-", ms=5, label="Koshizuka & Oka (1996)")
    for ln in ["coarse_L12", "fine_L24"]:
        c = cases[ln]
        ax.plot(c["t_series"], c["x_series"], "--", label=f"Sim {ln}", lw=1.5)
    ax.set(xlabel="T = t√(2g/L)", ylabel="X* = x_front/L", title="Dam-break: Koshizuka & Oka (1996)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(mm_T, mm_Z, "ks-", ms=5, label="Martin & Moyce (1952)")
    for ln in ["coarse_L12", "fine_L24"]:
        c = cases[ln]
        ax.plot(c["t_series"], [2*x-1 for x in c["x_series"]], "--", label=f"Sim {ln}", lw=1.5)
    ax.set(xlabel="T = t√(g/a)", ylabel="Z = 2X*-1", title="Dam-break: Martin & Moyce (1952)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / "dam_break_front_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    Ts = [r["T"] for r in error_table if r["level"] == "coarse_L12"]
    x = range(len(Ts)); w = 0.2
    ax.bar([i-1.5*w for i in x], [r["KO_err_pct"] or 0 for r in error_table if r["level"]=="coarse_L12"], w, label="KO coarse", color="#4477AA")
    ax.bar([i-0.5*w for i in x], [r["KO_err_pct"] or 0 for r in error_table if r["level"]=="fine_L24"], w, label="KO fine", color="#66CCEE")
    ax.bar([i+0.5*w for i in x], [r["MM_err_pct"] or 0 for r in error_table if r["level"]=="coarse_L12"], w, label="MM coarse", color="#EE6677")
    ax.bar([i+1.5*w for i in x], [r["MM_err_pct"] or 0 for r in error_table if r["level"]=="fine_L24"], w, label="MM fine", color="#FFBB33")
    ax.axhline(15, color="red", ls="--", lw=1, label="15% gate")
    ax.axhline(3, color="green", ls=":", lw=1, label="3% target")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"T={t}" for t in Ts])
    ax.set(ylabel="Error [%]", title="Dam-break error vs independent refs")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(FIG_DIR / "dam_break_error_bars.png", dpi=150)
    plt.close(fig)

    (OUT_DIR / "error_table.json").write_text(json.dumps(results, indent=2))
    print("DONE")
    return results


if __name__ == "__main__":
    evaluate()
