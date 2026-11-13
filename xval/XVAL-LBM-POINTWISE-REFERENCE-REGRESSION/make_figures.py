#!/usr/bin/env python3
"""Figures for XVAL-LBM-POINTWISE-REFERENCE-REGRESSION (reads summary.json only)."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "summary.json").read_text())
RAW = json.loads((HERE / "runs" / "raw_points.json").read_text())
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

# 1 -- eps vs tau per arm
by_arm = defaultdict(list)
for row in S["aggregate"]:
    by_arm[row["arm"]].append((row["tau"], row["eps_scaled_inf_max"]))
fig, ax = plt.subplots(figsize=(7, 4.2))
for arm, pts in sorted(by_arm.items()):
    pts = sorted(pts)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.semilogy(xs, ys, "o-", label=arm)
ax.axhline(float(S["provenance"]["predeclared"]["predeclared_thresholds"]["gate_fp64"]),
           color="k", ls="--", lw=1, label="predeclared gate 1e-10 (fp64)")
ax.axhline(1e-16, color="grey", lw=0.8, ls=":", label="fp64 round-off floor")
ax.set_xlabel(r"relaxation time $\tau$"); ax.set_ylabel(r"$\varepsilon_\infty$ = max$|f_{kern}-f_{ref}|$/max$|f_{ref}|$")
ax.set_title("TensorLBM collision kernels vs lbmpy 2.0 pointwise reference")
ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(FIG / "fig1_eps_vs_tau.png", dpi=160)

# 2 -- eps vs flow amplitude (Mach), D2Q9 arms
amp_pts = defaultdict(lambda: defaultdict(list))
for k, rec in RAW.items():
    amp_pts[rec["arm"]][rec["amplitude"]].append(rec["eps_scaled_inf"])
fig, ax = plt.subplots(figsize=(7, 4.2))
for arm, d in sorted(amp_pts.items()):
    xs = sorted(d); ys = [max(d[a]) for a in xs]
    ax.loglog(xs, ys, "s-", label=arm)
ax.set_xlabel(r"velocity-field amplitude (rms per component, max$|u|\approx3\times$this)")
ax.set_ylabel(r"$\varepsilon_\infty$ (worst over 5 seeds)")
ax.set_title("No amplitude/Mach dependence: agreement is round-off limited")
ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(FIG / "fig2_eps_vs_amplitude.png", dpi=160)

# 3 -- D3Q19 equilibrium-convention gap + physics arm
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
d3 = [(r["arm"], r["tau"], r["eps_scaled_inf_max"]) for r in S["aggregate"]
      if r["arm"].startswith("A6")]
labels = [f"{a.split('_')[0]}\nτ={t}" for a, t, _ in d3]
axes[0].bar(range(len(d3)), [v for _, _, v in d3], color=["#c44", "#c44", "#48a", "#48a"])
axes[0].set_xticks(range(len(d3))); axes[0].set_xticklabels(labels, fontsize=8)
axes[0].set_yscale("log"); axes[0].set_ylabel(r"$\varepsilon_\infty$")
axes[0].set_title("D3Q19: lbmpy continuous (red) vs discrete\nequilibrium (blue) reference")
for i, (_, _, v) in enumerate(d3):
    axes[0].annotate(f"{v:.1e}", (i, v), ha="center", va="bottom", fontsize=7)
phys = S["physics_shear_wave"]
names = [f"{p['side'].split('_')[0]}\nτ={p['tau']}" for p in phys]
axes[1].bar(range(len(phys)), [p["err_pct"] for p in phys], color="#2a7")
axes[1].axhline(3.0, color="k", ls="--", lw=1, label="3% target")
axes[1].axhline(15.0, color="r", ls="--", lw=1, label="15% acceptance")
axes[1].set_xticks(range(len(phys))); axes[1].set_xticklabels(names, fontsize=8)
axes[1].set_ylabel(r"$|\nu_{meas}-\nu_{th}|/\nu_{th}$  [%]")
axes[1].set_title("Physics arm: periodic shear-wave viscosity\n(TensorLBM kernel vs lbmpy reference operator)")
for i, p in enumerate(phys):
    axes[1].annotate(f"{p['err_pct']:.2f}%", (i, p["err_pct"]), ha="center", va="bottom", fontsize=8)
axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig3_d3q19_and_physics.png", dpi=160)
print("wrote", [p.name for p in sorted(FIG.glob('*.png'))])
