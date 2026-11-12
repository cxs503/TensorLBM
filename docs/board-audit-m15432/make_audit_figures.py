"""Audit figures for m15432 (CASE-CYLINDER-DRAG-RE100) — re-rendered from delivered raw data.

Panels:
  A ref-vs-got bars (Cd, St) with the declared <5% tolerance band
  B windowed Cd means over the 18000-step sampling window (stationarity check)
  C cross-run Cd vs blockage/resolution (all TensorLBM runs, confounded: D and B co-vary)
  D vorticity field re-rendered from the delivered `samples.wz` (anti-fabrication check)

Output: docs/board-audit-m15432/fig_audit_m15432.png
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = "/root/tp-agent/work/m15432/artifacts/cylinder_re100_D10/cylinder_re100_unconfined.json"
PILOT = os.path.join(HERE, "artifacts", "audit_pilot_rerun.json")

d = json.load(open(ART))
run = d["runs"][0]["derived"]
smp = d["samples"][0]
rec = json.load(open(os.path.join(HERE, "audit_recompute.json")))
main = rec["main_run"]

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# ---- A: ref vs got ---------------------------------------------------------
a = ax[0][0]
labels = ["Cd", "St"]
ref = [main["benchmark"]["cd"]["ref"], main["benchmark"]["st"]["ref"]]
got = [main["benchmark"]["cd"]["got"], main["benchmark"]["st"]["got"]]
err = [main["benchmark"]["cd"]["rel_err_pct"], main["benchmark"]["st"]["rel_err_pct"]]
x = np.arange(2)
a.bar(x - 0.18, ref, 0.36, label="ref (Schäfer & Turek 1996)", color="#4c72b0")
a.bar(x + 0.18, got, 0.36, label="got (TensorLBM D=10, 250x100)", color="#c44e52")
for i in range(2):
    a.text(x[i] - 0.18, ref[i] + 0.03, f"{ref[i]:.4g}", ha="center", fontsize=9)
    a.text(x[i] + 0.18, got[i] + 0.03, f"{got[i]:.4g}\n({err[i]:+.2f}%)", ha="center", fontsize=9)
a.set_xticks(x, labels)
a.set_ylim(0, 2.1)
a.set_ylabel("coefficient")
a.set_title("A. benchmark: Cd FAIL (+25.4% vs <5%), St PASS (+4.39%)")
a.legend(fontsize=8)
a.grid(alpha=0.3, axis="y")

# ---- B: windowed Cd means --------------------------------------------------
b = ax[0][1]
wins = main["recomputed"]["window_means_2000"]
b.plot(np.arange(len(wins)) * 2.0, wins, "o-", label="Cd mean per 2000-step window")
b.axhline(main["benchmark"]["cd"]["ref"], color="k", ls="--", lw=1, label="ref Cd = 1.3796")
b.fill_between([0, len(wins) * 2], 1.3796 * 0.95, 1.3796 * 1.05, color="k", alpha=0.08,
               label="ref ±5%")
b.set_xlabel("sampling window start (k steps after warm-up 8000)")
b.set_ylabel("Cd")
b.set_ylim(1.25, 1.85)
b.set_title(f"B. Cd is flat in the window (drift 0.028%) but {main['recomputed']['cd_mean']:.3f}, "
            "not 1.38")
b.legend(fontsize=8)
b.grid(alpha=0.3)

# ---- C: cross-run Cd ------------------------------------------------------
c = ax[1][0]
pts = [("D=10 pilot\nB=16.7%", 1.8134, 0.167),
       ("D=10 audited\nB=10%", main["recomputed"]["cd_mean"], 0.10),
       ("D=16 m15407\nB=7.14%", 1.5445, 0.0714)]
c.plot([p[2] * 100 for p in pts], [p[1] for p in pts], "s-", color="#c44e52",
       label="TensorLBM runs (D and B co-vary)")
for label, cd, bl in pts:
    c.annotate(f"{cd:.3f}", (bl * 100, cd), textcoords="offset points", xytext=(4, 6), fontsize=9)
c.axhspan(1.33, 1.3796, color="k", alpha=0.12, label="literature / anchor band 1.33–1.38")
c.set_xlabel("blockage D/H (%)")
c.set_ylabel("Cd")
c.set_title("C. Cd falls with finer grid + lower blockage — no converged unconfined point yet")
c.legend(fontsize=8)
c.grid(alpha=0.3)

# ---- D: vorticity re-render ------------------------------------------------
dd = ax[1][1]
wz = np.array(smp["wz"])
cx, cy, D = smp["cx"], smp["cy"], run["blockage"] * run["ny"]
lim = np.percentile(np.abs(wz[wz != 0]), 99.5)
im = dd.imshow(wz, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=[0, wz.shape[1], 0, wz.shape[0]])
dd.add_patch(plt.Circle((cx, cy), D / 2.0, color="k"))
dd.set_title("D. vorticity re-rendered from delivered samples.wz (D=10)")
dd.set_xlabel("x (lattice)")
dd.set_ylabel("y (lattice)")
fig.colorbar(im, ax=dd, shrink=0.85, label="ω_z")

fig.suptitle("Board audit m15712 · m15432 CASE-CYLINDER-DRAG-RE100 (TensorLBM) · verdict L3 = benchmark FAIL",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, "fig_audit_m15432.png")
fig.savefig(out, dpi=130)
print("wrote", out)