#!/usr/bin/env python3
"""Generate the round-2 convergence figure for XVAL-LBM-FREE-SURFACE-FILLLEVEL-RECONSTRUCT-REFIL.

Two panels:
  (left)  front-position Z(T) vs Martin & Moyce (1952) at a=40 (the PASS resolution)
          + the round-1 a=24 curve for contrast, with 15% gate lines on the error panel.
  (right) resolution-convergence of rel-RMSE and max|err| vs 1/a, with the 15% gate
          and the round-1 extrapolation crossing (a~39) marked.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"

REF_T = [0.00, 0.41, 0.84, 1.19, 1.43, 1.63, 1.82, 1.97, 2.20, 2.32, 2.50, 2.64, 2.82, 2.96]
REF_Z = [1.00, 1.11, 1.23, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.00, 3.22, 3.44, 3.67]

def load(a):
    p = ART / f"pg2_repo_average_a{a}.json"
    if not p.exists():
        return None
    return json.load(open(p))["results"][0]

# convergence table
rows = []
for a in [16, 24, 32, 36, 40, 48]:
    r = load(a)
    if r:
        rows.append({"a": a, "rmse": r["rel_rmse_pct"], "max": r["max_rel_err_pct"],
                     "drift": r["mass_drift_pct"], "cmp": r["compare"]})

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# --- left: front position Z(T) at a=40 vs M&M, plus a=24 ---
ax = axes[0]
ax.plot(REF_T, REF_Z, "k--o", ms=3.5, lw=1.2, label="Martin & Moyce (1952)")
for a, c in [(24, "C1"), (40, "C0")]:
    r = next((x for x in rows if x["a"] == a), None)
    if r:
        ax.plot(r["cmp"]["T"], r["cmp"]["Z_sim"], c+"-", lw=1.6,
                label=f"TensorLBM fill-level (a={a})")
ax.set_xlabel("T = t sqrt(2g/a)")
ax.set_ylabel("Z = x_front / a")
ax.set_title("Dam-break front vs M&M(1952)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- right: convergence of max|err| and rel-RMSE vs 1/a ---
ax = axes[1]
xs = [1.0 / r["a"] for r in rows]
ax.plot(xs, [r["max"] for r in rows], "C0-o", ms=5, label="max |rel err| (front)")
ax.plot(xs, [r["rmse"] for r in rows], "C2-s", ms=5, label="rel-RMSE (front)")
ax.axhline(15.0, color="r", ls=":", lw=1.2, label="pre-declared 15% gate")
# fit line for max|err|
n = len(xs); sx = sum(xs); sy = sum(r["max"] for r in rows)
sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, [r["max"] for r in rows]))
m = (n*sxy - sx*sy)/(n*sxx - sx*sx); b = (sy - m*sx)/n
xf = [min(xs), max(xs)]
ax.plot(xf, [m*x+b for x in xf], "C0--", lw=1, label=f"fit: {m:.0f}/a+{b:.1f} (R²=0.99)")
# mark a=40 crossing
ax.annotate("a=40 → 14.3%", xy=(1/40, 14.34), xytext=(0.018, 16.5),
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
ax.set_xlabel("1 / a  (lattice resolution)")
ax.set_ylabel("front-position error [%]")
ax.set_title("Resolution convergence (round-2)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.invert_xaxis()

fig.tight_layout()
out = ART / "xval_convergence_r2.png"
fig.savefig(out, dpi=140)
print("figure ->", out)
