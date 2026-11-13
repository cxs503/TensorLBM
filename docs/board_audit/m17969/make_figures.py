#!/usr/bin/env python3
"""Audit figures for m17865 (independent readings from artifacts/m17865_audit)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
AUD = json.loads(Path("/root/tp-agent/work/m17969/artifacts/m17865_audit"
                  "/audit_m17865.json").read_text())
DEL = Path("/root/tp-agent/work/m17865")
ART = DEL / "artifacts" / "xval_iso8_wall_normal_table"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

g = AUD["predeclared_gauges"]["per_case"]
names = list(g)
short = [n.replace("_offcenter", "").replace("plane_n40_", "plane ")
         .replace("alpha", "α=").replace("sphere_n", "sphere n=") for n in names]
mean = np.array([g[n]["K2_mean_rel"] for n in names]) * 100
mx = np.array([g[n]["K2_max_rel"] for n in names]) * 100
k1 = np.array([g[n]["K1_max_abs_dn"] for n in names])

fig, ax = plt.subplots(2, 2, figsize=(12.6, 8.6))
x = np.arange(len(names))

# (a) K2 accuracy vs the owner 15 % band
a = ax[0, 0]
a.bar(x - 0.19, mean, 0.38, label="mean rel err", color="#2b7a3d")
a.bar(x + 0.19, mx, 0.38, label="max rel err (untrimmed)", color="#c0392b")
a.axhline(15, color="k", ls="--", lw=1.2, label="owner/predeclared band 15 %")
for i, (m, v) in enumerate(zip(mean, mx)):
    a.text(i - 0.19, m + 0.8, f"{m:.2f}", ha="center", fontsize=8)
    a.text(i + 0.19, v + 0.8, f"{v:.1f}", ha="center", fontsize=8,
           color="#c0392b" if v > 15 else "k")
a.set_yscale("log")
a.set_ylim(1, 120)
a.set_xticks(x, short, rotation=12, fontsize=8)
a.set_ylabel("chordal rel. error [%]")
a.set_title("(a) K2 vs analytic normal: 4/4 mean pass, 2/4 max pass\n"
            "(auditor recompute from delivered OSS tables = delivery, digit for digit)")
a.legend(fontsize=7, loc="upper left")
a.grid(alpha=0.3, axis="y")

# (b) K1 exactness vs the compiled Fortran arm
b = ax[0, 1]
b.bar(x, k1, 0.55, color="#1f4e79")
b.axhline(1e-12, color="crimson", ls="--", lw=1.2, label="predeclared K1 ≤ 1e-12")
for i, v in enumerate(k1):
    b.text(i, v * 1.6, f"{v:.2e}", ha="center", fontsize=8)
b.set_yscale("log")
b.set_ylim(1e-17, 1e-9)
b.set_xticks(x, short, rotation=12, fontsize=8)
b.set_ylabel("max |n_ours − n_OSS|")
b.set_title("(b) K1 exactness vs gfortran-built upstream preprocessor\n"
            "float64 round-off floor; node sets / la_weight identical")
b.legend(fontsize=7)
b.grid(alpha=0.3, axis="y")

# (c) smoothing sensitivity
c = ax[1, 0]
rows = list(csv.DictReader(open(ART / "sensitivity_table.csv")))
for n in names:
    pts = sorted((int(r["key"]), float(r["mean_rel"]) * 100) for r in rows
                 if r["case"] == n and r["arm"] == "iso8_smooth_sweep")
    c.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3.5,
           label=n.replace("_offcenter", ""))
c.axhline(15, color="k", ls="--", lw=1.2, label="15 % band")
c.axvline(4, color="gray", ls=":", lw=1.5)
c.text(4.15, 13.0, "upstream = 4", fontsize=8, color="gray")
c.set_xscale("symlog", linthresh=1)
c.set_xlabel("D3Q27 smoothing iterations")
c.set_ylabel("mean rel. error [%]")
c.set_title("(c) sensitivity: spheres improve monotonically (12.6→1.1 %),\n"
            "45° plane degrades (2.1→5.1 %): upstream's 4 is not converged")
c.legend(fontsize=7)
c.grid(alpha=0.3)

# (d) curvature estimator (K4)
d = ax[1, 1]
sph = [n for n in names if n.startswith("sphere")]
xs = np.arange(len(sph))
kref = [g[n]["K4_raw"]["kappa_ref"] for n in sph]
kiso = [abs(g[n]["K4_raw"]["kappa_mean"]) for n in sph]
d.bar(xs - 0.19, kref, 0.38, label="analytic 2/R", color="#7f8c8d")
d.bar(xs + 0.19, kiso, 0.38, label="|κ| from ISO8 table", color="#8e44ad")
for i, n in enumerate(sph):
    d.text(i + 0.19, kiso[i] + 0.004,
           f"-{g[n]['K4_magnitude_rel_err'] * 100:.1f}%", ha="center", fontsize=9)
d.set_xticks(xs, [n.replace("_offcenter", "") for n in sph], fontsize=8)
d.set_ylabel("mean κ on the ±1-cell band")
d.set_title("(d) K4 curvature: |κ| biased low by 16.7–17.4 % > 15 % band\n"
            "→ honest FAIL (band estimator, not the table)")
d.legend(fontsize=7)
d.grid(alpha=0.3, axis="y")

fig.suptitle("m17865 XVAL-LBM-ISO8-WALL-NORMAL-TABLE — board audit (m17969), "
             "independent recompute 2026-11-13", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = FIG / "audit_m17865.png"
fig.savefig(out, dpi=135)
print(out, out.stat().st_size)