#!/usr/bin/env python3
"""m17742 审计图：数据全部来自 runs/m17742/verify_triple.json（零手抄）。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path("/root/tp-agent/work/m17742/runs/m17742")
ev = json.loads((RUN / "verify_triple.json").read_text())
rows = ev["parity_table"]

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

# (a) ref vs got：mask 格数 + 标记数
grids = [r["grid"] for r in rows]
x = np.arange(len(grids))
ax[0].bar(x - 0.2, [r["mask_cells_ref"] for r in rows], 0.38, label="ref: in-repo cylinder_mask", color="#4c72b0")
ax[0].bar(x + 0.2, [r["mask_cells_got"] for r in rows], 0.38, label="got: runtime geometry.config", color="#dd8452")
for i, r in enumerate(rows):
    ax[0].text(i, max(r["mask_cells_ref"], r["mask_cells_got"]) * 1.03,
               f"Δ={r['mask_delta_cells']}\nmarkers {r['markers_got']}/{r['markers_ref_formula']}",
               ha="center", fontsize=8)
ax[0].set_yscale("log")
ax[0].set_xticks(x, grids)
ax[0].set_ylabel("solid cells (log)")
ax[0].set_title("(a) geometry parity: Δmask = 0 cells, markers exact\n(exactness by construction, not validation)")
ax[0].legend(fontsize=7, loc="upper left")

# (b) perimeter error vs dx（二阶收敛）
dx = [r["dx_m"] for r in rows]
per = [abs(r["perimeter_err_pct"]) for r in rows]
ax[1].loglog(dx, per, "o-", color="#c44e52", label="|perimeter err| [%] (audit rerun)")
slope = np.polyfit(np.log(dx), np.log(per), 1)[0]
ref = np.array(dx)
ax[1].loglog(ref, per[0] * (ref / dx[0]) ** 2, "k--", lw=1, label=f"slope 2 reference (fit={slope:.2f})")
for d, p, r in zip(dx, per, rows):
    ax[1].annotate(f"{r['grid']}: {p:.4f}%", (d, p), textcoords="offset points", xytext=(4, 6), fontsize=8)
ax[1].invert_xaxis()
ax[1].set_xlabel("dx [m] (finer →)")
ax[1].set_ylabel("|chord perimeter − 2πr| / 2πr  [%]")
ax[1].set_title("(b) marker discretisation is 2nd order\n(ratios 4.00 / 4.07)")
ax[1].legend(fontsize=7)

# (c) 物理层：锚点 vs 未收敛半程值
ph = ev["physics_midrun"]
ax[2].axvspan(0, 1, color="#f2e394", alpha=0.35)
ax[2].axhline(ph["cd_anchor"], color="#4c72b0", lw=2, label=f"anchor (Nabh 1998): Cd={ph['cd_anchor']:.4f}")
ax[2].plot([0.5], [ph["cd_midrun"]], "x", ms=14, mew=3, color="#c44e52",
           label=f"delivered mid-run: Cd={ph['cd_midrun']} @ {ph['steps_done']}/{ph['steps_target']} steps")
ax[2].set_xlim(0, 1)
ax[2].set_ylim(4.8, 7.0)
ax[2].set_xticks([])
ax[2].set_ylabel("Cd")
ax[2].set_title("(c) physics layer: NOT RUN — cj1641 still queued\n"
                f"mid-run value is {ph['dev_pct_vs_anchor']:+.2f}% vs anchor, NOT converged → WITHHELD")
ax[2].legend(fontsize=7, loc="lower right")
ax[2].text(0.02, 5.05, "no verdict may be drawn\nfrom this panel", fontsize=8, color="#7f6000")

fig.suptitle("m17742 audit of m17673 — XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY (TensorLBM)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = RUN / "fig_m17742_audit_m17673.png"
fig.savefig(out, dpi=150)
print(out, out.stat().st_size)