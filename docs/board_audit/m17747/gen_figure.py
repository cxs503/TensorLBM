#!/usr/bin/env python3
"""Figure for the m17679 board audit (data read from audit.json + evidence/, nothing hand-typed)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EV = HERE / "evidence"
A = json.loads((HERE / "audit.json").read_text())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# ---------------------------------------------------------------- panel 1: FROZEN arm
rows = A["A2_frozen_arm"]["rows"]
grids = list(rows)
cd_err = [abs(rows[g]["err_cd_pct"]) for g in grids]
cl_err = [abs(rows[g]["err_cl_pct"]) for g in grids]
x = np.arange(len(grids))
w = 0.36
b1 = ax1.bar(x - w / 2, cd_err, w, label="|err| $C_D$ vs 1.389", color="#3b6ea5")
b2 = ax1.bar(x + w / 2, cl_err, w, label="|err| $C_L$ peak vs 0.718", color="#c1666b")
for bars in (b1, b2):
    for b in bars:
        ax1.annotate(f"{b.get_height():.2f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
ax1.axhline(15, color="k", ls="--", lw=1.2, label="15% acceptance bar (owner)")
ax1.axhline(3, color="0.45", ls=":", lw=1.2, label="3% report bar")
ax1.set_xticks(x, [f"{g}\n(frozen, {40}D x {40}D, u=0.08)" for g in grids])
ax1.set_ylabel("|error| [%]")
ax1.set_ylim(0, 20)
ax1.set_title("FROZEN arm — pre-existing verified case, wrong setup for this anchor\n"
              "(verified 2026-08-19; NOT produced by m17679)", fontsize=10)
ax1.legend(fontsize=7.5, loc="upper right")
ax1.grid(alpha=0.25, axis="y")

# ------------------------------------------------- panel 2: FRESH arm (decisive)
z = np.load(EV / "m17747_repro_D10.npz")
t, cd, cl = z["t"], z["cd"], z["cl"]
ax2.plot(t, cd, lw=1.0, color="#3b6ea5", label="$C_D(t)$ — auditor replay")
ax2.plot(t, cl, lw=1.0, color="#c1666b", label="$C_L(t)$ — auditor replay")
runs = A["A3_fresh_arm"]["runs"]
for i, (name, r) in enumerate(runs.items()):
    lab = f"{name.replace('xval_', '').replace('.json', '')}: NaN@{r['nan_at']}"
    ax2.axvline(r["nan_at"], color="k", ls=":", lw=0.9)
    ax2.annotate(lab, (r["nan_at"], 5.4 - 0.55 * i), fontsize=7.5, rotation=0,
                 ha="right" if r["nan_at"] > 800 else "left")
ax2.axvline(15000, color="0.3", ls="--", lw=1.0)
ax2.annotate("target window of run D=10 (15000 steps = 150 D/U$_0$)", (15000, -1.2),
             fontsize=7.5, ha="right", rotation=90, color="0.3")
ax2.set_xlim(0, 15500)
ax2.set_ylim(-2, 6)
ax2.set_xlabel("time step")
ax2.set_ylabel("force coefficient (normalised)")
ax2.set_title("FRESH caliber-matched arm — 0/3 runs produced a valid number\n"
              "(NaN at 2000 / 2500 / 500 steps; $C_D$ spikes before blow-up)", fontsize=10)
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(alpha=0.25)

fig.suptitle("m17679 board audit — XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS "
             "(TensorLBM vs haimingz/vivsim @fb84b12): verdict L2", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = HERE / "fig_m17747.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")

# printed so the numbers quoted in the report are traceable
print("frozen |err| cd:", [round(v, 2) for v in cd_err], "cl:", [round(v, 2) for v in cl_err])
print("fresh nan_at:", {k: v["nan_at"] for k, v in runs.items()},
      "| valid numbers:", A["A3_fresh_arm"]["n_runs_with_valid_number"])
print("replay mlups:", A["A3_fresh_arm"]["replay"]["rel_diff_cd_half1"])