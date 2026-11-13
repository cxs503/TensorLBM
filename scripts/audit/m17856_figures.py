#!/usr/bin/env python3
"""Figures for the m17856 board audit (all numbers read from audit artifacts)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUD = ROOT / "artifacts" / "m17856_audit"
OUT = ROOT / "docs" / "audit_m17856"
OUT.mkdir(parents=True, exist_ok=True)
aud = json.loads((AUD / "audit_m17856.json").read_text())
rec = json.loads((Path("/root/tp-agent/work/m17758/xval/fluctuating_lbm_ghost_modes/"
                       "artifacts/results.json")).read_text())
fresh = json.loads((AUD / "fresh_results.json").read_text())
modes = rec["modes"]
by_rec = {a["arm"]: a["report"]["ratios"] for a in rec["arms"]}
by_new = {a["arm"]: a["report"]["ratios"] for a in fresh["arms"]}
arms = ["A1_stress_only", "A2_ghost_thermostat", "A3_all_modes"]
made = []

# fig 1: recorded vs fresh mode ratios
fig, ax = plt.subplots(figsize=(8.2, 3.9))
x = np.arange(len(modes)); wd = 0.13
for i, arm in enumerate(arms):
    ax.bar(x + (2 * i - 2.5) * wd, [by_rec[arm][m] for m in modes], wd,
           label=f"{arm} (recorded artifact)", alpha=0.55)
    ax.bar(x + (2 * i - 1.5) * wd, [by_new[arm][m] for m in modes], wd,
           label=f"{arm} (fresh re-run)", alpha=0.95)
ax.axhline(1.0, color="k", lw=1.1, label="FDT target = 1")
ax.axhspan(0.85, 1.15, color="g", alpha=0.12, label="owner +/-15% band")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(modes)
ax.set_ylabel(r"$Var(m_a)\,/\,(3\rho k_BT\,W_a)$")
ax.set_title("m17758: recorded artifact vs fresh re-run of the delivered entry (max rel. diff 564%)")
ax.legend(fontsize=6, ncol=2)
fig.tight_layout(); p = OUT / "fig_m17856_recorded_vs_fresh.png"; fig.savefig(p, dpi=150); plt.close(fig)
made.append(str(p.relative_to(ROOT)))

# fig 2: independent theory referee
ref = aud["R5_theory_referee"]
labels = ["A1_stress", "A2_ghosts", "A3_all", "base_FluctuatingLBM_kt_only"]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
xx = np.arange(len(labels))
ax.bar(xx - 0.18, [ref[l]["var_ux"] / ref[l]["target_ux"] for l in labels], 0.36,
       label=r"$Var(u_x)\,/\,(k_BT/\rho)$")
ax.bar(xx + 0.18, [ref[l]["var_rho"] / ref[l]["target_rho"] for l in labels], 0.36,
       label=r"$Var(\rho)\,/\,(\rho^2 k_BT)$")
ax.axhline(1.0, color="k", lw=1.1)
ax.axhspan(0.85, 1.15, color="g", alpha=0.12, label="owner +/-15% band")
ax.set_xticks(xx); ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
ax.set_ylabel("measured / equipartition target")
ax.set_title("Independent referee (primitive variables, no module weights): tau=0.8, kT=2.5e-4")
ax.legend(fontsize=7)
fig.tight_layout(); p = OUT / "fig_m17856_theory_referee.png"; fig.savefig(p, dpi=150); plt.close(fig)
made.append(str(p.relative_to(ROOT)))
print("\n".join(made))
