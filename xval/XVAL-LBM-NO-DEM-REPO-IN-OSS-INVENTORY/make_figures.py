#!/usr/bin/env python3
"""Figures for XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY (reads results.json only)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
res = json.loads((HERE / "results.json").read_text())

fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))

# --- (a) restitution / contact-duration convergence -------------------------
ax = axes[0]
clip = res["arms"]["contact"]
ctrl = res["arms"]["contact_no_clip_control"]
n = [r["steps_per_contact"] for r in clip]
e_th = clip[0]["e_analytic"]
ax.axhline(e_th, color="k", ls="--", lw=1.0, label=f"analytic e = {e_th:.4f}")
ax.plot(n, [r["e_sim"] for r in clip], "o-", color="#c0392b", label="sim (no-tension clip)")
ax.plot(n, [r["e_sim"] for r in ctrl], "s-", color="#2471a3", label="sim (clip removed)")
ax.set_xscale("log")
ax.set_xlabel(r"time steps per contact $t_c/\Delta t$")
ax.set_ylabel("restitution coefficient e")
ax.set_title("(a) contact: restitution vs resolution")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# --- (b) settling: measured vs reference terminal velocity ------------------
ax = axes[1]
st = res["arms"]["stokes_settling"]
sn = res["arms"]["schiller_naumann_settling"]
d = sorted({r["diameter_m"] for r in sn})
ref = [next(r["u_reference_m_s"] for r in sn if r["diameter_m"] == dd) for dd in d]
u_st = [next(r["u_sim_m_s"] for r in st if r["diameter_m"] == dd) for dd in d]
u_sn = [next(r["u_sim_m_s"] for r in sn if r["diameter_m"] == dd) for dd in d]
ax.plot(np.array(d) * 1e6, ref, "k--", lw=1.0, label="Clift-Gauvin reference")
ax.plot(np.array(d) * 1e6, u_sn, "o-", color="#1e8449", label="DEM, Schiller-Naumann")
ax.plot(np.array(d) * 1e6, u_st, "x:", color="#7d3c98", label="DEM, Stokes closure")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("particle diameter D [um]")
ax.set_ylabel(r"terminal velocity $u_t$ [m/s]")
ax.set_title("(b) ice-sphere settling vs literature")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

# --- (c) fluid-side LBM leg -------------------------------------------------
ax = axes[2]
rows = [r for r in res["arms"]["lbm_leg"]["rows"] if r["variant"] == "raw_run_lambda"]
re_l = [r["Re"] for r in rows]
ax.plot(re_l, [r["cd_clift_gauvin"] for r in rows], "k--", lw=1.0, label="Clift-Gauvin")
ax.plot(re_l, [r["cd_lbm"] for r in rows], "o", color="#b9770e", label="LBM (run lambda, no correction)")
for r in rows:
    ax.annotate(f"{r['rel_err']*100:+.0f}%", (r["Re"], r["cd_lbm"]),
                textcoords="offset points", xytext=(4, 5), fontsize=7, color="#b9770e")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Re")
ax.set_ylabel(r"$C_D$")
ax.set_title("(c) fluid-side LBM leg (reported, not gating)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
out = HERE / "fig_xval_dem.png"
fig.savefig(out, dpi=110)
print(out, out.stat().st_size)