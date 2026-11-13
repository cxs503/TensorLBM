#!/usr/bin/env python3
"""Figures for XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK (reads artifacts/*.json only)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path(__file__).resolve().parent / "artifacts"
S = json.loads((ART / "summary.json").read_text())
load = lambda n: json.loads((ART / f"case_{n}.json").read_text())

# ---- fig 1: Couette ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for tag in ("couette_H16_tau0.8_U0.05_ladd_pre_local", "couette_H32_tau0.8_U0.05_ladd_pre_local"):
    r = load(tag); y = np.arange(len(r["u_num"]))
    lab = tag.split("_")[1]
    ax[0].plot(r["u_ana"], y, "-", lw=1.2, label=f"analytic {lab}")
    ax[0].plot(r["u_num"], y, "o", ms=3, mfc="none", label=f"LBM {lab}")
    d = np.abs(np.array(r["u_num"]) - np.array(r["u_ana"])) / max(abs(np.array(r["u_ana"])))
    ax[1].semilogy(np.clip(d, 1e-16, None), y, ".-", label=f"{lab} max={r['max_rel_pct_cf_umax']:.2e}%")
ax[0].set_xlabel("u_x/U0"); ax[0].set_ylabel("lattice row y"); ax[0].legend(fontsize=7)
ax[0].set_title("Couette: half-way BB moving wall vs exact linear")
ax[1].set_xlabel("|du|/max|u|"); ax[1].set_ylabel("lattice row y")
ax[1].axvline(0.02, color="r", ls="--", lw=1, label="2% case-pool tol")
ax[1].legend(fontsize=7); ax[1].set_title("pointwise error (float64)")
fig.tight_layout(); fig.savefig(ART / "fig1_couette.png", dpi=130); plt.close(fig)

# ---- fig 2: Stokes 2nd amplitude & phase ----
ref = load("stokes2_REF")
rows = np.array(ref["row_index"]); eta = (rows - 0.5) / ref["delta_bl"]
U = np.array(ref["u_profiles"]); A = np.array(ref["u_analytic_profiles"])
ts = np.array(ref["t_samples"]); w = ref["omega"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for arr, lab, st in ((U, "LBM", "o"), (A, "analytic", "-")):
    z = np.sum(arr * np.exp(-1j * w * ts)[:, None], axis=0)
    amp = 2.0 / len(ts) * np.abs(z); ph = np.degrees(np.angle(z)) % 360
    ax[0].semilogy(eta, amp / ref["U0"], st, ms=3, label=lab)
    ax[1].plot(eta, ph, st, ms=3, label=lab)
ax[0].set_xlabel(r"$\eta/\delta$"); ax[0].set_ylabel("A/U0"); ax[0].legend(fontsize=8)
ax[0].set_title("Stokes 2nd: fundamental amplitude")
ax[1].set_xlabel(r"$\eta/\delta$"); ax[1].set_ylabel("phase [deg]"); ax[1].legend(fontsize=8)
ax[1].set_title("Stokes 2nd: phase (effective wall location)")
fig.tight_layout(); fig.savefig(ART / "fig2_stokes2_amplitude_phase.png", dpi=130); plt.close(fig)

# ---- fig 3: space-time ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ext = [0, ts[-1] - ts[0], eta[0], eta[-1]]
im = ax[0].imshow(U / ref["U0"], aspect="auto", origin="lower", cmap="RdBu_r",
                  vmin=-1.2, vmax=1.2, extent=ext)
ax[0].set_xlabel("t (steps, last period)"); ax[0].set_ylabel(r"$\eta/\delta$")
ax[0].set_title(f"LBM u/U0  tau=0.8  delta={ref['delta_bl']:g}  Ma={ref['Ma']:.3f}")
fig.colorbar(im, ax=ax[0])
im2 = ax[1].imshow((U - A) / ref["U0"] * 100, aspect="auto", origin="lower", cmap="viridis",
                   vmin=-1, vmax=1, extent=ext)
ax[1].set_xlabel("t (steps, last period)"); ax[1].set_ylabel(r"$\eta/\delta$")
ax[1].set_title(f"(LBM-analytic)/U0 [%], max={ref['max_err_pct_U0_resolved']:.3f}%")
fig.colorbar(im2, ax=ax[1])
fig.tight_layout(); fig.savefig(ART / "fig3_stokes2_spacetime.png", dpi=130); plt.close(fig)

# ---- fig 4: ablation + sensitivity ----
abl = {c["variant"]: c["max_err_pct_U0_resolved"] for c in S["results"]["stokes2_ablation"]}
abl["ladd_pre_local"] = ref["max_err_pct_U0_resolved"]
order = ["ladd_pre_local", "ladd_pre_unitrho", "eq_wall_node", "ladd_precollision",
         "static_bb", "ladd_sign_flip"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
vals = [max(abl[k], 1e-9) for k in order]
cols = ["#2c7fb8" if k == "ladd_pre_local" else "#999999" for k in order]
bars = ax[0].barh(order[::-1], vals[::-1], color=cols[::-1])
ax[0].set_xscale("log"); ax[0].axvline(15, color="r", ls="--", lw=1, label="15% anchor tol")
ax[0].set_xlabel("max |du|/U0, eta<=3delta [%]"); ax[0].legend(fontsize=8)
ax[0].set_title("wall-closure ablation (Stokes 2nd)")
for b, v in zip(bars, vals[::-1]):
    ax[0].text(v * 1.1, b.get_y() + b.get_height() / 2, f"{v:.3g}%", va="center", fontsize=7)
par = [(r["tag"], r["max_err_pct_U0_resolved"]) for r in S["results"]["stokes2_parameters"]]
res = [(r["tag"], r["max_err_pct_U0_resolved"]) for r in S["results"]["stokes2_resolution"]]
labels = [t.replace("stokes2_", "") for t, _ in par + res]
vals2 = [v for _, v in par + res]
ax[1].barh(labels[::-1], vals2[::-1], color="#7fbc41")
ax[1].set_xscale("log"); ax[1].set_xlabel("max |du|/U0 [%]")
ax[1].axvline(15, color="r", ls="--", lw=1)
ax[1].set_title("parameter points (tau / U0 / resolution)")
for i, v in enumerate(vals2[::-1]):
    ax[1].text(v * 1.1, i, f"{v:.3g}%", va="center", fontsize=7)
fig.tight_layout(); fig.savefig(ART / "fig4_ablation_sensitivity.png", dpi=130); plt.close(fig)
print("figures ->", sorted(p.name for p in ART.glob("fig*.png")))
