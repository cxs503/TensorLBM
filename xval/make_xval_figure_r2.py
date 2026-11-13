"""Generate figures for XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S (Round 2).

Produces:
  xval/xval_nebb_bodyforce_r2_profile.png  — ux(y) profile comparison
  xval/xval_nebb_bodyforce_r2_convergence.png — convergence with grid refinement
  xval/xval_figure_b64_r2.json — base64-encoded figures for report embedding
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = Path(__file__).resolve().parent
result = json.loads((out / "xval_nebb_bodyforce_driver_r2.json").read_text())

figures = {}

# Figure 1: ux(y) profile comparison (NY=10)
run_ny10 = result["runs"]["tensorlbm_real_d3q19_ny10"]
y = np.array(run_ny10["y"])
u_got = np.array(run_ny10["u_got"])
u_ref = np.array(run_ny10["u_ref"])
oss_run = result["runs"].get("oss_vivsim_d2q9_ny10", {})
oss_profile = np.array(oss_run.get("ux_profile", []))

fig, ax = plt.subplots(figsize=(7, 4.5))
if len(oss_profile) > 0:
    y_oss = np.arange(len(oss_profile), dtype=float)
    ax.plot(y_oss, oss_profile, "o-", color="tab:orange", lw=1.2, ms=5,
            label=f"OSS vivsim D2Q9 (umax err {oss_run.get('u_max_rel_err', 0):.1%})")
ax.plot(y, u_got, "s-", color="tab:blue", lw=1.5, ms=5,
        label=f"REAL repo D3Q19 (umax err {run_ny10['u_max_rel_err']:.2%})")
ax.plot(y, u_ref, "--", color="k", lw=1.2, label="Analytic Poiseuille")
ax.set_xlabel("y (lattice row)", fontsize=11)
ax.set_ylabel(r"$u_x$", fontsize=11)
ax.set_title("Force-corrected NEBB + body-force driver: REAL repo vs OSS vs analytic", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
png_path = out / "xval_nebb_bodyforce_r2_profile.png"
fig.savefig(png_path, dpi=150)
plt.close(fig)
print(f"wrote {png_path}")

# Encode for report
buf = BytesIO()
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
if len(oss_profile) > 0:
    ax2.plot(np.arange(len(oss_profile), dtype=float), oss_profile, "o-", color="tab:orange", lw=1.2, ms=5, label="OSS vivsim D2Q9")
ax2.plot(y, u_got, "s-", color="tab:blue", lw=1.5, ms=5, label="REAL repo D3Q19")
ax2.plot(y, u_ref, "--", color="k", lw=1.2, label="Analytic")
ax2.set_xlabel("y (lattice row)")
ax2.set_ylabel("u_x")
ax2.set_title("ux(y) profile: REAL repo vs analytic (NY=10)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
fig2.tight_layout()
buf2 = BytesIO()
fig2.savefig(buf2, format="png", dpi=150)
plt.close(fig2)
figures["profile_ny10"] = {
    "name": "xval_nebb_bodyforce_r2_profile.png",
    "caption": "ux(y) profile: REAL repo D3Q19 (0.19% error) vs OSS D2Q9 (5.0%) vs analytic",
    "png_base64": base64.b64encode(buf2.getvalue()).decode("ascii"),
}

# Figure 2: Convergence
run_ny20 = result["runs"]["tensorlbm_real_d3q19_ny20"]
grids = ["NY=10", "NY=20"]
errs = [run_ny10["pointwise_max_rel_err"], run_ny20["pointwise_max_rel_err"]]

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.bar(grids, errs, color=["tab:blue", "tab:green"], width=0.5, edgecolor="k")
ax3.axhline(0.15, color="r", ls="--", lw=1, label="Owner threshold (15%)")
ax3.axhline(0.03, color="orange", ls="--", lw=1, label="Ecosystem threshold (3%)")
ax3.set_ylabel("Pointwise max relative error")
ax3.set_title("Convergence: grid refinement reduces error")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis="y")
fig3.tight_layout()
conv_path = out / "xval_nebb_bodyforce_r2_convergence.png"
fig3.savefig(conv_path, dpi=150)
plt.close(fig3)
print(f"wrote {conv_path}")

buf3 = BytesIO()
fig4, ax4 = plt.subplots(figsize=(6, 4))
ax4.bar(grids, errs, color=["tab:blue", "tab:green"], width=0.5, edgecolor="k")
ax4.axhline(0.15, color="r", ls="--", lw=1, label="15%")
ax4.axhline(0.03, color="orange", ls="--", lw=1, label="3%")
ax4.set_ylabel("Pointwise max rel err")
ax4.set_title("Convergence: NY=10 -> NY=20")
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3, axis="y")
fig4.tight_layout()
buf4 = BytesIO()
fig4.savefig(buf4, format="png", dpi=150)
plt.close(fig4)
figures["convergence"] = {
    "name": "xval_nebb_bodyforce_r2_convergence.png",
    "caption": f"Convergence: error drops from {errs[0]:.2%} (NY=10) to {errs[1]:.3%} (NY=20)",
    "png_base64": base64.b64encode(buf4.getvalue()).decode("ascii"),
}

(out / "xval_figure_b64_r2.json").write_text(json.dumps(figures, indent=2))
print(f"wrote {out / 'xval_figure_b64_r2.json'}")
