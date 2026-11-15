"""Campaign: Zou/He-driven 3D plane channel vs the analytic Poiseuille parabola.

Runs the resolution ladder of the pressure/pressure closure pair plus one
parabolic-inflow arm, writes ``artifacts/zou_he_channel3d/metrics.json`` and
the report figures.  Every number in the report is read back from that JSON.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.verification.zou_he_channel3d import simulate_channel_3d

OUT = pathlib.Path("artifacts/zou_he_channel3d")
OUT.mkdir(parents=True, exist_ok=True)
FIG = pathlib.Path("xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/figures")
FIG.mkdir(parents=True, exist_ok=True)

runs = []
for ny, tag in ((10, "H8"), (18, "H16")):
    runs.append((tag, simulate_channel_3d(nx=48, ny=ny, nz=2, steps=6000, block=800)))
runs.append(("H16-velocity-inlet",
             simulate_channel_3d(nx=48, ny=18, nz=2, steps=6000, block=800,
                                 inlet="velocity-profile")))

summary = {"arms": {tag: r.as_dict() for tag, r in runs}}
(OUT / "metrics.json").write_text(json.dumps(summary, indent=2))

# fig1: profiles vs analytic
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for tag, r in runs:
    yt = np.arange(1, r.ny - 1) - 0.5
    ax[0].plot(r.u_profile, yt, "o-", ms=3, label=f"{tag} (sim)")
    ax[0].plot(r.u_exact, yt, "k--", lw=1, label=f"{tag} (analytic)")
    ax[1].semilogy(yt, np.abs(r.u_profile - r.u_exact) / r.u_exact.max(), "o-", ms=3, label=tag)
ax[0].set_xlabel("$u_x$ [lattice]"); ax[0].set_ylabel("$y'$ [lattice]")
ax[0].set_title("Zou/He-driven 3D channel vs plane Poiseuille"); ax[0].legend(fontsize=7)
ax[1].axhspan(0, 0.03, color="g", alpha=0.08, label="3% gate")
ax[1].set_xlabel("$y'$ [lattice]"); ax[1].set_ylabel("$|u-u_{ex}|/u_{ex,max}$")
ax[1].legend(fontsize=7); ax[1].set_title("pointwise error")
fig.tight_layout(); fig.savefig(FIG / "fig1_profiles.png", dpi=130); plt.close(fig)

# fig2: convergence trace + ladder errors
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for tag, r in runs:
    ax[0].plot(r.trace["step"], r.trace["u_peak"], "o-", ms=3, label=tag)
ax[0].set_xlabel("time step"); ax[0].set_ylabel("mid-channel $u_{x,max}$")
ax[0].set_title("block-averaged convergence"); ax[0].legend(fontsize=7)
hs = [r.H for _, r in runs[:2]]
errs = [r.metrics["linf_rel_err"] for _, r in runs[:2]]
ax[1].loglog(hs, errs, "s-", label="measured $L_\\infty$ err")
ax[1].loglog(hs, [e * (hs[0] / h) ** 2 for e, h in zip(errs, hs)], "k--", lw=1,
             label="2nd-order reference")
ax[1].axhline(0.03, color="g", lw=1, label="3% gate")
ax[1].set_xlabel("channel width $H$"); ax[1].set_ylabel("$\\|u-u_{ex}\\|_\\infty/u_{ex,max}$")
ax[1].legend(fontsize=7); ax[1].set_title("resolution ladder")
fig.tight_layout(); fig.savefig(FIG / "fig2_convergence.png", dpi=130); plt.close(fig)

for tag, r in runs:
    m = r.metrics
    print(tag, {k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()})
