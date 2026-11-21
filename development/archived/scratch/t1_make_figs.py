"""T1 report figures: phi field at death step + gate trajectories (baseline vs prescribed, n=8)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "t1_figs"
OUT.mkdir(exist_ok=True)

phi = np.load(ROOT / "runs" / "t1_phi_n8_step26.npy")
base = json.loads((ROOT / "runs" / "t1_base_n8.json").read_text())
pres = json.loads((ROOT / "runs" / "t1_cac_w6_s03_n8.json").read_text())
smoke12 = json.loads((ROOT / "runs" / "t1_smoke_n12.json").read_text())

# --- Fig 1: phi field at the death step (prescribed w6/s0.03, n=8, step 26) ---
mid = phi[:, phi.shape[1] // 2, :]
fig, ax = plt.subplots(figsize=(5.2, 6.0))
cs = ax.contourf(mid, levels=20, cmap="viridis")
ax.contour(mid, levels=[0.5], colors="w", linewidths=1.2)
ax.set_title(r"$\phi$ mid-plane, n=8, w=6, s=0.03, step 26 (0.158 ms, pre-contact)")
ax.set_xlabel("y / cells")
ax.set_ylabel("z / cells")
fig.colorbar(cs, ax=ax, label="order parameter phi")
fig.tight_layout()
fig.savefig(OUT / "fig1_phi_field.png", dpi=150)
plt.close(fig)

# --- Fig 2: drift & mach trajectories vs gates ---
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
for r, lab in ((base, "baseline w=4, s=0.04"), (pres, "prescribed w=6, s=0.03")):
    t = [row["time_ms"] for row in r["history"]]
    a1.plot(t, [row["volume_drift"] for row in r["history"]], marker=".", label=lab)
    a2.plot(t, [row["mach_max"] for row in r["history"]], marker=".", label=lab)
a1.axhline(1e-4, color="r", ls="--", lw=1, label="gate 1e-4")
a1.set_yscale("log")
a1.set_xlabel("time / ms")
a1.set_ylabel("volume_drift (phi sum)")
a1.set_title("volume_drift (n=8)")
a1.legend(fontsize=8)
a2.axhline(0.3, color="r", ls="--", lw=1, label="gate 0.3")
a2.set_xlabel("time / ms")
a2.set_ylabel("mach_max")
a2.set_title("mach_max (n=8)")
a2.legend(fontsize=8)
fig.suptitle("n=8: baseline dies on mach_limit (0.065 ms); prescribed dies on volume_drift (0.158 ms)")
fig.tight_layout()
fig.savefig(OUT / "fig2_gate_trajectories.png", dpi=150)
plt.close(fig)

# --- Fig 3: drift creep rate vs resolution (prescribed config) ---
rate8 = pres["history"][-1]["volume_drift"] / pres["history"][-1]["time_ms"]
rate12 = smoke12["history"][-1]["volume_drift"] / smoke12["history"][-1]["time_ms"]
fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.bar(["n=8", "n=12"], [rate8, rate12], color=["#d62728", "#1f77b4"])
ax.set_yscale("log")
ax.set_ylabel("drift creep rate / ms$^{-1}$")
ax.set_title("phi-sum drift rate, prescribed w=6 s=0.03")
for i, v in enumerate((rate8, rate12)):
    ax.text(i, v * 1.15, f"{v:.1e}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig3_drift_rate_vs_resolution.png", dpi=150)
plt.close(fig)
print("figures written to", OUT)
