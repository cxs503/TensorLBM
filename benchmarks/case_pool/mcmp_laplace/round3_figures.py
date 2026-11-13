"""Render CASE-MCMP-LAPLACE round-3 figures from a fresh hero-slab run.

Writes artifacts/fig_r3_normal_stress.png (P_NN/P_TT profiles, sym vs closed)
and artifacts/fig_r3_sigma_estimators.png (three sigma estimators vs Laplace).
Numbers are read from artifacts/round3_closed_virial.json (zero hand-typed).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.mcmp_verification import estimate_coexistence  # noqa: E402
from tensorlbm.multiphase import collide_sc_two_component  # noqa: E402
from tensorlbm.scmc_pressure_tensor import scmc_pressure_tensor  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
doc = json.loads((ART / "round3_closed_virial.json").read_text())
G = -2.75
slab = doc["slabs"][f"{G}"]
drop = doc["droplet"]

torch.set_num_threads(1)
coex = estimate_coexistence(G_12=G, n=96, steps=3000)
r1l, r1g = coex["rho1_liq"], coex["rho1_gas"]
r2l, r2g = coex["rho2_liq"], coex["rho2_gas"]
NY, NX, HALF, W0, STEPS = 96, 8, 24.0, 2.0, 4000
ys = torch.arange(NY, dtype=torch.float32).view(NY, 1)
c = NY / 2.0
s_in = (
    0.5 * (1.0 + torch.tanh((ys - (c - HALF)) / W0))
    * 0.5 * (1.0 - torch.tanh((ys - (c + HALF)) / W0))
).expand(NY, NX).clone()
rho1 = r1g + s_in * (r1l - r1g)
rho2 = r2g + s_in * (r2l - r2g)
z = torch.zeros_like(rho1)
f1 = equilibrium(rho1, z, z)
f2 = equilibrium(rho2, z, z)
for _ in range(STEPS):
    f1, f2 = collide_sc_two_component(f1, f2, G_12=G, tau1=1.0, tau2=1.0)
    f1 = stream(f1)
    f2 = stream(f2)

prof = {}
for tag in ("sym", "closed"):
    Pxx, Pyy, _ = scmc_pressure_tensor(f1, f2, G, pairing=tag)
    prof[tag] = {"pn": Pyy.mean(1).numpy(), "pt": Pxx.mean(1).numpy()}

ysg = torch.arange(NY).float().numpy()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for tag, col in (("sym", "tab:red"), ("closed", "tab:blue")):
    ax[0].plot(prof[tag]["pn"], ysg, color=col, lw=1.6, label=f"P_NN {tag}")
    ax[0].plot(prof[tag]["pt"], ysg, color=col, ls="--", lw=1.2, label=f"P_TT {tag}")
ax[0].set_xlabel("pressure (lattice units)")
ax[0].set_ylabel("y (lattice units)")
ax[0].set_title(f"flat slab G12={G}: normal/tangential stress")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[0].invert_yaxis()

names = ["σ_Laplace\n(Δp_EOS·R, droplet)", "σ_mech_closed\n(Kirkwood–Buff)", "σ_EOS\n(profile integral)"]
vals = [drop["sigma_laplace"], slab["sigma_mech_closed"], slab["sigma_eos"]]
bars = ax[1].bar(names, vals, color=["tab:green", "tab:blue", "tab:orange"])
for b, v in zip(bars, vals):
    ax[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
ax[1].axhline(drop["sigma_laplace"] * 1.05, color="g", ls=":", lw=1)
ax[1].axhline(drop["sigma_laplace"] * 0.95, color="g", ls=":", lw=1)
ax[1].text(2.4, drop["sigma_laplace"] * 1.05, "±5% band", fontsize=8, color="g", va="bottom")
ax[1].set_ylabel("σ (lattice units)")
ax[1].set_title("three estimators, none within 5% of the droplet")
ax[1].grid(alpha=0.3, axis="y")
fig.tight_layout()
ART.mkdir(exist_ok=True)
fig.savefig(ART / "fig_r3_normal_stress.png", dpi=130)
plt.close(fig)

fig2, ax2 = plt.subplots(figsize=(6, 4))
bars2 = ax2.bar(names, vals, color=["tab:green", "tab:blue", "tab:orange"])
for b, v in zip(bars2, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
ax2.axhline(drop["sigma_laplace"] * 1.05, color="g", ls=":", lw=1)
ax2.axhline(drop["sigma_laplace"] * 0.95, color="g", ls=":", lw=1)
ax2.text(2.35, drop["sigma_laplace"] * 1.05, "±5% band", fontsize=8, color="g", va="bottom")
ax2.set_ylabel("σ (lattice units)")
ax2.set_title("three estimators, none within 5% of the droplet")
ax2.grid(alpha=0.3, axis="y")
fig2.tight_layout()
fig2.savefig(ART / "fig_r3_sigma_estimators.png", dpi=130)
plt.close(fig2)
print("figures written")
