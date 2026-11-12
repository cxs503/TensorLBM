"""Figures for docs/THERMAL_D12_DEBT_CLOSURE.md — purely from recorded artifacts."""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = pathlib.Path("artifacts/thermal_rb_debt")
outdir = pathlib.Path("figures/thermal_d12")
outdir.mkdir(parents=True, exist_ok=True)

rb = json.load(open(base / "base_dT1.json"))
dhc = json.load(open(base / "dhc_nx48_dT1_before.json"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# --- (a) RB spin-up: Nu vs step, with literature band ------------------------
h = rb["history"]
steps = [x["step"] for x in h]
nu = [x["nu"] for x in h]
umax = [x["u_max"] for x in h]
ax = axes[0]
ax.plot(steps, nu, "o-", color="C0", label="measured Nu (ny=32, Γ=2)")
ax.axhline(2.70, color="C3", ls="--", lw=1.0, label="reference Nu = 2.70")
ax.axhspan(2.2, 2.45, color="C2", alpha=0.12, label="2D RB literature band 2.2–2.45")
ax.axhline(1.0, color="0.5", ls=":", lw=1.0, label="conduction limit Nu = 1")
ax.axvline(5000, color="C1", ls="-.", lw=1.2, label="pre-fix test instant (5000 steps)")
ax.set_xlabel("step")
ax.set_ylabel("Nusselt number Nu")
ax.set_title("(a) RB Ra=1e4: spin-up of Nu (measured)")
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=0.3)

ax2 = ax.twinx()
ax2.semilogy(steps, umax, "s--", color="C4", ms=3, label="u_max")
ax2.set_ylabel("u_max (lattice)", color="C4")

# --- (b) DHC Nu vs step -----------------------------------------------------
h = dhc["history"]
steps = [x["step"] for x in h]
nu = [x["nu"] for x in h]
ax = axes[1]
ax.plot(steps, nu, "o-", color="C0", label="measured Nu (nx=48, grad2)")
ax.axhline(2.243, color="C3", ls="--", lw=1.0, label="de Vahl Davis (1983) 2.243")
ax.axhline(2.243 * 0.9, color="C3", ls=":", lw=0.8)
ax.axhline(2.243 * 1.1, color="C3", ls=":", lw=0.8, label="±10% band")
ax.axvline(3000, color="C1", ls="-.", lw=1.2, label="pre-fix test instant (3000 steps)")
ax.set_xlabel("step")
ax.set_ylabel("Nusselt number Nu")
ax.set_title("(b) DHC Ra=1e4, Pr=0.71: Nu convergence in steps")
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=0.3)

fig.tight_layout()
for name in ("thermal_d12_spinup.png",):
    fig.savefig(outdir / name, dpi=150)
print("wrote", outdir / "thermal_d12_spinup.png")