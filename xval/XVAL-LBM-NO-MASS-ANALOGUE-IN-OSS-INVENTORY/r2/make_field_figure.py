"""Field-level evidence for the interior-zeroing defect (R2 figure 3).

Two short arms, identical physics, differing only in the population-zeroing set of
the solid core:

  * ``interior="4"`` -- the library's original 4-neighbour definition (round-1 code):
    solid cells that are only diagonally adjacent to fluid get zeroed, so a zero
    population is injected into the flow at every step;
  * ``interior="8"`` -- the 8-neighbour definition, which cannot reach a fluid node.

Output: ``figures/fig3_interior_zeroing.png`` and ``figures/fig3_interior_zeroing.json``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorlbm.boundary.boundaries import bounce_back_cells
from tensorlbm.boussinesq import apply_guo_forcing
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.oscillatory_cylinder_inertia import circle_mask
from tensorlbm.solver import stream

from oc_added_mass_r2 import interior_nodes

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
D, NX, TAU = 16, 64, 0.654
OMEGA = 2.0 * math.pi / 200.0
A0 = OMEGA * 0.01
NSTEP = 800


def run(mode: str):
    torch.set_num_threads(4)
    mask = circle_mask(NX, NX, D)
    fluid = ~mask
    core = interior_nodes(mask, mode)
    rho0 = torch.ones((NX, NX), dtype=torch.float64)
    zero = torch.zeros_like(rho0)
    f = bounce_back_cells(equilibrium(rho0, zero, zero), mask)
    Fx = torch.zeros((NX, NX), dtype=torch.float64)
    hist = []
    mass0 = None
    for step in range(1, NSTEP + 1):
        ramp = 0.5 * (1.0 - math.cos(math.pi * min(1.0, step / 300.0)))
        Fx[fluid] = A0 * ramp * math.cos(OMEGA * step)
        old = f[:, mask].clone()
        f = apply_guo_forcing(f, torch.stack([Fx, torch.zeros_like(Fx)]), TAU)
        f[:, mask] = old
        f = stream(f)
        rho, ux, uy = macroscopic(f)
        f = bounce_back_cells(f, mask)
        f[:, core] = 0.0
        if step % 10 == 0:
            m = float(rho[fluid].sum())
            mass0 = m if mass0 is None else mass0
            hist.append((step, float((rho[fluid] - 1.0).abs().max()),
                         (m - mass0) / mass0))
    # final snapshot: curl of the velocity and the density error
    vort = np.zeros((NX, NX))
    vort[1:-1, 1:-1] = (np.roll(uy[1:-1, 1:-1], -1, 1) - np.roll(uy[1:-1, 1:-1], 1, 1)) \
        - (np.roll(ux[1:-1, 1:-1], -1, 0) - np.roll(ux[1:-1, 1:-1], 1, 0))
    return {
        "mode": mode,
        "n_zeroed": int(core.sum()),
        "hist": hist,
        "drho_field": (rho - 1.0).numpy(),
        "vort": vort,
        "mask": mask.numpy(),
        "drho_max": hist[-1][1],
        "mass_drift": hist[-1][2],
    }


def main() -> int:
    FIG.mkdir(exist_ok=True)
    out = {m: run(m) for m in ("4", "8")}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    for row, mode in enumerate(("4", "8")):
        d = out[mode]
        h = np.asarray(d["hist"], dtype=float)
        axes[row, 0].plot(h[:, 0], h[:, 1], lw=1.6,
                          color="#c0392b" if mode == "4" else "#1f77b4")
        axes[row, 0].set_ylabel(r"$\max_{\rm fluid}|\rho-1|$")
        axes[row, 0].set_title(f"interior zeroing = {mode}-neighbour  "
                              f"(max $|\\rho-1|$ = {d['drho_max']:.3f}, "
                              f"$\\Delta M/M$ = {d['mass_drift']:+.2e})", fontsize=10)
        axes[row, 0].set_xlabel("step")
        axes[row, 0].grid(alpha=.3)
        vmax = max(abs(out['4']['drho_field']).max(), 1e-6)
        im = axes[row, 1].imshow(d["drho_field"], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[row, 1].contour(d["mask"], levels=[0.5], colors="k", linewidths=0.6)
        axes[row, 1].set_title(r"$\rho-1$ at peak drive", fontsize=10)
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046)
        vv = np.abs(out['8']['vort']).max()
        im2 = axes[row, 2].imshow(d["vort"], cmap="viridis", vmin=-vv, vmax=vv)
        axes[row, 2].contour(d["mask"], levels=[0.5], colors="w", linewidths=0.6)
        axes[row, 2].set_title(r"$\omega_z$ (same run)", fontsize=10)
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046)
        for ax in axes[row, 1:]:
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("R2 fig3 — solid-core population zeroing: 4-neighbour rule injects a "
                 "zero population into the fluid (diagonal leakage)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig3_interior_zeroing.png", dpi=130)
    (FIG / "fig3_interior_zeroing.json").write_text(json.dumps({
        m: {"mode": m, "n_zeroed": out[m]["n_zeroed"], "drho_max": out[m]["drho_max"],
            "mass_drift": out[m]["mass_drift"]} for m in out}, indent=2) + "\n")
    for m in out:
        print(m, out[m]["n_zeroed"], round(out[m]["drho_max"], 4), f"{out[m]['mass_drift']:+.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())