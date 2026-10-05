"""W9-B P0b-Gsweep: find stable G window for 3D MCMP with walls, sym vs asym."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from tensorlbm.boundaries3d import (  # noqa: E402
    bounce_back_cells_3d,
    make_channel_wall_mask_3d,
    make_tank_wall_mask_3d,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.multiphase3d import collide_sc_two_component_3d  # noqa: E402
from tensorlbm.solver3d import stream3d  # noqa: E402

OUT = Path(__file__).resolve().parent
DEV = torch.device("cuda")
L, A_RAD, N_STEPS = 96, 12, 1500

z, y, x = torch.meshgrid(
    torch.arange(L, device=DEV, dtype=torch.float32),
    torch.arange(L, device=DEV, dtype=torch.float32),
    torch.arange(L, device=DEV, dtype=torch.float32),
    indexing="ij",
)
r2 = (x - L / 2 + 0.5) ** 2 + (y - L / 2 + 0.5) ** 2 + (z - L / 2 + 0.5) ** 2
SMOOTH = 0.5 * (1.0 - torch.tanh((r2.sqrt() - A_RAD) / 3.0))
EMPTY = torch.zeros((L, L, L), dtype=torch.bool, device=DEV)
WALL = make_channel_wall_mask_3d(L, L, L, EMPTY, DEV) | make_tank_wall_mask_3d(L, L, L, EMPTY, DEV)


def diags(f1, f2):
    rho1, rho2 = f1.sum(0), f2.sum(0)
    rho_tot = rho1 + rho2
    phi = rho1 - rho2
    _, ux, uy, uz = macroscopic3d(f1 + f2)
    u = (ux**2 + uy**2 + uz**2).sqrt()
    pos, neg = phi > 0.5 * phi.max(), phi < 0.5 * phi.min()
    # mutual solubility: dissolved comp2 inside droplet bulk / comp1 there
    r1_in = float(rho1[pos].mean())
    r2_in = float(rho2[pos].mean())
    return {
        "rho_in": float(rho_tot[pos].mean()),
        "rho_out": float(rho_tot[neg].mean()),
        "ratio": float(rho_tot[pos].mean() / rho_tot[neg].mean()),
        "r2_in": r2_in,
        "dissolved_frac": r2_in / (r1_in + r2_in),
        "max_u": float(u[~WALL].max()),
    }


def run(tag, rho1, rho2, G):
    zero = torch.zeros_like(rho1)
    f1, f2 = equilibrium3d(rho1, zero, zero, zero), equilibrium3d(rho2, zero, zero, zero)
    nan_step, last = None, {}
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        f1, f2 = collide_sc_two_component_3d(f1, f2, G_12=G, tau1=1.0, tau2=1.0, solid_mask=WALL)
        f1 = bounce_back_cells_3d(stream3d(f1), WALL)
        f2 = bounce_back_cells_3d(stream3d(f2), WALL)
        if step % 25 == 0:
            if torch.isnan(f1).any() or torch.isnan(f2).any():
                nan_step = step
                break
            last = diags(f1, f2)
    print(
        f"[{tag} G={G}] nan={nan_step} ratio={last.get('ratio', float('nan')):.4f} "
        f"diss={last.get('dissolved_frac', float('nan')):.3f} "
        f"max_u={last.get('max_u', float('nan')):.2e} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return {"tag": tag, "G": G, "nan_step": nan_step, "final": last}


if __name__ == "__main__":
    rho1s = 0.7 * SMOOTH + 0.3 * (1 - SMOOTH)
    rho2s = 0.3 * SMOOTH + 0.7 * (1 - SMOOTH)
    rho1a = 0.8 * SMOOTH + 0.1 * (1 - SMOOTH)
    rho2a = 0.1 * SMOOTH + 1.0 * (1 - SMOOTH)
    res = []
    for G in (0.3, 0.6, 0.9, 1.2, 1.5):
        res.append(run("sym", rho1s, rho2s, G))
    for G in (0.3, 0.6, 0.9, 1.2, 1.5):
        res.append(run("asym", rho1a, rho2a, G))
    with open(OUT / "p0b_gsweep.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("Gsweep complete")
