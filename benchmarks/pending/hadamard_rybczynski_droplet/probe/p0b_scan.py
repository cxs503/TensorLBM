"""W9-B P0b-scan: MCMP equal-tau asymmetric-init stability envelope.

Find the largest sustainable rho_tot contrast (droplet/ambient) before NaN.
Grid: (R1, R2) in {(0.8,1.0), (0.7,1.0), (0.55,1.0)} x G in {1.0, 2.5} x use_guo {F,T}.
Inside droplet: (R1, 0.1); outside: (0.1, R2). Seed solubility 0.1 both sides.
1500 steps, log every 50.
"""

from __future__ import annotations

import json
import math
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
L, A_RAD, N_STEPS, LOG_EVERY = 96, 12, 1500, 50

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
    fluid = ~WALL
    pos, neg = phi > 0.5 * phi.max(), phi < 0.5 * phi.min()
    r_eq = (3.0 * pos.sum() / (4.0 * math.pi)) ** (1.0 / 3.0)
    return {
        "rho_in": float(rho_tot[pos].mean()),
        "rho_out": float(rho_tot[neg].mean()),
        "ratio": float(rho_tot[pos].mean() / rho_tot[neg].mean()),
        "R_eq": float(r_eq),
        "max_u": float(u[fluid].max()),
    }


def run(R1, R2, G, guo):
    rho1 = R1 * SMOOTH + 0.1 * (1.0 - SMOOTH)
    rho2 = 0.1 * SMOOTH + R2 * (1.0 - SMOOTH)
    zero = torch.zeros_like(rho1)
    f1, f2 = equilibrium3d(rho1, zero, zero, zero), equilibrium3d(rho2, zero, zero, zero)
    trace, nan_step = [], None
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        f1, f2 = collide_sc_two_component_3d(
            f1, f2, G_12=G, tau1=1.0, tau2=1.0, solid_mask=WALL, use_guo=guo
        )
        f1 = bounce_back_cells_3d(stream3d(f1), WALL)
        f2 = bounce_back_cells_3d(stream3d(f2), WALL)
        if step % LOG_EVERY == 0:
            if torch.isnan(f1).any() or torch.isnan(f2).any():
                nan_step = step
                break
            d = diags(f1, f2)
            d["step"] = step
            trace.append(d)
    tag = f"R1={R1} R2={R2} G={G} guo={guo}"
    last = trace[-1] if trace else {"ratio": float("nan"), "max_u": float("nan")}
    print(
        f"[{tag}] nan_step={nan_step} last_step={trace[-1]['step'] if trace else 0} "
        f"ratio={last['ratio']:.4f} max_u={last['max_u']:.2e} R_eq={last.get('R_eq', -1):.3f} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return {"R1": R1, "R2": R2, "G": G, "guo": guo, "nan_step": nan_step, "trace": trace}


if __name__ == "__main__":
    results = []
    for R1, R2 in [(0.8, 1.0), (0.7, 1.0), (0.55, 1.0)]:
        for G in (1.0, 2.5):
            for guo in (False, True):
                results.append(run(R1, R2, G, guo))
    with open(OUT / "p0b_scan.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print("P0b scan complete")
