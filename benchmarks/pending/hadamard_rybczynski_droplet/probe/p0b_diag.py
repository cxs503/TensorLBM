"""W9-B P0b-diag: isolate NaN source — walls+bounce vs density contrast.

Matrix (all tau=1.0 equal, 800 steps, NaN check every 5, max_u trace):
 a) symmetric 0.7/0.3 swap, walls + bounce
 b) symmetric 0.7/0.3 swap, periodic no bounce
 c) asymmetric (0.8,0.1)/(0.1,1.0), periodic no bounce
 d) asymmetric, walls + bounce
 e) asymmetric no gravity-force at all (G_12=0, pure advection of contrast)
"""

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
L, A_RAD, N_STEPS = 96, 12, 800

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


def max_u(f1, f2):
    _, ux, uy, uz = macroscopic3d(f1 + f2)
    return float((ux**2 + uy**2 + uz**2).sqrt()[~WALL].max())


def run(tag, rho1, rho2, G, walls):
    zero = torch.zeros_like(rho1)
    f1, f2 = equilibrium3d(rho1, zero, zero, zero), equilibrium3d(rho2, zero, zero, zero)
    sm = WALL if walls else None
    nan_step, mu_trace = None, []
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        f1, f2 = collide_sc_two_component_3d(f1, f2, G_12=G, tau1=1.0, tau2=1.0, solid_mask=sm)
        f1 = stream3d(f1)
        f2 = stream3d(f2)
        if walls:
            f1 = bounce_back_cells_3d(f1, WALL)
            f2 = bounce_back_cells_3d(f2, WALL)
        if step % 5 == 0:
            if torch.isnan(f1).any() or torch.isnan(f2).any():
                nan_step = step
                break
            mu_trace.append(round(max_u(f1, f2), 6))
    print(
        f"[{tag}] nan={nan_step} max_u(first5)={mu_trace[:5]} last5={mu_trace[-5:]} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return {"tag": tag, "G": G, "walls": walls, "nan_step": nan_step, "max_u": mu_trace}


if __name__ == "__main__":
    rho1s = 0.7 * SMOOTH + 0.3 * (1 - SMOOTH)
    rho2s = 0.3 * SMOOTH + 0.7 * (1 - SMOOTH)
    rho1a = 0.8 * SMOOTH + 0.1 * (1 - SMOOTH)
    rho2a = 0.1 * SMOOTH + 1.0 * (1 - SMOOTH)
    res = [
        run("a-sym-walls-G5", rho1s, rho2s, 5.0, True),
        run("b-sym-per-G5", rho1s, rho2s, 5.0, False),
        run("c-asym-per-G5", rho1a, rho2a, 5.0, False),
        run("d-asym-walls-G5", rho1a, rho2a, 5.0, True),
        run("e-asym-walls-G0", rho1a, rho2a, 0.0, True),
    ]
    with open(OUT / "p0b_diag.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("diag complete")
