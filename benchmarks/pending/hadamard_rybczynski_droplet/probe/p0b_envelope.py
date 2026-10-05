"""W9-B P0b-envelope: sustainable rho_tot contrast for 3D MCMP, G<0, floors>=0.1.

Droplet (comp1-rich) small: R=12 in 96^3. in/out totals set by bulk values.
Success = no NaN to 4000 steps + steady contrast (ratio retention >90%) + segregated.
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

OUT = str(Path(__file__).resolve().parent / "p0b_envelope.json")
DEV = torch.device("cuda")
L, A_RAD, N = 96, 12, 4000

z, y, x = torch.meshgrid(*[torch.arange(L, device=DEV, dtype=torch.float32)] * 3, indexing="ij")
r2 = (x - L / 2 + 0.5) ** 2 + (y - L / 2 + 0.5) ** 2 + (z - L / 2 + 0.5) ** 2
S = 0.5 * (1.0 - torch.tanh((r2.sqrt() - A_RAD) / 3.0))
E = torch.zeros((L, L, L), dtype=torch.bool, device=DEV)
WALL = make_channel_wall_mask_3d(L, L, L, E, DEV) | make_tank_wall_mask_3d(L, L, L, E, DEV)


def run(c1, c2, G, seed=0.1):
    """c1, c2 = desired bulk totals inside/outside; floors=seed=0.1."""
    r1_in = c1 - seed
    r2_out = c2 - seed
    rho1 = r1_in * S + seed * (1 - S)
    rho2 = seed * S + r2_out * (1 - S)
    zz = torch.zeros_like(rho1)
    f1, f2 = equilibrium3d(rho1, zz, zz, zz), equilibrium3d(rho2, zz, zz, zz)
    nan, hist = None, []
    t0 = time.time()
    for step in range(1, N + 1):
        f1, f2 = collide_sc_two_component_3d(f1, f2, G_12=G, tau1=1.0, tau2=1.0, solid_mask=WALL)
        f1 = bounce_back_cells_3d(stream3d(f1), WALL)
        f2 = bounce_back_cells_3d(stream3d(f2), WALL)
        if step % 250 == 0:
            if torch.isnan(f1).any() or torch.isnan(f2).any():
                nan = step
                break
            ra, rb = f1.sum(0), f2.sum(0)
            rt, phi = ra + rb, ra - rb
            pos, neg = phi > 0.5 * phi.max(), phi < 0.5 * phi.min()
            _, ux, uy, uz = macroscopic3d(f1 + f2)
            u = (ux**2 + uy**2 + uz**2).sqrt()
            hist.append(
                {
                    "step": step,
                    "ratio": float(rt[pos].mean() / rt[neg].mean()),
                    "diss": float(rb[pos].mean() / rt[pos].mean()),
                    "Req": float((3 * pos.sum().float() / (4 * 3.14159265)) ** (1 / 3)),
                    "maxu": float(u[~WALL].max()),
                }
            )
    last = hist[-1] if hist else {}
    print(
        f"[in={c1} out={c2} G={G}] nan={nan} ratio={last.get('ratio', float('nan')):.4f} "
        f"diss={last.get('diss', float('nan')):.3f} Req={last.get('Req', -1):.2f} "
        f"maxu={last.get('maxu', float('nan')):.2e} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return {"c1": c1, "c2": c2, "G": G, "nan": nan, "hist": hist}


if __name__ == "__main__":
    res = []
    for c1, c2 in [(0.9, 1.1), (0.85, 1.15), (0.75, 1.15), (0.65, 1.15), (0.55, 1.15)]:
        for G in (-1.0, -1.5, -2.5):
            res.append(run(c1, c2, G))
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("envelope complete")
