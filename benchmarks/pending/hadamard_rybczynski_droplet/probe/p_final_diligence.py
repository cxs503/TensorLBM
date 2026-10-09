"""W9-B final due-diligence: (A) AC dissolution vs W/sigma sensitivity,
(B) D3Q27 MCMP contrast inversion (family-wide evidence), both signs of G."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from tensorlbm.allen_cahn_lbm import _phase_field_equilibrium, allen_cahn_step  # noqa: E402
from tensorlbm.boundaries3d import (  # noqa: E402
    make_channel_wall_mask_3d,
    make_tank_wall_mask_3d,
)
from tensorlbm.boundaries_d3q27 import bounce_back_cells_27  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.d3q27 import equilibrium27, stream27_roll  # noqa: E402
from tensorlbm.multiphase3d_d3q27 import collide_sc_two_component_27  # noqa: E402

OUT = str(Path(__file__).resolve().parent / "p_final_diligence.json")
DEV = torch.device("cuda")
L, A_RAD = 96, 12

z, y, x = torch.meshgrid(*[torch.arange(L, device=DEV, dtype=torch.float32)] * 3, indexing="ij")
r2 = (x - L / 2 + 0.5) ** 2 + (y - L / 4 + 0.5) ** 2 + (z - L / 2 + 0.5) ** 2
DIST = r2.sqrt()
EMPTY = torch.zeros((L, L, L), dtype=torch.bool, device=DEV)
WALL = make_channel_wall_mask_3d(L, L, L, EMPTY, DEV) | make_tank_wall_mask_3d(L, L, L, EMPTY, DEV)


def ac_case(w_int, sigma, n_steps=1000):
    phi0 = torch.tanh((DIST - A_RAD) / (w_int / 2))
    rho = 0.5 + 0.5 * (1.0 + phi0) / 2.0
    zz = torch.zeros_like(rho)
    f = equilibrium3d(rho, zz, zz, zz)
    g = _phase_field_equilibrium(phi0, zz, zz, zz, DEV)
    phi = phi0.clone()
    t0 = time.time()
    vols = []
    for step in range(1, n_steps + 1):
        f, g, phi = allen_cahn_step(
            f,
            g,
            phi,
            rho_h=1.0,
            rho_l=0.5,
            nu_h=1 / 6,
            nu_l=1 / 6,
            sigma=sigma,
            W=w_int,
            solid_mask=WALL,
        )
        if step % 100 == 0:
            nd = int((phi < -0.5).sum())
            vols.append(nd)
            if torch.isnan(phi).any():
                break
    print(
        f"[AC W={w_int} sig={sigma}] phi<-0.5 volume every100: {vols} "
        f"(init~{int((phi0 < -0.5).sum())}) ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return vols


def d27_case(G, n_steps=1500):
    """D3Q27 MCMP: light droplet (c1=0.75 total) in heavy ambient (c2=1.15)."""
    seed = 0.1
    s = 0.5 * (1.0 - torch.tanh((DIST - A_RAD) / 3.0))  # 1 inside
    rho1 = 0.65 * s + seed * (1 - s)
    rho2 = seed * s + 1.05 * (1 - s)
    zz = torch.zeros_like(rho1)
    f1 = equilibrium27(rho1, zz, zz, zz)
    f2 = equilibrium27(rho2, zz, zz, zz)
    t0 = time.time()
    nan, ctr = None, []
    for step in range(1, n_steps + 1):
        f1, f2 = collide_sc_two_component_27(f1, f2, G_12=G, tau1=1.0, tau2=1.0, solid_mask=WALL)
        # stream 27 via solver? use module's own stream if present; else roll by C
        f1 = bounce_back_cells_27(stream27_roll(f1), WALL)
        f2 = bounce_back_cells_27(stream27_roll(f2), WALL)
        if step % 250 == 0:
            if torch.isnan(f1).any() or torch.isnan(f2).any():
                nan = step
                break
            ra, rb = f1.sum(0), f2.sum(0)
            ctr.append(
                {
                    "step": step,
                    "ratio": float(
                        (ra + rb)[ra - rb > 0.5 * (ra - rb).max()].mean()
                        / (ra + rb)[ra - rb < 0.5 * (ra - rb).min()].mean()
                    ),
                    "r1_ctr": float(ra[L // 2, L // 4, L // 2]),
                    "r2_ctr": float(rb[L // 2, L // 4, L // 2]),
                }
            )
            print(f"[D27 G={G}] {step} {ctr[-1]}", flush=True)
    print(f"[D27 G={G}] nan={nan} ({time.time() - t0:.0f}s)", flush=True)
    return {"G": G, "nan": nan, "trace": ctr}


if __name__ == "__main__":
    res = {"ac": {}, "d27": []}
    for w in (2.0, 8.0):
        for sig in (0.0, 0.01):
            res["ac"][f"W{w}_sig{sig}"] = ac_case(w, sig)
    for G in (+2.5, -2.5):
        res["d27"].append(d27_case(G))
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("diligence complete")
