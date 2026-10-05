"""W9-B canonical evidence archive — one clean run per exclusion, into out/.

E1 CG3D:   rho-contrast collapse (r=0.5), trace to 2000 steps.
E2 MCMP19: pressure-crush inversion, G=-2.5, contrast 0.65/1.15, trace to 1500.
E3 MCMP19: G-sign scan summary (+2.5 mix/NaN, -2.5 invert), 300 steps each.
E4 AC:     corrected-init light droplet dissolution, W=4 sigma=0, 1000 steps.
E5 D27:    inversion G=-2.5 (+ G=+2.5 NaN), 600 steps.

Library-only kernels; composition scripts (grep self-check: no hand-written kernels).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from tensorlbm.allen_cahn_lbm import _phase_field_equilibrium, allen_cahn_step  # noqa: E402
from tensorlbm.boundaries3d import (  # noqa: E402
    bounce_back_cells_3d,
    make_channel_wall_mask_3d,
    make_tank_wall_mask_3d,
)
from tensorlbm.boundaries_d3q27 import bounce_back_cells_27  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.d3q27 import equilibrium27, stream27_roll  # noqa: E402
from tensorlbm.multiphase3d import (  # noqa: E402
    collide_sc_two_component_3d,
    color_gradient_step_3d,
)
from tensorlbm.multiphase3d_d3q27 import collide_sc_two_component_27  # noqa: E402
from tensorlbm.solver3d import stream3d  # noqa: E402

OUT = Path(__file__).resolve().parent
DEV = torch.device("cuda")
L, A_RAD = 96, 12

z, y, x = torch.meshgrid(*[torch.arange(L, device=DEV, dtype=torch.float32)] * 3, indexing="ij")
DIST = ((x - L / 2 + 0.5) ** 2 + (y - L / 4 + 0.5) ** 2 + (z - L / 2 + 0.5) ** 2).sqrt()
SMOOTH = 0.5 * (1.0 - torch.tanh((DIST - A_RAD) / 3.0))  # 1 inside
EMPTY = torch.zeros((L, L, L), dtype=torch.bool, device=DEV)
WALL = make_channel_wall_mask_3d(L, L, L, EMPTY, DEV) | make_tank_wall_mask_3d(L, L, L, EMPTY, DEV)


def phi_droplet_stats(ra, rb):
    rt, phi = ra + rb, ra - rb
    pos, neg = phi > 0.5 * phi.max(), phi < 0.5 * phi.min()
    if int(pos.sum()) == 0 or int(neg.sum()) == 0:
        return None
    return {
        "rho_in": float(rt[pos].mean()),
        "rho_out": float(rt[neg].mean()),
        "ratio": float(rt[pos].mean() / rt[neg].mean()),
        "r1_ctr": float(ra[L // 2, L // 4, L // 2]),
        "r2_ctr": float(rb[L // 2, L // 4, L // 2]),
    }


def e1_cg():
    fr = equilibrium3d(0.5 * SMOOTH, *(torch.zeros_like(SMOOTH) for _ in range(3)))
    fb = equilibrium3d(1.0 * (1 - SMOOTH), *(torch.zeros_like(SMOOTH) for _ in range(3)))
    trace, t0 = [], time.time()
    for step in range(1, 2001):
        fr, fb = color_gradient_step_3d(fr, fb, tau=1.0, A=0.04, beta=0.7, solid_mask=WALL)
        fr = bounce_back_cells_3d(stream3d(fr), WALL)
        fb = bounce_back_cells_3d(stream3d(fb), WALL)
        if step % 100 == 0:
            d = phi_droplet_stats(fr.sum(0), fb.sum(0))
            d["step"] = step
            trace.append(d)
    print(
        f"E1 CG collapse: ratio {trace[0]['ratio']:.4f} -> {trace[-1]['ratio']:.4f} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return trace


def e2_mcmp(G, n, tag):
    rho1 = 0.65 * SMOOTH + 0.1 * (1 - SMOOTH)
    rho2 = 0.1 * SMOOTH + 1.05 * (1 - SMOOTH)
    zz = torch.zeros_like(rho1)
    f1, f2 = equilibrium3d(rho1, zz, zz, zz), equilibrium3d(rho2, zz, zz, zz)
    trace, nan, t0 = [], None, time.time()
    for step in range(1, n + 1):
        f1, f2 = collide_sc_two_component_3d(f1, f2, G_12=G, tau1=1.0, tau2=1.0, solid_mask=WALL)
        f1 = bounce_back_cells_3d(stream3d(f1), WALL)
        f2 = bounce_back_cells_3d(stream3d(f2), WALL)
        if step % 50 == 0:
            if torch.isnan(f1).any() or torch.isnan(f2).any():
                nan = step
                break
            d = phi_droplet_stats(f1.sum(0), f2.sum(0))
            if d:
                d["step"] = step
                trace.append(d)
    print(
        f"E2 {tag}: nan={nan} last={trace[-1] if trace else None} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return {"G": G, "nan_step": nan, "trace": trace}


def e4_ac():
    phi0 = torch.tanh((DIST - A_RAD) / 2.0)  # -1 inside (light droplet)
    rho = 0.5 + 0.5 * (1.0 + phi0) / 2.0
    zz = torch.zeros_like(rho)
    f = equilibrium3d(rho, zz, zz, zz)
    g = _phase_field_equilibrium(phi0, zz, zz, zz, DEV)
    phi = phi0.clone()
    trace, t0 = [], time.time()
    for step in range(1, 1001):
        f, g, phi = allen_cahn_step(
            f,
            g,
            phi,
            rho_h=1.0,
            rho_l=0.5,
            nu_h=1 / 6,
            nu_l=1 / 6,
            sigma=0.0,
            W=4.0,
            solid_mask=WALL,
        )
        if step % 25 == 0:
            nd = int((phi < -0.5).sum())
            trace.append({"step": step, "vol_lt_-0.5": nd})
    print(
        f"E4 AC dissolution: vol {trace[0]['vol_lt_-0.5']} -> {trace[-1]['vol_lt_-0.5']} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return trace


def e5_d27(G, n, tag):
    rho1 = 0.65 * SMOOTH + 0.1 * (1 - SMOOTH)
    rho2 = 0.1 * SMOOTH + 1.05 * (1 - SMOOTH)
    zz = torch.zeros_like(rho1)
    f1 = equilibrium27(rho1, zz, zz, zz)
    f2 = equilibrium27(rho2, zz, zz, zz)
    trace, nan, t0 = [], None, time.time()
    for step in range(1, n + 1):
        f1, f2 = collide_sc_two_component_27(f1, f2, G_12=G, tau1=1.0, tau2=1.0, solid_mask=WALL)
        f1 = bounce_back_cells_27(stream27_roll(f1), WALL)
        f2 = bounce_back_cells_27(stream27_roll(f2), WALL)
        if step % 50 == 0:
            if torch.isnan(f1).any() or torch.isnan(f2).any():
                nan = step
                break
            d = phi_droplet_stats(f1.sum(0), f2.sum(0))
            if d:
                d["step"] = step
                trace.append(d)
    print(
        f"E5 D27 {tag}: nan={nan} last={trace[-1] if trace else None} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return {"G": G, "nan_step": nan, "trace": trace}


if __name__ == "__main__":
    ev = {
        "meta": {
            "worktree": "bm_w9 @ e717b464",
            "grid": f"{L}^3",
            "a": A_RAD,
            "walls": "channel|tank union, full-way bounce-back",
            "droplet_y0": L / 4,
            "purpose": "W9-B module-capability evidence",
        },
        "E1_cg3d_contrast_collapse": e1_cg(),
        "E2_mcmp19_G+2.5": e2_mcmp(2.5, 300, "G+2.5"),
        "E3_mcmp19_G-2.5": e2_mcmp(-2.5, 1500, "G-2.5"),
        "E4_ac3d_dissolution": e4_ac(),
        "E5_d27_G+2.5": e5_d27(2.5, 300, "G+2.5"),
        "E6_d27_G-2.5": e5_d27(-2.5, 600, "G-2.5"),
    }
    with open(OUT / "evidence.json", "w") as fh:
        json.dump(ev, fh, indent=1)
    print("evidence complete ->", OUT / "evidence.json")
