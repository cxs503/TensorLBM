"""W9-B P0 probes: static droplets, no gravity.

P0a: Color-Gradient 3D — does a total-density contrast (r=0.5 / r=2) survive?
P0b: SC-MCMP 3D equal-tau — asymmetric init (real rho_tot contrast), G_12 sign/strength.

Composition only: all kernels are library functions.
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
from tensorlbm.multiphase3d import (  # noqa: E402
    collide_sc_two_component_3d,
    color_gradient_step_3d,
)
from tensorlbm.solver3d import stream3d  # noqa: E402

OUT = Path(__file__).resolve().parent
DEV = torch.device("cuda")
L, A_RAD = 96, 12
N_STEPS = 6000
LOG_EVERY = 500


def make_geometry() -> torch.Tensor:
    z, y, x = torch.meshgrid(
        torch.arange(L, device=DEV, dtype=torch.float32),
        torch.arange(L, device=DEV, dtype=torch.float32),
        torch.arange(L, device=DEV, dtype=torch.float32),
        indexing="ij",
    )
    r2 = (x - L / 2 + 0.5) ** 2 + (y - L / 2 + 0.5) ** 2 + (z - L / 2 + 0.5) ** 2
    inside = r2 < A_RAD**2
    smooth = 0.5 * (1.0 - torch.tanh((r2.sqrt() - A_RAD) / 3.0))  # 1 inside
    empty = torch.zeros((L, L, L), dtype=torch.bool, device=DEV)
    wall = make_channel_wall_mask_3d(L, L, L, empty, DEV) | make_tank_wall_mask_3d(
        L, L, L, empty, DEV
    )
    return smooth, inside, wall


def diagnostics(fr, fb, wall):
    rho_r = fr.sum(0)
    rho_b = fb.sum(0)
    rho_tot = rho_r + rho_b
    phi = rho_r - rho_b
    f_tot = fr + fb
    _, ux, uy, uz = macroscopic3d(f_tot)
    u = torch.sqrt(ux**2 + uy**2 + uz**2)
    fluid = ~wall
    phi_pos = phi > 0.5 * phi.max()
    phi_neg = phi < 0.5 * phi.min()
    r_eq = (3.0 * phi_pos.sum() / (4.0 * math.pi)) ** (1.0 / 3.0)
    return {
        "rho_in": float(rho_tot[phi_pos].mean()),
        "rho_out": float(rho_tot[phi_neg].mean()),
        "ratio": float(rho_tot[phi_pos].mean() / rho_tot[phi_neg].mean()),
        "R_eq": float(r_eq),
        "max_u": float(u[fluid].max()),
        "rms_u": float(u[fluid].pow(2).mean().sqrt()),
    }


def run_cg(r_in, tag):
    smooth, inside, wall = make_geometry()
    rho_r = r_in * smooth  # droplet (red) bulk r_in, 0 outside
    rho_b = 1.0 * (1.0 - smooth)  # ambient (blue) bulk 1, 0 inside
    zero = torch.zeros_like(rho_r)
    fr = equilibrium3d(rho_r, zero, zero, zero)
    fb = equilibrium3d(rho_b, zero, zero, zero)
    trace = []
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        fr, fb = color_gradient_step_3d(fr, fb, tau=1.0, A=0.04, beta=0.7, solid_mask=wall)
        fr = stream3d(fr)
        fb = stream3d(fb)
        fr = bounce_back_cells_3d(fr, wall)
        fb = bounce_back_cells_3d(fb, wall)
        if step % LOG_EVERY == 0 or step == 1:
            d = diagnostics(fr, fb, wall)
            d["step"] = step
            trace.append(d)
            print(
                f"[CG {tag}] {step:6d} rho_in={d['rho_in']:.4f} rho_out={d['rho_out']:.4f} "
                f"ratio={d['ratio']:.4f} R_eq={d['R_eq']:.3f} max_u={d['max_u']:.2e} "
                f"rms_u={d['rms_u']:.2e} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    return {"model": "cg", "r_in": r_in, "trace": trace}


def run_mcmp(rho1_in, rho2_in, rho1_out, rho2_out, G12, tag):
    smooth, inside, wall = make_geometry()
    rho1 = rho1_in * smooth + rho1_out * (1.0 - smooth)
    rho2 = rho2_in * smooth + rho2_out * (1.0 - smooth)
    zero = torch.zeros_like(rho1)
    f1 = equilibrium3d(rho1, zero, zero, zero)
    f2 = equilibrium3d(rho2, zero, zero, zero)
    trace = []
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        f1, f2 = collide_sc_two_component_3d(f1, f2, G_12=G12, tau1=1.0, tau2=1.0, solid_mask=wall)
        f1 = stream3d(f1)
        f2 = stream3d(f2)
        f1 = bounce_back_cells_3d(f1, wall)
        f2 = bounce_back_cells_3d(f2, wall)
        if step % LOG_EVERY == 0 or step == 1:
            d = diagnostics(f1, f2, wall)
            d["step"] = step
            trace.append(d)
            print(
                f"[MC {tag}] {step:6d} rho_in={d['rho_in']:.4f} rho_out={d['rho_out']:.4f} "
                f"ratio={d['ratio']:.4f} R_eq={d['R_eq']:.3f} max_u={d['max_u']:.2e} "
                f"rms_u={d['rms_u']:.2e} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    return {
        "model": "mcmp",
        "G12": G12,
        "in": (rho1_in, rho2_in),
        "out": (rho1_out, rho2_out),
        "trace": trace,
    }


if __name__ == "__main__":
    results = {}
    # P0a CG: density contrast survival
    results["cg_r0.5"] = run_cg(0.5, "r0.5")
    results["cg_r2.0"] = run_cg(2.0, "r2.0")
    # P0b MCMP asymmetric: contrast 0.55/1.05 ~= 0.524, G sign/strength
    results["mc_G+2.5"] = run_mcmp(0.5, 0.05, 0.05, 1.0, +2.5, "G+2.5")
    results["mc_G+5.0"] = run_mcmp(0.5, 0.05, 0.05, 1.0, +5.0, "G+5.0")
    with open(OUT / "p0_static.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print("P0 complete ->", OUT / "p0_static.json")
