#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B2-D3Q27: sphere Re=100 drag — D3Q27 high-precision collision benchmark.

Library path (GeneralSimEngine D3Q27 enum is dead code — setup hard-codes
d3q19.equilibrium3d, collision selection returns only D3Q19 collide_*3d,
macroscopic uses d3q19.macroscopic3d; see /tmp/d3q27_gap.md):

  collide : tensorlbm.cumulant.collide_cumulant_geier_d3q27  (or d3q27.collide_mrt27)
  stream  : tensorlbm.d3q27.stream27  (solver3d.stream3d is D3Q19-hard-coded: q=19,
            d3q19 C vectors — cannot carry 27 populations)
  BC      : tensorlbm.boundaries_d3q27.far_field_bc_27 (free-stream feq on inlet +
            y±/z± lateral faces, zero-gradient outlet, then bounce_back_cells_27)
  force   : tensorlbm.obstacles.compute_obstacle_forces_27 (Ladd MEM, after
            streaming, before bounce-back) — Cd = Fx / (0.5*U^2*pi*R^2)

No hand-written collide/stream/equilibrium in this script (grep-checkable).
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import torch

from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.boundaries_d3q27 import far_field_bc_27
from tensorlbm.cumulant import collide_cumulant_geier_d3q27
from tensorlbm.d3q27 import (
    collide_mrt27,
    correct_mass27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from tensorlbm.obstacles import compute_obstacle_forces_27


def schiller_naumann_cd(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def clift_gauvin_cd(re: float) -> float:
    return 24.0 / re * (1.0 + 0.1315 * re ** (0.82 - 0.05 * math.log10(re)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diam", type=int, default=40, help="sphere diameter in cells (D)")
    ap.add_argument("--collision", choices=["cumulant_geier", "mrt27"], default="cumulant_geier")
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--outdir", type=str, default="/tmp/d3q27sphere")
    ap.add_argument("--u_in", type=float, default=0.06)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--lateral_ratio",
        type=float,
        default=4.5,
        help="lateral domain extent in units of D (ny=nz=lateral_ratio*D)",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile collide+stream (GPU memory reduction + speed)",
    )
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    D = args.diam
    R = D / 2.0
    U = args.u_in
    re = 100.0
    nu = U * D / re
    tau = 3.0 * nu + 0.5

    # Domain: >=10D streamwise (4D upstream / 6D downstream), lateral_ratio*D lateral
    nx = 10 * D
    ny = int(round(args.lateral_ratio * D))
    nz = int(round(args.lateral_ratio * D))
    cx = 4.0 * D
    cy = args.lateral_ratio * D / 2.0
    cz = args.lateral_ratio * D / 2.0

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    logf = open(os.path.join(outdir, "cd_history.csv"), "w")
    logf.write("step,fx,fy,fz,cd,cd_effd,rho_mean,max_u\n")

    device = torch.device(args.device)
    torch.manual_seed(0)

    obstacle = sphere_mask(nx, ny, nz, cx, cy, cz, R, device=device)  # (nz,ny,nx) bool
    n_solid = int(obstacle.sum().item())

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), U, device=device)
    uy0 = torch.zeros((nz, ny, nx), device=device)
    uz0 = torch.zeros((nz, ny, nx), device=device)
    ux0[obstacle] = 0.0
    f = equilibrium27(rho0, ux0, uy0, uz0, device=device)

    target_mass = float(rho0.sum().item())
    ref_area = math.pi * R * R
    dyn_pressure = 0.5 * U * U * ref_area

    if args.collision == "cumulant_geier":
        collide = collide_cumulant_geier_d3q27
    else:
        collide = collide_mrt27

    if args.compile:
        print("torch.compile: fusing collide + stream ...", flush=True)
        collide = torch.compile(collide)
        stream_fn = torch.compile(stream27)
    else:
        stream_fn = stream27

    print(
        f"=== D3Q27 sphere Re=100 | D={D} grid {nx}x{ny}x{nz} | collision={args.collision} | "
        f"tau={tau:.4f} nu={nu:.5f} | U={U} | n_solid={n_solid} ===",
        flush=True,
    )
    print(
        f"ref: Cd_SN(100)={schiller_naumann_cd(100):.4f}  Cd_CG(100)={clift_gauvin_cd(100):.4f}",
        flush=True,
    )

    t0 = time.time()
    t_last = t0
    cd_hist = []
    for step in range(1, args.steps + 1):
        f = collide(f, tau=tau)
        f = stream_fn(f)
        fx, fy, fz = compute_obstacle_forces_27(f, obstacle)  # MEM, pre bounce-back
        f = far_field_bc_27(f, U, obstacle_mask=obstacle)
        if step % 500 == 0:
            f = correct_mass27(f, target_mass)

        cd = float(fx.item()) / dyn_pressure
        cd_effd = float(fx.item()) / (0.5 * U * U * math.pi * (R + 0.5) ** 2)

        if step % 25 == 0 or step == args.steps:
            rho, ux, uy, uz = macroscopic27(f)
            rho_m = float(rho.mean().item())
            ux = ux.masked_fill(obstacle, 0.0)
            uy = uy.masked_fill(obstacle, 0.0)
            uz = uz.masked_fill(obstacle, 0.0)
            umax = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())
            logf.write(
                f"{step},{fx.item():.8e},{fy.item():.8e},{fz.item():.8e},"
                f"{cd:.6f},{cd_effd:.6f},{rho_m:.6f},{umax:.6f}\n"
            )
            logf.flush()
            if step % 500 == 0 or step == args.steps:
                el = time.time() - t_last
                t_last = time.time()
                print(
                    f"step={step:6d} Cd={cd:.4f} Cd_effD={cd_effd:.4f} rho={rho_m:.5f} "
                    f"u_max={umax:.5f}  [{el:.1f}s]",
                    flush=True,
                )
        cd_hist.append((step, cd, cd_effd))

    total = time.time() - t0
    logf.close()

    # ---- steady-state block analysis ----
    hist = cd_hist
    block = 2000
    blocks = []
    i = 0
    while i + block <= len(hist):
        seg = [c for _, c, _ in hist[i : i + block]]
        blocks.append((hist[i][0], sum(seg) / len(seg)))
        i += block
    # final steady value: mean over the last two full blocks if drift small
    if len(blocks) >= 2:
        cd_last = blocks[-1][1]
        cd_prev = blocks[-2][1]
        drift_pct = abs(cd_last - cd_prev) / cd_prev * 100.0
        window = 2 * block
    else:
        cd_last = blocks[-1][1]
        cd_prev = blocks[-1][1]
        drift_pct = 0.0
        window = block
    cd_final = sum(c for _, c, _ in hist[-window:]) / window
    cd_effd_final = sum(e for _, _, e in hist[-window:]) / window

    cd_ref_sn = schiller_naumann_cd(re)
    cd_ref_cg = clift_gauvin_cd(re)
    err_sn = (cd_final - cd_ref_sn) / cd_ref_sn * 100.0
    err_cg = (cd_final - cd_ref_cg) / cd_ref_cg * 100.0

    result = {
        "case": "sphere_re100_d3q27",
        "lattice": "D3Q27",
        "collision": args.collision,
        "diameter_cells": D,
        "grid": [nx, ny, nz],
        "sphere_center": [cx, cy, cz],
        "u_in": U,
        "re": re,
        "nu": nu,
        "tau": tau,
        "n_steps": args.steps,
        "ref_area": ref_area,
        "cd_nominal": cd_final,
        "cd_effD": cd_effd,
        "cd_ref_sn": cd_ref_sn,
        "err_pct_sn": err_sn,
        "cd_ref_cg": cd_ref_cg,
        "err_pct_cg": err_cg,
        "steady_drift_pct": drift_pct,
        "steady_window": window,
        "block_means": [[s, b] for s, b in blocks],
        "wall_seconds": total,
        "threads": args.threads,
        "n_solid": n_solid,
    }
    with open(os.path.join(outdir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(
        f"\nCd_final = {cd_final:.4f}  (SN ref {cd_ref_sn:.4f}, err {err_sn:+.2f}%; "
        f"CG ref {cd_ref_cg:.4f}, err {err_cg:+.2f}%)  drift={drift_pct:.3f}%  [{total:.0f}s]",
        flush=True,
    )
    print(f"blocks: {blocks}", flush=True)


if __name__ == "__main__":
    main()
