"""PG correction worker: runs one alpha on one SDAA card.

Usage:
  PYTHONPATH=src python pg_worker.py <alpha> <device_id> [--steps 1000] [--grid 160]

Runs bare_hull 160³, MRT+Smag Cs=0.05, Re=2e6, 1000 steps.
Reports Ct_total at steps 500/1000 and Ct_pres variance.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_model import wall_function_3d


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("alpha", type=float, help="Alpha for pressure gradient correction")
    parser.add_argument("device_id", type=int, help="SDAA device ID (0-3)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of LBM steps")
    parser.add_argument("--grid", type=int, default=160, help="Grid size (cube)")
    args = parser.parse_args()

    alpha = args.alpha
    did = args.device_id
    n_steps = args.steps
    grid = args.grid

    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    nx = ny = nz = grid
    hl = 120.0      # hull length in lattice units
    u_in = 0.06     # inlet velocity
    re = 2.0e6      # Reynolds number
    nu = u_in * hl / re
    tau = 3.0 * nu + 0.5
    warmup = 200

    log_path = Path(f"/tmp/pg_worker_a{alpha}_d{did}.log")

    def log(msg):
        # Write BOTH to stdout (for process monitoring) AND to the internal log file
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()

    log(f"Worker: alpha={alpha} device=sdaa:{did} grid={grid}³ steps={n_steps}")
    log(f"tau={tau:.6f} nu={nu:.6e} Re={re:.1e} u_in={u_in}")

    # Build geometry
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    solid_cpu, _ = build_suboff_mask(
        SuboffHullType.BARE_HULL, nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=hl, device="cpu")
    solid = solid_cpu.to(device)
    log(f"Solid cells: {solid.sum().item()}")

    S = _voxel_wetted_area(solid, 1.0)
    dpS = 0.5 * 1.0 * u_in ** 2 * S
    log(f"dpS={dpS:.6f} S={S:.1f}")

    # Initialize
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    ct_hist = []
    t0 = time.time()

    log(f"Start: alpha={alpha} dp_dx_correction=True")

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.05)
        f = stream3d(f)
        f, drag_fric, drag_pres = wall_function_3d(
            f, solid, nu, y_val=0.5,
            dp_dx_correction=True, alpha_pg=alpha,
        )
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, im)

        ct_fric = drag_fric / dpS if dpS > 0 else 0.0
        ct_pres = drag_pres / dpS if dpS > 0 else 0.0
        ct_hist.append({"step": step, "ct_fric": ct_fric, "ct_pres": ct_pres, "ct_total": ct_fric + ct_pres})

        if not torch.isfinite(f).all():
            log(f"DIVERGED at step {step}")
            break

        if step % 500 == 0 or step == n_steps:
            recent = [x["ct_pres"] for x in ct_hist[warmup:]] if step > warmup else [ct_pres]
            pvar = float(torch.tensor(recent).var().item()) if len(recent) > 1 else 0.0
            elapsed = time.time() - t0
            log(f"alpha={alpha:4.1f} step={step:4d} Ct_fric={ct_fric:.6f} Ct_pres={ct_pres:.6f} "
                f"Ct_tot={ct_fric + ct_pres:.6f} Ct_pres_var={pvar:.6e} t={elapsed:.1f}s")

    elapsed = time.time() - t0
    finite = torch.isfinite(f).all().item()

    # Compute summary stats
    pres_arr = [x["ct_pres"] for x in ct_hist[warmup:]]
    if len(pres_arr) > 1:
        pres_t = torch.tensor(pres_arr)
        ct_pres_mean = float(pres_t.mean().item())
        ct_pres_var = float(pres_t.var().item())
        ct_pres_std = float(pres_t.std().item())
    else:
        ct_pres_mean = ct_pres
        ct_pres_var = 0.0
        ct_pres_std = 0.0

    # Extract steps 500 and 1000
    step_500 = next((x for x in ct_hist if x["step"] == 500), None)
    step_1000 = next((x for x in ct_hist if x["step"] == 1000), None)

    result = {
        "alpha": alpha,
        "device_id": did,
        "grid": grid,
        "n_steps": n_steps,
        "tau": tau,
        "nu": nu,
        "u_in": u_in,
        "re": re,
        "finite": finite,
        "elapsed_s": elapsed,
        "ct_pres_mean": ct_pres_mean,
        "ct_pres_var": ct_pres_var,
        "ct_pres_std": ct_pres_std,
        "step_500": step_500,
        "step_1000": step_1000,
        "history": ct_hist,
    }

    # Write individual result
    out_path = Path(f"/tmp/pg_worker_a{alpha}_d{did}.json")
    out_path.write_text(json.dumps(result, indent=2))
    log(f"Result saved to {out_path}")
    log(f"DONE: alpha={alpha} finite={finite} elapsed={elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
