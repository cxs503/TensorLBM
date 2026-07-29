"""Sphere wall-function benchmark worker — tests Cd at Re=100/1000/10000.

D3Q19 MRT + Smagorinsky Cs=0.05 + wall_function_3d + farfield BC.
3D sphere D=24 cells, domain 120×60×60.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.boundaries3d import far_field_bc_3d, sphere_mask
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_model import wall_function_3d


def main():
    device_id = int(sys.argv[1])
    re = float(sys.argv[2])
    n_steps = int(sys.argv[3])
    warmup = int(sys.argv[4])
    output_path = sys.argv[5]

    # Fixed parameters
    nx, ny, nz = 120, 60, 60
    diameter = 24.0
    radius = diameter / 2.0
    u_in = 0.08
    cs_smag = 0.05

    nu = u_in * diameter / re
    tau = 3.0 * nu + 0.5

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{device_id} Re={int(re)}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={diameter} u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag}",
          flush=True)

    t0 = time.time()

    # Build sphere mask
    cx_sphere = nx * 0.25   # quarter from inlet
    cy_sphere = ny * 0.5    # centered
    cz_sphere = nz * 0.5    # centered
    solid = sphere_mask(nx, ny, nz, cx_sphere, cy_sphere, cz_sphere, radius, device=device)

    A_frontal = math.pi * radius ** 2   # Projected frontal area
    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_frontal

    # Initialize flow field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(torch.ones_like(rho0).sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s)", flush=True)

    # Accumulators for running average after warmup
    cd_hist = []  # per-step total Cd

    for step in range(1, n_steps + 1):
        # 1. Collision: MRT + Smagorinsky LES
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 2. Stream
        f = stream3d(f)

        # 3. Wall function (body force + drag computation)
        f, drag_fric, drag_pres = wall_function_3d(f, solid, nu, y_val=0.5)

        # 4. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # 5. Mass correction every 100 steps
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        # Compute Cd
        cd_fric = drag_fric / dyn_p if dyn_p > 0 else 0.0
        cd_pres = drag_pres / dyn_p if dyn_p > 0 else 0.0
        cd_total = cd_fric + cd_pres

        if step > warmup and math.isfinite(cd_total):
            cd_hist.append(cd_total)

        # Check for divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 200 == 0:
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
            elapsed = time.time() - t0
            print(f"{tag} step={step} Cd={cd_total:.4f} Cd_avg={cd_avg:.4f} ({elapsed:.0f}s)",
                  flush=True)

    elapsed = time.time() - t0

    # Final results
    cd_mean = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
    cd_std = (sum((c - cd_mean) ** 2 for c in cd_hist) / max(len(cd_hist) - 1, 1)) ** 0.5 if len(cd_hist) > 1 else 0.0

    # Schiller-Naumann reference Cd for sphere
    sn = 24.0 / re * (1.0 + 0.15 * re ** 0.687) if re > 0 else float("nan")
    ref_cd = sn
    err_pct = abs(cd_mean - ref_cd) / ref_cd * 100 if ref_cd > 0 and math.isfinite(cd_mean) else float("nan")

    result = {
        "case": f"sphere_Re{int(re)}",
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": f"MRT+Smag(Cs={cs_smag})",
        "boundary": "wall_function_3d+farfield",
        "grid": f"{nx}x{ny}x{nz}",
        "diameter": diameter,
        "radius": radius,
        "u_in": u_in,
        "Re": re,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd_mean": cd_mean,
        "Cd_std": cd_std,
        "Cd_ref": ref_cd,
        "error_pct": err_pct,
        "cd_samples": len(cd_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(f"{tag} DONE Cd={cd_mean:.4f} (ref={ref_cd:.4f}) err={err_pct:.1f}% time={elapsed:.0f}s",
          flush=True)
    Path(output_path).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
