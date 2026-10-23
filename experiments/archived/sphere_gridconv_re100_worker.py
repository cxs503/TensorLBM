#!/usr/bin/env python3
"""Sphere 3D grid convergence at Re=100 — 3 grid sizes in parallel.

Reference: Cd=1.09 (Henderson empirical, Re=100)

Grid sizes (domain grows with D to keep blockage low):
  D=20  → 120³  (6D domain)   tau=0.548
  D=40  → 180³  (4.5D domain) tau=0.596
  D=60  → 240³  (4D domain)   tau=0.644

u_in=0.08, Re=100, tau=3*0.08*D/100+0.5, 3000 steps each.
Uses sphere_mask from boundaries3d, MRT+Smagorinsky (Cs=0.05).

Usage:
  PYTHONPATH=src python sphere_gridconv_re100_worker.py <D> <device_id> <output_path>
  D: 20 | 40 | 60
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.boundaries3d import (
    sphere_mask,
    far_field_bc_3d,
    bounce_back_cells_3d,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import (
    collide_smagorinsky_mrt3d,
    collide_smagorinsky_bgk3d,
)
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)


def run_sphere_gridconv(D, device_id, output_path):
    """Run sphere drag at Re=100 for given diameter D."""
    Re = 100
    u_in = 0.08
    nu = u_in * D / Re
    tau = 3.0 * u_in * D / Re + 0.5
    cs_smag = 0.05
    n_steps = 3000

    # Domain size: grows with D (see module docstring)
    grid_map = {20: 120, 40: 180, 60: 240}
    n = grid_map[D]
    nx, ny, nz = n, n, n

    R = D / 2.0
    cx = nx * 0.25  # quarter from inlet
    cy = ny * 0.5
    cz = nz * 0.5

    # dpS = 0.5 * u_in^2 * pi * R^2 (dynamic pressure × frontal area)
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2

    tag = f"[SDAA:{device_id} Sphere D={D} Re=100 {n}³]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} D={D} R={R} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.6e}",
        flush=True,
    )

    t0 = time.time()

    # Build sphere solid mask using sphere_mask from boundaries3d
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # Precompute near-wall mask and surface mesh
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)

    # Normal statistics
    nx_n_vals = mesh.nx_n[near]
    ny_n_vals = mesh.ny_n[near]
    nz_n_vals = mesh.nz_n[near]
    print(
        f"{tag} normal stats: "
        f"nx_n=[{float(nx_n_vals.min()):.3f}, {float(nx_n_vals.max()):.3f}] "
        f"ny_n=[{float(ny_n_vals.min()):.3f}, {float(ny_n_vals.max()):.3f}] "
        f"nz_n=[{float(nz_n_vals.min()):.3f}, {float(nz_n_vals.max()):.3f}]",
        flush=True,
    )
    norm_check = torch.sqrt(mesh.nx_n ** 2 + mesh.ny_n ** 2 + mesh.nz_n ** 2)
    norm_near = norm_check[near]
    print(
        f"{tag} |n| stats: min={float(norm_near.min()):.6f} "
        f"max={float(norm_near.max()):.6f} mean={float(norm_near.mean()):.6f}",
        flush=True,
    )

    # Solid mask for NoDynamics (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize flow field: uniform flow, zero inside sphere
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    # Collision model: MRT for small grids, BGK for large (240³ OOM with MRT)
    use_bgk = (n >= 240)
    collide_fn = collide_smagorinsky_bgk3d if use_bgk else collide_smagorinsky_mrt3d
    collision_model = "BGK+Smagorinsky" if use_bgk else "MRT+Smagorinsky"
    print(f"{tag} collision: {collision_model} (Cs={cs_smag})", flush=True)

    # History
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []
    fz_hist = []

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT or BGK + Smagorinsky LES)
        f = collide_fn(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC (without obstacle_mask → don't touch solid)
        f = far_field_bc_3d(f, u_in)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Drag computation (after full step: post-stream, post-BC)
        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, fz_f = drag_friction_integration(f, mesh, dpS, nu)

        cd_p = fx_p
        cd_f = fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f
        fz_tot = fz_p + fz_f

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)
        fz_hist.append(fz_tot)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 200 == 0:
            n_avg = min(200, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_avg = sum(cl_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                f"Cd_tot={cd_tot_avg:.4f} Cl={cl_avg:.6f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # Final averages (last 500 steps or all if fewer)
    n_final = min(500, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final
    fz_final = sum(fz_hist[-n_final:]) / n_final

    # Reference: Henderson empirical Cd=1.09 at Re=100
    cd_ref = 1.09
    ref_name = "Henderson empirical (Re=100)"

    err_pct = abs(cd_tot_final - cd_ref) / cd_ref * 100

    result = {
        "case": tag,
        "benchmark": "sphere_3d_grid_convergence",
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "domain_ratio": f"{nx/D:.1f}D",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "collision_model": collision_model,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "fz": fz_final,
        "Cd_ref": cd_ref,
        "ref_name": ref_name,
        "error_pct": err_pct,
        "normal_stats": {
            "nx_n_min": float(nx_n_vals.min()),
            "nx_n_max": float(nx_n_vals.max()),
            "ny_n_min": float(ny_n_vals.min()),
            "ny_n_max": float(ny_n_vals.max()),
            "nz_n_min": float(nz_n_vals.min()),
            "nz_n_max": float(nz_n_vals.max()),
            "n_norm_min": float(norm_near.min()),
            "n_norm_max": float(norm_near.max()),
            "n_norm_mean": float(norm_near.mean()),
        },
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.4f} Cd_f={cd_f_final:.4f} "
        f"Cd_tot={cd_tot_final:.4f} Cl={cl_final:.6f} "
        f"(ref={cd_ref:.4f}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python sphere_gridconv_re100_worker.py <D> <device_id> <output_path>")
        print("  D: 20 | 40 | 60")
        sys.exit(1)

    D = int(sys.argv[1])
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if D not in (20, 40, 60):
        print(f"D must be 20, 40, or 60, got {D}")
        sys.exit(1)

    run_sphere_gridconv(D, device_id, output_path)


if __name__ == "__main__":
    main()
