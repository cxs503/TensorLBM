#!/usr/bin/env python3
"""SUBOFF grid convergence study at Re=1000.

Tests if Cd converges with grid refinement using:
  - from_gradient normals (generic, geometry-agnostic)
  - drag_pressure_integration (Bug22: background pressure subtraction)
  - drag_friction_integration (first-order wall shear)
  - WETTED AREA normalization: dpS = 0.5*u_in^2*pi*D*L
  - Laminar flat plate reference: Cf = 1.328/sqrt(Re)

Three grid levels (each 2x refinement):
  Level 0: L=40,  nx=100, ny=40,  nz=40   (0.16M cells)  — SDAA:9
  Level 1: L=80,  nx=200, ny=80,  nz=80   (1.28M cells)  — SDAA:10
  Level 2: L=160, nx=400, ny=160, nz=160  (10.2M cells)   — SDAA:11
           (fallback: 300x120x120 = 4.3M if OOM)

Usage:
  python suboff_grid_conv_re1000_worker.py <level> <device_id> <output_path>
  level: 0 | 1 | 2
"""
import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig


def run_grid_conv(
    device_id,
    Re,
    L,
    nx,
    ny,
    nz,
    u_in,
    tau,
    n_steps,
    win,
    tag,
    output_path=None,
):
    """Run SUBOFF bare-hull drag simulation and return results dict."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    config = SuboffConfig()
    r_over_l = config.r_over_l  # ≈ 0.0583
    radius = r_over_l * L  # R_max in lattice units
    D = 2.0 * radius  # max diameter

    cx = nx * 0.30   # bow at 30% of domain (room for wake)
    cy = ny * 0.5
    cz = nz * 0.5

    nu = u_in * L / Re
    cs_smag = 0.05

    # WETTED AREA normalization: dpS = 0.5 * u_in^2 * pi * D * L
    dpS = 0.5 * u_in ** 2 * math.pi * D * L

    # Laminar flat plate reference: Cf = 1.328 / sqrt(Re)
    Cf_ref = 1.328 / math.sqrt(Re)

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} L={L} R_max={radius:.3f} D={D:.3f} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
        f"dpS(wetted)={dpS:.6e} Cf_ref={Cf_ref:.6f}",
        flush=True,
    )

    t0 = time.time()

    # Build SUBOFF solid mask
    solid, stats = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=L, radius=radius,
        config=config,
        device=device,
    )
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}  L/D={stats['L_D_ratio']}", flush=True)

    # Precompute near-wall mask
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # Use from_gradient normals (generic, geometry-agnostic)
    mesh = SurfaceMesh.from_gradient(solid, near)

    # Normal statistics
    nx_n_vals = mesh.nx_n[near]
    ny_n_vals = mesh.ny_n[near]
    nz_n_vals = mesh.nz_n[near]
    norm_check = torch.sqrt(mesh.nx_n ** 2 + mesh.ny_n ** 2 + mesh.nz_n ** 2)
    norm_near = norm_check[near]
    print(
        f"{tag} normal stats: |n| min={float(norm_near.min()):.6f} "
        f"max={float(norm_near.max()):.6f} mean={float(norm_near.mean()):.6f}",
        flush=True,
    )
    n_nx_nonzero = int((nx_n_vals.abs() > 1e-6).sum().item())
    n_ny_nonzero = int((ny_n_vals.abs() > 1e-6).sum().item())
    n_nz_nonzero = int((nz_n_vals.abs() > 1e-6).sum().item())
    print(
        f"{tag} 3D check: nx_n nonzero={n_nx_nonzero}/{n_near} "
        f"ny_n nonzero={n_ny_nonzero}/{n_near} nz_n nonzero={n_nz_nonzero}/{n_near}",
        flush=True,
    )

    # Sign check: bow cells should have nx_n < 0, stern cells nx_n > 0
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing='ij')
    x_bow = cx - L / 2.0
    xi_field = (xx - x_bow) / L
    bow_mask = near & (xi_field < 0.233)
    stern_mask = near & (xi_field > 0.748)
    mid_mask = near & (xi_field >= 0.233) & (xi_field <= 0.748)
    if bow_mask.any():
        print(f"{tag} bow nx_n mean={float(mesh.nx_n[bow_mask].mean()):.4f} (expect < 0)", flush=True)
    if stern_mask.any():
        print(f"{tag} stern nx_n mean={float(mesh.nx_n[stern_mask].mean()):.4f} (expect > 0)", flush=True)
    if mid_mask.any():
        print(f"{tag} mid nx_n mean={float(mesh.nx_n[mid_mask].mean()):.6f} (expect ~ 0)", flush=True)

    # dA statistics
    dA_vals = mesh.dA[near]
    print(
        f"{tag} dA stats: min={float(dA_vals.min()):.4f} "
        f"max={float(dA_vals.max()):.4f} mean={float(dA_vals.mean()):.4f}",
        flush=True,
    )

    # Solid mask for NoDynamics (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize flow field: uniform flow, zero inside hull
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    # History
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []
    fz_hist = []

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky LES)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Drag computation
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

        # Check for divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_avg = sum(cl_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                f"Cd_tot={cd_tot_avg:.6f} Cl={cl_avg:.6f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # Final averages (last `win` steps or all if fewer)
    n_final = min(win, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final
    fz_final = sum(fz_hist[-n_final:]) / n_final

    # Reference: laminar flat plate Cf = 1.328/sqrt(Re)
    cd_ref = float(Cf_ref)
    ref_name = "Cf=1.328/sqrt(Re) (laminar flat plate)"

    err_pct = abs(cd_tot_final - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")

    result = {
        "case": tag,
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "L": L,
        "R_max": radius,
        "D": D,
        "L_D_ratio": stats["L_D_ratio"],
        "grid": f"{nx}x{ny}x{nz}",
        "n_cells": nx * ny * nz,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "win": win,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "dpS_type": "wetted_area_0.5*u^2*pi*D*L",
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "fz": fz_final,
        "Cf_ref": cd_ref,
        "ref_name": ref_name,
        "error_pct": err_pct,
        "normal_method": "from_gradient",
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
            "nx_n_nonzero": n_nx_nonzero,
            "ny_n_nonzero": n_ny_nonzero,
            "nz_n_nonzero": n_nz_nonzero,
            "bow_nx_mean": float(mesh.nx_n[bow_mask].mean()) if bow_mask.any() else None,
            "stern_nx_mean": float(mesh.nx_n[stern_mask].mean()) if stern_mask.any() else None,
            "mid_nx_mean": float(mesh.nx_n[mid_mask].mean()) if mid_mask.any() else None,
        },
        "dA_stats": {
            "min": float(dA_vals.min()),
            "max": float(dA_vals.max()),
            "mean": float(dA_vals.mean()),
        },
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"(ref Cf={cd_ref:.6f}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python suboff_grid_conv_re1000_worker.py <level> <device_id> <output_path>")
        print("  level: 0 (L=40) | 1 (L=80) | 2 (L=160)")
        sys.exit(1)

    level = int(sys.argv[1])
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    Re = 1000
    u_in = 0.06
    n_steps = 5000
    win = 500
    cs_smag = 0.05

    if level == 0:
        L = 40
        nx, ny, nz = 100, 40, 40
    elif level == 1:
        L = 80
        nx, ny, nz = 200, 80, 80
    elif level == 2:
        L = 160
        nx, ny, nz = 400, 160, 160
    else:
        print(f"Unknown level: {level}")
        sys.exit(1)

    tau = 3 * u_in * L / Re + 0.5
    tag = f"[SDAA:{device_id} SUBOFF grid-conv L={L} Re=1000]"

    try:
        run_grid_conv(
            device_id=device_id, Re=Re, L=L,
            nx=nx, ny=ny, nz=nz,
            u_in=u_in, tau=tau,
            n_steps=n_steps, win=win,
            tag=tag, output_path=output_path,
        )
    except RuntimeError as e:
        # Handle OOM for level 2 — fallback to 300x120x120
        if level == 2 and ("out of memory" in str(e).lower() or "oom" in str(e).lower()):
            print(f"{tag} OOM with 400x160x160, falling back to 300x120x120", flush=True)
            torch.sdaa.empty_cache()
            nx, ny, nz = 300, 120, 120
            run_grid_conv(
                device_id=device_id, Re=Re, L=L,
                nx=nx, ny=ny, nz=nz,
                u_in=u_in, tau=tau,
                n_steps=n_steps, win=win,
                tag=tag, output_path=output_path,
            )
        else:
            raise


if __name__ == "__main__":
    main()
