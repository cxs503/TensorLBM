#!/usr/bin/env python3
"""SUBOFF normal-method comparison: from_suboff (analytical) vs from_gradient (staircase).

Both runs use identical solver settings; only the SurfaceMesh normal computation differs:
  - from_suboff : analytical axisymmetric normal (n = (-dr/dx, cosθ, sinθ)/|n|)
  - from_gradient : staircase normal from central-difference of solid mask

Usage:
  python suboff_normal_cmp_worker.py <normal_method> <device_id> <output_path>
  normal_method: from_suboff | from_gradient
"""
import json
import math
import sys
import time
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


def run(device_id, normal_method, Re, L, nx, ny, nz, u_in, tau, n_steps,
        Cf_ref, tag, output_path=None):
    """Run SUBOFF bare-hull drag simulation and return results dict."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    config = SuboffConfig()
    radius = config.r_over_l * L          # R_max in lattice units
    D = 2.0 * radius                       # diameter
    cx = nx * 0.30
    cy = ny * 0.5
    cz = nz * 0.5

    nu = u_in * L / Re
    cs_smag = 0.05

    # dpS = 0.5 * u_in^2 * pi * D * L  (wetted-area normalisation)
    dpS = 0.5 * u_in ** 2 * math.pi * D * L

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} L={L} R_max={radius:.4f} D={D:.4f} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
        f"dpS={dpS:.6e} Cf_ref={Cf_ref}",
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

    # Near-wall mask
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # Build surface mesh with chosen normal method
    if normal_method == "from_suboff":
        mesh = SurfaceMesh.from_suboff(
            solid, near, cx, cy, cz, L, radius, config)
    elif normal_method == "from_gradient":
        mesh = SurfaceMesh.from_gradient(solid, near)
    else:
        raise ValueError(f"Unknown normal_method: {normal_method}")

    # --- Normal statistics ---
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

    # Sign check: bow nx_n<0, stern nx_n>0, midbody ~0
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
        print(f"{tag} bow nx_n mean={float(mesh.nx_n[bow_mask].mean()):.4f} (expect <0)", flush=True)
    if stern_mask.any():
        print(f"{tag} stern nx_n mean={float(mesh.nx_n[stern_mask].mean()):.4f} (expect >0)", flush=True)
    if mid_mask.any():
        print(f"{tag} mid nx_n mean={float(mesh.nx_n[mid_mask].mean()):.6f} (expect ~0)", flush=True)

    # dA statistics
    dA_vals = mesh.dA[near]
    print(
        f"{tag} dA stats: min={float(dA_vals.min()):.4f} "
        f"max={float(dA_vals.max()):.4f} mean={float(dA_vals.mean()):.4f}",
        flush=True,
    )

    # Solid mask for NoDynamics (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialise flow field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    # History
    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist, fz_hist = [], [], [], [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in)
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, fz_f = drag_friction_integration(f, mesh, dpS, nu)
        cd_p_hist.append(fx_p)
        cd_f_hist.append(fx_f)
        cd_tot_hist.append(fx_p + fx_f)
        cl_hist.append(fy_p + fy_f)
        fz_hist.append(fz_p + fz_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            print(
                f"{tag} step={step} Cd_p={sum(cd_p_hist[-n_avg:])/n_avg:.6f} "
                f"Cd_f={sum(cd_f_hist[-n_avg:])/n_avg:.6f} "
                f"Cd_tot={sum(cd_tot_hist[-n_avg:])/n_avg:.6f} "
                f"Cl={sum(cl_hist[-n_avg:])/n_avg:.6f} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # Final averages (last 1000 steps or all if fewer)
    n_final = min(1000, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final
    fz_final = sum(fz_hist[-n_final:]) / n_final

    err_pct = abs(cd_tot_final - Cf_ref) / Cf_ref * 100 if Cf_ref > 0 else float("nan")

    result = {
        "case": tag,
        "normal_method": normal_method,
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "L": L,
        "R_max": radius,
        "D": D,
        "L_D_ratio": stats["L_D_ratio"],
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "dpS_formula": "0.5*u_in^2*pi*D*L (wetted area)",
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "fz": fz_final,
        "Cf_ref": Cf_ref,
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
        f"(ref Cf={Cf_ref}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python suboff_normal_cmp_worker.py <normal_method> <device_id> <output_path>")
        print("  normal_method: from_suboff | from_gradient")
        sys.exit(1)

    normal_method = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    # Re=1000, L=80, nx=200, ny=80, nz=80
    # u_in=0.06, nu=0.06*80/1000=0.0048, tau=3*0.0048+0.5=0.5144
    run(
        device_id=device_id,
        normal_method=normal_method,
        Re=1000, L=80,
        nx=200, ny=80, nz=80,
        u_in=0.06, tau=0.5144,
        n_steps=5000,
        Cf_ref=0.042,
        tag=f"[SDAA:{device_id} SUBOFF Re=1000 {normal_method}]",
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
