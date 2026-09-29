#!/usr/bin/env python3
"""SUBOFF 3D drag — fast variant (smaller grid, fewer steps) for quick verification."""
import json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    get_near_wall_3d, SurfaceMesh,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig


def run(device_id, Re, L, nx, ny, nz, u_in, tau, n_steps, tag, output_path=None):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    config = SuboffConfig()
    radius = config.r_over_l * L
    cx, cy, cz = nx*0.3, ny*0.5, nz*0.5
    nu = u_in * L / Re
    cs_smag = 0.05
    dpS = 0.5 * u_in**2 * math.pi * radius**2
    Cf_ittc = 0.075 / (np.log10(Re) - 2.0)**2

    print(f"{tag} nx={nx} ny={ny} nz={nz} L={L} R={radius:.3f} "
          f"u_in={u_in} tau={tau:.6f} dpS={dpS:.6e} Cf_ITTC={Cf_ittc:.6f}", flush=True)

    t0 = time.time()
    solid, stats = build_suboff_mask(
        hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=L, radius=radius,
        config=config, device=device)
    n_solid = int(solid.sum().item())
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)

    # Normal stats
    nx_n_vals = mesh.nx_n[near]
    ny_n_vals = mesh.ny_n[near]
    nz_n_vals = mesh.nz_n[near]
    norm_near = torch.sqrt(mesh.nx_n**2 + mesh.ny_n**2 + mesh.nz_n**2)[near]
    print(f"{tag} solid={n_solid} near={n_near} |n|=[{float(norm_near.min()):.4f},{float(norm_near.max()):.4f}]", flush=True)

    # Sign check
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32), indexing='ij')
    x_bow = cx - L/2.0
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

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist, fz_hist = [], [], [], [], []

    for step in range(1, n_steps+1):
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

        if step % 200 == 0:
            n_avg = min(200, len(cd_tot_hist))
            print(f"{tag} step={step} Cd_p={sum(cd_p_hist[-n_avg:])/n_avg:.6f} "
                  f"Cd_f={sum(cd_f_hist[-n_avg:])/n_avg:.6f} "
                  f"Cd_tot={sum(cd_tot_hist[-n_avg:])/n_avg:.6f} "
                  f"Cl={sum(cl_hist[-n_avg:])/n_avg:.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(500, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_f = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_f = sum(cd_tot_hist[-n_final:]) / n_final
    cl_f = sum(cl_hist[-n_final:]) / n_final
    fz_f = sum(fz_hist[-n_final:]) / n_final
    cd_ref = float(Cf_ittc)
    err = abs(cd_tot_f - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")

    result = {
        "case": tag, "device": f"sdaa:{device_id}", "Re": Re, "L": L,
        "R_max": radius, "L_D_ratio": stats["L_D_ratio"],
        "grid": f"{nx}x{ny}x{nz}", "u_in": u_in, "nu": nu, "tau": tau,
        "n_steps": n_steps, "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cl": cl_f, "fz": fz_f, "Cd_ref": cd_ref,
        "ref_name": "ITTC-1957", "error_pct": err,
        "normal_stats": {
            "nx_n_min": float(nx_n_vals.min()), "nx_n_max": float(nx_n_vals.max()),
            "ny_n_min": float(ny_n_vals.min()), "ny_n_max": float(ny_n_vals.max()),
            "nz_n_min": float(nz_n_vals.min()), "nz_n_max": float(nz_n_vals.max()),
            "n_norm_min": float(norm_near.min()), "n_norm_max": float(norm_near.max()),
            "n_norm_mean": float(norm_near.mean()),
            "bow_nx_mean": float(mesh.nx_n[bow_mask].mean()) if bow_mask.any() else None,
            "stern_nx_mean": float(mesh.nx_n[stern_mask].mean()) if stern_mask.any() else None,
            "mid_nx_mean": float(mesh.nx_n[mid_mask].mean()) if mid_mask.any() else None,
        },
        "dA_stats": {"min": float(mesh.dA[near].min()), "max": float(mesh.dA[near].max()), "mean": float(mesh.dA[near].mean())},
        "finite": bool(torch.isfinite(f).all().item()), "elapsed_s": elapsed,
    }
    print(f"{tag} DONE Cd_p={cd_p_f:.6f} Cd_f={cd_f_f:.6f} Cd_tot={cd_tot_f:.6f} "
          f"Cl={cl_f:.6f} (ref={cd_ref:.6f}) err={err:.1f}% time={elapsed:.0f}s", flush=True)
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)
    return result


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "re1000"
    dev = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    out = sys.argv[3] if len(sys.argv) > 3 else None

    if bench == "re1000":
        # Smaller grid for faster turnaround: L=60, nx=200, ny=80, nz=80
        # Re=1000, u_in=0.06, nu=0.06*60/1000=0.0036, tau=3*0.0036+0.5=0.5108
        run(dev, Re=1000, L=60, nx=200, ny=80, nz=80,
            u_in=0.06, tau=0.5108, n_steps=3000,
            tag=f"[SDAA:{dev} SUBOFF Re=1000 fast]", output_path=out)
    elif bench == "re1e4":
        # Re=1e4, L=60, u_in=0.06, nu=0.06*60/1e4=0.00036, tau=3*0.00036+0.5=0.50108
        run(dev, Re=10000, L=60, nx=200, ny=80, nz=80,
            u_in=0.06, tau=0.50108, n_steps=3000,
            tag=f"[SDAA:{dev} SUBOFF Re=1e4 fast]", output_path=out)
    elif bench == "re1e5":
        # Re=1e5, L=60, u_in=0.06, nu=0.06*60/1e5=3.6e-5, tau=0.500108
        run(dev, Re=100000, L=60, nx=200, ny=80, nz=80,
            u_in=0.06, tau=0.500108, n_steps=3000,
            tag=f"[SDAA:{dev} SUBOFF Re=1e5 fast]", output_path=out)
