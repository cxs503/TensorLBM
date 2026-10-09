#!/usr/bin/env python3
"""Sphere 3D drag with pressure extrapolation — Re=100 + Stokes Re=0.1.

Validates quadratic pressure extrapolation at the wall:
  'none'      → p_wall = p1                 (cell 1, current)
  'linear'    → p_wall = 2·p1 - p2         (1st-order extrap)
  'quadratic' → p_wall = 3·p1 - 3·p2 + p3  (2nd-order extrap)

Benchmarks:
  1. Re=100:  D=40, nx=180³, u_in=0.08, tau=0.596, 3000 steps, Cd_ref=1.09
  2. Stokes:  D=20, nx=120³, u_in=0.0025, tau=3.5, 2000 steps, Cd_ref=240

Usage:
  python sphere_extrap_worker.py <benchmark> <extrap> <device_id> <output_path>
  benchmark: re100 | stokes
  extrap: none | linear | quadratic
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

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


def build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2


def run_sphere(device_id, Re, D, nx, ny, nz, u_in, tau, n_steps, extrap,
               tag, output_path=None):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    R = D / 2.0
    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    nu = u_in * D / Re
    cs_smag = 0.05
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2

    print(
        f"{tag} extrap={extrap} nx={nx} ny={ny} nz={nz} D={D} R={R} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.6e}",
        flush=True,
    )

    t0 = time.time()
    solid = build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

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

        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS, extrap=extrap)
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
    n_final = min(500, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final
    fz_final = sum(fz_hist[-n_final:]) / n_final

    if Re < 1.0:
        cd_ref = 24.0 / Re
        cd_p_ref = 8.0 / Re
        cd_f_ref = 16.0 / Re
        ref_name = "Stokes analytical (24/Re, 8/Re, 16/Re)"
    elif abs(Re - 100) < 1:
        cd_ref = 1.09
        cd_p_ref = None
        cd_f_ref = None
        ref_name = "Henderson empirical"
    else:
        cd_ref = 24.0 / Re * (1.0 + 0.15 * Re ** 0.687)
        cd_p_ref = None
        cd_f_ref = None
        ref_name = "Schiller-Naumann"

    err_pct = abs(cd_tot_final - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    err_p_pct = (
        abs(cd_p_final - cd_p_ref) / cd_p_ref * 100
        if cd_p_ref is not None else float("nan")
    )
    err_f_pct = (
        abs(cd_f_final - cd_f_ref) / cd_f_ref * 100
        if cd_f_ref is not None else float("nan")
    )

    result = {
        "case": tag,
        "extrap": extrap,
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "fz": fz_final,
        "Cd_ref": cd_ref,
        "Cd_p_ref": cd_p_ref,
        "Cd_f_ref": cd_f_ref,
        "ref_name": ref_name,
        "error_pct": err_pct,
        "error_p_pct": err_p_pct,
        "error_f_pct": err_f_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE extrap={extrap} Cd_p={cd_p_final:.4f} Cd_f={cd_f_final:.4f} "
        f"Cd_tot={cd_tot_final:.4f} Cl={cl_final:.6f} "
        f"(ref={cd_ref:.4f}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 5:
        print("Usage: python sphere_extrap_worker.py <benchmark> <extrap> <device_id> <output_path>")
        print("  benchmark: re100 | stokes")
        print("  extrap: none | linear | quadratic")
        sys.exit(1)

    benchmark = sys.argv[1]
    extrap = sys.argv[2]
    device_id = int(sys.argv[3])
    output_path = sys.argv[4]

    if benchmark == "re100":
        run_sphere(
            device_id=device_id, Re=100, D=40,
            nx=180, ny=180, nz=180,
            u_in=0.08, tau=0.596,
            n_steps=3000, extrap=extrap,
            tag=f"[SDAA:{device_id} Re=100 extrap={extrap}]",
            output_path=output_path,
        )
    elif benchmark == "stokes":
        # Stokes Re=0.1: Cd=240, Cd_p=80, Cd_f=160 (EXACT analytical)
        # tau=3.5 → nu_lattice=(3.5-0.5)/3=1.0
        # Re=0.1 → u_in = Re*nu/D = 0.1*1.0/20 = 0.005
        run_sphere(
            device_id=device_id, Re=0.1, D=20,
            nx=120, ny=120, nz=120,
            u_in=0.005, tau=3.5,
            n_steps=2000, extrap=extrap,
            tag=f"[SDAA:{device_id} Stokes Re=0.1 extrap={extrap}]",
            output_path=output_path,
        )
    else:
        print(f"Unknown benchmark: {benchmark}")
        sys.exit(1)


if __name__ == "__main__":
    main()
