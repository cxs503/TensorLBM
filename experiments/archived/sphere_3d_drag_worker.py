#!/usr/bin/env python3
"""Sphere 3D drag verification — Stokes(Re=0.1) + Re=100 + Re=1000 + grid convergence.

Validates 3D SurfaceMesh.from_sphere normal computation (x, y, z components).
Uses the verified main loop: NoDynamics + half-way BB + far-field BC.

Benchmarks:
  1. Stokes Re=0.1: Cd=24/Re=240 (Cd_p=80, Cd_f=160) — EXACT analytical
  2. Re=100: Cd=1.09 (Henderson empirical)
  3. Re=1000: Cd=0.47 (empirical)
  4. Grid convergence: D=10/20/40 at Re=0.1

Usage:
  python sphere_3d_drag_worker.py <benchmark> <device_id> <output_path>
  benchmark: stokes | re100 | re1000 | grid_D10 | grid_D20 | grid_D40
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
    """Vectorized sphere mask: (i-cx)^2+(j-cy)^2+(k-cz)^2 < R^2."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2


def run_sphere(
    device_id,
    Re,
    D,
    nx,
    ny,
    nz,
    u_in,
    tau,
    n_steps,
    tag,
    output_path=None,
):
    """Run sphere drag simulation and return results dict."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    R = D / 2.0
    cx = nx * 0.25  # quarter from inlet
    cy = ny * 0.5  # centered
    cz = nz * 0.5  # centered
    nu = u_in * D / Re
    cs_smag = 0.05

    # dpS = 0.5 * u_in^2 * pi * R^2 (dynamic pressure × frontal area)
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} D={D} R={R} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.6e}",
        flush=True,
    )

    t0 = time.time()

    # Build sphere solid mask
    solid = build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # Precompute near-wall mask and surface mesh
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)

    # --- Normal statistics (validate 3D normal) ---
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
    # Check normal is unit vector
    norm_check = torch.sqrt(mesh.nx_n ** 2 + mesh.ny_n ** 2 + mesh.nz_n ** 2)
    norm_near = norm_check[near]
    print(
        f"{tag} |n| stats: min={float(norm_near.min()):.6f} "
        f"max={float(norm_near.max()):.6f} mean={float(norm_near.mean()):.6f}",
        flush=True,
    )
    # Check all 3 components are non-zero (TRUE 3D normal)
    n_nx_nonzero = int((nx_n_vals.abs() > 1e-6).sum().item())
    n_ny_nonzero = int((ny_n_vals.abs() > 1e-6).sum().item())
    n_nz_nonzero = int((nz_n_vals.abs() > 1e-6).sum().item())
    print(
        f"{tag} 3D check: nx_n nonzero={n_nx_nonzero}/{n_near} "
        f"ny_n nonzero={n_ny_nonzero}/{n_near} nz_n nonzero={n_nz_nonzero}/{n_near}",
        flush=True,
    )

    # dA statistics (default dA=1.0 for from_sphere)
    dA_vals = mesh.dA[near]
    print(
        f"{tag} dA stats: min={float(dA_vals.min()):.4f} "
        f"max={float(dA_vals.max()):.4f} mean={float(dA_vals.mean()):.4f}",
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
    im = float(rho0.sum().item())  # initial mass = nx*ny*nz

    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    # History
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []  # fy (should be ~0 by symmetry)
    fz_hist = []  # fz (should be ~0 by symmetry)

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

        # 6. Far-field BC (without obstacle_mask → don't touch solid)
        f = far_field_bc_3d(f, u_in)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Drag computation (after full step: post-stream, post-BC)
        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, fz_f = drag_friction_integration(f, mesh, dpS, nu)

        cd_p = fx_p  # pressure drag (x-component, flow direction)
        cd_f = fx_f  # friction drag (x-component)
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f  # lift (y-component, should be ~0 by symmetry)
        fz_tot = fz_p + fz_f  # z-component (should be ~0 by symmetry)

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)
        fz_hist.append(fz_tot)

        # Check for divergence
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

    # Reference values
    if Re < 1.0:
        # Stokes: Cd = 24/Re, Cd_p = 8/Re, Cd_f = 16/Re
        cd_ref = 24.0 / Re
        cd_p_ref = 8.0 / Re
        cd_f_ref = 16.0 / Re
        ref_name = "Stokes analytical (24/Re, 8/Re, 16/Re)"
    elif abs(Re - 100) < 1:
        cd_ref = 1.09
        cd_p_ref = None
        cd_f_ref = None
        ref_name = "Henderson empirical"
    elif abs(Re - 1000) < 1:
        cd_ref = 0.47
        cd_p_ref = None
        cd_f_ref = None
        ref_name = "empirical"
    else:
        cd_ref = 24.0 / Re * (1.0 + 0.15 * Re ** 0.687)
        cd_p_ref = None
        cd_f_ref = None
        ref_name = "Schiller-Naumann"

    err_pct = abs(cd_tot_final - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    err_p_pct = (
        abs(cd_p_final - cd_p_ref) / cd_p_ref * 100
        if cd_p_ref is not None
        else float("nan")
    )
    err_f_pct = (
        abs(cd_f_final - cd_f_ref) / cd_f_ref * 100
        if cd_f_ref is not None
        else float("nan")
    )

    result = {
        "case": tag,
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
        print("Usage: python sphere_3d_drag_worker.py <benchmark> <device_id> <output_path>")
        print("  benchmark: stokes | re100 | re1000 | grid_D10 | grid_D20 | grid_D40")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if benchmark == "stokes":
        # Stokes Re=0.1: Cd=240, Cd_p=80, Cd_f=160 (EXACT analytical)
        run_sphere(
            device_id=device_id, Re=0.1, D=10,
            nx=60, ny=60, nz=60,
            u_in=0.01, tau=3.5,
            n_steps=2000,
            tag=f"[SDAA:{device_id} Stokes Re=0.1]",
            output_path=output_path,
        )
    elif benchmark == "re100":
        # Re=100: Cd=1.09 (Henderson)
        run_sphere(
            device_id=device_id, Re=100, D=40,
            nx=120, ny=120, nz=120,
            u_in=0.08, tau=0.596,
            n_steps=3000,
            tag=f"[SDAA:{device_id} Re=100]",
            output_path=output_path,
        )
    elif benchmark == "re1000":
        # Re=1000: Cd=0.47 (empirical)
        run_sphere(
            device_id=device_id, Re=1000, D=40,
            nx=120, ny=120, nz=120,
            u_in=0.08, tau=0.5096,
            n_steps=3000,
            tag=f"[SDAA:{device_id} Re=1000]",
            output_path=output_path,
        )
    elif benchmark == "grid_D10":
        # Grid convergence D=10 at Re=0.1
        run_sphere(
            device_id=device_id, Re=0.1, D=10,
            nx=60, ny=60, nz=60,
            u_in=0.01, tau=3.5,
            n_steps=2000,
            tag=f"[SDAA:{device_id} Grid D=10 Re=0.1]",
            output_path=output_path,
        )
    elif benchmark == "grid_D20":
        # Grid convergence D=20 at Re=0.1
        run_sphere(
            device_id=device_id, Re=0.1, D=20,
            nx=120, ny=120, nz=120,
            u_in=0.01, tau=6.5,
            n_steps=2000,
            tag=f"[SDAA:{device_id} Grid D=20 Re=0.1]",
            output_path=output_path,
        )
    elif benchmark == "grid_D40":
        # Grid convergence D=40 at Re=0.1
        run_sphere(
            device_id=device_id, Re=0.1, D=40,
            nx=240, ny=240, nz=240,
            u_in=0.01, tau=12.5,
            n_steps=2000,
            tag=f"[SDAA:{device_id} Grid D=40 Re=0.1]",
            output_path=output_path,
        )
    else:
        print(f"Unknown benchmark: {benchmark}")
        sys.exit(1)


if __name__ == "__main__":
    main()
