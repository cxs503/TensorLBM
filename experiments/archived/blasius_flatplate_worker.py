#!/usr/bin/env python3
"""Flat plate (Blasius boundary layer) benchmark worker.

Validates the LBM simulation against the Blasius analytical solution:
  Local Cf(x) = 0.664 / sqrt(Re_x)     (laminar boundary layer)
  Total  Cd   = 1.328 / sqrt(Re_L)     (average over plate of length L)

Uses the verified main loop (NoDynamics + half-way BB + far-field BC)
with MRT+Smagorinsky (Cs=0.05) and unified pressure-integration drag
(SurfaceMesh.from_gradient normal).

Geometry: thin plate at bottom (solid[:, 0:2, :] = True), 2 cells thick.
The plate spans the full domain; L=100 is the reference length for Re.

Usage:
  python blasius_flatplate_worker.py <device_id> <output_path>
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


def run_flatplate(device_id, output_path=None):
    """Run flat plate Blasius benchmark on sdaa:<device_id>."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id} Blasius]"

    # ---- Problem parameters (from task spec) ----
    L = 100.0          # reference length for Re
    nx, ny, nz = 400, 80, 4
    u_in = 0.05
    Re_L = 1000.0
    nu = u_in * L / Re_L          # = 0.005
    tau = 3.0 * nu + 0.5          # = 0.515
    n_steps = 10000
    warmup = 2000
    cs_smag = 0.05

    # Blasius reference
    # Local Cf(x) = 0.664 / sqrt(Re_x), Total Cd = 1.328 / sqrt(Re_L)
    cd_blasius_L = 1.328 / math.sqrt(Re_L)   # based on L=100

    # Dynamic pressure × plate area (full domain plate, one side)
    # Plate area = nx * nz (top surface of plate)
    plate_area = nx * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * plate_area

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} L={L} u_in={u_in} "
        f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} Re_L={Re_L}",
        flush=True,
    )
    print(f"{tag} Blasius Cd(L={L})={cd_blasius_L:.6f}", flush=True)
    print(f"{tag} dpS={dpS:.6e} plate_area={plate_area}", flush=True)

    t0 = time.time()

    # ---- Geometry: thin plate at bottom, 2 cells thick ----
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0:2, :] = True   # plate at y=0,1

    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid} (plate 2 cells thick at bottom)", flush=True)

    # ---- Precompute near-wall mask and surface mesh (from_gradient) ----
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_gradient(solid, near)

    # Normal stats
    ny_n_vals = mesh.ny_n[near]
    print(
        f"{tag} normal stats: ny_n=[{float(ny_n_vals.min()):.3f}, "
        f"{float(ny_n_vals.max()):.3f}] (expect ~1.0 at plate top)",
        flush=True,
    )

    # Solid mask for NoDynamics (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # ---- Initialize flow field ----
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={initial_mass}",
          flush=True)

    # ---- BC config: far-field at top (y+) only, periodic in z ----
    bc_config = {
        'far_field_faces': ['y+'],
        'periodic_faces': ['z-', 'z+'],
    }

    # ---- x-stations for local Cf measurement ----
    # Re_x = u_in * x / nu; Blasius Cf(x) = 0.664 / sqrt(Re_x)
    x_stations = [25, 50, 100, 200, 300, 399]
    print(f"{tag} Blasius Cf at x-stations:", flush=True)
    for xs in x_stations:
        re_x = u_in * xs / nu
        cf_b = 0.664 / math.sqrt(re_x)
        print(f"  x={xs:3d}  Re_x={re_x:7.0f}  Cf_blasius={cf_b:.6f}", flush=True)

    # ---- Time history ----
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cf_local_hist = {xs: [] for xs in x_stations}

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

        # 6. Far-field BC (top only, periodic in z)
        f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # 8. Drag computation (after warmup)
        if step > warmup:
            # Total drag via pressure integration
            fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
            fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_p_hist.append(fx_p)
            cd_f_hist.append(fx_f)
            cd_tot_hist.append(fx_p + fx_f)

            # Local Cf at x-stations: Cf(x) = 2*nu*ux(x, y_near) / (0.5*u_in^2)
            # = 4*nu*ux / u_in^2
            rho, ux, uy, uz = macroscopic3d(f)
            for xs in x_stations:
                # Average ux over z at (y=2, x=xs) — first fluid cell above plate
                ux_at_x = float(ux[:, 2, xs].mean().item())
                cf_local = 4.0 * nu * ux_at_x / (u_in ** 2)
                cf_local_hist[xs].append(cf_local)

        # Check for divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Progress
        if step % 1000 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            if n_avg > 0:
                cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
                cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
                cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            else:
                cd_p_avg = cd_f_avg = cd_tot_avg = 0.0
            elapsed = time.time() - t0
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                f"Cd_tot={cd_tot_avg:.6f} (Blasius Cd={cd_blasius_L:.6f}) "
                f"[{elapsed:.0f}s]",
                flush=True,
            )

    elapsed = time.time() - t0

    # ---- Final statistics: average over last 2000 samples ----
    win = min(2000, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-win:]) / max(win, 1)
    cd_f_final = sum(cd_f_hist[-win:]) / max(win, 1)
    cd_tot_final = sum(cd_tot_hist[-win:]) / max(win, 1)

    # Local Cf at x-stations (averaged)
    cf_results = []
    for xs in x_stations:
        hist = cf_local_hist[xs]
        w = min(2000, len(hist))
        cf_sim = sum(hist[-w:]) / max(w, 1) if w > 0 else float('nan')
        re_x = u_in * xs / nu
        cf_blas = 0.664 / math.sqrt(re_x)
        err = abs(cf_sim - cf_blas) / cf_blas * 100 if cf_blas > 0 else float('nan')
        cf_results.append({
            "x": xs,
            "Re_x": re_x,
            "Cf_simulated": cf_sim,
            "Cf_blasius": cf_blas,
            "error_pct": err,
        })

    # Total Cd based on full plate length (nx=400)
    re_full = u_in * nx / nu
    cd_blasius_full = 1.328 / math.sqrt(re_full)

    result = {
        "benchmark": "flat_plate_blasius",
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": "MRT+Smagorinsky",
        "C_s": cs_smag,
        "normal_method": "from_gradient",
        "L_ref": L,
        "nx": nx, "ny": ny, "nz": nz,
        "u_in": u_in,
        "Re_L": Re_L,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_samples": len(cd_tot_hist),
        "plate_area": plate_area,
        "dpS": dpS,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cd_blasius_L100": cd_blasius_L,
        "Cd_blasius_full": cd_blasius_full,
        "Re_full": re_full,
        "Cf_local": cf_results,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
    }

    print(f"\n{'='*60}")
    print(f"{tag} FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Cd_pressure  = {cd_p_final:.6f}")
    print(f"  Cd_friction  = {cd_f_final:.6f}")
    print(f"  Cd_total     = {cd_tot_final:.6f}")
    print(f"  Cd_blasius(L=100) = {cd_blasius_L:.6f}")
    print(f"  Cd_blasius(full={nx}) = {cd_blasius_full:.6f}")
    print(f"\n  Local Cf comparison:")
    print(f"  {'x':>5} {'Re_x':>8} {'Cf_sim':>10} {'Cf_blas':>10} {'err%':>8}")
    print(f"  {'-'*45}")
    for r in cf_results:
        print(f"  {r['x']:5d} {r['Re_x']:8.0f} {r['Cf_simulated']:10.6f} "
              f"{r['Cf_blasius']:10.6f} {r['error_pct']:8.1f}%")
    print(f"\n  Wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Finite: {result['finite']}")

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"  Results saved to {output_path}")

    return result


if __name__ == "__main__":
    did = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_flatplate(did, out)
