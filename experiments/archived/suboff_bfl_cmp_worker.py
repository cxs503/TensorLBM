#!/usr/bin/env python3
"""SUBOFF BFL vs standard BB comparison at Re=1000.

Three wall methods:
  standard_bb       — half-way bounce-back before streaming (staircase walls)
  bfl_from_suboff   — BFL interpolated bounce-back (analytical q-values from
                      SUBOFF axisymmetric profile) + from_suboff surface normal
  bfl_from_gradient — BFL interpolated bounce-back (same q-values) +
                      from_gradient (staircase) surface normal

All runs: SUBOFF bare hull L=80, nx=200, ny=80, nz=80,
u_in=0.06, Re=1000, tau=0.5144, Cs=0.05, MRT+Smagorinsky, 5000 steps.
dpS = 0.5*u_in^2*pi*D*L (wetted area), Cf_ref=0.042.

Usage:
  python suboff_bfl_cmp_worker.py <wall_method> <device_id> <output_path>
  wall_method: standard_bb | bfl_from_suboff | bfl_from_gradient
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
from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19
from tensorlbm.bfl_d3q19_vec import bouzidi_bounce_back_d3q19_vec
from tensorlbm.interpolated_bc_suboff import compute_q_suboff


def run(device_id, wall_method, Re, L, nx, ny, nz, u_in, tau, n_steps,
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

    use_bfl = wall_method.startswith("bfl_")
    normal_method = "from_suboff" if wall_method == "bfl_from_suboff" else \
                    "from_suboff" if wall_method == "standard_bb" else \
                    "from_gradient"

    print(
        f"{tag} wall_method={wall_method} normal={normal_method} "
        f"nx={nx} ny={ny} nz={nz} L={L} R_max={radius:.4f} D={D:.4f} "
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
    else:
        mesh = SurfaceMesh.from_gradient(solid, near)

    # Normal statistics
    nx_n_vals = mesh.nx_n[near]
    ny_n_vals = mesh.ny_n[near]
    nz_n_vals = mesh.nz_n[near]
    norm_check = torch.sqrt(mesh.nx_n ** 2 + mesh.ny_n ** 2 + mesh.nz_n ** 2)
    norm_near = norm_check[near]
    print(
        f"{tag} |n| stats: min={float(norm_near.min()):.6f} "
        f"max={float(norm_near.max()):.6f} mean={float(norm_near.mean()):.6f}",
        flush=True,
    )

    # BFL q-values (compute on CPU, move to device)
    bfl_mask = None
    bfl_q = None
    bfl_stats = {}
    if use_bfl:
        print(f"{tag} computing BFL q-values on CPU...", flush=True)
        t_q = time.time()
        bfl_mask, bfl_q = compute_q_suboff(
            nx, ny, nz, cx, cy, cz, L,
            hull_type="bare_hull", config=config,
            device="cpu", n_bisect=12,
        )
        n_links = int(bfl_mask.sum().item())
        q_at_boundary = bfl_q[bfl_mask]
        bfl_stats = {
            "n_links": n_links,
            "q_min": float(q_at_boundary.min()) if n_links > 0 else None,
            "q_max": float(q_at_boundary.max()) if n_links > 0 else None,
            "q_mean": float(q_at_boundary.mean()) if n_links > 0 else None,
            "q_lt05_frac": float((q_at_boundary < 0.5).float().mean()) if n_links > 0 else None,
        }
        print(
            f"{tag} BFL q-field: {n_links} links ({time.time()-t_q:.1f}s) "
            f"q=[{bfl_stats['q_min']:.4f}, {bfl_stats['q_max']:.4f}] "
            f"mean={bfl_stats['q_mean']:.4f}",
            flush=True,
        )
        bfl_mask = bfl_mask.to(device)
        bfl_q = bfl_q.to(device)

    # Solid mask for NoDynamics (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialise flow field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}",
          flush=True)

    # History
    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist, fz_hist = [], [], [], [], []

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky LES)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        if use_bfl:
            # BFL: stream first, then apply interpolated bounce-back
            f_pre_stream = f.clone()          # post-collision, pre-stream
            f = stream3d(f)                    # streaming
            f = far_field_bc_3d(f, u_in)       # far-field BC
            f = bouzidi_bounce_back_d3q19_vec(  # BFL after streaming (vectorized)
                f, f_pre_stream, bfl_mask, bfl_q)
        else:
            # Standard BB: bounce-back before streaming
            f = bounce_back_cells_3d(f, solid)
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in)

        # 4. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 5. Drag computation
        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, fz_f = drag_friction_integration(f, mesh, dpS, nu)

        cd_p_hist.append(fx_p)
        cd_f_hist.append(fx_f)
        cd_tot_hist.append(fx_p + fx_f)
        cl_hist.append(fy_p + fy_f)
        fz_hist.append(fz_p + fz_f)

        # Check for divergence
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
        "wall_method": wall_method,
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
        "bfl_stats": bfl_stats,
        "normal_stats": {
            "n_norm_min": float(norm_near.min()),
            "n_norm_max": float(norm_near.max()),
            "n_norm_mean": float(norm_near.mean()),
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
        print("Usage: python suboff_bfl_cmp_worker.py <wall_method> <device_id> <output_path>")
        print("  wall_method: standard_bb | bfl_from_suboff | bfl_from_gradient")
        sys.exit(1)

    wall_method = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    # Re=1000, L=80, nx=200, ny=80, nz=80
    # u_in=0.06, nu=0.06*80/1000=0.0048, tau=3*0.0048+0.5=0.5144
    run(
        device_id=device_id,
        wall_method=wall_method,
        Re=1000, L=80,
        nx=200, ny=80, nz=80,
        u_in=0.06, tau=0.5144,
        n_steps=5000,
        Cf_ref=0.042,
        tag=f"[SDAA:{device_id} SUBOFF Re=1000 {wall_method}]",
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
