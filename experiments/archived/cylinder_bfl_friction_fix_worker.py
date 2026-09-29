#!/usr/bin/env python3
"""Cylinder BFL with friction fix (τ=ν·u_t/q) at Re=200.

Fix: when BFL is active, the friction drag formula uses
  τ = ν · u_t / q
instead of the standard half-way BB formula
  τ = 2ν · u_t = ν · u_t / 0.5

The effective q for each near-wall cell is computed by averaging the
per-direction BFL q-values over all boundary links at that cell.

Parameters: D=48, nx=400, ny=160, nz=4, u_in=0.08, Re=200, tau=0.5576,
Cs=0.05, MRT+Smagorinsky, 8000 steps, Cd_ref=1.30.

Usage:
  python cylinder_bfl_friction_fix_worker.py <mode> <device_id> <output_path>
  mode: bfl_fix | standard_bb
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
    get_near_wall_2d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.bfl_d3q19_vec import bouzidi_bounce_back_d3q19_vec
from tensorlbm.bfl_d3q19 import compute_q_cylinder_d3q19


def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def compute_effective_q_wall(bfl_mask, bfl_q, near, device):
    """Compute effective per-cell q_wall from per-direction BFL q-field.

    For each near-wall cell, average the q values across all boundary
    directions at that cell.  Non-boundary directions have q=0.5 and
    mask=False, so they don't contribute.

    For near-wall cells with NO boundary links (e.g., diagonal-only
    adjacency), use q=0.5 (standard half-way BB default).

    Returns: (nz, ny, nx) float32 tensor with effective q at near-wall
    cells, 0.5 elsewhere.
    """
    nz, ny, nx = near.shape
    # Sum q over directions where mask is True
    mask_f = bfl_mask.float()  # (19, nz, ny, nx)
    q_weighted = (bfl_q * mask_f).sum(dim=0)  # (nz, ny, nx)
    n_dirs_raw = mask_f.sum(dim=0)  # (nz, ny, nx), 0 for no links
    n_dirs = n_dirs_raw.clamp(min=1.0)  # avoid div-by-zero
    q_avg = q_weighted / n_dirs  # (nz, ny, nx)

    # For cells with no boundary links, use q=0.5 (standard half-way BB)
    q_avg = torch.where(n_dirs_raw > 0, q_avg, torch.full_like(q_avg, 0.5))

    # Only apply at near-wall cells; default 0.5 elsewhere
    q_wall = torch.full((nz, ny, nx), 0.5, dtype=torch.float32, device=device)
    q_wall = torch.where(near, q_avg, q_wall)

    # Stats
    q_at_near = q_wall[near]
    n_zero = int((n_dirs_raw[near] == 0).sum().item())
    print(
        f"  q_wall stats: n_near={int(near.sum().item())} "
        f"n_no_links={n_zero} "
        f"q_min={float(q_at_near.min()):.4f} q_max={float(q_at_near.max()):.4f} "
        f"q_mean={float(q_at_near.mean()):.4f}",
        flush=True,
    )
    return q_wall


def run(device_id, mode, Re, D, nx, ny, nz, u_in, tau, n_steps,
        Cd_ref, tag, output_path=None):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    radius = D / 2.0
    cx_c = nx * 0.25   # cylinder center x (quarter from inlet)
    cy_c = ny * 0.5    # cylinder center y
    nu = u_in * D / Re
    cs_smag = 0.05
    # Frontal area normalization: A_frontal = D * nz
    A_frontal = D * nz
    dpS = 0.5 * u_in ** 2 * A_frontal

    use_bfl = mode.startswith("bfl")
    use_fix = mode == "bfl_fix"

    print(
        f"{tag} mode={mode} use_bfl={use_bfl} use_fix={use_fix} "
        f"nx={nx} ny={ny} nz={nz} D={D} R={radius:.4f} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
        f"dpS={dpS:.6e} Cd_ref={Cd_ref}",
        flush=True,
    )

    t0 = time.time()
    solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    near = get_near_wall_2d(solid, axis='z')
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius, axis='z')

    # BFL q-values
    bfl_mask = None
    bfl_q = None
    q_wall = None
    bfl_stats = {}
    if use_bfl:
        print(f"{tag} computing BFL q-values...", flush=True)
        t_q = time.time()
        bfl_mask, bfl_q = compute_q_cylinder_d3q19(
            nx, ny, nz, cx_c, cy_c, radius, device, axis='z'
        )
        n_links = int(bfl_mask.sum().item())
        q_at_boundary = bfl_q[bfl_mask]
        bfl_stats = {
            "n_links": n_links,
            "q_min": float(q_at_boundary.min()) if n_links > 0 else None,
            "q_max": float(q_at_boundary.max()) if n_links > 0 else None,
            "q_mean": float(q_at_boundary.mean()) if n_links > 0 else None,
        }
        print(
            f"{tag} BFL q-field: {n_links} links ({time.time()-t_q:.1f}s) "
            f"q=[{bfl_stats['q_min']:.4f}, {bfl_stats['q_max']:.4f}] "
            f"mean={bfl_stats['q_mean']:.4f}",
            flush=True,
        )

        if use_fix:
            print(f"{tag} computing effective q_wall for friction fix...", flush=True)
            q_wall = compute_effective_q_wall(bfl_mask, bfl_q, near, device)

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = max(1000, n_steps // 4)

    bc_config = {'far_field_faces': ['y-', 'y+'], 'periodic_faces': ['z-', 'z+']}

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        if use_bfl:
            f_pre_stream = f.clone()
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in, bc_config=bc_config)
            f = bouzidi_bounce_back_d3q19_vec(f, f_pre_stream, bfl_mask, bfl_q)
        else:
            f = bounce_back_cells_3d(f, solid)
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in, bc_config=bc_config)

        if step % 200 == 0:
            f = correct_mass3d(f, im)

        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu, q_wall=q_wall)

        cd_tot = fx_p + fx_f
        cl = fy_p + fy_f

        # Record post-warmup
        if step > warmup:
            if math.isfinite(cd_tot):
                cd_p_hist.append(fx_p)
                cd_f_hist.append(fx_f)
                cd_tot_hist.append(cd_tot)
                cl_hist.append(cl)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            if n_avg > 0:
                print(
                    f"{tag} step={step} Cd_p={sum(cd_p_hist[-n_avg:])/n_avg:.6f} "
                    f"Cd_f={sum(cd_f_hist[-n_avg:])/n_avg:.6f} "
                    f"Cd_tot={sum(cd_tot_hist[-n_avg:])/n_avg:.6f} "
                    f"Cl={sum(cl_hist[-n_avg:])/n_avg:.6f} "
                    f"({time.time()-t0:.0f}s)",
                    flush=True,
                )
            else:
                print(f"{tag} step={step} (warmup, {time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = max(1, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist) / n_final if cd_p_hist else float("nan")
    cd_f_final = sum(cd_f_hist) / n_final if cd_f_hist else float("nan")
    cd_tot_final = sum(cd_tot_hist) / n_final if cd_tot_hist else float("nan")
    cl_final = sum(cl_hist) / n_final if cl_hist else float("nan")

    err_pct = abs(cd_tot_final - Cd_ref) / Cd_ref * 100 if Cd_ref > 0 and math.isfinite(cd_tot_final) else float("nan")

    result = {
        "case": tag,
        "mode": mode,
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "Cd_ref": Cd_ref,
        "error_pct": err_pct,
        "bfl_stats": bfl_stats,
        "friction_formula": "nu*u_t/q" if use_fix else "2*nu*u_t",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE mode={mode} Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"(ref Cd={Cd_ref}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python cylinder_bfl_friction_fix_worker.py <mode> <device_id> <output_path>")
        print("  mode: bfl_fix | standard_bb")
        sys.exit(1)

    mode = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    # D=48, Re=200, u_in=0.08, nu=0.08*48/200=0.0192, tau=3*0.0192+0.5=0.5576
    run(
        device_id=device_id, mode=mode,
        Re=200, D=48,
        nx=400, ny=160, nz=4,
        u_in=0.08, tau=0.5576,
        n_steps=8000,
        Cd_ref=1.30,
        tag=f"[SDAA:{device_id} CYL Re=200 {mode}]",
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
