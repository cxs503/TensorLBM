#!/usr/bin/env python3
"""BFL common-module verification worker — SDAA 12-15.

Tests the unified ``bfl_common`` module on curved and flat surfaces:

  SDAA:12 — Stationary cylinder (Re=200, D=48, from_cylinder)
            BFL with exact analytical q.  Compare Cd with half-way BB.
  SDAA:13 — Stationary sphere (Re=100, D=40, from_sphere)
            BFL with exact analytical q.  Compare Cd with half-way BB.
  SDAA:14 — Moving boundary (Couette, lid-driven)
            BFL moving-wall correction.  Compare with exact solution.
  SDAA:15 — STL geometry (KVLCC2, Re=1000, from_gradient)
            BFL with generic q (voxelised).  Compare Cd_p with half-way BB.

Usage:
  python bfl_common_sdaa_worker.py <test> <device_id> <output_path>
  test: cylinder_bfl | cylinder_bb | sphere_bfl | sphere_bb |
        couette_bfl | kvlcc2_bfl | kvlcc2_bb
"""
from __future__ import annotations

import functools
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.collision_common import (
    equilibrium3d,
    macroscopic3d,
    collide_bgk3d,
    collide_smagorinsky_mrt3d,
    correct_mass3d,
    stream3d,
)
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    get_near_wall_2d,
    get_near_wall_3d,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.bfl_common import (
    bfl_bounce_back_common,
    bfl_moving_wall_correction,
    compute_q_cylinder_common,
    compute_q_sphere_common,
    compute_q_flat_walls_common,
    compute_q_generic_common,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def drag_friction_bfl(f, mesh, dpS, nu, q_wall=None, u_wall=None):
    """Friction drag with BFL q-correction and optional wall velocity.

    τ = ν·(u_t - u_wall_t) / q   (BFL corrected)
    Falls back to τ = 2ν·u_t when q_wall is None (standard half-way BB).
    """
    rho, ux, uy, uz = macroscopic3d(f)
    nx_n, ny_n, nz_n = mesh.nx_n, mesh.ny_n, mesh.nz_n
    u_dot_n = ux * nx_n + uy * ny_n + uz * nz_n
    ut_x = ux - u_dot_n * nx_n
    ut_y = uy - u_dot_n * ny_n
    ut_z = uz - u_dot_n * nz_n

    if u_wall is not None:
        uw_x, uw_y, uw_z = u_wall[0], u_wall[1], u_wall[2]
        uw_dot_n = uw_x * nx_n + uw_y * ny_n + uw_z * nz_n
        uwt_x = uw_x - uw_dot_n * nx_n
        uwt_y = uw_y - uw_dot_n * ny_n
        uwt_z = uw_z - uw_dot_n * nz_n
        ut_x = ut_x - uwt_x
        ut_y = ut_y - uwt_y
        ut_z = ut_z - uwt_z

    if q_wall is not None:
        inv_q = 1.0 / q_wall.clamp(min=1e-6)
        tau_x = nu * ut_x * inv_q
        tau_y = nu * ut_y * inv_q
        tau_z = nu * ut_z * inv_q
    else:
        tau_x = 2.0 * nu * ut_x
        tau_y = 2.0 * nu * ut_y
        tau_z = 2.0 * nu * ut_z

    # For moving walls, the formula above computes the shear stress exerted
    # BY the wall ON the fluid (negative when u_wall > u_fluid).  The drag
    # force ON the wall is the reaction: F_wall = -τ_fluid.  For stationary
    # walls (u_wall=None) the formula already gives the correct positive
    # drag (shear stress exerted by the fluid on the wall).
    if u_wall is not None:
        tau_x = -tau_x
        tau_y = -tau_y
        tau_z = -tau_z

    mask = mesh.near.float() * mesh.dA
    ffx = (tau_x * mask).sum()
    ffy = (tau_y * mask).sum()
    ffz = (tau_z * mask).sum()
    return (
        float(ffx.item() / dpS),
        float(ffy.item() / dpS),
        float(ffz.item() / dpS),
    )


def compute_q_wall_per_cell(bfl_mask, bfl_q, near, device):
    """Average per-direction q to a per-cell q_wall field."""
    mask_f = bfl_mask.float()
    q_weighted = (bfl_q * mask_f).sum(dim=0)
    n_dirs_raw = mask_f.sum(dim=0)
    n_dirs = n_dirs_raw.clamp(min=1.0)
    q_avg = q_weighted / n_dirs
    q_avg = torch.where(n_dirs_raw > 0, q_avg, torch.full_like(q_avg, 0.5))
    q_wall = torch.full(near.shape, 0.5, dtype=torch.float32, device=device)
    q_wall = torch.where(near, q_avg, q_wall)
    return q_wall


# --------------------------------------------------------------------------- #
# TEST 1 (SDAA:12): Stationary cylinder — BFL vs half-way BB
# --------------------------------------------------------------------------- #
def run_cylinder(device_id, mode, output_path):
    """Cylinder Re=200, D=48.  mode: 'bfl' or 'bb'."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    D = 48
    nx, ny, nz = 400, 160, 4
    Re = 200
    u_in = 0.08
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 8000
    warmup = 2000
    Cd_ref = 1.30
    radius = D / 2.0
    cx_c = nx * 0.25
    cy_c = ny * 0.5
    A_frontal = D * nz
    dpS = 0.5 * u_in ** 2 * A_frontal

    use_bfl = mode == "bfl"
    tag = f"[Cyl-Re200-{mode} SDAA:{device_id}]"
    print(
        f"{tag} nx={nx} ny={ny} nz={nz} D={D} R={radius:.4f} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} "
        f"dpS={dpS:.6e} Cd_ref={Cd_ref} use_bfl={use_bfl}",
        flush=True,
    )

    t0 = time.time()

    # Build cylinder mask
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx_c) ** 2 + (yy - cy_c) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    near = get_near_wall_2d(solid, axis="z")
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius, axis="z")

    # BFL q-values (analytical)
    bfl_mask = None
    bfl_q = None
    q_wall = None
    bfl_stats = {}
    if use_bfl:
        print(f"{tag} computing BFL q-values (analytical cylinder)...", flush=True)
        t_q = time.time()
        bfl_mask, bfl_q = compute_q_cylinder_common(
            nx, ny, nz, cx_c, cy_c, radius, device, axis="z", lattice="D3Q19"
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
        q_wall = compute_q_wall_per_cell(bfl_mask, bfl_q, near, device)
        q_at_near = q_wall[near]
        print(
            f"  q_wall: n_near={n_near} "
            f"q_min={float(q_at_near.min()):.4f} q_max={float(q_at_near.max()):.4f} "
            f"q_mean={float(q_at_near.mean()):.4f}",
            flush=True,
        )

    # Initialize
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_bgk3d, tau, solid, u_in, far_field_fn,
            correct_mass_fn=correct_mass3d, target_mass=im,
            step=step, mass_interval=200,
            wall_treatment="bfl" if use_bfl else "bb",
            bfl_mask=bfl_mask, bfl_q=bfl_q,
        )

        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, _ = drag_friction_bfl(f, mesh, dpS, nu, q_wall=q_wall)
        cd_tot = fx_p + fx_f
        cl = fy_p + fy_f

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

    elapsed = time.time() - t0
    n_final = max(1, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist) / n_final if cd_p_hist else float("nan")
    cd_f_final = sum(cd_f_hist) / n_final if cd_f_hist else float("nan")
    cd_tot_final = sum(cd_tot_hist) / n_final if cd_tot_hist else float("nan")
    cl_final = sum(cl_hist) / n_final if cl_hist else float("nan")
    err_pct = (
        abs(cd_tot_final - Cd_ref) / Cd_ref * 100
        if Cd_ref > 0 and math.isfinite(cd_tot_final)
        else float("nan")
    )

    result = {
        "case": "cylinder_Re200",
        "mode": mode,
        "device": f"sdaa:{device_id}",
        "Re": int(Re),
        "D": float(D),
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": float(u_in),
        "nu": float(nu),
        "tau": float(tau),
        "n_steps": int(n_steps),
        "warmup": int(warmup),
        "n_solid": int(n_solid),
        "n_near": int(n_near),
        "dpS": float(dpS),
        "Cd_pressure": float(cd_p_final) if cd_p_final == cd_p_final else None,
        "Cd_friction": float(cd_f_final) if cd_f_final == cd_f_final else None,
        "Cd_total": float(cd_tot_final) if cd_tot_final == cd_tot_final else None,
        "Cl": float(cl_final) if cl_final == cl_final else None,
        "Cd_ref": float(Cd_ref),
        "error_pct": float(err_pct) if err_pct == err_pct else None,
        "bfl_stats": bfl_stats,
        "friction_formula": "nu*u_t/q" if use_bfl else "2*nu*u_t",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"(ref Cd={Cd_ref}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# TEST 2 (SDAA:13): Stationary sphere — BFL vs half-way BB
# --------------------------------------------------------------------------- #
def run_sphere(device_id, mode, output_path):
    """Sphere Re=100, D=40.  mode: 'bfl' or 'bb'."""
    from tensorlbm.turbulence import collide_smagorinsky_mrt3d

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    D = 40
    nx, ny, nz = 160, 120, 120
    Re = 100
    u_in = 0.08
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 6000
    warmup = 1500
    Cd_ref = 1.09
    cs_smag = 0.05
    R = D / 2.0
    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2

    use_bfl = mode == "bfl"
    tag = f"[Sph-Re100-{mode} SDAA:{device_id}]"
    print(
        f"{tag} nx={nx} ny={ny} nz={nz} D={D} R={R} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
        f"dpS={dpS:.6e} Cd_ref={Cd_ref} use_bfl={use_bfl}",
        flush=True,
    )

    t0 = time.time()

    # Build sphere mask
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    solid = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)

    bfl_mask = None
    bfl_q = None
    q_wall = None
    bfl_stats = {}
    if use_bfl:
        print(f"{tag} computing BFL q-values (analytical sphere)...", flush=True)
        t_q = time.time()
        bfl_mask, bfl_q = compute_q_sphere_common(
            nx, ny, nz, cx, cy, cz, R, device, lattice="D3Q19"
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
        q_wall = compute_q_wall_per_cell(bfl_mask, bfl_q, near, device)
        q_at_near = q_wall[near]
        print(
            f"  q_wall: n_near={n_near} "
            f"q_min={float(q_at_near.min()):.4f} q_max={float(q_at_near.max()):.4f} "
            f"q_mean={float(q_at_near.mean()):.4f}",
            flush=True,
        )

    # Initialize
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
        f = no_dynamics(f, f_pre, solid)

        if use_bfl:
            f = bounce_back_cells_3d(f, solid)
            f_pre_stream = f.clone()
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in)
            f = bfl_bounce_back_common(
                f, f_pre_stream, bfl_mask, bfl_q, lattice="D3Q19"
            )
        else:
            f = bounce_back_cells_3d(f, solid)
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in)

        if step % 200 == 0:
            f = correct_mass3d(f, im)

        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, fz_f = drag_friction_bfl(f, mesh, dpS, nu, q_wall=q_wall)
        cd_tot = fx_p + fx_f
        cl = fy_p + fy_f
        fz_tot = fz_p + fz_f

        if step > warmup:
            if math.isfinite(cd_tot):
                cd_p_hist.append(fx_p)
                cd_f_hist.append(fx_f)
                cd_tot_hist.append(cd_tot)
                cl_hist.append(cl)
                fz_hist.append(fz_tot)

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

    elapsed = time.time() - t0
    n_final = max(1, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist) / n_final if cd_p_hist else float("nan")
    cd_f_final = sum(cd_f_hist) / n_final if cd_f_hist else float("nan")
    cd_tot_final = sum(cd_tot_hist) / n_final if cd_tot_hist else float("nan")
    cl_final = sum(cl_hist) / n_final if cl_hist else float("nan")
    err_pct = (
        abs(cd_tot_final - Cd_ref) / Cd_ref * 100
        if Cd_ref > 0 and math.isfinite(cd_tot_final)
        else float("nan")
    )

    result = {
        "case": "sphere_Re100",
        "mode": mode,
        "device": f"sdaa:{device_id}",
        "Re": int(Re),
        "D": float(D),
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": float(u_in),
        "nu": float(nu),
        "tau": float(tau),
        "Cs": float(cs_smag),
        "n_steps": int(n_steps),
        "warmup": int(warmup),
        "n_solid": int(n_solid),
        "n_near": int(n_near),
        "dpS": float(dpS),
        "Cd_pressure": float(cd_p_final) if cd_p_final == cd_p_final else None,
        "Cd_friction": float(cd_f_final) if cd_f_final == cd_f_final else None,
        "Cd_total": float(cd_tot_final) if cd_tot_final == cd_tot_final else None,
        "Cl": float(cl_final) if cl_final == cl_final else None,
        "Cd_ref": float(Cd_ref),
        "error_pct": float(err_pct) if err_pct == err_pct else None,
        "bfl_stats": bfl_stats,
        "friction_formula": "nu*u_t/q" if use_bfl else "2*nu*u_t",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"(ref Cd={Cd_ref}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# TEST 3 (SDAA:14): Moving boundary — Couette with BFL moving wall
# --------------------------------------------------------------------------- #
def run_couette(device_id, output_path):
    """3D Couette flow with BFL bounce-back + moving wall.

    Flat walls → q=0.5 everywhere → BFL reduces to standard BB.
    Moving wall correction added to BFL.
    """
    from tensorlbm.solver3d import collide_bgk3d

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0  # = 1/6
    u_top = 0.05
    n_steps = 3000
    warmup = 500
    H = ny - 2  # =10
    Cf_exact = 2.0 * nu / (H * u_top)

    tag = f"[Couette-BFL SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f} u_top={u_top}", flush=True)
    print(f"{tag} H={H} Cf_exact={Cf_exact:.6f} n_steps={n_steps}", flush=True)

    t0 = time.time()

    # Solid mask: top and bottom walls
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # BFL mask and q for flat walls (q=0.5)
    bfl_mask, bfl_q = compute_q_flat_walls_common(
        nx, ny, nz, device, wall_axis="y", lattice="D3Q19"
    )
    n_links = int(bfl_mask.sum().item())
    print(f"{tag} BFL links={n_links} (q=0.5 everywhere, flat walls)", flush=True)

    # Moving wall correction for top wall
    u_w = (u_top, 0.0, 0.0)
    moving_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    moving_mask[:, ny - 2, :] = True
    wall_corr = bfl_moving_wall_correction(
        bfl_mask, moving_mask, u_w, lattice="D3Q19", rho_w=1.0
    )
    print(f"{tag} moving wall correction computed (u_top={u_top})", flush=True)

    # q_wall for friction: 0.5 everywhere
    q_wall = torch.full((nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    # u_wall field for friction
    u_wall_field = torch.zeros((3, nz, ny, nx), dtype=torch.float32, device=device)
    u_wall_field[0, :, ny - 2, :] = u_top

    # Initialize: zero velocity
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )

    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    cf_hist = []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        f = no_dynamics(f, f_pre, solid)

        # BFL: bounce_back → stream → BFL with moving wall correction
        f = bounce_back_cells_3d(f, solid)
        f_pre_stream = f.clone()
        f = stream3d(f)
        # Periodic in x and z (torch.roll handles this)
        f = bfl_bounce_back_common(
            f, f_pre_stream, bfl_mask, bfl_q,
            lattice="D3Q19", wall_correction=wall_corr,
        )

        if step % 200 == 0:
            im = float(rho0.sum().item())
            f = correct_mass3d(f, im)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            # Compute Cf on top wall only
            near_top = near.clone()
            near_top[:, : ny // 2 + 1, :] = False
            mesh_top = SurfaceMesh.from_gradient(solid, near_top)

            A_wall = nx * nz
            dpS_wall = 0.5 * 1.0 * u_top ** 2 * A_wall

            u_wall_top = torch.zeros((3, nz, ny, nx), dtype=torch.float32, device=device)
            u_wall_top[0, :, ny - 2, :] = u_top

            ffx, _, _ = drag_friction_bfl(
                f, mesh_top, dpS_wall, nu, q_wall=q_wall, u_wall=u_wall_top
            )
            cf_hist.append(ffx)

        if step % 500 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_prof = ux.mean(dim=(0, 2))
            cf_avg = sum(cf_hist) / max(len(cf_hist), 1)
            print(
                f"{tag} step={step} Cf={cf_avg:.6f} (exact={Cf_exact:.6f}) "
                f"u[ny//2]={float(u_prof[ny//2]):.6f} ({time.time()-t0:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0
    cf_mean = sum(cf_hist) / max(len(cf_hist), 1) if cf_hist else float("nan")
    cf_err = abs(cf_mean - Cf_exact) / Cf_exact * 100 if Cf_exact > 0 else float("nan")

    # u profile
    _, ux, _, _ = macroscopic3d(f)
    u_prof = ux.mean(dim=(0, 2)).cpu().numpy()

    # Analytical: u(y) = u_top * (y - 0.5) / H
    y_vals = np.arange(ny, dtype=np.float32)
    u_exact = np.zeros(ny, dtype=np.float32)
    for y in range(1, ny - 1):
        u_exact[y] = u_top * (y - 0.5) / H

    u_err_max = 0.0
    for y in range(1, ny - 1):
        if abs(u_exact[y]) > 1e-10:
            err = abs(u_prof[y] - u_exact[y]) / max(abs(u_exact[y]), 1e-10) * 100
            u_err_max = max(u_err_max, err)

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cf      = {cf_mean:.6f}  (exact={Cf_exact:.6f}, err={cf_err:.2f}%)", flush=True)
    print(f"{tag} u_err   = {u_err_max:.2f}% (max relative error)", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
        "case": "couette_bfl",
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": "BGK",
        "boundary": "BFL_moving_wall",
        "grid": f"{nx}x{ny}x{nz}",
        "tau": float(tau),
        "nu": float(nu),
        "u_top": float(u_top),
        "H": int(H),
        "n_steps": int(n_steps),
        "warmup": int(warmup),
        "Cf_mean": float(cf_mean) if cf_mean == cf_mean else None,
        "Cf_exact": float(Cf_exact),
        "Cf_err_pct": float(cf_err) if cf_err == cf_err else None,
        "u_err_max_pct": float(u_err_max),
        "n_bfl_links": int(n_links),
        "friction_formula": "nu*(u_t-u_wall)/q",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} results saved to {output_path}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# TEST 4 (SDAA:15): STL geometry — KVLCC2 with BFL (generic q)
# --------------------------------------------------------------------------- #
def run_kvlcc2(device_id, mode, output_path):
    """KVLCC2 Re=1000 with BFL (generic q) or half-way BB.

    Uses from_gradient normals (NOT STL normals) per task spec.
    mode: 'bfl' or 'bb'
    """
    from tensorlbm.turbulence import collide_smagorinsky_mrt3d
    from tensorlbm.ship_cad import build_hull_mask, ShipHullType

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    L = 80.0
    nx, ny, nz = 300, 120, 120
    Re = 1000
    u_in = 0.06
    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5
    n_steps = 5000
    warmup = 1000
    cs_smag = 0.05

    use_bfl = mode == "bfl"
    tag = f"[KVLCC2-Re1000-{mode} SDAA:{device_id}]"
    print(
        f"{tag} nx={nx} ny={ny} nz={nz} L={L} u_in={u_in} "
        f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} Re={Re} use_bfl={use_bfl}",
        flush=True,
    )

    t0 = time.time()

    cx = nx * 0.30
    cy = ny * 0.5
    cz_keel = nz * 0.5 - 10
    beam = ny * 0.20
    draft = nz * 0.15

    solid, stats = build_hull_mask(
        hull_type=ShipHullType.KVLCC2,
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy,
        cz_keel=cz_keel,
        length=L,
        beam=beam,
        draft=draft,
        device=str(device),
    )
    n_solid = int(solid.sum().item())
    cb_num = stats.get("Cb_numerical", 0)
    print(f"{tag} solid cells: {n_solid}  Cb_numerical={cb_num:.4f}", flush=True)

    D = draft
    dpS = 0.5 * 1.0 * u_in ** 2 * math.pi * D * L
    print(f"{tag} dpS = {dpS:.6e} (D=draft={D})", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # from_gradient normals (NOT STL normals) per task spec
    mesh = SurfaceMesh.from_gradient(solid, near)

    bfl_mask = None
    bfl_q = None
    q_wall = None
    bfl_stats = {}
    if use_bfl:
        print(f"{tag} computing BFL q-values (generic voxelised)...", flush=True)
        t_q = time.time()
        bfl_mask, bfl_q = compute_q_generic_common(solid, device, lattice="D3Q19")
        n_links = int(bfl_mask.sum().item())
        bfl_stats = {
            "n_links": n_links,
            "q_value": 0.5,
            "method": "generic_voxelised",
        }
        print(
            f"{tag} BFL q-field: {n_links} links ({time.time()-t_q:.1f}s) "
            f"(q=0.5, generic voxelised)",
            flush=True,
        )
        q_wall = torch.full((nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    # Initialize
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    bc_config = {
        "far_field_faces": ["y-", "y+", "z-", "z+"],
        "periodic_faces": [],
    }

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = no_dynamics(f, f_pre, solid)

        if use_bfl:
            f = bounce_back_cells_3d(f, solid)
            f_pre_stream = f.clone()
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)
            f = bfl_bounce_back_common(
                f, f_pre_stream, bfl_mask, bfl_q, lattice="D3Q19"
            )
        else:
            f = bounce_back_cells_3d(f, solid)
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)

        if step % 200 == 0:
            f = correct_mass3d(f, im)

        if step > warmup:
            fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
            fx_f, fy_f, fz_f = drag_friction_bfl(f, mesh, dpS, nu, q_wall=q_wall)
            cd_p_hist.append(fx_p)
            cd_f_hist.append(fx_f)
            cd_tot_hist.append(fx_p + fx_f)
            cl_hist.append(fy_p + fy_f)

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

    elapsed = time.time() - t0
    n_final = min(1000, len(cd_tot_hist))
    if n_final == 0:
        n_final = 1
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final if cd_p_hist else float("nan")
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final if cd_f_hist else float("nan")
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final if cd_tot_hist else float("nan")
    cl_final = sum(cl_hist[-n_final:]) / n_final if cl_hist else float("nan")

    result = {
        "case": "kvlcc2_Re1000",
        "mode": mode,
        "device": f"sdaa:{device_id}",
        "Re": int(Re),
        "L": float(L),
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": float(u_in),
        "nu": float(nu),
        "tau": float(tau),
        "Cs": float(cs_smag),
        "n_steps": int(n_steps),
        "n_solid": int(n_solid),
        "n_near": int(n_near),
        "dpS": float(dpS),
        "Cd_pressure": float(cd_p_final) if cd_p_final == cd_p_final else None,
        "Cd_friction": float(cd_f_final) if cd_f_final == cd_f_final else None,
        "Cd_total": float(cd_tot_final) if cd_tot_final == cd_tot_final else None,
        "Cl": float(cl_final) if cl_final == cl_final else None,
        "bfl_stats": bfl_stats,
        "friction_formula": "nu*u_t/q" if use_bfl else "2*nu*u_t",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) < 4:
        print("Usage: python bfl_common_sdaa_worker.py <test> <device_id> <output_path>")
        print("  test: cylinder_bfl | cylinder_bb | sphere_bfl | sphere_bb |")
        print("        couette_bfl | kvlcc2_bfl | kvlcc2_bb")
        sys.exit(1)

    test = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if test == "cylinder_bfl":
        run_cylinder(device_id, "bfl", output_path)
    elif test == "cylinder_bb":
        run_cylinder(device_id, "bb", output_path)
    elif test == "sphere_bfl":
        run_sphere(device_id, "bfl", output_path)
    elif test == "sphere_bb":
        run_sphere(device_id, "bb", output_path)
    elif test == "couette_bfl":
        run_couette(device_id, output_path)
    elif test == "kvlcc2_bfl":
        run_kvlcc2(device_id, "bfl", output_path)
    elif test == "kvlcc2_bb":
        run_kvlcc2(device_id, "bb", output_path)
    else:
        print(f"Unknown test: {test}")
        sys.exit(1)


if __name__ == "__main__":
    main()
