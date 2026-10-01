#!/usr/bin/env python3
"""Complete BFL implementation: bounce-back + friction integrated.

Tests:
  TEST 1: Couette with BFL (SDAA:0) — flat walls, moving wall, q=0.5
  TEST 2: Cylinder with BFL (SDAA:1) + standard BB (SDAA:2)
  TEST 3: SUBOFF with BFL (SDAA:3)

Key fixes vs previous BFL friction fix:
  1. Moving-wall momentum correction added to BFL bounce-back
  2. Friction integration accounts for wall velocity (u_wall)
     τ = ν·(u_t - u_wall_t)/q  instead of  τ = ν·u_t/q
  3. Couette near-wall mask properly filtered (top wall only)

Usage:
  python bfl_complete_worker.py <test> <device_id> <output_path>
  test: couette_bfl | cylinder_bfl | cylinder_bb | suboff_bfl | suboff_bb
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    get_near_wall_2d,
    get_near_wall_3d,
    drag_pressure_integration,
)
from tensorlbm.bfl_d3q19_vec import bouzidi_bounce_back_d3q19_vec
from tensorlbm.bfl_d3q19 import compute_q_cylinder_d3q19


# ---------------------------------------------------------------------------
# Friction integration with wall-velocity correction
# ---------------------------------------------------------------------------
def drag_friction_with_uwall(f, mesh, dpS, nu, q_wall=None, u_wall=None):
    """Friction drag with wall-velocity correction.

    Standard (u_wall=None):  τ = ν·u_t/q   (assumes stationary wall)
    Corrected (u_wall given): τ = ν·(u_t - u_wall_t)/q

    where u_wall_t is the tangential wall velocity.

    For a moving wall (Couette), this gives the exact shear stress:
      τ = ν·(u_wall - u_fluid)/Δy  →  force on wall = ν·(u_t - u_wall)/q·A

    Args:
        f: distribution (19, nz, ny, nx)
        mesh: SurfaceMesh with normals
        dpS: dynamic pressure scale
        nu: kinematic viscosity
        q_wall: (nz, ny, nx) fractional wall distance, None→0.5
        u_wall: (3, nz, ny, nx) wall velocity components, or None

    Returns: (fx, fy, fz) drag coefficients
    """
    rho, ux, uy, uz = macroscopic3d(f)
    nx_n, ny_n, nz_n = mesh.nx_n, mesh.ny_n, mesh.nz_n

    # Tangential velocity: u_t = u - (u·n)·n
    u_dot_n = ux * nx_n + uy * ny_n + uz * nz_n
    ut_x = ux - u_dot_n * nx_n
    ut_y = uy - u_dot_n * ny_n
    ut_z = uz - u_dot_n * nz_n

    # Subtract wall tangential velocity
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

    mask = mesh.near.float() * mesh.dA
    ffx = (tau_x * mask).sum()
    ffy = (tau_y * mask).sum()
    ffz = (tau_z * mask).sum()
    return (
        float(ffx.item() / dpS),
        float(ffy.item() / dpS),
        float(ffz.item() / dpS),
    )


# ---------------------------------------------------------------------------
# BFL q-values for flat walls (Couette)
# ---------------------------------------------------------------------------
def compute_q_flat_walls(nx, ny, nz, device, wall_axis="y"):
    """BFL mask and q for flat walls at y=0 and y=ny-1.

    For flat walls, q=0.5 everywhere (half-way BB).
    Boundary links are directions with non-zero y-component.

    Returns: (fluid_boundary_mask, q_field) each (19, nz, ny, nx)
    """
    c = C.to(device).float()
    mask = torch.zeros((19, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((19, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    if wall_axis == "y":
        wall_dim = 1  # y
    elif wall_axis == "x":
        wall_dim = 0  # x
    elif wall_axis == "z":
        wall_dim = 2  # z
    else:
        raise ValueError(f"wall_axis must be x/y/z, got {wall_axis}")

    dims = [nz, ny, nx]
    dim_size = dims[wall_dim]

    for d in range(1, 19):
        cv = float(c[d, wall_dim].item())
        if cv == 0.0:
            continue

        if cv > 0:
            # Direction points toward +axis → boundary at top wall
            # Fluid cell at index dim_size-2, solid at dim_size-1
            fluid_idx = dim_size - 2
        else:
            # Direction points toward -axis → boundary at bottom wall
            # Fluid cell at index 1, solid at index 0
            fluid_idx = 1

        if wall_dim == 0:  # z
            mask[d, fluid_idx, :, :] = True
        elif wall_dim == 1:  # y
            mask[d, :, fluid_idx, :] = True
        elif wall_dim == 2:  # x
            mask[d, :, :, fluid_idx] = True

    return mask, q_field


# ---------------------------------------------------------------------------
# Moving-wall correction for BFL
# ---------------------------------------------------------------------------
def compute_moving_wall_correction(
    bfl_mask, device, u_w, wall_axis="y", top=True, rho_w=1.0
):
    """Compute wall_correction tensor for BFL moving wall.

    For a wall moving with velocity u_w=(uwx, uwy, uwz), the correction
    for the unknown population f[opp_d] is:
        corr[opp_d] = 2·ρ·w[opp_d]·(c[opp_d]·u_w)/cs²

    Since the BFL scatter sets f_out[e] = f_bc[opp[e]], the correction
    added to f_bc[d] is corr[opp[d]].

    Returns: (19, nz, ny, nx) tensor, zero except at moving-wall cells.
    """
    c = C.to(device).float()
    w = W.to(device).float()
    opp = OPPOSITE.to(device)
    cs2 = 1.0 / 3.0

    # correction[i] = 2*rho*w[i]*(c[i]·u_w)/cs2 = 6*rho*w[i]*(c[i]·u_w)
    c_dot_u = (
        c[:, 0] * u_w[0] + c[:, 1] * u_w[1] + c[:, 2] * u_w[2]
    )  # (19,)
    correction_dir = 6.0 * rho_w * w * c_dot_u  # (19,) = correction[i]

    # wall_correction[d] = correction[opp[d]]
    wall_corr_dir = correction_dir[opp]  # (19,)

    # Broadcast to (19, nz, ny, nx), masked to moving-wall boundary cells
    nz, ny, nx = bfl_mask.shape[1:]
    wall_corr = wall_corr_dir.view(19, 1, 1, 1).expand(19, nz, ny, nx).clone()

    # Only apply at moving-wall boundary cells
    # Find cells that are boundary cells for the moving wall
    # Determine wall dimension index
    if wall_axis == "y":
        wall_dim = 1
    elif wall_axis == "x":
        wall_dim = 0
    elif wall_axis == "z":
        wall_dim = 2
    else:
        raise ValueError(f"wall_axis must be x/y/z, got {wall_axis}")

    dims = [nz, ny, nx]
    dim_size = dims[wall_dim]

    # Moving wall is at top (index dim_size-1) or bottom (index 0)
    # Boundary cells are at dim_size-2 (top) or 1 (bottom)
    if top:
        fluid_idx = dim_size - 2
    else:
        fluid_idx = 1

    # Create mask for moving-wall boundary cells
    moving_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    if wall_dim == 0:
        moving_mask[fluid_idx, :, :] = True
    elif wall_dim == 1:
        moving_mask[:, fluid_idx, :] = True
    elif wall_dim == 2:
        moving_mask[:, :, fluid_idx] = True

    # Apply correction only at moving-wall cells AND where bfl_mask is True
    # (bfl_mask already restricts to boundary links)
    wall_corr = wall_corr * moving_mask.unsqueeze(0).float()

    return wall_corr


# ---------------------------------------------------------------------------
# NoDynamics: restore solid cells to pre-collision values
# ---------------------------------------------------------------------------
def no_dynamics(f, f_pre, solid):
    """Restore solid cells to pre-collision values."""
    sm = solid.unsqueeze(0).expand_as(f)
    return torch.where(sm, f_pre, f)


# ---------------------------------------------------------------------------
# TEST 1: Couette with BFL
# ---------------------------------------------------------------------------
def run_couette_bfl(device_id, output_path):
    """3D Couette flow with BFL bounce-back + moving wall.

    Flat walls → q=0.5 everywhere → BFL reduces to standard BB.
    Moving wall correction added to BFL.
    Friction: τ=ν·(u_t - u_wall)/q with u_wall=u_top at top, 0 at bottom.
    """
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
    solid[:, 0, :] = True    # bottom wall (y=0)
    solid[:, -1, :] = True   # top wall (y=ny-1)

    # Near-wall mask (3D)
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # Surface mesh with from_gradient normals
    mesh_all = SurfaceMesh.from_gradient(solid, near)

    # BFL mask and q for flat walls
    bfl_mask, bfl_q = compute_q_flat_walls(nx, ny, nz, device, wall_axis="y")
    n_links = int(bfl_mask.sum().item())
    print(f"{tag} BFL links={n_links} (q=0.5 everywhere, flat walls)", flush=True)

    # Moving wall correction for top wall
    u_w = (u_top, 0.0, 0.0)
    wall_corr = compute_moving_wall_correction(
        bfl_mask, device, u_w, wall_axis="y", top=True, rho_w=1.0
    )
    print(f"{tag} moving wall correction computed (u_top={u_top})", flush=True)

    # q_wall for friction: 0.5 everywhere at near-wall cells
    q_wall = torch.full((nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    # u_wall field for friction: (3, nz, ny, nx)
    u_wall_field = torch.zeros((3, nz, ny, nx), dtype=torch.float32, device=device)
    # Top wall near-cells (y=ny-2) have u_wall=(u_top, 0, 0)
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
        # 1. Save pre-collision
        f_pre = f.clone()

        # 2. Collision (BGK)
        from tensorlbm.solver3d import collide_bgk3d
        f = collide_bgk3d(f, tau=tau)

        # 3. NoDynamics: restore solid cells
        f = no_dynamics(f, f_pre, solid)

        # 4. BFL bounce-back (replaces standard BB)
        #    Key: bounce_back solid cells BEFORE streaming so BFL receives
        #    properly bounced-back values. Then stream, then BFL interpolates.
        f = bounce_back_cells_3d(f, solid)
        f_pre_stream = f.clone()
        f = stream3d(f)
        # Periodic in x and z (torch.roll handles this)
        f = bouzidi_bounce_back_d3q19_vec(
            f, f_pre_stream, bfl_mask, bfl_q, wall_correction=wall_corr
        )

        # 5. Mass correction
        if step % 200 == 0:
            im = float(rho0.sum().item())
            f = correct_mass3d(f, im)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Record post-warmup
        if step > warmup:
            # Compute Cf on top wall only
            near_top = near.clone()
            near_top[:, : ny // 2 + 1, :] = False  # keep only top wall cells
            mesh_top = SurfaceMesh.from_gradient(solid, near_top)

            A_wall = nx * nz
            dpS_wall = 0.5 * 1.0 * u_top ** 2 * A_wall

            # u_wall field for top wall only
            u_wall_top = torch.zeros((3, nz, ny, nx), dtype=torch.float32, device=device)
            u_wall_top[0, :, ny - 2, :] = u_top

            ffx, _, _ = drag_friction_with_uwall(
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
    import numpy as np
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


# ---------------------------------------------------------------------------
# TEST 2: Cylinder with BFL
# ---------------------------------------------------------------------------
def run_cylinder(device_id, mode, output_path):
    """Cylinder Re=200 with BFL or standard BB.

    mode: 'bfl' or 'standard_bb'
    """
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

    # BFL q-values
    bfl_mask = None
    bfl_q = None
    q_wall = None
    bfl_stats = {}
    if use_bfl:
        print(f"{tag} computing BFL q-values...", flush=True)
        t_q = time.time()
        bfl_mask, bfl_q = compute_q_cylinder_d3q19(
            nx, ny, nz, cx_c, cy_c, radius, device, axis="z"
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

        # Effective q_wall per cell
        mask_f = bfl_mask.float()
        q_weighted = (bfl_q * mask_f).sum(dim=0)
        n_dirs_raw = mask_f.sum(dim=0)
        n_dirs = n_dirs_raw.clamp(min=1.0)
        q_avg = q_weighted / n_dirs
        q_avg = torch.where(n_dirs_raw > 0, q_avg, torch.full_like(q_avg, 0.5))
        q_wall = torch.full((nz, ny, nx), 0.5, dtype=torch.float32, device=device)
        q_wall = torch.where(near, q_avg, q_wall)
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

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        # BGK collision
        from tensorlbm.solver3d import collide_bgk3d
        f = collide_bgk3d(f, tau=tau)
        # NoDynamics
        f = no_dynamics(f, f_pre, solid)

        if use_bfl:
            # BFL: bounce_back → stream → far_field → BFL interpolation
            # bounce_back solid cells BEFORE streaming so BFL receives
            # properly bounced-back values from solid cells
            f = bounce_back_cells_3d(f, solid)
            f_pre_stream = f.clone()
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in, bc_config=bc_config)
            f = bouzidi_bounce_back_d3q19_vec(f, f_pre_stream, bfl_mask, bfl_q)
        else:
            # Standard: bounce_back → stream → far_field
            f = bounce_back_cells_3d(f, solid)
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in, bc_config=bc_config)

        if step % 200 == 0:
            f = correct_mass3d(f, im)

        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, _ = drag_friction_with_uwall(
            f, mesh, dpS, nu, q_wall=q_wall, u_wall=None
        )

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
        "case": tag,
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


# ---------------------------------------------------------------------------
# TEST 3: SUBOFF with BFL
# ---------------------------------------------------------------------------
def run_suboff(device_id, mode, output_path):
    """SUBOFF Re=1000 with BFL or standard BB.

    mode: 'bfl' or 'standard_bb'
    """
    from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig
    from tensorlbm.interpolated_bc_suboff import compute_q_suboff
    from tensorlbm.turbulence import collide_smagorinsky_mrt3d

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    L = 80
    nx, ny, nz = 200, 80, 80
    Re = 1000
    u_in = 0.06
    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    n_steps = 5000
    Cf_ref = 0.042

    config = SuboffConfig()
    radius = config.r_over_l * L
    D = 2.0 * radius
    cx = nx * 0.30
    cy = ny * 0.5
    cz = nz * 0.5
    dpS = 0.5 * u_in ** 2 * math.pi * D * L

    use_bfl = mode == "bfl"
    tag = f"[SUBOFF-Re1000-{mode} SDAA:{device_id}]"
    print(
        f"{tag} nx={nx} ny={ny} nz={nz} L={L} R_max={radius:.4f} D={D:.4f} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
        f"dpS={dpS:.6e} Cf_ref={Cf_ref} use_bfl={use_bfl}",
        flush=True,
    )

    t0 = time.time()
    solid, stats = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=L, radius=radius,
        config=config, device=device,
    )
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}  L/D={stats['L_D_ratio']}", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)

    # BFL q-values
    bfl_mask = None
    bfl_q = None
    q_wall = None
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
        }
        print(
            f"{tag} BFL q-field: {n_links} links ({time.time()-t_q:.1f}s) "
            f"q=[{bfl_stats['q_min']:.4f}, {bfl_stats['q_max']:.4f}] "
            f"mean={bfl_stats['q_mean']:.4f}",
            flush=True,
        )
        bfl_mask = bfl_mask.to(device)
        bfl_q = bfl_q.to(device)

        # Effective q_wall per cell
        mask_f = bfl_mask.float()
        q_weighted = (bfl_q * mask_f).sum(dim=0)
        n_dirs_raw = mask_f.sum(dim=0)
        n_dirs = n_dirs_raw.clamp(min=1.0)
        q_avg = q_weighted / n_dirs
        q_avg = torch.where(n_dirs_raw > 0, q_avg, torch.full_like(q_avg, 0.5))
        q_wall = torch.full((nz, ny, nx), 0.5, dtype=torch.float32, device=device)
        q_wall = torch.where(near, q_avg, q_wall)
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
        # NoDynamics
        f = no_dynamics(f, f_pre, solid)

        if use_bfl:
            # BFL: bounce_back → stream → far_field → BFL interpolation
            f = bounce_back_cells_3d(f, solid)
            f_pre_stream = f.clone()
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in)
            f = bouzidi_bounce_back_d3q19_vec(f, f_pre_stream, bfl_mask, bfl_q)
        else:
            f = bounce_back_cells_3d(f, solid)
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in)

        if step % 200 == 0:
            f = correct_mass3d(f, im)

        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, fz_f = drag_friction_with_uwall(
            f, mesh, dpS, nu, q_wall=q_wall, u_wall=None
        )

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
    n_final = min(1000, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final
    fz_final = sum(fz_hist[-n_final:]) / n_final

    err_pct = abs(cd_tot_final - Cf_ref) / Cf_ref * 100 if Cf_ref > 0 else float("nan")

    result = {
        "case": tag,
        "mode": mode,
        "device": f"sdaa:{device_id}",
        "Re": int(Re),
        "L": int(L),
        "R_max": float(radius),
        "D": float(D),
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
        "fz": float(fz_final) if fz_final == fz_final else None,
        "Cf_ref": float(Cf_ref),
        "error_pct": float(err_pct) if err_pct == err_pct else None,
        "bfl_stats": bfl_stats,
        "friction_formula": "nu*u_t/q" if use_bfl else "2*nu*u_t",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"(ref Cf={Cf_ref}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print("Usage: python bfl_complete_worker.py <test> <device_id> <output_path>")
        print("  test: couette_bfl | cylinder_bfl | cylinder_bb | suboff_bfl | suboff_bb")
        sys.exit(1)

    test = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if test == "couette_bfl":
        run_couette_bfl(device_id, output_path)
    elif test == "cylinder_bfl":
        run_cylinder(device_id, "bfl", output_path)
    elif test == "cylinder_bb":
        run_cylinder(device_id, "standard_bb", output_path)
    elif test == "suboff_bfl":
        run_suboff(device_id, "bfl", output_path)
    elif test == "suboff_bb":
        run_suboff(device_id, "standard_bb", output_path)
    else:
        print(f"Unknown test: {test}")
        sys.exit(1)


if __name__ == "__main__":
    main()
