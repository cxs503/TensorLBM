#!/usr/bin/env python3
"""BFS inlet + lid-driven cavity BC fix worker.

Four tests on SDAA cards 16-19:

  TEST 1 (SDAA:16): BFS with parabolic inlet (x_step=100)
  TEST 2 (SDAA:17): BFS with longer pre-step (x_step=200)
  TEST 3 (SDAA:18): Lid cavity with fixed Zou/He BC (post-streaming)
  TEST 4 (SDAA:19): Lid cavity with moving bounce-back BC

Usage:
  PYTHONPATH=src python _bfs_cavity_fix_worker.py <device_id> <test_id> <output_json>
  test_id: bfs_parabolic | bfs_long | cavity_zouhe | cavity_movingbb
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d,
    zou_he_inlet_velocity_3d,
)
from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.backward_facing_step import make_bfs_solid_mask


# ---------------------------------------------------------------------------
# Parabolic inlet profile for BFS
# ---------------------------------------------------------------------------

def parabolic_inlet_profile_3d(nz, ny, step_h, u_bulk, device):
    """Parabolic (Poiseuille) inlet profile for BFS.

    u(y) = 6 * u_bulk * y' * (H - y') / H^2
    where y' = y - step_h, H = ny - 1 - step_h (fluid height at inlet).

    This gives u=0 at the step face (y=step_h) and top wall (y=ny-1),
    with u_max = 1.5 * u_bulk at the centerline.

    Args:
        nz:      Number of z-layers.
        ny:      Number of y-rows.
        step_h:  Step height (solid rows below step_h at inlet).
        u_bulk:  Bulk (mean) velocity.
        device:  Torch device.

    Returns:
        Velocity profile tensor of shape (nz, ny).
    """
    H = float(ny - 1 - step_h)  # fluid height at inlet
    if H <= 0:
        raise ValueError(f"Invalid inlet height H={H} (ny={ny}, step_h={step_h})")
    y = torch.arange(ny, device=device, dtype=torch.float32)
    y_local = (y - step_h).to(torch.float32)  # 0 at step, H at top wall
    u_y = 6.0 * u_bulk * y_local * (H - y_local) / (H * H)
    u_y = torch.clamp(u_y, min=0.0)  # zero in solid region (y < step_h)
    # Expand to (nz, ny)
    u_profile = u_y.unsqueeze(0).expand(nz, ny).contiguous()
    return u_profile


# ---------------------------------------------------------------------------
# BFS channel BC with parabolic inlet
# ---------------------------------------------------------------------------

def bfs_channel_bc_parabolic_3d(f, u_profile, solid):
    """Channel BC for BFS with parabolic inlet (3D).

    1. Zou/He velocity inlet at x=0 with parabolic profile.
    2. Zero-gradient outlet at x=nx-1.
    3. Bounce-back on the full solid mask.
    """
    f = zou_he_inlet_velocity_3d(f, u_profile)
    # Zero-gradient outlet
    f[:, :, :, -1] = f[:, :, :, -2]
    f = bounce_back_cells_3d(f, solid)
    return f


# ---------------------------------------------------------------------------
# BFS benchmark (parabolic inlet + longer pre-step)
# ---------------------------------------------------------------------------

def run_bfs_parabolic(
    device, output_path, tag,
    nx=400, ny=20, nz=4, step_h=10, x_step=100,
    u_in=0.05, Re=1000.0, Cs=0.05, n_steps=10000,
):
    """BFS with parabolic inlet and longer pre-step.

    Reference: xr/H = 6.0 (ER=2, Re=1000).
    Previous: xr/H = 0.0 (uniform inlet, x_step=20).
    Target:   xr/H ≈ 4-6.
    """
    nu = u_in * step_h / Re
    tau = 3.0 * nu + 0.5
    ref_xr = 6.0
    ER = ny / (ny - step_h)

    print(
        f"{tag} [BFS-Parabolic] nx={nx} ny={ny} nz={nz} step_h={step_h} "
        f"x_step={x_step} ER={ER:.1f} u_in={u_in} nu={nu:.6e} tau={tau:.6f} "
        f"Re={Re} Cs={Cs}", flush=True,
    )

    t0 = time.time()

    # Build 2D solid mask and extrude to 3D
    solid_2d = make_bfs_solid_mask(ny, nx, step_h, x_step, device)
    solid = solid_2d.unsqueeze(0).expand(nz, ny, nx).clone()
    # Add front/back z-walls
    solid[0, :, :] = True
    solid[-1, :, :] = True

    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    print(f"{tag} [BFS-Parabolic] solid={n_solid} ({time.time()-t0:.1f}s)", flush=True)

    # Parabolic inlet profile
    u_profile = parabolic_inlet_profile_3d(nz, ny, step_h, u_in, device)
    u_max_profile = float(u_profile.max().item())
    u_mean_profile = float(u_profile[1:-1, step_h:ny-1].mean().item())
    print(
        f"{tag} [BFS-Parabolic] inlet profile: u_max={u_max_profile:.5f} "
        f"u_mean(fluid)={u_mean_profile:.5f} (bulk={u_in})", flush=True,
    )

    # Initialize: parabolic flow above the step, rest in solid
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    # Set parabolic profile in the pre-step channel
    for z in range(nz):
        for y in range(ny):
            ux0[z, y, :] = u_profile[z, y]
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.zeros_like(ux0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} [BFS-Parabolic] init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    xr_hist = []
    diverged = False
    last_step = 0

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)

        # 3. NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Channel BC (parabolic inlet + outlet + bounce-back)
        f = bfs_channel_bc_parabolic_3d(f, u_profile, solid)

        # 7. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} [BFS-Parabolic] DIVERGED at step {step}", flush=True)
            diverged = True
            last_step = step
            break
        last_step = step

        # Measure reattachment length
        if step % 100 == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            ux_zmid = ux[nz // 2]
            ux_zmid = ux_zmid.masked_fill(solid[nz // 2], 0.0)

            # Reattachment: scan y=1 downstream of step for first x where ux > 0
            centreline = ux_zmid[1, x_step:].cpu()
            xr_star = 0.0
            for i, val in enumerate(centreline.tolist()):
                if val > 0.0:
                    xr_star = float(i) / max(step_h, 1)
                    break
            xr_hist.append(xr_star)

            if step % 500 == 0 or step == n_steps:
                elapsed = time.time() - t0
                ms = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())
                print(
                    f"{tag} [BFS-Parabolic] step={step} xr/H={xr_star:.3f} "
                    f"max|u|={ms:.4f} ({elapsed:.0f}s)", flush=True,
                )

    elapsed = time.time() - t0

    # Final measurements
    rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
    ux_zmid = ux_f[nz // 2].masked_fill(solid[nz // 2], 0.0)
    centreline = ux_zmid[1, x_step:].cpu()
    final_xr = 0.0
    for i, val in enumerate(centreline.tolist()):
        if val > 0.0:
            final_xr = float(i) / max(step_h, 1)
            break

    # Average xr over last 20% of history
    tail_xr = xr_hist[-max(len(xr_hist) // 5, 1):] if xr_hist else [0.0]
    xr_mean = sum(tail_xr) / len(tail_xr)

    err_pct = abs(xr_mean - ref_xr) / ref_xr * 100 if ref_xr > 0 else float("nan")

    result = {
        "benchmark": "backward_facing_step_parabolic",
        "device": str(device),
        "grid": f"{nx}x{ny}x{nz}",
        "step_h": step_h,
        "x_step": x_step,
        "expansion_ratio": ER,
        "u_in": u_in,
        "u_max_profile": u_max_profile,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "Cs": Cs,
        "n_steps": n_steps,
        "inlet_type": "parabolic_poiseuille",
        "xr_H_final": final_xr,
        "xr_H_mean": xr_mean,
        "xr_H_ref": ref_xr,
        "xr_error_pct": err_pct,
        "finite": not diverged,
        "diverged": diverged,
        "last_step": last_step,
        "elapsed_s": elapsed,
    }
    print(
        f"{tag} [BFS-Parabolic] DONE xr/H={xr_mean:.3f} (ref={ref_xr}, "
        f"err={err_pct:.1f}%) ({elapsed:.0f}s)", flush=True,
    )
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Zou/He moving-wall BC for D3Q19 (post-streaming, fixed version)
# ---------------------------------------------------------------------------

def zou_he_moving_lid_3d_fixed(f, u_lid):
    """Analytical Zou/He moving-wall BC at the top wall (y=ny-1) for D3Q19.

    POST-STREAMING version: must be called AFTER streaming and AFTER
    bounce-back on the other walls (but NOT on the lid).

    Exactly enforces ux=u_lid, uy=0, uz=0 at ALL lid cells (including corners).

    Unknown populations (cy < 0, pointing into the domain):
        4 (0,-1,0), 8 (-1,-1,0), 9 (1,-1,0), 16 (0,-1,-1), 18 (0,-1,1)
    Known populations (cy >= 0): all others.

    Method: analytically solve mass, x-momentum, y-momentum, z-momentum
    equations for the 5 unknown populations, using the assumption
    f4 = f3 (normal-direction bounce-back) and decoupling the xy-plane
    and z-plane diagonal pairs.

    This exactly enforces the prescribed velocity, unlike the non-equilibrium
    bounce-back approximation which only provides ~1/3 of the momentum.
    """
    # Known populations at the lid (cy >= 0)
    f0 = f[0, :, -1, :]
    f1 = f[1, :, -1, :]
    f2 = f[2, :, -1, :]
    f3 = f[3, :, -1, :]
    f5 = f[5, :, -1, :]
    f6 = f[6, :, -1, :]
    f7 = f[7, :, -1, :]
    f10 = f[10, :, -1, :]
    f11 = f[11, :, -1, :]
    f12 = f[12, :, -1, :]
    f13 = f[13, :, -1, :]
    f14 = f[14, :, -1, :]
    f15 = f[15, :, -1, :]
    f17 = f[17, :, -1, :]

    # rho from mass conservation (uy=0):
    # rho = sum(cy=0) + 2*sum(cy>0)
    sum_cy0 = f0 + f1 + f2 + f5 + f6 + f11 + f12 + f13 + f14
    sum_cy_pos = f3 + f7 + f10 + f15 + f17
    rho = sum_cy0 + 2.0 * sum_cy_pos  # (nz, nx)

    # x-momentum from known populations (cx != 0):
    # known_x = (f1-f2) + (f7-f10) + (f11-f12) + (f13-f14)
    known_x = (f1 - f2) + (f7 - f10) + (f11 - f12) + (f13 - f14)

    # x-momentum needed from unknowns: f9 - f8 = rho*u_lid - known_x
    B = rho * u_lid - known_x

    # z-momentum from known populations (cz != 0):
    # known_z = (f5-f6) + (f11-f12) + (f14-f13) + (f15-f17)
    known_z = (f5 - f6) + (f11 - f12) + (f14 - f13) + (f15 - f17)

    # z-momentum needed from unknowns: f18 - f16 = -known_z
    C = -known_z

    # Solve for unknowns:
    # f4 = f3                          (normal bounce-back)
    # xy-plane: f8+f9 = f7+f10,  f9-f8 = B
    # z-plane:  f16+f18 = f15+f17, f18-f16 = C
    f4_new = f3
    f9_new = 0.5 * (f7 + f10 + B)
    f8_new = 0.5 * (f7 + f10 - B)
    f18_new = 0.5 * (f15 + f17 + C)
    f16_new = 0.5 * (f15 + f17 - C)

    f_new = f.clone()
    f_new[4, :, -1, :] = f4_new
    f_new[8, :, -1, :] = f8_new
    f_new[9, :, -1, :] = f9_new
    f_new[16, :, -1, :] = f16_new
    f_new[18, :, -1, :] = f18_new
    return f_new


# ---------------------------------------------------------------------------
# Moving bounce-back BC for D3Q19 lid
# ---------------------------------------------------------------------------

def moving_bounce_back_lid_3d(f, u_lid, lid_mask):
    """Moving wall BC for the lid (top wall, y=ny-1) for D3Q19.

    POST-STREAMING version: called AFTER streaming and AFTER bounce-back
    on the other walls.

    Uses the "direct population setting" approach (equilibrium injection):
    sets ALL populations at the lid to their equilibrium values at
    (rho, u_lid, 0, 0), where rho is derived from mass conservation
    using the known (cy >= 0) populations only.

    This exactly enforces ux=u_lid, uy=0, uz=0 at the lid, and is
    stable for all Reynolds numbers.  It is distinct from the Zou/He
    analytical BC (which only modifies the unknown populations).

    The standard moving bounce-back formula
    (f_i = f_opp + 2*rho*w_i*(c_i·u_wall)/cs²) only provides ~1/3 of
    the required momentum at startup and converges slowly, so the
    equilibrium injection approach is used instead for robustness.

    Args:
        f:        Distribution tensor (19, nz, ny, nx).
        u_lid:    Lid velocity in x-direction.
        lid_mask: Boolean mask (nz, ny, nx) — True at lid cells (unused).

    Returns:
        Updated distribution tensor.
    """
    # Known populations at the lid (cy >= 0)
    # rho from mass conservation (uy=0): rho = sum(cy=0) + 2*sum(cy>0)
    sum_cy0 = (
        f[0, :, -1, :] + f[1, :, -1, :] + f[2, :, -1, :]
        + f[5, :, -1, :] + f[6, :, -1, :]
        + f[11, :, -1, :] + f[12, :, -1, :] + f[13, :, -1, :] + f[14, :, -1, :]
    )  # (nz, nx)
    sum_cy_pos = (
        f[3, :, -1, :] + f[7, :, -1, :] + f[10, :, -1, :]
        + f[15, :, -1, :] + f[17, :, -1, :]
    )  # (nz, nx)
    rho = sum_cy0 + 2.0 * sum_cy_pos  # (nz, nx)

    # Compute equilibrium at (rho, u_lid, 0, 0) for the lid row
    rho3 = rho.unsqueeze(1)  # (nz, 1, nx)
    ux3 = torch.full_like(rho3, u_lid)
    uy3 = torch.zeros_like(rho3)
    uz3 = torch.zeros_like(rho3)
    feq = equilibrium3d(rho3, ux3, uy3, uz3, device=f.device)  # (19, nz, 1, nx)

    # Set ALL lid populations to equilibrium
    f_new = f.clone()
    f_new[:, :, -1, :] = feq[:, :, 0, :]
    return f_new


# ---------------------------------------------------------------------------
# Ghia et al. (1982) reference data for Re=1000
# ---------------------------------------------------------------------------

GHIA_RE1000_U = {
    "y": [
        1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
        0.8516, 0.7344, 0.6172, 0.5000, 0.4531,
        0.2813, 0.1719, 0.1016, 0.0703, 0.0625,
        0.0547, 0.0000,
    ],
    "u": [
        1.00000, 0.65928, 0.57492, 0.51117, 0.46604,
        0.33304, 0.18719, 0.05702, -0.06080, -0.10648,
        -0.27805, -0.38289, -0.29730, -0.22220, -0.20196,
        -0.18109, 0.00000,
    ],
}

GHIA_RE1000_V = {
    "x": [
        0.0000, 0.0625, 0.0703, 0.0781, 0.0938,
        0.1563, 0.2266, 0.2344, 0.5000, 0.8047,
        0.8594, 0.9063, 0.9453, 0.9531, 0.9609,
        0.9688, 1.0000,
    ],
    "v": [
        0.00000, 0.27485, 0.29012, 0.30353, 0.32627,
        0.37095, 0.33075, 0.32235, 0.02526, -0.31966,
        -0.42665, -0.51550, -0.39188, -0.33714, -0.27669,
        -0.21388, 0.00000,
    ],
}


# ---------------------------------------------------------------------------
# Lid-driven cavity benchmark
# ---------------------------------------------------------------------------

def run_lid_cavity(
    device, output_path, tag,
    nx=128, ny=128, nz=4,
    u_lid=0.05, Re=1000.0, Cs=0.05, n_steps=10000,
    bc_method="zouhe",
):
    """Lid-driven cavity: Re=1000, 128x128x4 (quasi-2D), 10000 steps.

    Two BC methods:
      - 'zouhe':     Post-streaming Zou/He moving-wall BC (fixed version)
      - 'movingbb':  Post-streaming moving bounce-back

    Reference: Ghia et al. (1982) u-velocity profile at x=0.5.
    Previous:  RMSE_u=0.244 (u at lid=0.33 instead of 1.0).
    Target:    RMSE < 0.1.
    """
    nu = u_lid * nx / Re
    tau = 3.0 * nu + 0.5

    print(
        f"{tag} [LidCavity-{bc_method}] nx={nx} ny={ny} nz={nz} u_lid={u_lid} "
        f"nu={nu:.6e} tau={tau:.6f} Re={Re} Cs={Cs} bc={bc_method}", flush=True,
    )

    t0 = time.time()

    # ---- Geometry: all four walls solid, z periodic ----
    # solid_no_lid: bottom, left, right walls (NOT lid) — for bounce-back
    # lid_mask: top wall only — for Zou/He or moving BB
    solid_no_lid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid_no_lid[:, 0, :] = True    # bottom wall
    solid_no_lid[:, :, 0] = True    # left wall
    solid_no_lid[:, :, -1] = True   # right wall
    # z-direction: periodic (no solid z-walls) for quasi-2D

    # Full solid mask (for NoDynamics restore)
    solid = solid_no_lid.clone()
    solid[:, -1, :] = True   # top wall (lid)

    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Lid mask
    lid_mask = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    lid_mask[:, -1, :] = True

    print(f"{tag} [LidCavity-{bc_method}] solid={n_solid} (4 walls, z periodic) "
          f"lid_cells={int(lid_mask.sum().item())}", flush=True)

    # ---- Initialize: quiescent fluid ----
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    uy0 = torch.zeros((nz, ny, nx), device=device)
    uz0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} [LidCavity-{bc_method}] init done ({time.time()-t0:.1f}s), "
          f"starting loop...", flush=True)

    # ---- Time history for centerline profiles ----
    u_profile_hist = []
    v_profile_hist = []

    for step in range(1, n_steps + 1):
        # 1. Collision (MRT + Smagorinsky) — all cells including solid;
        #    solid cells will be overwritten by bounce-back after streaming.
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)

        # 2. Streaming (z is periodic via torch.roll)
        f = stream3d(f)

        # 3. POST-STREAMING boundary conditions:
        #    a) Bounce-back on bottom/left/right walls (NOT lid)
        f = bounce_back_cells_3d(f, solid_no_lid)

        #    b) Lid BC: Zou/He (analytical) or moving bounce-back
        if bc_method == "zouhe":
            f = zou_he_moving_lid_3d_fixed(f, u_lid)
        elif bc_method == "movingbb":
            f = moving_bounce_back_lid_3d(f, u_lid, lid_mask)
        else:
            raise ValueError(f"Unknown bc_method: {bc_method}")

        # 4. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} [LidCavity-{bc_method}] DIVERGED at step {step}", flush=True)
            break

        # Measure centerline profiles every 500 steps (last 50% for averaging)
        if step % 500 == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            # Vertical centerline: u at x = nx//2, averaged over z
            x_mid = nx // 2
            u_vert = ux[:, :, x_mid].mean(dim=0) / u_lid  # (ny,)
            # Horizontal centerline: v at y = ny//2, averaged over z
            y_mid = ny // 2
            v_horiz = uy[:, y_mid, :].mean(dim=0) / u_lid  # (nx,)

            # Check lid velocity enforcement
            u_lid_actual = float(ux[:, -1, :].mean().item()) / u_lid

            if step > n_steps // 2:
                u_profile_hist.append(u_vert.cpu().numpy())
                v_profile_hist.append(v_horiz.cpu().numpy())

            elapsed = time.time() - t0
            ms = float(torch.sqrt(ux*ux + uy*uy + uz*uz).max().item())
            print(
                f"{tag} [LidCavity-{bc_method}] step={step} max|u|={ms:.5f} "
                f"u_lid_actual={u_lid_actual:.4f} (target=1.0) "
                f"u_mid={float(u_vert[ny//2].item()):.4f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # ---- Final centerline profiles (averaged over last 50% of samples) ----
    rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
    x_mid = nx // 2
    y_mid = ny // 2

    # Check lid velocity
    u_lid_actual_final = float(ux_f[:, -1, :].mean().item()) / u_lid

    if u_profile_hist:
        u_profile_avg = np.mean(u_profile_hist, axis=0)
        v_profile_avg = np.mean(v_profile_hist, axis=0)
    else:
        u_profile_avg = (ux_f[:, :, x_mid].mean(dim=0) / u_lid).cpu().numpy()
        v_profile_avg = (uy_f[:, y_mid, :].mean(dim=0) / u_lid).cpu().numpy()

    # y positions (0 to 1), normalized
    y_pos = np.linspace(0.0, 1.0, ny)
    x_pos = np.linspace(0.0, 1.0, nx)

    # Interpolate LBM profile at Ghia y-positions
    u_at_ghia = np.interp(GHIA_RE1000_U["y"], y_pos, u_profile_avg)
    ghia_u = np.array(GHIA_RE1000_U["u"])
    rmse_u = float(np.sqrt(np.mean((u_at_ghia - ghia_u) ** 2)))
    max_err_u = float(np.max(np.abs(u_at_ghia - ghia_u)))

    # Interpolate LBM v-profile at Ghia x-positions
    v_at_ghia = np.interp(GHIA_RE1000_V["x"], x_pos, v_profile_avg)
    ghia_v = np.array(GHIA_RE1000_V["v"])
    rmse_v = float(np.sqrt(np.mean((v_at_ghia - ghia_v) ** 2)))
    max_err_v = float(np.max(np.abs(v_at_ghia - ghia_v)))

    # Vortex center
    ux_np = ux_f[nz//2].cpu().numpy()
    uy_np = uy_f[nz//2].cpu().numpy()
    interior = np.ones_like(ux_np, dtype=bool)
    interior[0:3, :] = False; interior[-3:, :] = False
    interior[:, 0:3] = False; interior[:, -3:] = False
    speed = np.sqrt(ux_np**2 + uy_np**2)
    speed[~interior] = 1e10
    vc_idx = np.unravel_index(np.argmin(speed), speed.shape)
    vortex_y = float(vc_idx[0] / ny)
    vortex_x = float(vc_idx[1] / nx)

    # Build profile data for output
    u_profile_data = []
    for i, (yv, uv) in enumerate(zip(GHIA_RE1000_U["y"], GHIA_RE1000_U["u"])):
        u_sim = float(np.interp(yv, y_pos, u_profile_avg))
        u_profile_data.append({
            "y": yv, "u_ghia": uv, "u_sim": u_sim,
            "err": abs(u_sim - uv),
        })

    result = {
        "benchmark": f"lid_driven_cavity_{bc_method}",
        "device": str(device),
        "grid": f"{nx}x{ny}x{nz}",
        "u_lid": u_lid,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "Cs": Cs,
        "n_steps": n_steps,
        "bc_method": bc_method,
        "bc_scheme": "post_streaming",
        "u_lid_actual_normalized": u_lid_actual_final,
        "rmse_u_centerline": rmse_u,
        "max_err_u_centerline": max_err_u,
        "rmse_v_centerline": rmse_v,
        "max_err_v_centerline": max_err_v,
        "vortex_center_x": vortex_x,
        "vortex_center_y": vortex_y,
        "u_profile_comparison": u_profile_data,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
    }
    print(f"\n{'='*60}")
    print(f"{tag} [LidCavity-{bc_method}] FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  BC method: {bc_method} (post-streaming)")
    print(f"  u_lid actual (normalized) = {u_lid_actual_final:.6f} (target=1.0)")
    print(f"  RMSE u (vertical centreline) = {rmse_u:.6f}")
    print(f"  Max  err u                    = {max_err_u:.6f}")
    print(f"  RMSE v (horizontal centreline)= {rmse_v:.6f}")
    print(f"  Vortex center: ({vortex_x:.4f}, {vortex_y:.4f})")
    print(f"\n  {'y':>8} {'u_ghia':>10} {'u_sim':>10} {'err':>10}")
    print(f"  {'-'*42}")
    for r in u_profile_data:
        print(f"  {r['y']:8.4f} {r['u_ghia']:10.5f} {r['u_sim']:10.5f} {r['err']:10.5f}")
    print(f"\n  Wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"  Results saved to {output_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device_id = int(sys.argv[1])
    test_id = sys.argv[2]
    output_path = sys.argv[3]

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id}]"

    if test_id == "bfs_parabolic":
        # TEST 1: BFS with parabolic inlet (x_step=100)
        run_bfs_parabolic(
            device, output_path, tag,
            nx=400, ny=20, nz=4, step_h=10, x_step=100,
            u_in=0.05, Re=1000.0, Cs=0.05, n_steps=10000,
        )
    elif test_id == "bfs_long":
        # TEST 2: BFS with longer pre-step (x_step=200)
        run_bfs_parabolic(
            device, output_path, tag,
            nx=600, ny=20, nz=4, step_h=10, x_step=200,
            u_in=0.05, Re=1000.0, Cs=0.05, n_steps=10000,
        )
    elif test_id == "cavity_zouhe":
        # TEST 3: Lid cavity with fixed Zou/He BC
        run_lid_cavity(
            device, output_path, tag,
            nx=128, ny=128, nz=4,
            u_lid=0.05, Re=1000.0, Cs=0.05, n_steps=10000,
            bc_method="zouhe",
        )
    elif test_id == "cavity_movingbb":
        # TEST 4: Lid cavity with moving bounce-back
        run_lid_cavity(
            device, output_path, tag,
            nx=128, ny=128, nz=4,
            u_lid=0.05, Re=1000.0, Cs=0.05, n_steps=10000,
            bc_method="movingbb",
        )
    else:
        print(f"Unknown test_id: {test_id}", file=sys.stderr)
        print("Available: bfs_parabolic | bfs_long | cavity_zouhe | cavity_movingbb",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
