#!/usr/bin/env python3
"""Lid-driven cavity + cosine hill benchmark worker.

Two classic CFD benchmarks using D3Q19 MRT+Smagorinsky on SDAA:

  1. lid_driven_cavity  — Re=1000, 128x128x4 (quasi-2D), 10000 steps
     Reference: Ghia et al. (1982) u-velocity profile at x=0.5
     Moving top wall via Zou/He moving-wall BC (D3Q19).

  2. cosine_hill        — Re=100/500, 300x100x4, 5000 steps
     Hill: h(x) = H*cos²(πx/L) for |x|<L/2, H=20, L=100
     Reference: separation length, reattachment point

Usage:
  PYTHONPATH=src python lid_cavity_cosine_hill_worker.py <device_id> <benchmark> <output_json>
  benchmark: lid_driven_cavity | cosine_hill
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
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, OPPOSITE
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
    get_near_wall_3d,
    get_near_wall_2d,
)


# ---------------------------------------------------------------------------
# Zou/He moving-wall BC for D3Q19 (lid-driven cavity top wall)
# ---------------------------------------------------------------------------

def zou_he_moving_lid_3d(f, u_lid):
    """Zou/He moving-wall BC at the top wall (y=ny-1) for D3Q19.

    Prescribes ux=u_lid, uy=0, uz=0 at interior lid cells (x=1..nx-2).
    Corner cells (x=0, x=nx-1) are left to bounce-back.

    Unknown populations (cy < 0, pointing into the domain):
        4 (0,-1,0), 8 (-1,-1,0), 9 (1,-1,0), 16 (0,-1,-1), 18 (0,-1,1)
    Known populations (cy >= 0):
        0,1,2,3,5,6,7,10,11,12,13,14,15,17

    rho = sum(cy=0) + 2*sum(cy>0)   (for uy=0)
    Unknown f[i] = feq[i] - feq[opp(i)] + f[opp(i)]   (non-eq bounce-back)
    """
    # Compute rho at the lid row from mass conservation (uy=0)
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

    # Equilibrium at (rho, u_lid, 0, 0) for the lid row only
    rho3 = rho.unsqueeze(1)  # (nz, 1, nx)
    ux3 = torch.full_like(rho3, u_lid)
    uy3 = torch.zeros_like(rho3)
    uz3 = torch.zeros_like(rho3)
    feq = equilibrium3d(rho3, ux3, uy3, uz3, device=f.device)  # (19, nz, 1, nx)

    # Non-equilibrium bounce-back for unknown directions
    unk_dirs = [4, 8, 9, 16, 18]
    opp = OPPOSITE.to(f.device)
    f_new = f.clone()
    for i in unk_dirs:
        j = int(opp[i].item())
        # Only update interior cells (x=1..nx-2); corners left to bounce-back
        f_new[i, :, -1, 1:-1] = (
            feq[i, :, 0, 1:-1] - feq[j, :, 0, 1:-1] + f[j, :, -1, 1:-1]
        )
    return f_new


# ---------------------------------------------------------------------------
# Ghia et al. (1982) reference data for Re=1000
# ---------------------------------------------------------------------------

# u/u_lid along vertical centreline x=0.5 (Re=1000, 129x129 grid)
GHIA_RE1000_U = {
    "y": [
        1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
        0.8516, 0.7344, 0.6172, 0.5000, 0.4531,
        0.2813, 0.1719, 0.1016, 0.0703, 0.0625,
        0.0547, 0.0000,
    ],
    "u": [
        1.00000,  0.65928,  0.57492,  0.51117,  0.46604,
        0.33304,  0.18719,  0.05702, -0.06080, -0.10648,
       -0.27805, -0.38289, -0.29730, -0.22220, -0.20196,
       -0.18109,  0.00000,
    ],
}

# v/u_lid along horizontal centreline y=0.5 (Re=1000)
GHIA_RE1000_V = {
    "x": [
        0.0000, 0.0625, 0.0703, 0.0781, 0.0938,
        0.1563, 0.2266, 0.2344, 0.5000, 0.8047,
        0.8594, 0.9063, 0.9453, 0.9531, 0.9609,
        0.9688, 1.0000,
    ],
    "v": [
        0.00000,  0.27485,  0.29012,  0.30353,  0.32627,
        0.37095,  0.33075,  0.32235,  0.02526, -0.31966,
       -0.42665, -0.51550, -0.39188, -0.33714, -0.27669,
       -0.21388,  0.00000,
    ],
}


# ---------------------------------------------------------------------------
# Benchmark 1: Lid-driven cavity (Re=1000)
# ---------------------------------------------------------------------------

def run_lid_driven_cavity(device, output_path, tag,
                          nx=128, ny=128, nz=4,
                          u_lid=0.05, Re=1000.0, Cs=0.05, n_steps=10000):
    """Lid-driven cavity: Re=1000, 128x128x4 (quasi-2D), 10000 steps.

    All four walls are solid (no-slip bounce-back). The top wall (lid)
    moves in +x with velocity u_lid via moving bounce-back.
    z-direction is periodic (no solid z-walls) for quasi-2D behavior.

    Reference: Ghia et al. (1982) u-velocity profile at x=0.5.
    """
    nu = u_lid * nx / Re
    tau = 3.0 * nu + 0.5

    print(
        f"{tag} [LidCavity] nx={nx} ny={ny} nz={nz} u_lid={u_lid} "
        f"nu={nu:.6e} tau={tau:.6f} Re={Re} Cs={Cs}",
        flush=True,
    )

    t0 = time.time()

    # ---- Geometry: all four walls solid, z periodic ----
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True    # bottom wall
    solid[:, -1, :] = True   # top wall (lid)
    solid[:, :, 0] = True    # left wall
    solid[:, :, -1] = True   # right wall
    # z-direction: periodic (no solid z-walls) for quasi-2D

    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    print(f"{tag} [LidCavity] solid={n_solid} (4 walls, z periodic)",
          flush=True)

    # ---- Initialize: quiescent fluid ----
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    uy0 = torch.zeros((nz, ny, nx), device=device)
    uz0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} [LidCavity] init done ({time.time()-t0:.1f}s), starting loop...",
          flush=True)

    # ---- Time history for centerline profiles ----
    u_profile_hist = []  # u/u_lid at vertical centerline (averaged over z)
    v_profile_hist = []  # v/u_lid at horizontal centerline (averaged over z)

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Bounce-back on all solid cells + Zou/He moving lid
        #    (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)
        # Overwrite interior lid cells with Zou/He moving-wall BC
        f = zou_he_moving_lid_3d(f, u_lid)

        # 5. Streaming (z is periodic via torch.roll)
        f = stream3d(f)

        # 6. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} [LidCavity] DIVERGED at step {step}", flush=True)
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

            if step > n_steps // 2:
                u_profile_hist.append(u_vert.cpu().numpy())
                v_profile_hist.append(v_horiz.cpu().numpy())

            elapsed = time.time() - t0
            ms = float(torch.sqrt(ux*ux + uy*uy + uz*uz).max().item())
            print(
                f"{tag} [LidCavity] step={step} max|u|={ms:.5f} "
                f"u_mid={float(u_vert[ny//2].item()):.4f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # ---- Final centerline profiles (averaged over last 50% of samples) ----
    rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
    x_mid = nx // 2
    y_mid = ny // 2

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

    # Primary vortex center (location of min |u| near center, or max streamfunction)
    # Simple: find center of primary vortex as point where u≈0, v≈0 near center
    ux_np = ux_f[nz//2].cpu().numpy()
    uy_np = uy_f[nz//2].cpu().numpy()
    # Streamfunction via integration (approximate)
    # Vortex center: where velocity magnitude is minimum in interior
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
        "benchmark": "lid_driven_cavity",
        "device": str(device),
        "grid": f"{nx}x{ny}x{nz}",
        "u_lid": u_lid,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "Cs": Cs,
        "n_steps": n_steps,
        "normal_method": "from_gradient (not used — internal flow)",
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
    print(f"{tag} [LidCavity] FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  RMSE u (vertical centreline) = {rmse_u:.6f}")
    print(f"  Max  err u                    = {max_err_u:.6f}")
    print(f"  RMSE v (horizontal centreline)= {rmse_v:.6f}")
    print(f"  Max  err v                    = {max_err_v:.6f}")
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
# Benchmark 2: Flow past cosine hill (Re=100)
# ---------------------------------------------------------------------------

def run_cosine_hill(device, output_path, tag,
                    nx=300, ny=100, nz=4,
                    H=20, L=100, u_in=0.05, Re=100.0, Cs=0.05, n_steps=5000):
    """Flow past cosine hill: Re=100, 300x100x4, 5000 steps.

    Hill: h(x) = H*cos²(π*(x-x_c)/L) for |x-x_c| < L/2, else 0
    H=20, L=100, centered at x_c = L/2 + 50 = 100
    Re = u_in * H / nu (based on hill height)
    Channel: inlet (x=0) Zou/He velocity, outlet (x=nx-1) zero-gradient
    Top wall: no-slip bounce-back. Bottom wall + hill: no-slip bounce-back.

    Reference: separation length, reattachment point.
    """
    nu = u_in * H / Re
    tau = 3.0 * nu + 0.5

    # Hill center: place hill starting at x=50, centered at x=100
    x_c = L / 2 + 50  # = 100

    print(
        f"{tag} [CosineHill] nx={nx} ny={ny} nz={nz} H={H} L={L} x_c={x_c} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Re={Re} Cs={Cs}",
        flush=True,
    )

    t0 = time.time()

    # ---- Build solid mask: bottom wall + cosine hill ----
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    # Bottom wall (y=0)
    solid[:, 0, :] = True
    # Top wall (y=ny-1)
    solid[:, -1, :] = True
    # z-direction: periodic (no solid z-walls) for quasi-2D

    # Cosine hill: h(x) = H * cos²(π*(x-x_c)/L) for |x-x_c| < L/2
    xx = torch.arange(nx, device=device, dtype=torch.float32)
    dx_from_center = (xx - x_c) / L  # normalized distance
    hill_height = torch.where(
        torch.abs(dx_from_center) < 0.5,
        H * torch.cos(torch.pi * dx_from_center) ** 2,
        torch.zeros_like(xx),
    )  # (nx,)

    # Fill solid below hill
    for x in range(nx):
        h_int = int(hill_height[x].item())
        if h_int > 0:
            solid[:, 1:h_int+1, x] = True  # y=0 already solid (bottom wall)

    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Hill-only mask (for drag measurement, without channel walls)
    hill_only = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    for x in range(nx):
        h_int = int(hill_height[x].item())
        if h_int > 0:
            hill_only[:, 1:h_int+1, x] = True

    n_hill = int(hill_only.sum().item())
    print(f"{tag} [CosineHill] solid={n_solid} hill_cells={n_hill} "
          f"max_hill_h={float(hill_height.max().item()):.1f}", flush=True)

    # Near-wall mask and surface mesh for the hill (from_gradient normal)
    near_hill = get_near_wall_3d(hill_only)
    mesh_hill = SurfaceMesh.from_gradient(hill_only, near_hill)
    n_near = int(near_hill.sum().item())
    print(f"{tag} [CosineHill] near_hill={n_near} mesh built ({time.time()-t0:.1f}s)",
          flush=True)

    # ---- Initialize: uniform flow above hill ----
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    uy0 = torch.zeros((nz, ny, nx), device=device)
    uz0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} [CosineHill] init done ({time.time()-t0:.1f}s), starting loop...",
          flush=True)

    # dpS for hill drag (frontal area = H * nz)
    A_hill = H * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_hill

    # Hill crest x-position
    x_crest = int(x_c)  # x=100

    sep_hist = []
    reattach_hist = []
    cd_hill_hist = []
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

        # 6. Channel BC: Zou/He inlet + zero-gradient outlet + bounce-back
        f = zou_he_inlet_velocity_3d(f, u_in)
        # Zero-gradient outlet
        f[:, :, :, -1] = f[:, :, :, -2]
        # Bounce-back on solid (after inlet/outlet to enforce no-slip)
        f = bounce_back_cells_3d(f, solid)

        # 7. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} [CosineHill] DIVERGED at step {step}", flush=True)
            diverged = True
            last_step = step
            break
        last_step = step

        # Measure separation, reattachment, and hill drag
        if step % 100 == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            ux_zmid = ux[nz // 2]  # middle z-layer (ny, nx)
            ux_zmid = ux_zmid.masked_fill(solid[nz // 2], 0.0)

            # Separation: scan along the hill surface for first x downstream
            # of crest where ux < 0 at the first fluid cell above the hill
            hill_h_np = hill_height.cpu()  # (nx,)
            sep_x = None
            for x in range(x_crest, nx - 1):
                h_int = int(hill_h_np[x].item())
                y_fluid = h_int + 1  # first fluid cell above hill
                if y_fluid < ny - 1:
                    val = float(ux_zmid[y_fluid, x].item())
                    if val < -1e-6:
                        sep_x = x
                        break

            # Reattachment: first x after separation where ux > 0 at y=1
            # (first fluid row above bottom wall)
            if sep_x is not None:
                reattach_x = None
                for x in range(sep_x, nx - 1):
                    val = float(ux_zmid[1, x].item())
                    if val > 1e-6:
                        reattach_x = x
                        break
            else:
                reattach_x = None

            sep_dist = float(sep_x - x_crest) / H if sep_x else 0.0
            reattach_dist = float(reattach_x - x_crest) / H if reattach_x else 0.0
            sep_hist.append(sep_dist)
            reattach_hist.append(reattach_dist)

            # Hill pressure drag (from_gradient normal)
            cd_x, _, _ = drag_pressure_integration(f, mesh_hill, dpS)
            cd_hill_hist.append(cd_x)

            if step % 500 == 0 or step == n_steps:
                elapsed = time.time() - t0
                ms = float(torch.sqrt(ux*ux + uy*uy + uz*uz).max().item())
                print(
                    f"{tag} [CosineHill] step={step} sep/H={sep_dist:.3f} "
                    f"reattach/H={reattach_dist:.3f} Cd_hill={cd_x:.4f} "
                    f"max|u|={ms:.4f} ({elapsed:.0f}s)",
                    flush=True,
                )

    elapsed = time.time() - t0

    # Final measurements
    rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
    ux_zmid = ux_f[nz // 2].masked_fill(solid[nz // 2], 0.0)

    # Final separation/reattachment (same improved detection)
    hill_h_np = hill_height.cpu()
    final_sep_x = None
    for x in range(x_crest, nx - 1):
        h_int = int(hill_h_np[x].item())
        y_fluid = h_int + 1
        if y_fluid < ny - 1:
            val = float(ux_zmid[y_fluid, x].item())
            if val < -1e-6:
                final_sep_x = x
                break
    if final_sep_x is not None:
        final_reattach_x = None
        for x in range(final_sep_x, nx - 1):
            val = float(ux_zmid[1, x].item())
            if val > 1e-6:
                final_reattach_x = x
                break
    else:
        final_reattach_x = None

    final_sep = float(final_sep_x - x_crest) / H if final_sep_x else 0.0
    final_reattach = float(final_reattach_x - x_crest) / H if final_reattach_x else 0.0

    # Average over last 20% of history
    tail_n = max(len(sep_hist) // 5, 1)
    sep_mean = sum(sep_hist[-tail_n:]) / tail_n if sep_hist else 0.0
    reattach_mean = sum(reattach_hist[-tail_n:]) / tail_n if reattach_hist else 0.0
    cd_mean = sum(cd_hill_hist[-tail_n:]) / tail_n if cd_hill_hist else 0.0

    # Velocity profile at x = x_crest + 2H (downstream of hill)
    x_downstream = min(int(x_crest + 2 * H), nx - 2)
    u_downstream = ux_zmid[:, x_downstream].cpu().numpy()
    y_positions = np.arange(ny)

    # Velocity profile at x = x_crest (at hill crest)
    u_at_crest = ux_zmid[:, x_crest].cpu().numpy()

    # Separation bubble length
    bubble_length = (reattach_mean - sep_mean) if (reattach_mean > sep_mean) else 0.0

    result = {
        "benchmark": "cosine_hill",
        "device": str(device),
        "grid": f"{nx}x{ny}x{nz}",
        "H": H,
        "L": L,
        "x_crest": x_crest,
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "Cs": Cs,
        "n_steps": n_steps,
        "normal_method": "from_gradient",
        "separation_x": final_sep_x,
        "reattachment_x": final_reattach_x,
        "separation_H": final_sep,
        "reattachment_H": final_reattach,
        "separation_H_mean": sep_mean,
        "reattachment_H_mean": reattach_mean,
        "bubble_length_H": bubble_length,
        "Cd_hill_pressure": cd_mean,
        "u_profile_at_crest": u_at_crest.tolist(),
        "u_profile_downstream": u_downstream.tolist(),
        "x_downstream_profile": x_downstream,
        "finite": not diverged,
        "diverged": diverged,
        "last_step": last_step,
        "wall_time_s": elapsed,
    }

    print(f"\n{'='*60}")
    print(f"{tag} [CosineHill] FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Separation x   = {final_sep_x} ({final_sep:.3f} H)")
    print(f"  Reattachment x = {final_reattach_x} ({final_reattach:.3f} H)")
    print(f"  Bubble length  = {bubble_length:.3f} H")
    print(f"  Cd_hill (pressure) = {cd_mean:.4f}")
    print(f"  Finite: {not diverged}")
    print(f"  Wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"  Results saved to {output_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device_id = int(sys.argv[1])
    benchmark = sys.argv[2]
    output_path = sys.argv[3]

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id}]"

    if benchmark == "lid_driven_cavity":
        kwargs = {}
        if len(sys.argv) > 4:
            kwargs["Re"] = float(sys.argv[4])
        run_lid_driven_cavity(device, output_path, tag, **kwargs)
    elif benchmark == "cosine_hill":
        kwargs = {}
        if len(sys.argv) > 4:
            kwargs["Re"] = float(sys.argv[4])
        run_cosine_hill(device, output_path, tag, **kwargs)
    else:
        print(f"Unknown benchmark: {benchmark}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
