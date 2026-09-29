"""3D Couette & Poiseuille flow verification on SDAA.

Unified module for both benchmarks:
  BENCHMARK 2: 3D Couette flow (moving top wall)
    - Reference: Cf = 2ν/(H*u_top) (exact)
    - nx=80, ny=12, nz=12 (true 3D, not extruded)
    - tau=1.0, u_top=0.05
    - 3000 steps
    - Measure: Cf, u profile
    - Verify 3D pressure integration gives 0.00%

  BENCHMARK 3: 3D Poiseuille flow (body force driven)
    - Reference: u_max = G*H²/(2ν) (exact)
    - nx=80, ny=12, nz=12 (true 3D)
    - tau=1.0, G=2ν*u_max/H²
    - 3000 steps
    - Measure: u profile, Cd (should match body force)

Uses verified modules:
  - drag_pressure (SurfaceMesh.from_gradient, drag_pressure/friction_integration)
  - boundaries3d (bounce_back_cells_3d)
  - d3q19 (equilibrium3d, macroscopic3d, C, W, OPPOSITE)
  - solver3d (stream3d)
  - turbulence (collide_bgk3d)

Usage:
    SDAA_VISIBLE_DEVICES=17 PYTHONPATH=src python couette_poiseuille_3d.py couette 17
    SDAA_VISIBLE_DEVICES=18 PYTHONPATH=src python couette_poiseuille_3d.py poiseuille 18
"""
from __future__ import annotations
import sys, json, math, time
from pathlib import Path
import numpy as np
import torch
from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d, collide_bgk3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_3d,
    drag_pressure_integration, drag_friction_integration,
)


# ---------------------------------------------------------------------------
# Moving-wall bounce-back for Couette flow
# ---------------------------------------------------------------------------
def moving_wall_bounce_back_3d(f, solid, top_wall_mask, u_top, rho_w=1.0):
    """Half-way bounce-back with moving wall correction for top wall.

    Standard bounce-back: f[i] = f[opp[i]] at solid cells.
    Moving wall correction: f[i] += 2*rho*w[i]*(c[i]·u_w)/cs² at top wall.

    The correction adds +x momentum when the top wall moves in +x (u_top>0).
    Only populations with cy=-1 stream from the top wall into the fluid,
    but the correction is applied to all directions (only relevant ones
    have non-zero c[i,0] since u_w=(u_top,0,0)).
    """
    opp = OPPOSITE.to(f.device)
    # Standard bounce-back at all solid cells
    f = torch.where(solid.unsqueeze(0), f[opp], f)

    # Moving wall correction at top wall solid cells
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    cs2 = 1.0 / 3.0

    # correction[i] = 2 * rho_w * w[i] * (c[i,0] * u_top) / cs2
    #              = 6 * rho_w * w[i] * c[i,0] * u_top
    correction = 6.0 * rho_w * u_top * w * c[:, 0]  # (19,)
    top_mask = top_wall_mask.unsqueeze(0).float()  # (1, nz, ny, nx)
    f = f + correction.view(19, 1, 1, 1) * top_mask

    return f


# ---------------------------------------------------------------------------
# Guo body force for Poiseuille flow
# ---------------------------------------------------------------------------
def apply_body_force_guo(f, Fx, tau):
    """Guo body force for D3Q19 (x-direction only, uniform).

    f_i += (1 - 1/(2*tau)) * w_i * 3 * c_i,x * Fx
    """
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    factor = (1.0 - 1.0 / (2.0 * tau))
    forcing = factor * w.view(19, 1, 1, 1) * 3.0 * c[:, 0].view(19, 1, 1, 1) * Fx
    return f + forcing


# ---------------------------------------------------------------------------
# BENCHMARK 2: 3D Couette flow
# ---------------------------------------------------------------------------
def run_couette(device_id, output_path):
    """3D Couette flow: moving top wall, stationary bottom wall."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # Parameters
    nx, ny, nz = 80, 12, 12
    tau = 1.0
    nu = (tau - 0.5) / 3.0  # = 1/6
    u_top = 0.05
    n_steps = 3000
    warmup = 500

    # Channel geometry
    # Walls at y=0 (bottom, stationary) and y=ny-1 (top, moving at u_top)
    # Half-way bounce-back: effective walls at y=0.5 and y=ny-1.5
    H = ny - 2  # channel height = 10 (effective wall-to-wall distance)
    # Cf_exact = 2*nu / (H * u_top)  [using H = full channel height]
    # But with half-way BB, effective H = ny-2 = 10
    Cf_exact = 2.0 * nu / (H * u_top)

    tag = f"[Couette3D SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f} u_top={u_top}", flush=True)
    print(f"{tag} H={H} Cf_exact={Cf_exact:.6f} n_steps={n_steps}", flush=True)

    t0 = time.time()

    # Solid mask: top and bottom walls
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True   # bottom wall (y=0)
    solid[:, -1, :] = True  # top wall (y=ny-1)

    # Top wall mask (for moving wall correction)
    top_wall_mask = torch.zeros_like(solid)
    top_wall_mask[:, -1, :] = True

    # Near-wall mask (3D)
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # Surface mesh using from_gradient normal (for arbitrary geometry)
    mesh = SurfaceMesh.from_gradient(solid, near)
    print(f"{tag} SurfaceMesh built (from_gradient normal)", flush=True)

    # Initialize: zero velocity everywhere
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)

    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    # Accumulators
    cf_hist = []
    u_profiles = []

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision
        f_pre = f.clone()

        # 2. Collision (BGK)
        f = collide_bgk3d(f, tau=tau)

        # 3. NoDynamics: restore solid cells
        sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back with moving wall correction (BEFORE streaming)
        f = moving_wall_bounce_back_3d(f, solid, top_wall_mask, u_top)

        # 5. Stream (periodic in x and z via torch.roll)
        f = stream3d(f)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Record post-warmup
        if step > warmup:
            # Compute Cf from friction drag on top wall
            # dpS for Cf: 0.5 * rho * u_top^2 * A_wall (A_wall = nx * nz)
            A_wall = nx * nz
            dpS_wall = 0.5 * 1.0 * u_top ** 2 * A_wall

            # Friction drag on ALL near-wall cells (both walls)
            # For Couette, force on top wall = +tau_w*A, force on bottom = -tau_w*A
            # Total = 0. We need just the top wall.
            # Create separate mesh for top wall only
            near_top = near.clone()
            near_top[:, 0, :] = False  # only keep top wall near-cells
            mesh_top = SurfaceMesh.from_gradient(solid, near_top)

            _, ffx, _ = drag_friction_integration(f, mesh_top, dpS_wall, nu)
            # ffx is the friction force on the top wall / dpS_wall = Cf
            cf_hist.append(ffx)

            # Also check pressure drag (should be 0)
            # We'll compute this at the end

        if step % 500 == 0:
            _, ux, _, _ = macroscopic3d(f)
            # Average u profile in x and z
            u_prof = ux.mean(dim=(0, 2))  # (ny,)
            cf_avg = sum(cf_hist) / max(len(cf_hist), 1)
            elapsed = time.time() - t0
            print(f"{tag} step={step} Cf={cf_avg:.6f} (exact={Cf_exact:.6f}) "
                  f"u[ny//2]={float(u_prof[ny//2]):.6f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # Final measurements
    cf_mean = sum(cf_hist) / max(len(cf_hist), 1) if cf_hist else float("nan")
    cf_err = abs(cf_mean - Cf_exact) / Cf_exact * 100 if Cf_exact > 0 and math.isfinite(cf_mean) else float("nan")

    # u profile (average over x and z)
    _, ux, _, _ = macroscopic3d(f)
    u_prof = ux.mean(dim=(0, 2)).cpu().numpy()  # (ny,)

    # Analytical: u(y) = u_top * (y - 0.5) / H for y=1..ny-2
    y_vals = np.arange(ny, dtype=np.float32)
    u_exact = np.zeros(ny, dtype=np.float32)
    for y in range(1, ny - 1):
        u_exact[y] = u_top * (y - 0.5) / H  # half-way BB: wall at y=0.5

    # u error
    u_err_max = 0.0
    for y in range(1, ny - 1):
        if abs(u_exact[y]) > 1e-10:
            err = abs(u_prof[y] - u_exact[y]) / max(abs(u_exact[y]), 1e-10) * 100
            u_err_max = max(u_err_max, err)

    # Pressure drag check (should be 0 for Couette)
    A_wall = nx * nz
    dpS_wall = 0.5 * 1.0 * u_top ** 2 * A_wall
    near_top = near.clone()
    near_top[:, 0, :] = False
    mesh_top = SurfaceMesh.from_gradient(solid, near_top)
    cdp_x, _, _ = drag_pressure_integration(f, mesh_top, dpS_wall)
    # Also check bottom wall
    near_bot = near.clone()
    near_bot[:, -1, :] = False
    mesh_bot = SurfaceMesh.from_gradient(solid, near_bot)
    cdp_x_bot, _, _ = drag_pressure_integration(f, mesh_bot, dpS_wall)

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cf      = {cf_mean:.6f}  (exact={Cf_exact:.6f}, err={cf_err:.2f}%)", flush=True)
    print(f"{tag} u_err   = {u_err_max:.2f}% (max relative error)", flush=True)
    print(f"{tag} Cd_p_top  = {cdp_x:.6f}  (should be 0)", flush=True)
    print(f"{tag} Cd_p_bot  = {cdp_x_bot:.6f}  (should be 0)", flush=True)
    print(f"{tag} u_profile:", flush=True)
    for y in range(ny):
        print(f"{tag}   y={y:2d}  u={u_prof[y]:.6f}  exact={u_exact[y]:.6f}", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
        "case": "couette_3d",
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": "BGK",
        "boundary": "halfway_BB_moving_wall",
        "grid": f"{nx}x{ny}x{nz}",
        "tau": tau,
        "nu": nu,
        "u_top": u_top,
        "H": H,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cf_mean": cf_mean,
        "Cf_exact": Cf_exact,
        "Cf_err_pct": cf_err,
        "u_err_max_pct": u_err_max,
        "Cd_pressure_top": cdp_x,
        "Cd_pressure_bot": cdp_x_bot,
        "u_profile": u_prof.tolist(),
        "u_exact": u_exact.tolist(),
        "n_samples": len(cf_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} results saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
# BENCHMARK 3: 3D Poiseuille flow
# ---------------------------------------------------------------------------
def run_poiseuille(device_id, output_path):
    """3D Poiseuille flow: body force driven, stationary walls."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # Parameters
    nx, ny, nz = 80, 12, 12
    tau = 1.0
    nu = (tau - 0.5) / 3.0  # = 1/6
    u_max_target = 0.05
    n_steps = 3000
    warmup = 500

    # Channel geometry
    # Walls at y=0 and y=ny-1 (both stationary)
    # Half-way BB: effective walls at y=0.5 and y=ny-1.5
    H_full = ny - 2  # full channel height = 10
    H_half = H_full / 2.0  # half-channel height = 5

    # Body force: u_max = G * H_half^2 / (2*nu)  =>  G = 2*nu*u_max / H_half^2
    G = 2.0 * nu * u_max_target / (H_half ** 2)
    u_max_exact = G * H_half ** 2 / (2.0 * nu)  # should equal u_max_target

    tag = f"[Poiseuille3D SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f}", flush=True)
    print(f"{tag} H_full={H_full} H_half={H_half} G={G:.6e} u_max_target={u_max_target}", flush=True)
    print(f"{tag} u_max_exact={u_max_exact:.6f} n_steps={n_steps}", flush=True)

    t0 = time.time()

    # Solid mask: top and bottom walls (both stationary)
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True   # bottom wall
    solid[:, -1, :] = True  # top wall

    # Near-wall mask (3D)
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # Surface mesh using from_gradient normal
    mesh = SurfaceMesh.from_gradient(solid, near)
    print(f"{tag} SurfaceMesh built (from_gradient normal)", flush=True)

    # Initialize: zero velocity everywhere
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)

    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    # Accumulators
    cd_f_hist = []
    cd_p_hist = []

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision
        f_pre = f.clone()

        # 2. Collision (BGK)
        f = collide_bgk3d(f, tau=tau)

        # 3. NoDynamics: restore solid cells
        sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming, stationary walls)
        f = bounce_back_cells_3d(f, solid)

        # 5. Stream (periodic in x and z)
        f = stream3d(f)

        # 6. Body force (Guo forcing)
        f = apply_body_force_guo(f, G, tau)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Record post-warmup
        if step > warmup:
            # Compute drag from friction and pressure
            # dpS: 0.5 * rho * u_max^2 * A_frontal (A_frontal = H_full * nz)
            A_frontal = H_full * nz
            dpS = 0.5 * 1.0 * u_max_target ** 2 * A_frontal

            cdp_x, _, _ = drag_pressure_integration(f, mesh, dpS)
            cdf_x, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_p_hist.append(cdp_x)
            cd_f_hist.append(cdf_x)

        if step % 500 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_prof = ux.mean(dim=(0, 2))  # (ny,)
            u_max_sim = float(u_prof[ny // 2])
            elapsed = time.time() - t0
            print(f"{tag} step={step} u_max={u_max_sim:.6f} (target={u_max_target:.6f}) "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # Final measurements
    cd_f_mean = sum(cd_f_hist) / max(len(cd_f_hist), 1) if cd_f_hist else float("nan")
    cd_p_mean = sum(cd_p_hist) / max(len(cd_p_hist), 1) if cd_p_hist else float("nan")

    # u profile (average over x and z)
    _, ux, _, _ = macroscopic3d(f)
    u_prof = ux.mean(dim=(0, 2)).cpu().numpy()  # (ny,)

    # Analytical: u(y) = G/(2*nu) * (y-0.5) * (H_full - (y-0.5))
    # = G/(2*nu) * (y-0.5) * (ny-1.5 - y)
    y_vals = np.arange(ny, dtype=np.float32)
    u_exact = np.zeros(ny, dtype=np.float32)
    for y in range(1, ny - 1):
        y_eff = y - 0.5  # effective distance from bottom wall
        u_exact[y] = G / (2.0 * nu) * y_eff * (H_full - y_eff)

    # u error
    u_err_max = 0.0
    for y in range(1, ny - 1):
        if abs(u_exact[y]) > 1e-10:
            err = abs(u_prof[y] - u_exact[y]) / max(abs(u_exact[y]), 1e-10) * 100
            u_err_max = max(u_err_max, err)

    # Body force drag: F_body = G * rho * V_fluid
    # Cd_body = F_body / dpS = G * V_fluid / dpS
    V_fluid = nx * (ny - 2) * nz
    A_frontal = H_full * nz
    dpS = 0.5 * 1.0 * u_max_target ** 2 * A_frontal
    cd_body = G * V_fluid / dpS  # body force coefficient

    # Friction drag should match body force
    cd_f_err = abs(cd_f_mean - cd_body) / abs(cd_body) * 100 if cd_body != 0 and math.isfinite(cd_f_mean) else float("nan")

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} u_max   = {float(u_prof[ny//2]):.6f}  (exact={u_max_exact:.6f})", flush=True)
    print(f"{tag} u_err   = {u_err_max:.2f}% (max relative error)", flush=True)
    print(f"{tag} Cd_f    = {cd_f_mean:.6f}  (friction drag)", flush=True)
    print(f"{tag} Cd_body = {cd_body:.6f}  (body force)", flush=True)
    print(f"{tag} Cd_f_err= {cd_f_err:.2f}% (friction vs body force)", flush=True)
    print(f"{tag} Cd_p    = {cd_p_mean:.6f}  (pressure, should be ~0)", flush=True)
    print(f"{tag} u_profile:", flush=True)
    for y in range(ny):
        print(f"{tag}   y={y:2d}  u={u_prof[y]:.6f}  exact={u_exact[y]:.6f}", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
        "case": "poiseuille_3d",
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": "BGK",
        "boundary": "halfway_BB_body_force",
        "grid": f"{nx}x{ny}x{nz}",
        "tau": tau,
        "nu": nu,
        "G": G,
        "u_max_target": u_max_target,
        "u_max_exact": u_max_exact,
        "H_full": H_full,
        "H_half": H_half,
        "n_steps": n_steps,
        "warmup": warmup,
        "u_max_sim": float(u_prof[ny // 2]),
        "u_err_max_pct": u_err_max,
        "Cd_f_mean": cd_f_mean,
        "Cd_body": cd_body,
        "Cd_f_err_pct": cd_f_err,
        "Cd_p_mean": cd_p_mean,
        "u_profile": u_prof.tolist(),
        "u_exact": u_exact.tolist(),
        "n_samples": len(cd_f_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} results saved to {output_path}", flush=True)
    return result


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "couette"
    device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 17
    output_path = sys.argv[3] if len(sys.argv) > 3 else f"{mode}_3d_sdaa{device_id}.json"

    if mode == "couette":
        run_couette(device_id, output_path)
    elif mode == "poiseuille":
        run_poiseuille(device_id, output_path)
    else:
        print(f"Unknown mode: {mode}. Use 'couette' or 'poiseuille'.")
        sys.exit(1)
