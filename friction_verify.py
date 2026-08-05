"""Friction drag verification on 100% analytical solutions + grid convergence.

TEST 1: 3D Poiseuille friction verification (SDAA:8)
  - nx=80, ny=12, nz=12 (true 3D), tau=1.0
  - Body force G drives flow
  - Reference: u(y) = G/(2ν) * y_eff * (H-y_eff) (exact parabolic)
  - Measure: u profile, Cd_f (should match body force)
  - Verify friction formula τ=2ν·u_t gives correct result

TEST 2: Couette friction grid convergence (SDAA:9-10)
  - nx=80, ny=8/16/32, nz=4 (grid convergence)
  - tau=1.0, u_top=0.05
  - Reference: Cf = 2ν/(H*u_top) (exact)
  - Check if Cf converges with grid refinement
  - Measures BOTTOM wall (stationary) — formula τ=2ν·u_t is exact for linear profiles

TEST 3: Poiseuille friction grid convergence (SDAA:11)
  - nx=80, ny=8/16/32, nz=4
  - tau=1.0, G=2ν*u_max/H²
  - Check if Cd_f converges with grid refinement
  - Tests if the friction formula is grid-convergent for parabolic profiles

Key physics:
  - For STATIONARY walls: τ = 2ν·u_t is correct (u_wall=0, gradient = u_t/0.5 = 2·u_t)
  - For MOVING walls: τ = 2ν·(u_t - u_wall) — NOT supported by drag_friction_integration
  - Couette (linear profile): formula is EXACT at all resolutions (bottom wall)
  - Poiseuille (parabolic profile): formula has O(1/H) discretization error → grid convergence

Usage:
    PYTHONPATH=src python friction_verify.py test1 <device>
    PYTHONPATH=src python friction_verify.py couette <device> <ny>
    PYTHONPATH=src python friction_verify.py poiseuille <device> <ny>
    PYTHONPATH=src python friction_verify.py poiseuille_conv <device>
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
    """Half-way bounce-back with moving wall correction for top wall."""
    opp = OPPOSITE.to(f.device)
    f = torch.where(solid.unsqueeze(0), f[opp], f)
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    cs2 = 1.0 / 3.0
    correction = 6.0 * rho_w * u_top * w * c[:, 0]
    top_mask = top_wall_mask.unsqueeze(0).float()
    f = f + correction.view(19, 1, 1, 1) * top_mask
    return f


# ---------------------------------------------------------------------------
# Guo body force for Poiseuille flow
# ---------------------------------------------------------------------------
def apply_body_force_guo(f, Fx, tau):
    """Guo body force for D3Q19 (x-direction only, uniform).

    Post-streaming forcing: factor = 1.0 (not (1-1/(2τ))).
    The (1-1/(2τ)) factor is for collision-time application; when applied
    after streaming, the full force Fx is added directly.
    """
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    forcing = w.view(19, 1, 1, 1) * 3.0 * c[:, 0].view(19, 1, 1, 1) * Fx
    return f + forcing


def _make_channel_solid(nz, ny, nx, device):
    """Create solid mask for channel: walls at y=0 and y=ny-1."""
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    return solid


def _near_wall_bottom_only(near, ny):
    """Keep only bottom-wall near cells (y=1), zero out top (y=ny-2)."""
    near_b = near.clone()
    near_b[:, ny - 2, :] = False  # remove top near-wall cells
    return near_b


def _near_wall_top_only(near, ny):
    """Keep only top-wall near cells (y=ny-2), zero out bottom (y=1)."""
    near_t = near.clone()
    near_t[:, 1, :] = False  # remove bottom near-wall cells
    return near_t


# ---------------------------------------------------------------------------
# TEST 1: 3D Poiseuille friction verification
# ---------------------------------------------------------------------------
def run_poiseuille_3d(device_id, output_path):
    """3D Poiseuille: body force driven, stationary walls. Cd_f vs body force."""
    device = torch.device("sdaa:0")  # SDAA_VISIBLE_DEVICES remaps to 0
    torch.sdaa.set_device(device)

    nx, ny, nz = 80, 12, 12
    tau = 1.0
    nu = (tau - 0.5) / 3.0  # = 1/6
    u_max_target = 0.05
    n_steps = 4000
    warmup = 500

    H = ny - 2  # effective channel height = 10
    H_half = H / 2.0

    # Body force: u_max = G * H_half² / (2ν)  =>  G = 2ν·u_max / H_half²
    G = 2.0 * nu * u_max_target / (H_half ** 2)
    u_max_exact = G * H_half ** 2 / (2.0 * nu)

    tag = f"[Pois3D SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f}", flush=True)
    print(f"{tag} H={H} H_half={H_half} G={G:.6e} u_max={u_max_target}", flush=True)

    t0 = time.time()
    solid = _make_channel_solid(nz, ny, nx, device)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)
    print(f"{tag} near-wall cells={int(near.sum().item())}", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    cd_f_hist, cd_p_hist = [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        # NoDynamics: restore solid cells
        sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        # Bounce-back (stationary walls)
        f = bounce_back_cells_3d(f, solid)
        # Stream (periodic in x and z)
        f = stream3d(f)
        # Body force (Guo forcing)
        f = apply_body_force_guo(f, G, tau)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            A_frontal = H * nz
            dpS = 0.5 * 1.0 * u_max_target ** 2 * A_frontal
            cdp_x, _, _ = drag_pressure_integration(f, mesh, dpS)
            cdf_x, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_p_hist.append(cdp_x)
            cd_f_hist.append(cdf_x)

        if step % 1000 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_prof = ux.mean(dim=(0, 2))
            u_max_sim = float(u_prof[ny // 2])
            cf_avg = sum(cd_f_hist) / max(len(cd_f_hist), 1)
            print(f"{tag} step={step} u_max={u_max_sim:.6f} Cd_f={cf_avg:.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    cd_f_mean = sum(cd_f_hist) / max(len(cd_f_hist), 1) if cd_f_hist else float("nan")
    cd_p_mean = sum(cd_p_hist) / max(len(cd_p_hist), 1) if cd_p_hist else float("nan")

    _, ux, _, _ = macroscopic3d(f)
    u_prof = ux.mean(dim=(0, 2)).cpu().numpy()

    # Analytical: u(y) = G/(2ν) * y_eff * (H - y_eff), y_eff = y - 0.5
    u_exact = np.zeros(ny, dtype=np.float32)
    for y in range(1, ny - 1):
        y_eff = y - 0.5
        u_exact[y] = G / (2.0 * nu) * y_eff * (H - y_eff)

    u_err_max = 0.0
    for y in range(1, ny - 1):
        if abs(u_exact[y]) > 1e-10:
            err = abs(u_prof[y] - u_exact[y]) / max(abs(u_exact[y]), 1e-10) * 100
            u_err_max = max(u_err_max, err)

    # Body force drag: F_body = G * rho * V_fluid
    V_fluid = nx * (ny - 2) * nz
    A_frontal = H * nz
    dpS = 0.5 * 1.0 * u_max_target ** 2 * A_frontal
    cd_body = G * V_fluid / dpS

    cd_f_err = abs(cd_f_mean - cd_body) / abs(cd_body) * 100 if cd_body != 0 and math.isfinite(cd_f_mean) else float("nan")

    # Analytical friction: τ_w = G*H/2 (force balance), Cd_f_exact = G*H/2 * A_wall / dpS
    A_wall = nx * nz
    cd_f_exact = (G * H / 2.0) * A_wall / dpS  # = cd_body (force balance)
    # Formula prediction: τ_formula = 2ν·u_t at y=1, u_t = G/(2ν)·0.5·(H-0.5)
    u_t_wall = G / (2.0 * nu) * 0.5 * (H - 0.5)
    cd_f_formula = (2.0 * nu * u_t_wall) * A_wall / dpS  # both walls
    cd_f_formula *= 2  # two walls

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} u_max   = {float(u_prof[ny//2]):.6f}  (exact={u_max_exact:.6f})", flush=True)
    print(f"{tag} u_err   = {u_err_max:.2f}% (max relative error)", flush=True)
    print(f"{tag} Cd_f    = {cd_f_mean:.6f}  (friction drag, both walls)", flush=True)
    print(f"{tag} Cd_body = {cd_body:.6f}  (body force = exact friction)", flush=True)
    print(f"{tag} Cd_f_err= {cd_f_err:.2f}% (friction vs body force)", flush=True)
    print(f"{tag} Cd_p    = {cd_p_mean:.6f}  (pressure, should be ~0)", flush=True)
    print(f"{tag} Cd_f_formula = {cd_f_formula:.6f}  (analytical formula prediction)", flush=True)
    print(f"{tag} u_profile:", flush=True)
    for y in range(ny):
        print(f"{tag}   y={y:2d}  u={u_prof[y]:.6f}  exact={u_exact[y]:.6f}", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
        "case": "poiseuille_3d_friction",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "tau": tau, "nu": float(nu), "G": float(G),
        "u_max_target": u_max_target, "u_max_exact": float(u_max_exact),
        "H": H, "n_steps": n_steps, "warmup": warmup,
        "u_max_sim": float(u_prof[ny // 2]),
        "u_err_max_pct": float(u_err_max),
        "Cd_f_mean": float(cd_f_mean),
        "Cd_body": float(cd_body),
        "Cd_f_err_pct": float(cd_f_err),
        "Cd_f_formula": float(cd_f_formula),
        "Cd_p_mean": float(cd_p_mean),
        "u_profile": [float(v) for v in u_prof],
        "u_exact": [float(v) for v in u_exact],
        "n_samples": len(cd_f_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} results saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
# TEST 2: Couette friction grid convergence
# ---------------------------------------------------------------------------
def run_couette(device_id, ny, output_path):
    """3D Couette: moving top wall. Measure BOTTOM wall friction (stationary)."""
    device = torch.device("sdaa:0")  # SDAA_VISIBLE_DEVICES remaps to 0
    torch.sdaa.set_device(device)

    nx, nz = 80, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0  # = 1/6
    u_top = 0.05
    n_steps = 4000
    warmup = 500

    H = ny - 2  # effective channel height
    Cf_exact = 2.0 * nu / (H * u_top)

    tag = f"[Couette SDAA:{device_id} ny={ny}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f} u_top={u_top}", flush=True)
    print(f"{tag} H={H} Cf_exact={Cf_exact:.6f} n_steps={n_steps}", flush=True)

    t0 = time.time()
    solid = _make_channel_solid(nz, ny, nx, device)
    top_wall_mask = torch.zeros_like(solid)
    top_wall_mask[:, -1, :] = True

    near = get_near_wall_3d(solid)
    # Bottom wall only (stationary — formula τ=2ν·u_t is correct)
    near_bottom = _near_wall_bottom_only(near, ny)
    mesh_bottom = SurfaceMesh.from_gradient(solid, near_bottom)
    print(f"{tag} near-wall cells (bottom only)={int(near_bottom.sum().item())}", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    cf_hist = []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = moving_wall_bounce_back_3d(f, solid, top_wall_mask, u_top)
        f = stream3d(f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            A_wall = nx * nz
            dpS_wall = 0.5 * 1.0 * u_top ** 2 * A_wall
            ffx, _, _ = drag_friction_integration(f, mesh_bottom, dpS_wall, nu)
            cf_hist.append(ffx)

        if step % 1000 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_prof = ux.mean(dim=(0, 2))
            cf_avg = sum(cf_hist) / max(len(cf_hist), 1)
            print(f"{tag} step={step} Cf={cf_avg:.6f} (exact={Cf_exact:.6f}) "
                  f"u[1]={float(u_prof[1]):.6f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    cf_mean = sum(cf_hist) / max(len(cf_hist), 1) if cf_hist else float("nan")
    cf_err = abs(cf_mean - Cf_exact) / Cf_exact * 100 if Cf_exact > 0 and math.isfinite(cf_mean) else float("nan")

    _, ux, _, _ = macroscopic3d(f)
    u_prof = ux.mean(dim=(0, 2)).cpu().numpy()

    # Analytical: u(y) = u_top * (y-0.5) / H for y=1..ny-2
    u_exact = np.zeros(ny, dtype=np.float32)
    for y in range(1, ny - 1):
        u_exact[y] = u_top * (y - 0.5) / H

    u_err_max = 0.0
    for y in range(1, ny - 1):
        if abs(u_exact[y]) > 1e-10:
            err = abs(u_prof[y] - u_exact[y]) / max(abs(u_exact[y]), 1e-10) * 100
            u_err_max = max(u_err_max, err)

    # Analytical formula prediction: τ = 2ν·u_t at y=1, u_t = u_top·0.5/H
    u_t_wall = u_top * 0.5 / H
    Cf_formula = 2.0 * nu * u_t_wall / (0.5 * u_top ** 2)

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cf      = {cf_mean:.6f}  (exact={Cf_exact:.6f}, err={cf_err:.2f}%)", flush=True)
    print(f"{tag} Cf_formula = {Cf_formula:.6f}  (analytical formula prediction)", flush=True)
    print(f"{tag} u_err   = {u_err_max:.2f}% (max relative error)", flush=True)
    print(f"{tag} u_profile:", flush=True)
    for y in range(ny):
        print(f"{tag}   y={y:2d}  u={u_prof[y]:.6f}  exact={u_exact[y]:.6f}", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
        "case": "couette_friction",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "tau": tau, "nu": float(nu), "u_top": u_top,
        "H": H, "n_steps": n_steps, "warmup": warmup,
        "Cf_mean": float(cf_mean),
        "Cf_exact": float(Cf_exact),
        "Cf_err_pct": float(cf_err),
        "Cf_formula": float(Cf_formula),
        "u_err_max_pct": float(u_err_max),
        "u_profile": [float(v) for v in u_prof],
        "u_exact": [float(v) for v in u_exact],
        "n_samples": len(cf_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} results saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
# TEST 3: Poiseuille friction grid convergence
# ---------------------------------------------------------------------------
def run_poiseuille(device_id, ny, output_path):
    """3D Poiseuille: body force driven. Measure both-wall friction vs body force."""
    device = torch.device("sdaa:0")  # SDAA_VISIBLE_DEVICES remaps to 0
    torch.sdaa.set_device(device)

    nx, nz = 80, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0  # = 1/6
    u_max_target = 0.05
    n_steps = 4000
    warmup = 500

    H = ny - 2
    H_half = H / 2.0
    G = 2.0 * nu * u_max_target / (H_half ** 2)
    u_max_exact = G * H_half ** 2 / (2.0 * nu)

    tag = f"[Pois SDAA:{device_id} ny={ny}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f}", flush=True)
    print(f"{tag} H={H} H_half={H_half} G={G:.6e} u_max={u_max_target}", flush=True)

    t0 = time.time()
    solid = _make_channel_solid(nz, ny, nx, device)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)
    print(f"{tag} near-wall cells={int(near.sum().item())}", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    cd_f_hist, cd_p_hist = [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        f = apply_body_force_guo(f, G, tau)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            A_frontal = H * nz
            dpS = 0.5 * 1.0 * u_max_target ** 2 * A_frontal
            cdp_x, _, _ = drag_pressure_integration(f, mesh, dpS)
            cdf_x, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_p_hist.append(cdp_x)
            cd_f_hist.append(cdf_x)

        if step % 1000 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_prof = ux.mean(dim=(0, 2))
            cdf_avg = sum(cd_f_hist) / max(len(cd_f_hist), 1)
            print(f"{tag} step={step} u_max={float(u_prof[ny//2]):.6f} Cd_f={cdf_avg:.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    cd_f_mean = sum(cd_f_hist) / max(len(cd_f_hist), 1) if cd_f_hist else float("nan")
    cd_p_mean = sum(cd_p_hist) / max(len(cd_p_hist), 1) if cd_p_hist else float("nan")

    _, ux, _, _ = macroscopic3d(f)
    u_prof = ux.mean(dim=(0, 2)).cpu().numpy()

    u_exact = np.zeros(ny, dtype=np.float32)
    for y in range(1, ny - 1):
        y_eff = y - 0.5
        u_exact[y] = G / (2.0 * nu) * y_eff * (H - y_eff)

    u_err_max = 0.0
    for y in range(1, ny - 1):
        if abs(u_exact[y]) > 1e-10:
            err = abs(u_prof[y] - u_exact[y]) / max(abs(u_exact[y]), 1e-10) * 100
            u_err_max = max(u_err_max, err)

    V_fluid = nx * (ny - 2) * nz
    A_frontal = H * nz
    dpS = 0.5 * 1.0 * u_max_target ** 2 * A_frontal
    cd_body = G * V_fluid / dpS

    cd_f_err = abs(cd_f_mean - cd_body) / abs(cd_body) * 100 if cd_body != 0 and math.isfinite(cd_f_mean) else float("nan")

    # Formula prediction: τ = 2ν·u_t at y=1, u_t = G/(2ν)·0.5·(H-0.5), both walls
    u_t_wall = G / (2.0 * nu) * 0.5 * (H - 0.5)
    A_wall = nx * nz
    cd_f_formula = 2.0 * (2.0 * nu * u_t_wall) * A_wall / dpS  # both walls

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} u_max   = {float(u_prof[ny//2]):.6f}  (exact={u_max_exact:.6f})", flush=True)
    print(f"{tag} u_err   = {u_err_max:.2f}%", flush=True)
    print(f"{tag} Cd_f    = {cd_f_mean:.6f}  (friction, both walls)", flush=True)
    print(f"{tag} Cd_body = {cd_body:.6f}  (body force = exact)", flush=True)
    print(f"{tag} Cd_f_err= {cd_f_err:.2f}% (friction vs body force)", flush=True)
    print(f"{tag} Cd_f_formula = {cd_f_formula:.6f}  (analytical formula prediction)", flush=True)
    print(f"{tag} Cd_p    = {cd_p_mean:.6f}  (pressure, ~0)", flush=True)
    print(f"{tag} u_profile:", flush=True)
    for y in range(ny):
        print(f"{tag}   y={y:2d}  u={u_prof[y]:.6f}  exact={u_exact[y]:.6f}", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
        "case": "poiseuille_friction",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "tau": tau, "nu": float(nu), "G": float(G),
        "u_max_target": u_max_target, "u_max_exact": float(u_max_exact),
        "H": H, "n_steps": n_steps, "warmup": warmup,
        "u_max_sim": float(u_prof[ny // 2]),
        "u_err_max_pct": float(u_err_max),
        "Cd_f_mean": float(cd_f_mean),
        "Cd_body": float(cd_body),
        "Cd_f_err_pct": float(cd_f_err),
        "Cd_f_formula": float(cd_f_formula),
        "Cd_p_mean": float(cd_p_mean),
        "u_profile": [float(v) for v in u_prof],
        "u_exact": [float(v) for v in u_exact],
        "n_samples": len(cd_f_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} results saved to {output_path}", flush=True)
    return result


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test1"
    device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    if mode == "test1":
        out = f"friction_test1_poiseuille3d_sdaa{device_id}.json"
        run_poiseuille_3d(device_id, out)
    elif mode == "couette":
        ny = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        out = f"friction_test2_couette_ny{ny}_sdaa{device_id}.json"
        run_couette(device_id, ny, out)
    elif mode == "poiseuille":
        ny = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        out = f"friction_test3_poiseuille_ny{ny}_sdaa{device_id}.json"
        run_poiseuille(device_id, ny, out)
    elif mode == "poiseuille_conv":
        # Run all three resolutions sequentially on one card
        results = []
        for ny in [8, 16, 32]:
            out = f"friction_test3_poiseuille_ny{ny}_sdaa{device_id}.json"
            r = run_poiseuille(device_id, ny, out)
            results.append(r)
        # Summary
        print("\n=== POISEUILLE GRID CONVERGENCE SUMMARY ===", flush=True)
        print(f"{'ny':>4} {'H':>4} {'Cd_f':>10} {'Cd_body':>10} {'err%':>8} {'formula':>10}", flush=True)
        for r in results:
            print(f"{r['grid'].split('x')[1]:>4} {r['H']:>4} {r['Cd_f_mean']:>10.6f} "
                  f"{r['Cd_body']:>10.6f} {r['Cd_f_err_pct']:>8.2f} {r['Cd_f_formula']:>10.6f}", flush=True)
    else:
        print(f"Unknown mode: {mode}. Use 'test1', 'couette', 'poiseuille', or 'poiseuille_conv'.")
        sys.exit(1)
