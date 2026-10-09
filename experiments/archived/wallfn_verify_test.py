#!/usr/bin/env python3
"""Verify and fix wall function on Poiseuille/Couette flow.

Tests on SDAA cards 4-7:
  TEST 1 (SDAA:4): Poiseuille with wall function, y_val=0.5
  TEST 2 (SDAA:5): Poiseuille with wall function, y_val=1.0
  TEST 3 (SDAA:6): Couette with wall function
  TEST 4 (SDAA:7): Debug wall function body force

The wall function (wall_model.py) has 6 bug fixes (7-12) but still gives
15.2% on Poiseuille. The issue is in the body force application:
  1. Force magnitude: F = -(τ_w/y_val) should be F = -τ_w
  2. Missing (1-1/(2τ)) Guo factor in ibm_apply_body_force_3d
"""
from __future__ import annotations
import sys, json, math, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
import torch
from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d, correct_mass3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.wall_model import wall_function_3d, compute_wall_normal, _near_wall_mask_no_wrap
from tensorlbm.ibm import ibm_apply_body_force_3d

DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Guo body force with correct (1-1/(2τ)) factor
# ---------------------------------------------------------------------------
def guo_body_force_3d(f, fx, fy, fz, tau):
    """Guo body force for D3Q19 with correct (1-1/(2τ)) factor.

    f_i += (1 - 1/(2τ)) * w_i * 3 * (c_i · F)
    """
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    factor = (1.0 - 1.0 / (2.0 * tau))
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
    wv = w.view(19, 1, 1, 1)
    forcing = factor * wv * 3.0 * (
        cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0) + cz * fz.unsqueeze(0)
    )
    return f + forcing


# ---------------------------------------------------------------------------
# NoDynamics: restore solid cells to pre-collision state
# ---------------------------------------------------------------------------
def nodynamics_restore(f, f_pre, solid):
    """Restore solid cells to pre-collision values (NoDynamics)."""
    sm = solid.unsqueeze(0).expand_as(f)
    return torch.where(sm, f_pre, f)


# ---------------------------------------------------------------------------
# Set solid cells to equilibrium at rest (enforce no-slip during streaming)
# ---------------------------------------------------------------------------
def set_solid_equilibrium_rest(f, solid, device):
    """Set solid cell populations to equilibrium at rest (u=0, ρ=1).

    This ensures streaming from solid to fluid carries zero momentum,
    providing a no-slip wall without bounce-back.
    """
    nz, ny, nx = solid.shape
    rho_w = torch.ones((nz, ny, nx), dtype=DTYPE, device=device)
    f_eq_rest = equilibrium3d(rho_w, torch.zeros_like(rho_w),
                             torch.zeros_like(rho_w), torch.zeros_like(rho_w),
                             device=device)
    sm = solid.unsqueeze(0).expand_as(f)
    return torch.where(sm, f_eq_rest, f)


# ---------------------------------------------------------------------------
# TEST 1: Poiseuille with wall function (y_val=0.5)
# ---------------------------------------------------------------------------
def run_poiseuille_wallfn(device_id, y_val=0.5, use_guo_factor=True,
                           set_solid_eq=True, nsteps=3000, label=""):
    """Poiseuille flow with wall function instead of bounce-back.

    Reference: u_max = G*H²/(2ν) (exact parabolic profile)
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0  # = 1/6
    u_max_target = 0.05
    warmup = 500

    H = ny - 2  # channel height = 10
    # Body force: u_max = G*H²/(8ν)  =>  G = 8ν*u_max/H²
    G = 8.0 * nu * u_max_target / (H * H)

    tag = f"[Pois-WF y_val={y_val} SDAA:{device_id}]{label}"
    print(f"\n{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f}", flush=True)
    print(f"{tag} H={H} G={G:.6e} u_max_target={u_max_target}", flush=True)
    print(f"{tag} y_val={y_val} use_guo_factor={use_guo_factor} set_solid_eq={set_solid_eq}", flush=True)

    # Solid mask: top and bottom walls
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    # Initialize: equilibrium at rest
    rho0 = torch.ones((nz, ny, nx), dtype=DTYPE, device=device)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=device)
    target_mass = f.sum().item()

    t0 = time.time()
    for step in range(1, nsteps + 1):
        f_pre = f.clone()

        # 1. Collision (BGK)
        f = collide_bgk3d(f, tau)

        # 2. NoDynamics: restore solid cells
        f = nodynamics_restore(f, f_pre, solid)

        # 3. Set solid cells to equilibrium at rest (no-slip during streaming)
        if set_solid_eq:
            f = set_solid_equilibrium_rest(f, solid, device)

        # 4. Wall function (instead of bounce-back)
        f, drag_fric, drag_pres = wall_function_3d(
            f, solid, nu, y_val=y_val, wall_law="gradient")

        # 5. Stream (periodic in x and z)
        f = stream3d(f)

        # 6. Driving force (Guo body force)
        fx = torch.full((nz, ny, nx), G, dtype=DTYPE, device=device)
        fy = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
        fz = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
        if use_guo_factor:
            f = guo_body_force_3d(f, fx, fy, fz, tau)
        else:
            f = ibm_apply_body_force_3d(f, fx, fy, fz)

        # Mass correction
        f = correct_mass3d(f, target_mass)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_prof = ux.mean(dim=(0, 2))
            u_max_sim = float(u_prof[ny // 2])
            print(f"{tag} step={step} u_max={u_max_sim:.6f} ({time.time()-t0:.0f}s)", flush=True)

    # Final measurements
    _, ux, _, _ = macroscopic3d(f)
    u_prof = ux.mean(dim=(0, 2)).cpu().numpy()

    # Exact Poiseuille: u(y) = (G/(2ν)) * y' * (H - y'), y' = y - 0.5
    u_exact = np.zeros(ny)
    for y in range(1, ny - 1):
        y_eff = y - 0.5
        u_exact[y] = (G / (2.0 * nu)) * y_eff * (H - y_eff)

    # Error
    u_err_max = 0.0
    for y in range(1, ny - 1):
        if abs(u_exact[y]) > 1e-10:
            err = abs(u_prof[y] - u_exact[y]) / abs(u_exact[y]) * 100
            u_err_max = max(u_err_max, err)

    u_max_sim = float(u_prof[ny // 2])
    u_max_err = abs(u_max_sim - u_max_target) / u_max_target * 100

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} u_max   = {u_max_sim:.6f}  (exact={u_max_target:.6f}, err={u_max_err:.2f}%)", flush=True)
    print(f"{tag} u_err   = {u_err_max:.2f}% (max relative error)", flush=True)
    print(f"{tag} drag_fric = {drag_fric:.6f}", flush=True)
    print(f"{tag} drag_pres = {drag_pres:.6f}", flush=True)
    print(f"{tag} u_profile:", flush=True)
    for y in range(ny):
        print(f"{tag}   y={y:2d}  u={u_prof[y]:.6f}  exact={u_exact[y]:.6f}  err={abs(u_prof[y]-u_exact[y])/max(abs(u_exact[y]),1e-10)*100:.2f}%", flush=True)

    return {
        "y_val": y_val, "use_guo_factor": use_guo_factor,
        "set_solid_eq": set_solid_eq,
        "u_max_sim": u_max_sim, "u_max_target": u_max_target,
        "u_max_err_pct": u_max_err, "u_err_max_pct": u_err_max,
        "u_profile": u_prof.tolist(), "u_exact": u_exact.tolist(),
        "drag_fric": drag_fric, "drag_pres": drag_pres,
    }


# ---------------------------------------------------------------------------
# TEST 3: Couette with wall function
# ---------------------------------------------------------------------------
def run_couette_wallfn(device_id, y_val=0.5, use_guo_factor=True,
                       set_solid_eq=True, nsteps=3000, label=""):
    """Couette flow with wall function instead of bounce-back.

    Top wall moving at u_top, bottom stationary.
    Reference: Cf = 2ν/(H*u_top) (exact)
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    u_top = 0.05
    warmup = 500

    H = ny - 2  # channel height = 10
    Cf_exact = 2.0 * nu / (H * u_top)

    tag = f"[Cout-WF y_val={y_val} SDAA:{device_id}]{label}"
    print(f"\n{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f} u_top={u_top}", flush=True)
    print(f"{tag} H={H} Cf_exact={Cf_exact:.6f}", flush=True)

    # Solid mask
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True   # bottom (stationary)
    solid[:, -1, :] = True  # top (moving)

    # Initialize: linear velocity profile
    rho0 = torch.ones((nz, ny, nx), dtype=DTYPE, device=device)
    y_vals = torch.arange(ny, dtype=DTYPE, device=device).view(1, ny, 1)
    u_init = u_top * (y_vals - 0.5) / H
    u_init[:, 0, :] = 0
    u_init[:, -1, :] = u_top
    f = equilibrium3d(rho0, u_init.expand(nz, ny, nx),
                      torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)
    target_mass = f.sum().item()

    t0 = time.time()
    cf_hist = []

    for step in range(1, nsteps + 1):
        f_pre = f.clone()

        # 1. Collision
        f = collide_bgk3d(f, tau)

        # 2. NoDynamics
        f = nodynamics_restore(f, f_pre, solid)

        # 3. Set solid to equilibrium (top wall with u_top, bottom at rest)
        if set_solid_eq:
            # Top wall: equilibrium with u_top
            rho_w = torch.ones_like(solid, dtype=DTYPE)
            ux_w = torch.zeros_like(solid, dtype=DTYPE)
            ux_w[:, -1, :] = u_top  # top wall moving
            f_eq_wall = equilibrium3d(rho_w, ux_w,
                                      torch.zeros_like(rho_w), torch.zeros_like(rho_w),
                                      device=device)
            sm = solid.unsqueeze(0).expand_as(f)
            f = torch.where(sm, f_eq_wall, f)

        # 4. Wall function
        f, drag_fric, drag_pres = wall_function_3d(
            f, solid, nu, y_val=y_val, wall_law="gradient")

        # 5. Stream
        f = stream3d(f)

        # 6. Mass correction
        f = correct_mass3d(f, target_mass)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            cf_hist.append(drag_fric)

        if step % 500 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_prof = ux.mean(dim=(0, 2))
            print(f"{tag} step={step} u[mid]={float(u_prof[ny//2]):.6f} ({time.time()-t0:.0f}s)", flush=True)

    # Final
    _, ux, _, _ = macroscopic3d(f)
    u_prof = ux.mean(dim=(0, 2)).cpu().numpy()

    # Exact Couette: u(y') = u_top * y' / H, y' = y - 0.5
    u_exact = np.zeros(ny)
    for y in range(1, ny - 1):
        u_exact[y] = u_top * (y - 0.5) / H

    u_err_max = 0.0
    for y in range(1, ny - 1):
        if abs(u_exact[y]) > 1e-10:
            err = abs(u_prof[y] - u_exact[y]) / abs(u_exact[y]) * 100
            u_err_max = max(u_err_max, err)

    cf_mean = sum(cf_hist) / max(len(cf_hist), 1) if cf_hist else 0.0

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} u_err   = {u_err_max:.2f}% (max relative error)", flush=True)
    print(f"{tag} drag_fric = {drag_fric:.6f}", flush=True)
    print(f"{tag} u_profile:", flush=True)
    for y in range(ny):
        print(f"{tag}   y={y:2d}  u={u_prof[y]:.6f}  exact={u_exact[y]:.6f}", flush=True)

    return {
        "y_val": y_val, "u_err_max_pct": u_err_max,
        "u_profile": u_prof.tolist(), "u_exact": u_exact.tolist(),
        "drag_fric": drag_fric, "Cf_exact": Cf_exact,
    }


# ---------------------------------------------------------------------------
# TEST 4: Debug wall function body force
# ---------------------------------------------------------------------------
def debug_wall_force(device_id):
    """Debug the wall function body force.

    Print force magnitude and direction at each near-wall cell.
    Check if force is tangent (F·n=0) and magnitude matches τ_w/y_val.
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    y_val = 0.5

    tag = f"[Debug SDAA:{device_id}]"
    print(f"\n{tag} === WALL FUNCTION BODY FORCE DEBUG ===", flush=True)
    print(f"{tag} nu={nu:.6f} tau={tau} y_val={y_val}", flush=True)

    # Solid mask
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    near = _near_wall_mask_no_wrap(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # Create a test velocity field: parabolic Poiseuille profile
    H = ny - 2
    G = 8.0 * nu * 0.05 / (H * H)
    u_max = 0.05

    rho = torch.ones((nz, ny, nx), dtype=DTYPE, device=device)
    ux = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
    uy = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
    uz = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
    for y in range(1, ny - 1):
        y_eff = y - 0.5
        u_val = (G / (2 * nu)) * y_eff * (H - y_eff)
        ux[:, y, :] = u_val

    f = equilibrium3d(rho, ux, uy, uz, device=device)

    # Compute wall normal
    nx_n, ny_n, nz_n = compute_wall_normal(solid, near)

    # Compute tangential velocity
    u_dot_n = ux * nx_n + uy * ny_n + uz * nz_n
    ut_x = ux - u_dot_n * nx_n
    ut_y = uy - u_dot_n * ny_n
    ut_z = uz - u_dot_n * nz_n
    u_tan_mag = torch.sqrt(ut_x**2 + ut_y**2 + ut_z**2).clamp(min=1e-12)

    # Gradient law: τ_w = ν * u_tan / y_val
    tau_w = nu * u_tan_mag / y_val
    u_tau = torch.sqrt(tau_w.clamp(min=1e-30))

    # Current body force: F = -(τ_w / y_val) * û_tan
    coef_current = -(tau_w / y_val) * near.to(DTYPE)
    fx_current = coef_current * (ut_x / u_tan_mag)
    fy_current = coef_current * (ut_y / u_tan_mag)
    fz_current = coef_current * (ut_z / u_tan_mag)

    # Correct body force: F = -τ_w * (1-1/(2τ)) * û_tan
    guo_factor = (1.0 - 1.0 / (2.0 * tau))
    coef_correct = -tau_w * guo_factor * near.to(DTYPE)
    fx_correct = coef_correct * (ut_x / u_tan_mag)
    fy_correct = coef_correct * (ut_y / u_tan_mag)
    fz_correct = coef_correct * (ut_z / u_tan_mag)

    # Alternative: F = -τ_w * û_tan (no guo factor, for split forcing)
    coef_alt = -tau_w * near.to(DTYPE)
    fx_alt = coef_alt * (ut_x / u_tan_mag)
    fy_alt = coef_alt * (ut_y / u_tan_mag)
    fz_alt = coef_alt * (ut_z / u_tan_mag)

    print(f"\n{tag} --- Force analysis at near-wall cells ---", flush=True)
    print(f"{tag} guo_factor = (1-1/(2τ)) = {guo_factor:.6f}", flush=True)
    print(f"{tag} y_val = {y_val}", flush=True)
    print(f"{tag} 1/y_val = {1.0/y_val:.6f}", flush=True)
    print(f"{tag} Current force factor: 1/y_val = {1.0/y_val:.6f} (NO guo factor)", flush=True)
    print(f"{tag} Correct force factor: guo_factor = {guo_factor:.6f}", flush=True)
    print(f"{tag} Ratio current/correct = {(1.0/y_val)/guo_factor:.6f}", flush=True)

    # Check F·n = 0 (force is tangent)
    f_dot_n_current = fx_current * nx_n + fy_current * ny_n + fz_current * nz_n
    f_dot_n_correct = fx_correct * nx_n + fy_correct * ny_n + fz_correct * nz_n

    print(f"\n{tag} F·n (current) = {float(f_dot_n_current[near].abs().max()):.2e} (should be 0)", flush=True)
    print(f"{tag} F·n (correct) = {float(f_dot_n_correct[near].abs().max()):.2e} (should be 0)", flush=True)

    # Print force at bottom near-wall cells (y=1)
    print(f"\n{tag} --- Bottom near-wall cells (y=1) ---", flush=True)
    for x in [0, nx//4, nx//2, nx-1]:
        z = nz // 2
        if near[z, 1, x]:
            u_val = float(ux[z, 1, x])
            tw = float(tau_w[z, 1, x])
            ut = float(u_tau[z, 1, x])
            fx_cur = float(fx_current[z, 1, x])
            fx_cor = float(fx_correct[z, 1, x])
            fx_al = float(fx_alt[z, 1, x])
            nx_norm = float(nx_n[z, 1, x])
            ny_norm = float(ny_n[z, 1, x])
            print(f"{tag}   x={x:3d}: u={u_val:.6f} τ_w={tw:.6e} u_tau={ut:.6e}", flush=True)
            print(f"{tag}          n=({nx_norm:.3f},{ny_norm:.3f},0)", flush=True)
            print(f"{tag}          F_current={fx_cur:.6e} F_correct={fx_cor:.6e} F_alt={fx_al:.6e}", flush=True)
            print(f"{tag}          ratio cur/cor={fx_cur/max(abs(fx_cor),1e-30):.4f} cur/alt={fx_cur/max(abs(fx_al),1e-30):.4f}", flush=True)

    # Print force at top near-wall cells (y=ny-2)
    print(f"\n{tag} --- Top near-wall cells (y={ny-2}) ---", flush=True)
    for x in [0, nx//4, nx//2, nx-1]:
        z = nz // 2
        if near[z, ny-2, x]:
            u_val = float(ux[z, ny-2, x])
            tw = float(tau_w[z, ny-2, x])
            ut = float(u_tau[z, ny-2, x])
            fx_cur = float(fx_current[z, ny-2, x])
            fx_cor = float(fx_correct[z, ny-2, x])
            fx_al = float(fx_alt[z, ny-2, x])
            nx_norm = float(nx_n[z, ny-2, x])
            ny_norm = float(ny_n[z, ny-2, x])
            print(f"{tag}   x={x:3d}: u={u_val:.6f} τ_w={tw:.6e} u_tau={ut:.6e}", flush=True)
            print(f"{tag}          n=({nx_norm:.3f},{ny_norm:.3f},0)", flush=True)
            print(f"{tag}          F_current={fx_cur:.6e} F_correct={fx_cor:.6e} F_alt={fx_al:.6e}", flush=True)
            print(f"{tag}          ratio cur/cor={fx_cur/max(abs(fx_cor),1e-30):.4f} cur/alt={fx_cur/max(abs(fx_al),1e-30):.4f}", flush=True)

    # Summary
    print(f"\n{tag} === SUMMARY ===", flush=True)
    print(f"{tag} Current: F = -(τ_w/y_val) * û_tan, applied via ibm_apply_body_force_3d (no guo factor)", flush=True)
    print(f"{tag}   Effective force = -(τ_w/y_val) / guo_factor = -(τ_w/y_val) / {guo_factor:.4f} = -{1.0/y_val/guo_factor:.4f}*τ_w", flush=True)
    print(f"{tag} Correct: F = -τ_w * guo_factor * û_tan, applied via ibm_apply_body_force_3d", flush=True)
    print(f"{tag}   Effective force = -τ_w * guo_factor / guo_factor = -τ_w", flush=True)
    print(f"{tag} Alt (split): F = -τ_w * û_tan, applied via ibm_apply_body_force_3d", flush=True)
    print(f"{tag}   Effective force = -τ_w / guo_factor = -{1.0/guo_factor:.4f}*τ_w", flush=True)
    print(f"{tag} Over-factor (current vs correct) = {1.0/y_val/guo_factor:.4f}x", flush=True)

    return {
        "y_val": y_val, "guo_factor": guo_factor,
        "current_over_factor": (1.0/y_val) / guo_factor,
        "F_dot_n_max": float(f_dot_n_current[near].abs().max()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test = sys.argv[1] if len(sys.argv) > 1 else "all"

    results = {}

    if test in ("1", "all"):
        print("=" * 70)
        print("TEST 1: Poiseuille with wall function (y_val=0.5) — SDAA:4")
        print("=" * 70)
        r = run_poiseuille_wallfn(4, y_val=0.5, use_guo_factor=True,
                                  set_solid_eq=True, nsteps=3000)
        results["test1"] = r

    if test in ("2", "all"):
        print("=" * 70)
        print("TEST 2: Poiseuille with wall function (y_val=1.0) — SDAA:5")
        print("=" * 70)
        r = run_poiseuille_wallfn(5, y_val=1.0, use_guo_factor=True,
                                  set_solid_eq=True, nsteps=3000)
        results["test2"] = r

    if test in ("3", "all"):
        print("=" * 70)
        print("TEST 3: Couette with wall function — SDAA:6")
        print("=" * 70)
        r = run_couette_wallfn(6, y_val=0.5, use_guo_factor=True,
                               set_solid_eq=True, nsteps=3000)
        results["test3"] = r

    if test in ("4", "all"):
        print("=" * 70)
        print("TEST 4: Debug wall function body force — SDAA:7")
        print("=" * 70)
        r = debug_wall_force(7)
        results["test4"] = r

    # Save results
    out = Path("wallfn_verify_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")
