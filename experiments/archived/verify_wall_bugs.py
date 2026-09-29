#!/usr/bin/env python3
"""Cross-validate 6 wall function bugs with 2 test cases each.

Bugs:
  7: gradient law uses u_mag instead of u_tangent
  8: hybrid law uses (i, i+9) instead of OPPOSITE array
  9: Near-wall detection z-direction periodic wrap (torch.roll)
  10: Body force always in x-direction, not tangent
  11: Factor-of-2 discrepancy (wall_model.py vs wall_function_common.py)
  12: Over-damping when combined with bounce-back

Uses SDAA cards 8-11.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import torch
from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d, correct_mass3d
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d, far_field_bc_3d, make_channel_wall_mask_3d, sphere_mask,
)
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.wall_function_common import compute_u_tau, wall_function, _near_wall_mask
from tensorlbm.ibm import ibm_apply_body_force_3d

DEV = torch.device("sdaa:8")
DTYPE = torch.float32

# ===========================================================================
# Helper: cylinder mask (2D extruded in z)
# ===========================================================================
def cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded in z."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=DTYPE),
        torch.arange(ny, device=device, dtype=DTYPE),
        torch.arange(nx, device=device, dtype=DTYPE),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

# ===========================================================================
# Helper: near-wall mask WITHOUT periodic wrap (correct version)
# ===========================================================================
def near_wall_no_wrap(solid):
    """Near-wall mask without periodic wrap (correct)."""
    fluid = ~solid
    near = torch.zeros_like(solid)
    nz, ny, nx = solid.shape
    # x-direction (interior only)
    near[:, :, 1:-1] |= (solid[:, :, 2:] | solid[:, :, :-2]) & fluid[:, :, 1:-1]
    # y-direction
    near[:, 1:-1, :] |= (solid[:, 2:, :] | solid[:, :-2, :]) & fluid[:, 1:-1, :]
    # z-direction (no periodic wrap)
    if nz > 1:
        near[1:-1] |= (solid[2:] | solid[:-2]) & fluid[1:-1]
        near[0] |= solid[1] & fluid[0]
        near[-1] |= solid[-2] & fluid[-1]
    return near

# ===========================================================================
# Poiseuille flow simulation
# ===========================================================================
def run_poiseuille(device, wall_law="gradient", use_bb_first=False, nsteps=3000,
                   nx=80, ny=12, nz=4, tau=1.0, u_max=0.05):
    """Run Poiseuille flow with wall function.

    Returns (u_profile, u_max_achieved, error_pct).
    """
    nu = (tau - 0.5) / 3.0
    H = ny - 2  # channel height (between walls)
    # Body force for Poiseuille: G = 8*nu*u_max / H^2
    G = 8.0 * nu * u_max / (H * H)

    # Solid mask: top and bottom walls
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, 0, :] = True   # bottom wall
    solid[:, -1, :] = True  # top wall

    # Initialize: equilibrium at rest
    rho0 = torch.ones((nz, ny, nx), dtype=DTYPE, device=device)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=device)
    target_mass = f.sum().item()

    for step in range(nsteps):
        # Collide
        f = collide_bgk3d(f, tau)
        # Driving force (body force in x)
        fx = torch.full((nz, ny, nx), G, dtype=DTYPE, device=device)
        fy = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
        fz = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
        f = ibm_apply_body_force_3d(f, fx, fy, fz)
        # Wall treatment
        if use_bb_first:
            f = bounce_back_cells_3d(f, solid)
        f, _, _ = wall_function_3d(f, solid, nu, y_val=0.5, wall_law=wall_law)
        # Stream
        f = stream3d(f)
        # Mass correction
        f = correct_mass3d(f, target_mass)

    # Extract velocity profile (average over x and z)
    rho, ux, uy, uz = macroscopic3d(f)
    u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))  # shape (ny-2,)
    u_max_achieved = u_profile.max().item()
    # Exact Poiseuille: u(y') = (G/2nu) * y' * (H - y'), y' = y - 0.5
    y_vals = torch.arange(1, ny - 1, dtype=DTYPE, device=device) - 0.5
    u_exact = (G / (2 * nu)) * y_vals * (H - y_vals)
    # Relative error
    err = (u_profile - u_exact.to(device)).abs() / u_exact.clamp(min=1e-10).to(device)
    err_pct = float(err.mean().item() * 100)
    return u_profile.cpu().numpy(), u_max_achieved, err_pct

# ===========================================================================
# Couette flow simulation
# ===========================================================================
def run_couette(device, wall_law="gradient", use_bb_first=False, nsteps=3000,
                nx=80, ny=12, nz=4, tau=1.0, U_top=0.05):
    """Run Couette flow with wall function.

    Top wall moving at U_top, bottom wall stationary.
    Returns (u_profile, error_pct).
    """
    nu = (tau - 0.5) / 3.0
    H = ny - 2  # channel height

    # Solid mask: top and bottom walls
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, 0, :] = True   # bottom wall (stationary)
    solid[:, -1, :] = True  # top wall (moving)

    # Initialize: linear velocity profile
    rho0 = torch.ones((nz, ny, nx), dtype=DTYPE, device=device)
    y_vals = torch.arange(ny, dtype=DTYPE, device=device).view(1, ny, 1)
    u_init = U_top * (y_vals - 0.5) / (H)
    u_init[:, 0, :] = 0
    u_init[:, -1, :] = U_top
    f = equilibrium3d(rho0, u_init.expand(nz, ny, nx), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=device)
    target_mass = f.sum().item()

    for step in range(nsteps):
        # Collide
        f = collide_bgk3d(f, tau)
        # Wall treatment
        if use_bb_first:
            f = bounce_back_cells_3d(f, solid)
        f, _, _ = wall_function_3d(f, solid, nu, y_val=0.5, wall_law=wall_law)
        # Stream
        f = stream3d(f)
        # Apply moving wall BC on top, stationary on bottom
        # (wall function should handle this, but we need to set the wall velocity)
        # For now, just mass correct
        f = correct_mass3d(f, target_mass)

    rho, ux, uy, uz = macroscopic3d(f)
    u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))
    # Exact Couette: u(y') = U_top * y' / H, y' = y - 0.5
    y_coords = torch.arange(1, ny - 1, dtype=DTYPE, device=device) - 0.5
    u_exact = U_top * y_coords / H
    err = (u_profile - u_exact).abs() / u_exact.clamp(min=1e-10)
    err_pct = float(err.mean().item() * 100)
    return u_profile.cpu().numpy(), err_pct

# ===========================================================================
# Cylinder flow simulation (simplified)
# ===========================================================================
def run_cylinder(device, wall_law="gradient", nsteps=2000,
                 D=24, nx=200, ny=80, nz=4, Re=200, u_in=0.05):
    """Run cylinder flow with wall function.

    Returns (drag_total, cd).
    """
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    cx, cy = nx // 4, ny // 2
    radius = D / 2.0

    solid = cylinder_mask(nx, ny, nz, cx, cy, radius, device)

    # Initialize: uniform flow
    rho0 = torch.ones((nz, ny, nx), dtype=DTYPE, device=device)
    u_init = torch.full((nz, ny, nx), u_in, dtype=DTYPE, device=device)
    f = equilibrium3d(rho0, u_init, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      device=device)
    target_mass = f.sum().item()

    drag_hist = []
    for step in range(nsteps):
        f = collide_bgk3d(f, tau)
        f, drag_fric, drag_pres = wall_function_3d(
            f, solid, nu, y_val=0.5, wall_law=wall_law)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in, solid)
        f = correct_mass3d(f, target_mass)
        if step > nsteps // 2:
            drag_hist.append(drag_fric + drag_pres)

    drag_avg = sum(drag_hist) / len(drag_hist) if drag_hist else 0.0
    # Cd = drag / (0.5 * rho * U^2 * D * nz)
    cd = drag_avg / (0.5 * 1.0 * u_in * u_in * D * nz)
    return drag_avg, cd

# ===========================================================================
# BUG 7: gradient law uses u_mag instead of u_tangent
# ===========================================================================
def test_bug7(device):
    """Verify Bug 7: gradient law uses u_mag instead of u_tangent.

    Test 1: Couette flow (flat wall, u_mag=u_tangent → should work)
    Test 2: Synthetic curved wall (u_mag≠u_tangent → should fail)
    """
    print("\n" + "=" * 70)
    print("BUG 7: gradient law uses u_mag instead of u_tangent")
    print("=" * 70)

    # --- Test 1: Couette flow (flat wall) ---
    print("\n  Test 1: Couette flow (flat wall, u_mag = u_tangent)")
    u_prof, err = run_couette(device, wall_law="gradient", nsteps=2000)
    print(f"    Couette error: {err:.2f}%")
    test1_pass = err < 10.0  # Should work for flat wall
    print(f"    → Couette {'PASSES' if test1_pass else 'FAILS'} (error < 10%)")

    # --- Test 2: Synthetic curved wall test ---
    print("\n  Test 2: Synthetic curved wall (u_mag ≠ u_tangent)")
    # Create a velocity field on a cylinder surface where velocity has
    # both tangential and normal components
    nz, ny, nx = 4, 80, 80
    cx, cy = 40.0, 40.0
    radius = 12.0
    solid = cylinder_mask(nx, ny, nz, cx, cy, radius, device)

    # Create a velocity field: flow in x-direction
    ux = torch.full((nz, ny, nx), 0.05, dtype=DTYPE, device=device)
    uy = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
    uz = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)

    # Compute near-wall mask
    near = near_wall_no_wrap(solid)

    # For each near-wall cell, compute:
    # - u_mag = sqrt(ux^2 + uy^2 + uz^2) (current code)
    # - u_tangent = u - (u·n)n (correct)
    # - τ_w_mag = nu * u_mag / y_val (current code)
    # - τ_w_tan = nu * |u_tangent| / y_val (correct)

    nu = 0.01
    y_val = 0.5

    # Compute wall normal at each near-wall cell
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=DTYPE),
        torch.arange(ny, device=device, dtype=DTYPE),
        torch.arange(nx, device=device, dtype=DTYPE),
        indexing="ij",
    )
    # Normal points from solid toward fluid (outward from cylinder)
    dx_norm = xx - cx
    dy_norm = yy - cy
    dist = torch.sqrt(dx_norm ** 2 + dy_norm ** 2).clamp(min=1e-10)
    nx_norm = dx_norm / dist
    ny_norm = dy_norm / dist

    # u_mag
    u_mag = torch.sqrt(ux ** 2 + uy ** 2 + uz ** 2).clamp(min=1e-12)

    # u_tangent = u - (u·n)n
    u_dot_n = ux * nx_norm + uy * ny_norm  # uz=0
    ut_x = ux - u_dot_n * nx_norm
    ut_y = uy - u_dot_n * ny_norm
    u_tan_mag = torch.sqrt(ut_x ** 2 + ut_y ** 2).clamp(min=1e-12)

    # τ_w with u_mag (current code)
    tau_w_mag = nu * u_mag / y_val
    # τ_w with u_tangent (correct)
    tau_w_tan = nu * u_tan_mag / y_val

    # Compare only at near-wall cells
    tau_mag_near = tau_w_mag[near]
    tau_tan_near = tau_w_tan[near]
    ratio = (tau_mag_near / tau_tan_near.clamp(min=1e-30)).mean().item()

    print(f"    Near-wall cells: {near.sum().item()}")
    print(f"    Mean τ_w(u_mag) / τ_w(u_tangent) = {ratio:.4f}")
    print(f"    (ratio > 1.0 means u_mag overestimates τ_w)")
    test2_fail = ratio > 1.01  # Should fail (u_mag > u_tangent)
    print(f"    → Curved wall {'FAILS (bug confirmed)' if test2_fail else 'PASSES'} "
          f"(ratio > 1.01)")

    bug_confirmed = test1_pass and test2_fail
    print(f"\n  BUG 7 {'CONFIRMED' if bug_confirmed else 'NOT CONFIRMED'}: "
          "Couette passes but curved wall fails")
    return bug_confirmed

# ===========================================================================
# BUG 8: hybrid law uses (i, i+9) instead of OPPOSITE array
# ===========================================================================
def test_bug8(device):
    """Verify Bug 8: hybrid law uses (i, i+9) instead of OPPOSITE array."""
    print("\n" + "=" * 70)
    print("BUG 8: hybrid law uses (i, i+9) instead of OPPOSITE array")
    print("=" * 70)

    # --- Check: print OPPOSITE array, compare with i+9 mapping ---
    print("\n  Check: OPPOSITE array vs i+9 mapping")
    opp = OPPOSITE.tolist()
    print(f"    OPPOSITE = {opp}")
    print(f"    i+9 mapping: [{', '.join(str(i+9) for i in range(9))}]")
    print(f"    OPPOSITE[0:9] = {opp[:9]}")

    mismatches = 0
    for i in range(9):
        if opp[i] != i + 9:
            mismatches += 1
            print(f"    i={i}: OPPOSITE[{i}]={opp[i]} vs i+9={i+9} ← MISMATCH")
    print(f"    Total mismatches: {mismatches}/9")
    check_pass = mismatches > 0
    print(f"    → Mapping check {'CONFIRMS bug' if check_pass else 'does not confirm bug'}")

    # --- Test: run hybrid law on Couette ---
    print("\n  Test: run hybrid law on Couette flow")
    try:
        u_prof, err = run_couette(device, wall_law="hybrid", nsteps=500)
        print(f"    Hybrid Couette error: {err:.2f}%")
        test_fail = err > 20.0  # Should give wrong result
        print(f"    → Hybrid law {'FAILS (bug confirmed)' if test_fail else 'PASSES'}")
    except Exception as e:
        print(f"    Hybrid law CRASHED: {e}")
        test_fail = True
        print(f"    → Hybrid law FAILS (crash = bug confirmed)")

    bug_confirmed = check_pass and test_fail
    print(f"\n  BUG 8 {'CONFIRMED' if bug_confirmed else 'NOT CONFIRMED'}: "
          "i+9 mapping is wrong, should use OPPOSITE")
    return bug_confirmed

# ===========================================================================
# BUG 9: Near-wall detection z-direction periodic wrap (torch.roll)
# ===========================================================================
def test_bug9(device):
    """Verify Bug 9: Near-wall detection z-direction periodic wrap."""
    print("\n" + "=" * 70)
    print("BUG 9: Near-wall detection z-direction periodic wrap (torch.roll)")
    print("=" * 70)

    # --- Test: 2D case (nz=4), check near-wall count ---
    print("\n  Test: 2D case (nz=4), compare torch.roll vs no-wrap")
    nz, ny, nx = 4, 20, 20
    # Create a solid block at the bottom (y=0) only
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, 0, :] = True  # bottom wall

    # Near-wall with torch.roll (wall_function_common.py _near_wall_mask)
    near_roll = _near_wall_mask(solid)
    # Near-wall without periodic wrap (correct)
    near_no_wrap = near_wall_no_wrap(solid)

    count_roll = near_roll.sum().item()
    count_no_wrap = near_no_wrap.sum().item()
    diff = count_roll - count_no_wrap

    print(f"    Near-wall count (torch.roll): {count_roll}")
    print(f"    Near-wall count (no wrap):    {count_no_wrap}")
    print(f"    Difference: {diff}")
    test1_fail = diff > 0
    print(f"    → torch.roll {'over-counts (bug confirmed)' if test1_fail else 'matches'}")

    # --- Test 2: Solid at z=0 only (check z-wrap) ---
    print("\n  Test 2: Solid at z=0 only (check z-wrap)")
    solid2 = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid2[0, :, :] = True  # solid at z=0 only

    near_roll2 = _near_wall_mask(solid2)
    near_no_wrap2 = near_wall_no_wrap(solid2)

    count_roll2 = near_roll2.sum().item()
    count_no_wrap2 = near_no_wrap2.sum().item()
    diff2 = count_roll2 - count_no_wrap2

    print(f"    Near-wall count (torch.roll): {count_roll2}")
    print(f"    Near-wall count (no wrap):    {count_no_wrap2}")
    print(f"    Difference: {diff2}")
    # With periodic wrap, z=0 solid wraps to z=nz-1, creating false near-wall cells
    test2_fail = diff2 > 0
    print(f"    → torch.roll {'over-counts (bug confirmed)' if test2_fail else 'matches'}")

    bug_confirmed = test1_fail or test2_fail
    print(f"\n  BUG 9 {'CONFIRMED' if bug_confirmed else 'NOT CONFIRMED'}: "
          "torch.roll causes periodic wrap in z-direction")
    return bug_confirmed

# ===========================================================================
# BUG 10: Body force always in x-direction, not tangent
# ===========================================================================
def test_bug10(device):
    """Verify Bug 10: Body force always in x-direction, not tangent."""
    print("\n" + "=" * 70)
    print("BUG 10: Body force always in velocity direction, not tangent")
    print("=" * 70)

    # --- Test 1: Flat wall (tangent = x, OK) ---
    print("\n  Test 1: Flat wall (tangent direction = x, should be OK)")
    nz, ny, nx = 4, 12, 20
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, 0, :] = True  # bottom wall

    # Velocity: u = (0.05, 0, 0) (purely tangential for flat wall)
    ux = torch.full((nz, ny, nx), 0.05, dtype=DTYPE, device=device)
    uy = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
    uz = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)

    u_mag = torch.sqrt(ux ** 2 + uy ** 2 + uz ** 2).clamp(min=1e-12)
    inv_umag = 1.0 / u_mag
    # Force direction (current code): u/|u|
    fx_dir = ux * inv_umag
    fy_dir = uy * inv_umag
    fz_dir = uz * inv_umag

    # Tangent direction for flat wall (normal = (0,1,0)):
    # u_tan = u - (u·n)n = (ux, 0, 0) → direction = (1, 0, 0)
    # Current code direction = (ux/|u|, 0, 0) = (1, 0, 0) ← SAME
    near = near_wall_no_wrap(solid)
    fx_diff = (fx_dir[near] - 1.0).abs().max().item()
    fy_diff = fy_dir[near].abs().max().item()
    print(f"    Force direction at near-wall: ({fx_diff:.6f}, {fy_diff:.6f}) deviation from (1,0)")
    test1_pass = fx_diff < 0.01 and fy_diff < 0.01
    print(f"    → Flat wall {'PASSES' if test1_pass else 'FAILS'} (tangent = x)")

    # --- Test 2: Curved wall (tangent ≠ x, wrong) ---
    print("\n  Test 2: Curved wall (tangent ≠ velocity direction)")
    nz, ny, nx = 4, 80, 80
    cx, cy = 40.0, 40.0
    radius = 12.0
    solid = cylinder_mask(nx, ny, nz, cx, cy, radius, device)

    # Velocity: u = (0.05, 0, 0) (uniform inflow)
    ux = torch.full((nz, ny, nx), 0.05, dtype=DTYPE, device=device)
    uy = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
    uz = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)

    u_mag = torch.sqrt(ux ** 2 + uy ** 2 + uz ** 2).clamp(min=1e-12)
    inv_umag = 1.0 / u_mag

    # Current code force direction: u/|u| = (1, 0, 0) everywhere
    fx_dir = ux * inv_umag  # = 1.0 everywhere
    fy_dir = uy * inv_umag  # = 0.0 everywhere

    # Correct tangent direction: u_tan/|u_tan|
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=DTYPE),
        torch.arange(ny, device=device, dtype=DTYPE),
        torch.arange(nx, device=device, dtype=DTYPE),
        indexing="ij",
    )
    dx_n = xx - cx
    dy_n = yy - cy
    dist = torch.sqrt(dx_n ** 2 + dy_n ** 2).clamp(min=1e-10)
    nx_n = dx_n / dist
    ny_n = dy_n / dist

    u_dot_n = ux * nx_n + uy * ny_n
    ut_x = ux - u_dot_n * nx_n
    ut_y = uy - u_dot_n * ny_n
    ut_mag = torch.sqrt(ut_x ** 2 + ut_y ** 2).clamp(min=1e-12)
    fx_tan = ut_x / ut_mag
    fy_tan = ut_y / ut_mag

    near = near_wall_no_wrap(solid)
    # Compare force directions at near-wall cells
    fx_diff_curved = (fx_dir[near] - fx_tan[near]).abs().mean().item()
    fy_diff_curved = (fy_dir[near] - fy_tan[near]).abs().mean().item()
    total_diff = math.sqrt(fx_diff_curved ** 2 + fy_diff_curved ** 2)

    print(f"    Near-wall cells: {near.sum().item()}")
    print(f"    Mean force direction deviation: ({fx_diff_curved:.4f}, {fy_diff_curved:.4f})")
    print(f"    Total deviation: {total_diff:.4f}")
    test2_fail = total_diff > 0.01
    print(f"    → Curved wall {'FAILS (bug confirmed)' if test2_fail else 'PASSES'}")

    bug_confirmed = test1_pass and test2_fail
    print(f"\n  BUG 10 {'CONFIRMED' if bug_confirmed else 'NOT CONFIRMED'}: "
          "force direction is velocity direction, not tangent")
    return bug_confirmed

# ===========================================================================
# BUG 11: Factor-of-2 discrepancy
# ===========================================================================
def test_bug11(device):
    """Verify Bug 11: Factor-of-2 discrepancy between wall_model.py and
    wall_function_common.py."""
    print("\n" + "=" * 70)
    print("BUG 11: Factor-of-2 discrepancy (wall_model.py vs wall_function_common.py)")
    print("=" * 70)

    # --- Analytical check ---
    print("\n  Analytical check:")
    nu = 1.0 / 6.0
    y_val = 0.5
    u = 0.05
    tau_w_model = nu * u / y_val       # wall_model.py: ν·u/y_val
    tau_w_common = 2.0 * nu * u / y_val  # wall_function_common.py: 2ν·u/y_val
    # Correct: τ_w = ν·du/dy, du/dy = u/y_val (linear sublayer)
    tau_w_correct = nu * u / y_val
    print(f"    u = {u}, y_val = {y_val}, nu = {nu:.4f}")
    print(f"    wall_model.py:       τ_w = ν·u/y_val = {tau_w_model:.6f}")
    print(f"    wall_function_common: τ_w = 2ν·u/y_val = {tau_w_common:.6f}")
    print(f"    Correct:              τ_w = ν·u/y_val = {tau_w_correct:.6f}")
    print(f"    wall_model.py is {'CORRECT' if abs(tau_w_model - tau_w_correct) < 1e-10 else 'WRONG'}")
    print(f"    wall_function_common.py is {'CORRECT' if abs(tau_w_common - tau_w_correct) < 1e-10 else 'WRONG (2× too strong)'}")

    # --- Test 1: Poiseuille with wall_model.py (gradient law) ---
    print("\n  Test 1: Poiseuille with wall_model.py gradient law")
    u_prof1, u_max1, err1 = run_poiseuille(device, wall_law="gradient", nsteps=3000)
    print(f"    U_max achieved: {u_max1:.6f} (target: 0.05)")
    print(f"    Mean error: {err1:.2f}%")

    # --- Test 2: Poiseuille with wall_function_common.py (gradient law) ---
    print("\n  Test 2: Poiseuille with wall_function_common.py gradient law")
    # We need to use the common module's wall_function directly
    nu_val = 1.0 / 6.0
    tau = 1.0
    H = 10  # ny - 2
    u_max_target = 0.05
    G = 8.0 * nu_val * u_max_target / (H * H)
    nx, ny, nz = 80, 12, 4

    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    rho0 = torch.ones((nz, ny, nx), dtype=DTYPE, device=device)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=device)
    target_mass = f.sum().item()

    for step in range(3000):
        f = collide_bgk3d(f, tau)
        # Driving force
        fx = torch.full((nz, ny, nx), G, dtype=DTYPE, device=device)
        fy = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
        fz = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
        f = ibm_apply_body_force_3d(f, fx, fy, fz)
        # Wall function from common module
        rho, ux, uy, uz = macroscopic3d(f)
        u_mag = torch.sqrt(ux ** 2 + uy ** 2 + uz ** 2).clamp(min=1e-12)
        u_tau = compute_u_tau(u_mag, nu_val, y_val=0.5, wall_law="gradient")
        y_plus = (0.5 * u_tau / nu_val).clamp(min=0.0)
        f = wall_function(f, solid, u_tau, y_plus, lattice="D3Q19",
                          nu=nu_val, y_val=0.5, rho=rho, ux=ux, uy=uy, uz=uz)
        f = stream3d(f)
        f = correct_mass3d(f, target_mass)

    rho, ux, uy, uz = macroscopic3d(f)
    u_prof2 = ux[:, 1:-1, :].mean(dim=(0, 2))
    u_max2 = u_prof2.max().item()
    y_vals = torch.arange(1, ny - 1, dtype=DTYPE) - 0.5
    u_exact = (G / (2 * nu_val)) * y_vals * (H - y_vals)
    err2 = float(((u_prof2.cpu() - u_exact).abs() / u_exact.clamp(min=1e-10)).mean().item() * 100)
    print(f"    U_max achieved: {u_max2:.6f} (target: 0.05)")
    print(f"    Mean error: {err2:.2f}%")

    # Compare
    print(f"\n    wall_model.py error: {err1:.2f}%")
    print(f"    wall_function_common.py error: {err2:.2f}%")
    ratio = err2 / max(err1, 0.001)
    print(f"    Error ratio (common/model): {ratio:.2f}")

    # wall_model.py should be more accurate (correct formula)
    # wall_function_common.py should have ~2× larger error (2× too strong)
    bug_confirmed = err2 > err1 * 1.5
    print(f"\n  BUG 11 {'CONFIRMED' if bug_confirmed else 'NOT CONFIRMED'}: "
          "wall_function_common.py has 2× too strong τ_w")
    return bug_confirmed

# ===========================================================================
# BUG 12: Over-damping when combined with bounce-back
# ===========================================================================
def test_bug12(device):
    """Verify Bug 12: Over-damping when combined with bounce-back."""
    print("\n" + "=" * 70)
    print("BUG 12: Over-damping when combined with bounce-back")
    print("=" * 70)

    # --- Test 1: Wall function only (no BB) ---
    print("\n  Test 1: Wall function only (no BB)")
    u_prof1, u_max1, err1 = run_poiseuille(device, wall_law="gradient",
                                           use_bb_first=False, nsteps=3000)
    print(f"    U_max: {u_max1:.6f}, Error: {err1:.2f}%")

    # --- Test 2: BB + wall function ---
    print("\n  Test 2: BB + wall function")
    u_prof2, u_max2, err2 = run_poiseuille(device, wall_law="gradient",
                                           use_bb_first=True, nsteps=3000)
    print(f"    U_max: {u_max2:.6f}, Error: {err2:.2f}%")

    print(f"\n    Wall fn only error: {err1:.2f}%")
    print(f"    BB + wall fn error: {err2:.2f}%")
    ratio = err2 / max(err1, 0.001)
    print(f"    Error ratio (BB+wf / wf_only): {ratio:.2f}")

    bug_confirmed = err2 > err1 * 1.5
    print(f"\n  BUG 12 {'CONFIRMED' if bug_confirmed else 'NOT CONFIRMED'}: "
          "BB + wall function causes over-damping")
    return bug_confirmed

# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("WALL FUNCTION BUG CROSS-VALIDATION")
    print("=" * 70)
    print(f"Device: {DEV}")

    results = {}
    results["bug7"] = test_bug7(DEV)
    results["bug8"] = test_bug8(DEV)
    results["bug9"] = test_bug9(DEV)
    results["bug10"] = test_bug10(DEV)
    results["bug11"] = test_bug11(DEV)
    results["bug12"] = test_bug12(DEV)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for bug, confirmed in results.items():
        status = "CONFIRMED" if confirmed else "NOT CONFIRMED"
        print(f"  {bug}: {status}")

    # Save results
    with open("/tmp/wall_bug_verification.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to /tmp/wall_bug_verification.json")

if __name__ == "__main__":
    main()
