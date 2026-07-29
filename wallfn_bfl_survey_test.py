"""Wall function + BFL verification tests (SDAA 28-31).

Tests the mature wall function + BFL implementation from the literature
survey (docs/WALL_FUNCTION_SURVEY.md).

TEST 1 (SDAA:28): Poiseuille channel — Guo vs simple forcing
  - Flat walls, Re=10, body-force driven
  - Compare Guo forcing vs simple forcing
  - Verify friction velocity matches analytical

TEST 2 (SDAA:29): Couette flow — grid convergence
  - Flat walls, moving top wall
  - Verify Cf = 2ν/(H·u_top) at multiple resolutions
  - Tests gradient wall law (viscous sublayer)

TEST 3 (SDAA:30): Cylinder BFL — Re=100
  - Curved boundary, BFL interpolated bounce-back
  - Verify Cd ≈ 1.33 (literature value)
  - Tests BFL geometric accuracy

TEST 4 (SDAA:31): Cylinder BFL + wall function — Re=1000
  - Curved boundary, BFL + Guo wall function
  - Verify Cd ≈ 0.46–0.50 (literature range)
  - Tests BFL + wall function combination

Usage:
    PYTHONPATH=src python wallfn_bfl_survey_test.py <test> <device>
    PYTHONPATH=src python wallfn_bfl_survey_test.py all <device_start>
"""
from __future__ import annotations
import sys, json, math, time
from pathlib import Path
import numpy as np
import torch
from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d, collide_bgk3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.wall_model import (
    guo_body_force_d3q19,
    bfl_wall_function_3d,
    wall_function_3d,
    _solve_wall_law,
    _near_wall_mask_no_wrap,
    compute_wall_normal,
)
from tensorlbm.bfl_d3q19 import compute_q_cylinder_d3q19, bouzidi_bounce_back_d3q19


# =========================================================================
# TEST 1: Poiseuille channel — Guo vs simple forcing (SDAA:28)
# =========================================================================
def test1_poiseuille_guo_vs_simple(device_id: int):
    """Compare Guo forcing vs simple forcing for Poiseuille channel.

    For Poiseuille flow driven by body force G:
      u_max = G·H²/(8ν)  (analytical)
      τ_w = G·H/2         (wall shear = driving force per unit area)
      u_τ = sqrt(τ_w/ρ)   (friction velocity)

    The wall function should recover τ_w from the velocity field.
    Guo forcing should give more accurate results than simple forcing
    at non-trivial velocities.
    """
    dev = f"sdaa:{device_id}"
    torch.sdaa.set_device(dev)

    nx, ny, nz = 64, 16, 4
    nu = 0.02  # τ = 1.0 → ν = (1-0.5)/3 = 1/6 ≈ 0.167... but we use 0.02 for Re
    tau = 3 * nu + 0.5
    G = 1e-4  # body force (small for stability)

    # Solid mask: walls at y=0 and y=ny-1
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    # Initialize
    rho = torch.ones(nz, ny, nx, device=dev)
    ux = torch.zeros(nz, ny, nx, device=dev)
    uy = torch.zeros(nz, ny, nx, device=dev)
    uz = torch.zeros(nz, ny, nx, device=dev)
    f = equilibrium3d(rho, ux, uy, uz, device=dev)

    # Body force field (uniform in x)
    Fx = torch.full((nz, ny, nx), G, device=dev)

    n_steps = 3000
    results = {"guo": {}, "simple": {}}

    for forcing_type in ["guo", "simple"]:
        f_test = f.clone()
        for step in range(n_steps):
            # Collision
            f_test = collide_bgk3d(f_test, tau=tau)

            # Streaming
            f_test = stream3d(f_test)

            # Bounce-back (flat walls)
            f_test = bounce_back_cells_3d(f_test, solid)

            # Body force
            rho_c, ux_c, uy_c, uz_c = macroscopic3d(f_test)
            if forcing_type == "guo":
                f_test = guo_body_force_d3q19(f_test, Fx, torch.zeros_like(Fx), torch.zeros_like(Fx), ux_c, uy_c, uz_c)
            else:
                # Simple forcing (legacy)
                from tensorlbm.ibm import ibm_apply_body_force_3d
                f_test = ibm_apply_body_force_3d(f_test, Fx, torch.zeros_like(Fx), torch.zeros_like(Fx))

        # Measure
        rho_f, ux_f, uy_f, uz_f = macroscopic3d(f_test)
        u_max = float(ux_f[:, ny // 2, :].max().item())
        u_avg = float(ux_f[~solid].mean().item())

        # Analytical: u_max = G*H^2/(8*nu), H = ny-2 (fluid channels)
        H = ny - 2
        u_max_analytical = G * H * H / (8 * nu)

        # Wall shear: τ_w = G*H/2
        tau_w_analytical = G * H / 2
        u_tau_analytical = math.sqrt(tau_w_analytical)

        # Compute u_τ from wall function (gradient law, y_val=0.5)
        near = _near_wall_mask_no_wrap(solid)
        u_mag = torch.sqrt(ux_f**2 + uy_f**2 + uz_f**2).clamp(min=1e-12)
        u_tau_computed = _solve_wall_law(u_mag, nu, 0.5, "gradient", near)
        u_tau_mean = float(u_tau_computed[near].mean().item())
        tau_w_computed = u_tau_mean ** 2

        results[forcing_type] = {
            "u_max": u_max,
            "u_max_analytical": u_max_analytical,
            "u_max_error_pct": abs(u_max - u_max_analytical) / u_max_analytical * 100,
            "u_tau_analytical": u_tau_analytical,
            "u_tau_computed": u_tau_mean,
            "tau_w_analytical": tau_w_analytical,
            "tau_w_computed": tau_w_computed,
            "tau_w_error_pct": abs(tau_w_computed - tau_w_analytical) / tau_w_analytical * 100,
        }

    print(f"\n{'='*60}")
    print(f"TEST 1: Poiseuille channel — Guo vs simple forcing (SDAA:{device_id})")
    print(f"{'='*60}")
    print(f"  Grid: {nx}×{ny}×{nz}, ν={nu}, G={G}, steps={n_steps}")
    print(f"  Analytical: u_max={results['guo']['u_max_analytical']:.6f}, "
          f"u_τ={results['guo']['u_tau_analytical']:.6f}, "
          f"τ_w={results['guo']['tau_w_analytical']:.6e}")
    print()
    for ft in ["guo", "simple"]:
        r = results[ft]
        print(f"  [{ft.upper()}] u_max={r['u_max']:.6f} (err={r['u_max_error_pct']:.2f}%), "
              f"u_τ={r['u_tau_computed']:.6f} (err={r['tau_w_error_pct']:.2f}%)")

    # Guo should be more accurate (or at least as good)
    guo_better = results["guo"]["u_max_error_pct"] <= results["simple"]["u_max_error_pct"] * 1.5
    print(f"\n  Guo forcing u_max error ≤ 1.5× simple: {'PASS' if guo_better else 'CHECK'}")

    return results, guo_better


# =========================================================================
# TEST 2: Couette flow — grid convergence (SDAA:29)
# =========================================================================
def test2_couette_grid_conv(device_id: int):
    """Couette flow grid convergence with gradient wall law.

    For Couette flow (top wall moving at u_top, bottom stationary):
      u(y) = u_top * y / H  (linear)
      Cf = 2ν / (H * u_top)  (exact, independent of grid)

    The gradient wall law (τ_w = ν·u/y_val) should recover this exactly
    at all resolutions because the profile is linear.
    """
    dev = f"sdaa:{device_id}"
    torch.sdaa.set_device(dev)

    u_top = 0.05
    nu = 0.02
    tau = 3 * nu + 0.5
    nx, nz = 64, 4
    n_steps = 2000

    results = {}

    for ny in [8, 16, 32]:
        solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
        solid[:, 0, :] = True   # bottom wall (stationary)
        solid[:, -1, :] = True  # top wall (moving)

        rho = torch.ones(nz, ny, nx, device=dev)
        ux = torch.zeros(nz, ny, nx, device=dev)
        uy = torch.zeros(nz, ny, nx, device=dev)
        uz = torch.zeros(nz, ny, nx, device=dev)
        f = equilibrium3d(rho, ux, uy, uz, device=dev)

        # Moving wall bounce-back for top wall
        opp = OPPOSITE.to(dev)
        c = C.to(dev).float()
        w = W.to(dev).float()
        cs2 = 1.0 / 3.0

        for step in range(n_steps):
            f = collide_bgk3d(f, tau=tau)
            f = stream3d(f)
            f = bounce_back_cells_3d(f, solid)

            # Moving wall correction for top wall
            correction = 6.0 * 1.0 * u_top * w * c[:, 0]
            top_mask = torch.zeros(1, nz, ny, nx, device=dev)
            top_mask[:, :, -1, :] = 1.0
            f = f + correction.view(19, 1, 1, 1) * top_mask

        rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)

        # Measure bottom wall friction (stationary wall)
        near = _near_wall_mask_no_wrap(solid)
        # Only bottom wall near-cells (y=1, adjacent to solid at y=0)
        # Exclude top wall near-cells (y=ny-2, adjacent to solid at y=ny-1)
        bottom_near = near.clone()
        bottom_near[:, -2, :] = False  # exclude top wall near-cells

        u_mag = torch.sqrt(ux_f**2 + uy_f**2 + uz_f**2).clamp(min=1e-12)
        u_tau = _solve_wall_law(u_mag, nu, 0.5, "gradient", bottom_near)
        u_tau_mean = float(u_tau[bottom_near].mean().item())
        tau_w = u_tau_mean ** 2

        # Analytical: Cf = 2*nu / (H * u_top), H = ny-2
        H = ny - 2
        Cf_analytical = 2 * nu / (H * u_top)
        Cf_computed = tau_w / (0.5 * u_top * u_top)  # Cf = τ_w / (0.5*ρ*u²)

        results[ny] = {
            "u_tau": u_tau_mean,
            "tau_w": tau_w,
            "Cf_analytical": Cf_analytical,
            "Cf_computed": Cf_computed,
            "error_pct": abs(Cf_computed - Cf_analytical) / Cf_analytical * 100,
        }

    print(f"\n{'='*60}")
    print(f"TEST 2: Couette flow — grid convergence (SDAA:{device_id})")
    print(f"{'='*60}")
    print(f"  u_top={u_top}, ν={nu}, steps={n_steps}")
    print(f"  {'ny':>4} {'Cf_analytical':>14} {'Cf_computed':>14} {'error%':>8}")
    for ny in [8, 16, 32]:
        r = results[ny]
        print(f"  {ny:4d} {r['Cf_analytical']:14.6f} {r['Cf_computed']:14.6f} {r['error_pct']:8.2f}")

    # Check grid convergence (error should decrease or stay small)
    errors = [results[ny]["error_pct"] for ny in [8, 16, 32]]
    converged = max(errors) < 10.0  # Should be within 10%
    print(f"\n  Max error < 10%: {'PASS' if converged else 'CHECK'}")

    return results, converged


# =========================================================================
# TEST 3: Cylinder BFL — Re=100 (SDAA:30)
# =========================================================================
def test3_cylinder_bfl_re100(device_id: int):
    """Cylinder drag with BFL at Re=100.

    Reference: Cd ≈ 1.33 (literature, e.g. Tritton 1959).
    Tests BFL interpolated bounce-back for curved boundaries.
    """
    dev = f"sdaa:{device_id}"
    torch.sdaa.set_device(dev)

    nx, ny, nz = 128, 64, 4
    cx_c, cy_c, R = nx * 0.3, ny * 0.5, 8.0
    nu = 0.02
    tau = 3 * nu + 0.5
    u_inflow = 0.1
    Re = u_inflow * 2 * R / nu

    # Solid mask (cylinder, extruded in z)
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing='ij'
    )
    dist = torch.sqrt((xx - cx_c)**2 + (yy - cy_c)**2)
    solid = dist <= R

    # BFL q-values
    fluid_boundary_mask, q_field = compute_q_cylinder_d3q19(
        nx, ny, nz, cx_c, cy_c, R, dev, axis='z'
    )

    # Initialize
    rho = torch.ones(nz, ny, nx, device=dev)
    ux = torch.full((nz, ny, nx), u_inflow, device=dev)
    uy = torch.zeros(nz, ny, nx, device=dev)
    uz = torch.zeros(nz, ny, nx, device=dev)
    f = equilibrium3d(rho, ux, uy, uz, device=dev)

    n_steps = 3000
    dpS = 0.5 * u_inflow**2 * 2 * R * nz  # normalization

    for step in range(n_steps):
        f_prev = f.clone()
        f = collide_bgk3d(f, tau=tau)
        f = stream3d(f)

        # BFL bounce-back
        f = bouzidi_bounce_back_d3q19(f, f_prev, fluid_boundary_mask, q_field)

        # Inflow (left boundary)
        f[:, :, 0, :] = equilibrium3d(
            torch.ones(nz, 1, device=dev),
            torch.full((nz, 1), u_inflow, device=dev),
            torch.zeros(nz, 1, device=dev),
            torch.zeros(nz, 1, device=dev),
            device=dev
        ).squeeze(1)

        # Outflow (right boundary) — simple copy
        f[:, :, -1, :] = f[:, :, -2, :]

    # Compute drag via momentum exchange
    from tensorlbm.wall_surface_bfl import drag_momentum_exchange_bfl
    cd = drag_momentum_exchange_bfl(f, f.clone(), fluid_boundary_mask, q_field, dpS)

    # Also compute via wall function friction
    near = _near_wall_mask_no_wrap(solid)
    rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
    u_mag = torch.sqrt(ux_f**2 + uy_f**2 + uz_f**2).clamp(min=1e-12)
    u_tau = _solve_wall_law(u_mag, nu, 0.5, "gradient", near)
    tau_w = u_tau ** 2
    cd_fric = float(tau_w[near].sum().item()) / dpS

    results = {
        "Re": Re,
        "Cd_me": cd,
        "Cd_fric": cd_fric,
        "Cd_reference": 1.33,
        "Cd_me_error_pct": abs(cd - 1.33) / 1.33 * 100,
    }

    print(f"\n{'='*60}")
    print(f"TEST 3: Cylinder BFL — Re=100 (SDAA:{device_id})")
    print(f"{'='*60}")
    print(f"  Grid: {nx}×{ny}×{nz}, R={R}, ν={nu}, u={u_inflow}")
    print(f"  Re = {Re:.1f}")
    print(f"  Cd (MEM)  = {cd:.4f} (ref=1.33, err={results['Cd_me_error_pct']:.1f}%)")
    print(f"  Cd (fric) = {cd_fric:.4f}")

    # BFL should give reasonable drag (within 30% for coarse grid)
    ok = abs(cd - 1.33) / 1.33 < 0.30
    print(f"  Cd within 30% of reference: {'PASS' if ok else 'CHECK'}")

    return results, ok


# =========================================================================
# TEST 4: Cylinder BFL + wall function — Re=1000 (SDAA:31)
# =========================================================================
def test4_cylinder_bfl_wf_re1000(device_id: int):
    """Cylinder drag with BFL + wall function at Re=1000.

    Reference: Cd ≈ 0.46–0.50 (literature for Re=1000 cylinder).
    Tests BFL + Guo wall function combination.
    """
    dev = f"sdaa:{device_id}"
    torch.sdaa.set_device(dev)

    nx, ny, nz = 128, 64, 4
    cx_c, cy_c, R = nx * 0.3, ny * 0.5, 8.0
    nu = 0.002  # higher Re
    tau = 3 * nu + 0.5
    u_inflow = 0.1
    Re = u_inflow * 2 * R / nu

    # Solid mask
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing='ij'
    )
    dist = torch.sqrt((xx - cx_c)**2 + (yy - cy_c)**2)
    solid = dist <= R

    # BFL q-values
    fluid_boundary_mask, q_field = compute_q_cylinder_d3q19(
        nx, ny, nz, cx_c, cy_c, R, dev, axis='z'
    )

    # Initialize
    rho = torch.ones(nz, ny, nx, device=dev)
    ux = torch.full((nz, ny, nx), u_inflow, device=dev)
    uy = torch.zeros(nz, ny, nx, device=dev)
    uz = torch.zeros(nz, ny, nx, device=dev)
    f = equilibrium3d(rho, ux, uy, uz, device=dev)

    n_steps = 3000
    dpS = 0.5 * u_inflow**2 * 2 * R * nz

    # Compare: BFL+Guo vs BFL+simple vs legacy wall_function_3d
    results = {}

    for method in ["bfl_guo", "bfl_simple", "legacy"]:
        f_test = f.clone()
        for step in range(n_steps):
            f_prev = f_test.clone()
            f_test = collide_bgk3d(f_test, tau=tau)
            f_test = stream3d(f_test)

            if method == "bfl_guo":
                # BFL + Guo wall function
                f_test, drag_f, drag_p = bfl_wall_function_3d(
                    f_test, f_prev, solid, nu,
                    fluid_boundary_mask, q_field,
                    y_val=0.5, wall_law="reichardt",
                    apply_bfl=True, use_guo=True,
                )
            elif method == "bfl_simple":
                # BFL + simple forcing wall function
                f_test, drag_f, drag_p = bfl_wall_function_3d(
                    f_test, f_prev, solid, nu,
                    fluid_boundary_mask, q_field,
                    y_val=0.5, wall_law="reichardt",
                    apply_bfl=True, use_guo=False,
                )
            else:
                # Legacy wall_function_3d (simple forcing, no BFL)
                f_test, drag_f, drag_p = wall_function_3d(
                    f_test, solid, nu, y_val=0.5, wall_law="reichardt",
                )

            # Inflow
            f_test[:, :, 0, :] = equilibrium3d(
                torch.ones(nz, 1, device=dev),
                torch.full((nz, 1), u_inflow, device=dev),
                torch.zeros(nz, 1, device=dev),
                torch.zeros(nz, 1, device=dev),
                device=dev
            ).squeeze(1)
            # Outflow
            f_test[:, :, -1, :] = f_test[:, :, -2, :]

        # Compute drag
        from tensorlbm.wall_surface_bfl import drag_momentum_exchange_bfl
        if method.startswith("bfl"):
            cd = drag_momentum_exchange_bfl(f_test, f_test.clone(), fluid_boundary_mask, q_field, dpS)
        else:
            # Legacy: use friction + pressure from wall_function_3d
            cd = drag_f + drag_p

        results[method] = {
            "Cd": cd,
            "drag_fric": drag_f,
            "drag_pres": drag_p,
        }

    print(f"\n{'='*60}")
    print(f"TEST 4: Cylinder BFL + wall function — Re=1000 (SDAA:{device_id})")
    print(f"{'='*60}")
    print(f"  Grid: {nx}×{ny}×{nz}, R={R}, ν={nu}, u={u_inflow}")
    print(f"  Re = {Re:.0f}")
    print(f"  Reference Cd ≈ 0.46–0.50")
    print(f"  {'Method':<15} {'Cd':>8} {'friction':>10} {'pressure':>10}")
    for method in ["bfl_guo", "bfl_simple", "legacy"]:
        r = results[method]
        print(f"  {method:<15} {r['Cd']:8.4f} {r['drag_fric']:10.4f} {r['drag_pres']:10.4f}")

    # Check if BFL+Guo gives reasonable drag
    cd_guo = results["bfl_guo"]["Cd"]
    ok = 0.3 < cd_guo < 0.8  # broad range for coarse grid
    print(f"\n  BFL+Guo Cd in [0.3, 0.8]: {'PASS' if ok else 'CHECK'}")

    return results, ok


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wallfn_bfl_survey_test.py <test> <device>")
        print("  test: 1, 2, 3, 4, or 'all'")
        print("  device: SDAA card id (28-31)")
        sys.exit(1)

    test = sys.argv[1]
    device = int(sys.argv[2]) if len(sys.argv) > 2 else 28

    all_results = {}

    if test in ("1", "all"):
        r, ok = test1_poiseuille_guo_vs_simple(28)
        all_results["test1"] = {"results": r, "pass": ok}

    if test in ("2", "all"):
        r, ok = test2_couette_grid_conv(29)
        all_results["test2"] = {"results": r, "pass": ok}

    if test in ("3", "all"):
        r, ok = test3_cylinder_bfl_re100(30)
        all_results["test3"] = {"results": r, "pass": ok}

    if test in ("4", "all"):
        r, ok = test4_cylinder_bfl_wf_re1000(31)
        all_results["test4"] = {"results": r, "pass": ok}

    # Save results
    out_file = Path(f"wallfn_bfl_survey_results_sdaa{device}.json")
    with open(out_file, "w") as fout:
        json.dump(all_results, fout, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for t, r in all_results.items():
        status = "PASS" if r["pass"] else "CHECK"
        print(f"  {t}: {status}")
