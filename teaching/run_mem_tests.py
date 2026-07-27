#!/usr/bin/env python
"""MEM + BB Bug Fix Test Suite — SDAA cards 8-15.

Runs all tests specified in the task:
1. Couette (SDAA:8): MEM vs pressure integration, target 0.00%
2. Poiseuille (SDAA:9): MEM vs pressure integration
3. Cylinder Re=200 (SDAA:10-11): MEM vs pressure, compare Cd
4. Sphere Re=100 (SDAA:12): MEM vs pressure
5. SUBOFF Re=1000 (SDAA:13): MEM vs pressure, target <6%
6. BB bug fix verification (SDAA:14-15): old BB vs corrected BB

Usage:
    PYTHONPATH=src python teaching/run_mem_tests.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback

import torch
import torch_sdaa

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, OPPOSITE
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_3d,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.momentum_exchange import (
    momentum_exchange_standard,
    momentum_exchange_galilean,
    momentum_exchange_bfl,
    compare_all_methods,
)


def test_couette(device="sdaa:8"):
    """Test 1: Couette flow — MEM vs pressure integration, target 0.00%."""
    print("\n" + "=" * 60)
    print("TEST 1: Couette Flow (SDAA:8)")
    print("=" * 60)

    dev = torch.device(device)
    ny, nx, nz = 16, 32, 4
    u_top, tau = 0.01, 1.0
    n_steps = 2000

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    rho0 = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=dev)

    rho_top = torch.ones(nz, nx, device=dev)
    feq_top = equilibrium3d(rho_top,
                            torch.full((nz, nx), u_top, device=dev),
                            torch.zeros(nz, nx, device=dev),
                            torch.zeros(nz, nx, device=dev), device=dev)

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f[:, :, -1, :] = feq_top[:, :, :]
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    _, ux, _, _ = macroscopic3d(f)
    u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))
    y_int = torch.arange(1, ny - 1, device=dev, dtype=torch.float32)
    u_exact = u_top * (y_int - 0.5) / (ny - 1 - 0.5)
    u_num = u_profile.cpu().numpy()
    u_ex = u_exact.cpu().numpy()
    max_err = max(abs(a - b) for a, b in zip(u_num, u_ex)) / abs(u_top) * 100

    # MEM on Couette (flat wall, should give 0 pressure drag)
    near = torch.zeros_like(solid)
    near[:, 1, :] = ~solid[:, 1, :]
    near[:, -2, :] = ~solid[:, -2, :]
    fx_me, _, _ = momentum_exchange_standard(f, solid, near)
    nu = (tau - 0.5) / 3.0
    dpS = 0.5 * u_top ** 2 * (ny - 2) * nz
    cd_me = fx_me / dpS if abs(dpS) > 1e-20 else 0.0

    # Pressure drag (should be ~0 for Couette)
    mesh = SurfaceMesh.from_gradient(solid, near)
    cd_p, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
    cd_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)

    print(f"  u_max error: {max_err:.2f}%")
    print(f"  Cd_MEM:      {cd_me:.6f}")
    print(f"  Cd_pressure: {cd_p:.6f} (should be ~0 for Couette)")
    print(f"  Cd_friction: {cd_f:.6f}")

    result = {
        "test": "couette", "device": device,
        "u_max_err_pct": max_err, "cd_mem": cd_me,
        "cd_pressure": cd_p, "cd_friction": cd_f,
        "pass": max_err < 0.01,
    }
    print(f"  Result: {'PASS' if result['pass'] else 'CHECK'}")
    return result


def test_poiseuille(device="sdaa:9"):
    """Test 2: Poiseuille flow — MEM vs pressure integration."""
    print("\n" + "=" * 60)
    print("TEST 2: Poiseuille Flow (SDAA:9)")
    print("=" * 60)

    dev = torch.device(device)
    ny, nx, nz = 16, 32, 4
    force, tau = 1e-5, 1.0
    n_steps = 5000

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    rho0 = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=dev)

    c = C.to(dev).float()
    w_force = c[:, 0] * force * (1.0 - 0.5 / tau)

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        rho, ux, uy, uz = macroscopic3d(f)
        ux = ux + force * tau / rho
        feq = equilibrium3d(rho, ux, uy, uz, device=dev)
        f = f - (f - feq) / tau
        f = f + w_force.view(19, 1, 1, 1).expand(19, nz, ny, nx)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    _, ux, _, _ = macroscopic3d(f)
    u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))
    nu = (tau - 0.5) / 3.0
    H = ny - 1.0
    y_int = torch.arange(1, ny - 1, device=dev, dtype=torch.float32)
    u_exact = force / (2.0 * nu) * (y_int - 0.5) * (H - 0.5 - (y_int - 0.5))
    u_num = u_profile.cpu().numpy()
    u_ex = u_exact.cpu().numpy()
    u_max_ex = float(u_ex.max())
    max_err = max(abs(a - b) for a, b in zip(u_num, u_ex)) / abs(u_max_ex) * 100

    # MEM
    near = torch.zeros_like(solid)
    near[:, 1, :] = ~solid[:, 1, :]
    near[:, -2, :] = ~solid[:, -2, :]
    fx_me, _, _ = momentum_exchange_standard(f, solid, near)
    dpS = 0.5 * u_max_ex ** 2 * (ny - 2) * nz if u_max_ex > 0 else 1.0
    cd_me = fx_me / dpS

    mesh = SurfaceMesh.from_gradient(solid, near)
    cd_p, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
    cd_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)

    print(f"  u_max error: {max_err:.2f}%")
    print(f"  Cd_MEM:      {cd_me:.6f}")
    print(f"  Cd_pressure: {cd_p:.6f}")
    print(f"  Cd_friction: {cd_f:.6f}")

    result = {
        "test": "poiseuille", "device": device,
        "u_max_err_pct": max_err, "cd_mem": cd_me,
        "cd_pressure": cd_p, "cd_friction": cd_f,
    }
    print(f"  Result: {'PASS' if max_err < 0.1 else 'CHECK'}")
    return result


def test_cylinder(device="sdaa:10"):
    """Test 3: Cylinder Re=200 — MEM vs pressure, compare Cd."""
    print("\n" + "=" * 60)
    print(f"TEST 3: Cylinder Re=200 (SDAA:10)")
    print("=" * 60)

    dev = torch.device(device)
    nx, ny, nz = 128, 64, 4
    R, u_in, tau = 8.0, 0.05, 0.55
    n_steps, warmup = 3000, 1500
    D = 2 * R
    nu = (tau - 0.5) / 3.0
    Re = u_in * D / nu
    dpS = 0.5 * u_in ** 2 * D * nz

    cx, cy = nx // 4, ny // 2
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis='z')

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    f[:, solid] = 0

    cd_me_hist, cd_pf_hist, cl_hist = [], [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in=u_in, obstacle_mask=solid)

        if step > warmup and step % 10 == 0:
            fx, fy, _ = momentum_exchange_standard(f, solid, near)
            cd_me_hist.append(fx / dpS)
            cl_hist.append(fy / dpS)
            cd_p, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
            cd_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_pf_hist.append(cd_p + cd_f)

        if step % 500 == 0:
            print(f"  Step {step}: Cd_MEM={cd_me_hist[-1]:.4f}, Cd_PF={cd_pf_hist[-1]:.4f}" if cd_me_hist else f"  Step {step}")

    n = len(cd_me_hist)
    cd_me = sum(cd_me_hist) / n if n else 0
    cd_pf = sum(cd_pf_hist) / n if n else 0
    cl_amp = max(abs(max(cl_hist)), abs(min(cl_hist))) if cl_hist else 0

    # Strouhal
    st = 0.0
    if len(cl_hist) > 20:
        crossings = sum(1 for i in range(1, len(cl_hist)) if cl_hist[i-1] < 0 and cl_hist[i] >= 0)
        if crossings > 1:
            st = (1.0 / (len(cl_hist) / crossings)) * D / u_in

    print(f"  Re={Re:.0f}")
    print(f"  Cd_MEM (mean): {cd_me:.4f}  (target ~1.33)")
    print(f"  Cd_PF  (mean): {cd_pf:.4f}")
    print(f"  Cl amplitude:  {cl_amp:.4f}")
    print(f"  Strouhal:      {st:.4f}  (target ~0.20)")

    return {"test": "cylinder", "device": device, "Re": Re,
            "cd_mem": cd_me, "cd_pf": cd_pf, "cl_amp": cl_amp, "st": st}


def test_sphere(device="sdaa:12"):
    """Test 4: Sphere Re=100 — MEM vs pressure."""
    print("\n" + "=" * 60)
    print(f"TEST 4: Sphere Re=100 (SDAA:12)")
    print("=" * 60)

    dev = torch.device(device)
    nx, ny, nz = 64, 48, 48
    R, u_in, tau = 6.0, 0.05, 0.6
    n_steps, warmup = 2000, 800
    D = 2 * R
    nu = (tau - 0.5) / 3.0
    Re = u_in * D / nu
    dpS = 0.5 * u_in ** 2 * 3.14159 * R ** 2

    cx, cy, cz = nx // 4, ny // 2, nz // 2
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    solid = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= R ** 2
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    f[:, solid] = 0

    cd_me_hist, cd_pf_hist = [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in=u_in, obstacle_mask=solid)

        if step > warmup and step % 20 == 0:
            fx, _, _ = momentum_exchange_standard(f, solid, near)
            cd_me_hist.append(fx / dpS)
            cd_p, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
            cd_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_pf_hist.append(cd_p + cd_f)

        if step % 500 == 0:
            print(f"  Step {step}: Cd_MEM={cd_me_hist[-1]:.4f}, Cd_PF={cd_pf_hist[-1]:.4f}" if cd_me_hist else f"  Step {step}")

    n = len(cd_me_hist)
    cd_me = sum(cd_me_hist) / n if n else 0
    cd_pf = sum(cd_pf_hist) / n if n else 0

    print(f"  Re={Re:.0f}")
    print(f"  Cd_MEM (mean): {cd_me:.4f}  (target ~1.09)")
    print(f"  Cd_PF  (mean): {cd_pf:.4f}")

    return {"test": "sphere", "device": device, "Re": Re,
            "cd_mem": cd_me, "cd_pf": cd_pf}


def test_suboff(device="sdaa:13"):
    """Test 5: SUBOFF Re=1000 — MEM vs pressure, target <6%."""
    print("\n" + "=" * 60)
    print(f"TEST 5: SUBOFF Re=1000 (SDAA:13)")
    print("=" * 60)

    dev = torch.device(device)
    from tensorlbm.suboff_cad import build_suboff_mask
    from tensorlbm.suboff_resistance import _voxel_wetted_area
    from tensorlbm.turbulence import collide_smagorinsky_mrt3d

    nx, ny, nz = 80, 40, 40
    u_in, tau, cs = 0.05, 0.55, 0.05
    n_steps, warmup = 1500, 300
    hull_length = nx * 0.6
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    nu = (tau - 0.5) / 3.0
    Re = u_in * hull_length / nu

    solid, _ = build_suboff_mask(
        hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=hull_length, device=dev,
    )
    near = get_near_wall_3d(solid)
    S_wet = _voxel_wetted_area(solid, 1.0)
    dpS = 0.5 * u_in ** 2 * S_wet
    mesh = SurfaceMesh.from_gradient(solid, near)

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    f[:, solid] = 0

    cd_me_hist, cd_pf_hist = [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in=u_in, obstacle_mask=solid)

        if step > warmup and step % 50 == 0:
            fx, _, _ = momentum_exchange_standard(f, solid, near)
            cd_me_hist.append(fx / dpS)
            cd_p, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
            cd_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_pf_hist.append(cd_p + cd_f)

        if step % 500 == 0:
            print(f"  Step {step}: Cd_MEM={cd_me_hist[-1]:.6f}, Cd_PF={cd_pf_hist[-1]:.6f}" if cd_me_hist else f"  Step {step}")

    n = len(cd_me_hist)
    cd_me = sum(cd_me_hist) / n if n else 0
    cd_pf = sum(cd_pf_hist) / n if n else 0
    target = 0.042
    err = abs(cd_pf - target) / target * 100 if target > 0 else 0

    print(f"  Re={Re:.0f}")
    print(f"  Cd_MEM (mean): {cd_me:.6f}")
    print(f"  Cd_PF  (mean): {cd_pf:.6f}  (target ~{target})")
    print(f"  PF error:      {err:.1f}%")

    return {"test": "suboff", "device": device, "Re": Re,
            "cd_mem": cd_me, "cd_pf": cd_pf, "pf_err_pct": err}


def test_bb_bug_fix(device="sdaa:14"):
    """Test 6: BB bug fix verification — old BB vs corrected BB."""
    print("\n" + "=" * 60)
    print(f"TEST 6: BB Bug Fix Verification (SDAA:14)")
    print("=" * 60)

    dev = torch.device(device)
    ny, nx, nz = 16, 32, 4
    u_top, tau = 0.01, 1.0
    n_steps = 2000

    def run_couette(use_f_pre):
        solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
        solid[:, 0, :] = True
        solid[:, -1, :] = True
        rho0 = torch.ones(nz, ny, nx, device=dev)
        f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                          torch.zeros_like(rho0), device=dev)
        rho_top = torch.ones(nz, nx, device=dev)
        feq_top = equilibrium3d(rho_top,
                                torch.full((nz, nx), u_top, device=dev),
                                torch.zeros(nz, nx, device=dev),
                                torch.zeros(nz, nx, device=dev), device=dev)
        for step in range(1, n_steps + 1):
            f_pre = f.clone()
            f = collide_bgk3d(f, tau=tau)
            if use_f_pre:
                f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
            else:
                f = bounce_back_cells_3d(f, solid)
            f[:, :, -1, :] = feq_top[:, :, :]
            f = stream3d(f)
            f[:, :, :, 0] = f[:, :, :, -2]
            f[:, :, :, -1] = f[:, :, :, -2]
        _, ux, _, _ = macroscopic3d(f)
        u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))
        y_int = torch.arange(1, ny - 1, device=dev, dtype=torch.float32)
        u_exact = u_top * (y_int - 0.5) / (ny - 1 - 0.5)
        u_num = u_profile.cpu().numpy()
        u_ex = u_exact.cpu().numpy()
        return max(abs(a - b) for a, b in zip(u_num, u_ex)) / abs(u_top) * 100

    print("  Running OLD BB (post-collision f)...")
    err_old = run_couette(use_f_pre=False)
    print(f"    OLD BB u_max error: {err_old:.2f}%")

    print("  Running CORRECTED BB (pre-collision f_pre)...")
    err_new = run_couette(use_f_pre=True)
    print(f"    CORRECTED BB u_max error: {err_new:.2f}%")

    print(f"  Improvement: {err_old:.2f}% → {err_new:.2f}%")
    print(f"  Result: {'PASS' if err_new < 0.01 else 'CHECK'}")

    return {"test": "bb_bug_fix", "device": device,
            "old_bb_err": err_old, "corrected_bb_err": err_new,
            "pass": err_new < 0.01}


def main():
    print("=" * 60)
    print("MEM + BB Bug Fix Test Suite")
    print("SDAA cards 8-15")
    print("=" * 60)

    results = []
    tests = [
        ("couette", lambda: test_couette("sdaa:8")),
        ("poiseuille", lambda: test_poiseuille("sdaa:9")),
        ("cylinder", lambda: test_cylinder("sdaa:10")),
        ("sphere", lambda: test_sphere("sdaa:12")),
        ("suboff", lambda: test_suboff("sdaa:13")),
        ("bb_bug_fix", lambda: test_bb_bug_fix("sdaa:14")),
    ]

    for name, test_fn in tests:
        try:
            t0 = time.time()
            result = test_fn()
            result["elapsed_s"] = time.time() - t0
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"test": name, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        if "error" in r:
            print(f"  {r['test']}: ERROR - {r['error']}")
        elif r.get("pass", False):
            print(f"  {r['test']}: PASS")
        else:
            print(f"  {r['test']}: completed")

    # Save results
    with open("mem_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to mem_test_results.json")

    return results


if __name__ == "__main__":
    main()
