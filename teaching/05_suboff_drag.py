#!/usr/bin/env python3
"""05 — SUBOFF drag: Cd, Cf with fixed BB.

SUBOFF is a standard axisymmetric submarine hull benchmark:
  - DARPA SUBOFF AFF-8 bare hull configuration
  - Re=1e6 (based on hull length)
  - Reference: Cf ≈ 0.042 (friction drag coefficient)

This example demonstrates:
  1. Axisymmetric body geometry (body of revolution)
  2. Pre-collision BB for curved 3D surface
  3. Pressure + friction integration with analytical normals
  4. Smagorinsky LES turbulence model

Usage:
  python 05_suboff_drag.py [device_id]

Expected output:
  Cd ≈ 0.04-0.05 (ref Cf=0.042)
"""
from __future__ import annotations

import sys
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d, correct_mass3d
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    get_near_wall_3d,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.momentum_exchange import (
    drag_momentum_exchange_pre,
    bounce_back_pre_collision,
)


def run_suboff(device_id=4, n_steps=5000, warmup=1000):
    """Run SUBOFF flow and compute drag."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # Import SUBOFF geometry
    try:
        from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig
        from tensorlbm.turbulence import collide_smagorinsky_mrt3d
    except ImportError as e:
        print(f"Cannot import SUBOFF modules: {e}")
        print("Running simplified version with cylinder proxy...")
        return _run_suboff_proxy(device_id, n_steps, warmup)

    # Parameters
    L = 80
    nx, ny, nz = 200, 80, 80
    Re = 1000  # Lower Re for stability
    u_in = 0.06
    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    Cf_ref = 0.042

    config = SuboffConfig()
    radius = config.r_over_l * L
    D = 2.0 * radius
    cx = nx * 0.30
    cy = ny * 0.5
    cz = nz * 0.5
    dpS = 0.5 * u_in ** 2 * math.pi * D * L

    print(f"=== SUBOFF Drag Re={Re} (SDAA:{device_id}) ===")
    print(f"Grid: {nx}x{ny}x{nz}, L={L}, R_max={radius:.4f}, D={D:.4f}")
    print(f"u_in={u_in}, nu={nu:.6e}, tau={tau:.6f}, Cs={cs_smag}")
    print(f"dpS={dpS:.6e}, Cf_ref={Cf_ref}")
    print(f"Steps: {n_steps} (warmup={warmup})")

    t0 = time.time()

    solid, stats = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=L, radius=radius,
        config=config, device=device,
    )
    n_solid = int(solid.sum().item())
    print(f"Solid cells: {n_solid}  L/D={stats['L_D_ratio']}")

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"Near-wall cells: {n_near}")

    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)

    # Initialize
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    target_mass = float(rho0.sum().item())

    cd_p_hist, cd_f_hist, cd_tot_hist = [], [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # NoDynamics
        sm = solid.unsqueeze(0).expand_as(f)
        f = torch.where(sm, f_pre, f)

        # FIXED bounce-back
        f = bounce_back_pre_collision(f, f_pre, solid)

        # Streaming
        f = stream3d(f)

        # Far-field BC
        f = far_field_bc_3d(f, u_in)

        # Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, target_mass)

        if not torch.isfinite(f).all():
            print(f"DIVERGED at step {step}")
            break

        # Forces
        fx_p, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
        fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu, formula='standard')
        cd_tot = fx_p + fx_f

        if step > warmup:
            cd_p_hist.append(fx_p)
            cd_f_hist.append(fx_f)
            cd_tot_hist.append(cd_tot)

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            if n_avg > 0:
                print(
                    f"  step={step} Cd_p={sum(cd_p_hist[-n_avg:])/n_avg:.6f} "
                    f"Cd_f={sum(cd_f_hist[-n_avg:])/n_avg:.6f} "
                    f"Cd_tot={sum(cd_tot_hist[-n_avg:])/n_avg:.6f} "
                    f"({time.time()-t0:.0f}s)"
                )

    elapsed = time.time() - t0
    n_final = min(1000, max(1, len(cd_tot_hist)))

    cd_p_mean = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_mean = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_mean = sum(cd_tot_hist[-n_final:]) / n_final

    err_pct = abs(cd_tot_mean - Cf_ref) / Cf_ref * 100

    print(f"\n=== FINAL RESULTS ===")
    print(f"Cd_pressure = {cd_p_mean:.6f}")
    print(f"Cd_friction = {cd_f_mean:.6f}")
    print(f"Cd_total   = {cd_tot_mean:.6f}  (ref Cf={Cf_ref}, err={err_pct:.1f}%)")
    print(f"Time: {elapsed:.0f}s")

    return {
        "Cd_pressure": cd_p_mean,
        "Cd_friction": cd_f_mean,
        "Cd_total": cd_tot_mean,
        "Cf_ref": Cf_ref,
        "error_pct": err_pct,
    }


def _run_suboff_proxy(device_id, n_steps, warmup):
    """Simplified SUBOFF proxy using elongated ellipsoid."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    from tensorlbm.solver3d import collide_bgk3d

    L = 80
    nx, ny, nz = 200, 80, 80
    Re = 1000
    u_in = 0.06
    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5
    Cf_ref = 0.042

    a, b, c = L / 2, 8.0, 8.0
    cx, cy, cz = nx * 0.3, ny * 0.5, nz * 0.5
    D = 2 * b
    dpS = 0.5 * u_in ** 2 * math.pi * D * L

    print(f"=== SUBOFF Proxy (ellipsoid) Re={Re} (SDAA:{device_id}) ===")

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    solid = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 + ((zz - cz) / c) ** 2 <= 1.0
    n_solid = int(solid.sum().item())
    print(f"Solid cells: {n_solid}")

    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_ellipsoid(solid, near, cx, cy, cz, a, b, c)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    target_mass = float(rho0.sum().item())

    cd_p_hist, cd_f_hist, cd_tot_hist = [], [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        sm = solid.unsqueeze(0).expand_as(f)
        f = torch.where(sm, f_pre, f)
        f = bounce_back_pre_collision(f, f_pre, solid)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in)
        if step % 200 == 0:
            f = correct_mass3d(f, target_mass)

        if step > warmup:
            fx_p, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
            fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu, formula='standard')
            cd_p_hist.append(fx_p)
            cd_f_hist.append(fx_f)
            cd_tot_hist.append(fx_p + fx_f)

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            if n_avg > 0:
                print(f"  step={step} Cd_tot={sum(cd_tot_hist[-n_avg:])/n_avg:.6f}")

    n_final = max(1, len(cd_tot_hist))
    cd_tot_mean = sum(cd_tot_hist[-n_final:]) / n_final
    err = abs(cd_tot_mean - Cf_ref) / Cf_ref * 100
    print(f"\nCd_total = {cd_tot_mean:.6f} (ref={Cf_ref}, err={err:.1f}%)")
    return {"Cd_total": cd_tot_mean, "Cf_ref": Cf_ref, "error_pct": err}


if __name__ == "__main__":
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_suboff(device_id)
