#!/usr/bin/env python3
"""Re-test all geometries with BB fix (Bug 27) — SDAA 8-11.

Bug 27 fix: bounce_back_cells_3d now accepts f_pre parameter.
  OLD (buggy):  f = bounce_back_cells_3d(f, solid)          # post-collision f
  NEW (fixed):  f = bounce_back_cells_3d(f, solid, f_pre=f_pre)  # pre-collision f

The BB bug caused u_max 16.66% too high → friction too high → grid divergence.
Fix: use pre-collision f → u_max 0.00%.

TEST 1: Cylinder Re=200 D=48 (SDAA:8)
  5000 steps, MRT+Smag(Cs=0.05), from_cylinder normals
  Previous (pre-BB-fix): Cd=1.63 (25.3%), ref Cd=1.30

TEST 2: Sphere Re=100 D=40 (SDAA:9)
  3000 steps, MRT+Smag(Cs=0.05), from_sphere normals
  Previous: Cd=1.069 (2.0%), ref Cd=1.09

TEST 3: SUBOFF Re=1000 L=80 (SDAA:10)
  5000 steps, MRT+Smag(Cs=0.05), from_suboff normals
  Previous: Cd=0.044 (5.6%), ref Cf=0.042

TEST 4: NACA 0012 Re=1000 (SDAA:11)
  10000 steps, MRT+Smag(Cs=0.05), from_naca normals, 6L domain
  Previous: Cd=0.036 (28.3%), ref Cd=0.05

Usage:
  PYTHONPATH=src python bb_fix_allgeo_worker.py <test_name> <device_id> <output_json>
  test_name: cylinder | sphere | suboff | naca0012
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d,
    far_field_bc_3d,
)
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
    get_near_wall_2d,
    get_near_wall_3d,
)


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device):
    """Vectorized sphere mask: (i-cx)^2+(j-cy)^2+(k-cz)^2 < R^2."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2


def build_naca(chord, nx, ny, x_le, y_c, device):
    """Build NACA 0012 solid mask (2D extruded in z, nz=4).

    Standard NACA 4-digit thickness formula (symmetric, no camber):
      yt = 5*t*(0.2969*sqrt(xc) - 0.1260*xc - 0.3516*xc^2 + 0.2843*xc^3 - 0.1015*xc^4)
    where t=0.12 for NACA 0012, xc = (i - x_le) / chord, 0 <= xc <= 1.
    """
    nz = 4
    t = 0.12
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    for k in range(nz):
        for i in range(nx):
            xc = (i - x_le) / chord
            if 0 <= xc <= 1:
                yt = 5.0 * t * (
                    0.2969 * math.sqrt(xc)
                    - 0.1260 * xc
                    - 0.3516 * xc ** 2
                    + 0.2843 * xc ** 3
                    - 0.1015 * xc ** 4
                )
                j_lo = max(0, int(y_c - yt * chord))
                j_hi = min(ny - 1, int(y_c + yt * chord))
                solid[k, j_lo : j_hi + 1, i] = True
    return solid


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def run_cylinder(device_id, output_path):
    """TEST 1: Cylinder Re=200 D=48 with BB fix."""
    Re = 200
    D = 48
    nx = 400
    ny = 160
    nz = 4
    u_in = 0.08
    nu = u_in * D / Re  # 0.0192
    tau = 3.0 * nu + 0.5  # 0.5576
    cs_smag = 0.05
    n_steps = 5000
    Cd_ref = 1.30
    radius = D / 2.0
    cx = nx * 0.25
    cy = ny * 0.5
    A_frontal = D * nz
    dpS = 0.5 * u_in ** 2 * A_frontal

    tag = f"[SDAA:{device_id} CYL Re=200 BB-fix]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(
        f"{tag} D={D} nx={nx} ny={ny} nz={nz} u_in={u_in} "
        f"nu={nu:.6f} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.6f} "
        f"Cd_ref={Cd_ref} n_steps={n_steps}",
        flush=True,
    )

    t0 = time.time()
    solid = build_cylinder_mask(nx, ny, nz, cx, cy, radius, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    near = get_near_wall_2d(solid, axis='z')
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, radius, axis='z')

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = max(1000, n_steps // 5)
    bc_config = {'far_field_faces': ['y-', 'y+'], 'periodic_faces': ['z-', 'z+']}

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky LES)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming) — BB FIX: pass f_pre
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in, bc_config=bc_config)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Drag computation
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)

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
            else:
                print(f"{tag} step={step} (warmup, {time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = max(1, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist) / n_final if cd_p_hist else float("nan")
    cd_f_final = sum(cd_f_hist) / n_final if cd_f_hist else float("nan")
    cd_tot_final = sum(cd_tot_hist) / n_final if cd_tot_hist else float("nan")
    cl_final = sum(cl_hist) / n_final if cl_hist else float("nan")

    err_pct = abs(cd_tot_final - Cd_ref) / Cd_ref * 100 if Cd_ref > 0 and math.isfinite(cd_tot_final) else float("nan")

    result = {
        "case": tag,
        "test": "cylinder_re200_bb_fix",
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "normal_method": "from_cylinder",
        "bb_fix": True,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "Cd_ref": Cd_ref,
        "error_pct": err_pct,
        "previous_pre_bb_fix": {"Cd_total": 1.63, "error_pct": 25.3},
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"(ref={Cd_ref}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )
    print(
        f"{tag} Previous (pre-BB-fix): Cd_tot=1.63 err=25.3% → "
        f"BB-fix: Cd_tot={cd_tot_final:.4f} err={err_pct:.1f}%",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def run_sphere(device_id, output_path):
    """TEST 2: Sphere Re=100 D=40 with BB fix."""
    Re = 100
    D = 40
    nx = 120
    ny = 120
    nz = 120
    u_in = 0.08
    nu = u_in * D / Re  # 0.032
    tau = 3.0 * nu + 0.5  # 0.596
    cs_smag = 0.05
    n_steps = 3000
    Cd_ref = 1.09
    R = D / 2.0
    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2

    tag = f"[SDAA:{device_id} SPH Re=100 BB-fix]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(
        f"{tag} D={D} nx={nx} ny={ny} nz={nz} u_in={u_in} "
        f"nu={nu:.6f} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.6f} "
        f"Cd_ref={Cd_ref} n_steps={n_steps}",
        flush=True,
    )

    t0 = time.time()
    solid = build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
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
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky LES)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming) — BB FIX: pass f_pre
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Drag computation
        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, fz_f = drag_friction_integration(f, mesh, dpS, nu)

        cd_p = fx_p
        cd_f = fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f
        fz_tot = fz_p + fz_f

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)
        fz_hist.append(fz_tot)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 200 == 0:
            n_avg = min(200, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_avg = sum(cl_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                f"Cd_tot={cd_tot_avg:.4f} Cl={cl_avg:.6f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0
    n_final = min(500, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final
    fz_final = sum(fz_hist[-n_final:]) / n_final

    err_pct = abs(cd_tot_final - Cd_ref) / Cd_ref * 100 if Cd_ref > 0 and math.isfinite(cd_tot_final) else float("nan")

    result = {
        "case": tag,
        "test": "sphere_re100_bb_fix",
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "normal_method": "from_sphere",
        "bb_fix": True,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "fz": fz_final,
        "Cd_ref": Cd_ref,
        "error_pct": err_pct,
        "previous_pre_bb_fix": {"Cd_total": 1.069, "error_pct": 2.0},
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"(ref={Cd_ref}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )
    print(
        f"{tag} Previous (pre-BB-fix): Cd_tot=1.069 err=2.0% → "
        f"BB-fix: Cd_tot={cd_tot_final:.4f} err={err_pct:.1f}%",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def run_suboff(device_id, output_path):
    """TEST 3: SUBOFF Re=1000 L=80 with BB fix."""
    from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig

    Re = 1000
    L = 80
    nx = 200
    ny = 80
    nz = 80
    u_in = 0.06
    nu = u_in * L / Re  # 0.0048
    tau = 3.0 * nu + 0.5  # 0.5144
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

    tag = f"[SDAA:{device_id} SUBOFF Re=1000 BB-fix]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(
        f"{tag} L={L} nx={nx} ny={ny} nz={nz} u_in={u_in} "
        f"nu={nu:.6f} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.6f} "
        f"Cf_ref={Cf_ref} n_steps={n_steps}",
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

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
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
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky LES)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming) — BB FIX: pass f_pre
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Drag computation
        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, fz_f = drag_friction_integration(f, mesh, dpS, nu)

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

    err_pct = abs(cd_tot_final - Cf_ref) / Cf_ref * 100 if Cf_ref > 0 and math.isfinite(cd_tot_final) else float("nan")

    result = {
        "case": tag,
        "test": "suboff_re1000_bb_fix",
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "L": L,
        "R_max": radius,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "normal_method": "from_suboff",
        "bb_fix": True,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "fz": fz_final,
        "Cf_ref": Cf_ref,
        "error_pct": err_pct,
        "previous_pre_bb_fix": {"Cd_total": 0.044, "error_pct": 5.6},
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"(ref Cf={Cf_ref}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )
    print(
        f"{tag} Previous (pre-BB-fix): Cd_tot=0.044 err=5.6% → "
        f"BB-fix: Cd_tot={cd_tot_final:.4f} err={err_pct:.1f}%",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def run_naca0012(device_id, output_path):
    """TEST 4: NACA 0012 Re=1000 with BB fix, 6L domain, from_naca normals."""
    Re = 1000
    chord = 100
    nx = 600  # 6L domain
    ny = 200  # 2L
    nz = 4
    u_in = 0.05
    nu = u_in * chord / Re  # 0.005
    tau = 3.0 * nu + 0.5    # 0.515
    cs_smag = 0.05
    n_steps = 10000
    ref_cd = 0.05

    x_le = int(nx * 0.25)  # 1.5 chords from inlet
    y_c = ny // 2           # centered
    dpS = 0.5 * u_in ** 2 * chord * nz

    tag = f"[SDAA:{device_id} NACA0012 Re=1000 BB-fix 6L]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(
        f"{tag} chord={chord} nx={nx} ny={ny} nz={nz} "
        f"u_in={u_in} nu={nu:.6f} tau={tau:.6f} Cs={cs_smag} "
        f"x_le={x_le} y_c={y_c} dpS={dpS:.6f} "
        f"Cd_ref={ref_cd} n_steps={n_steps}",
        flush=True,
    )

    t0 = time.time()
    solid = build_naca(chord, nx, ny, x_le, y_c, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    near = get_near_wall_2d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # NACA 0012: symmetric airfoil, m=0, p=0.40 (avoid p=0 div-by-zero), t=0.12
    mesh = SurfaceMesh.from_naca(solid, near, x_le, y_c, chord, m=0.0, p=0.40, t=0.12)

    # Normal statistics
    nx_n_vals = mesh.nx_n[near]
    ny_n_vals = mesh.ny_n[near]
    print(
        f"{tag} normal stats: "
        f"nx_n=[{float(nx_n_vals.min()):.3f}, {float(nx_n_vals.max()):.3f}] "
        f"ny_n=[{float(ny_n_vals.min()):.3f}, {float(ny_n_vals.max()):.3f}]",
        flush=True,
    )

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky LES)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming) — BB FIX: pass f_pre
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Drag computation
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)

        cd_p = fx_p
        cd_f = fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_avg = sum(cl_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                f"Cd_tot={cd_tot_avg:.4f} Cl={cl_avg:.6f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0
    n_final = min(1000, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final

    err_pct = abs(cd_tot_final - ref_cd) / ref_cd * 100 if ref_cd > 0 and math.isfinite(cd_tot_final) else float("nan")

    result = {
        "case": tag,
        "test": "naca0012_re1000_bb_fix",
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "chord": chord,
        "grid": f"{nx}x{ny}x{nz}",
        "domain_ratio": f"{nx/chord:.0f}L",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "x_le": x_le,
        "y_c": y_c,
        "normal_method": "from_naca (m=0, p=0.40, t=0.12)",
        "bb_fix": True,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "Cd_ref": ref_cd,
        "error_pct": err_pct,
        "previous_pre_bb_fix": {"Cd_total": 0.036, "error_pct": 28.3},
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"(ref={ref_cd:.4f}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )
    print(
        f"{tag} Previous (pre-BB-fix): Cd_tot=0.036 err=28.3% → "
        f"BB-fix: Cd_tot={cd_tot_final:.4f} err={err_pct:.1f}%",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage: PYTHONPATH=src python bb_fix_allgeo_worker.py <test_name> <device_id> <output_json>")
        print("  test_name: cylinder | sphere | suboff | naca0012")
        sys.exit(1)

    test_name = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if test_name == "cylinder":
        run_cylinder(device_id, output_path)
    elif test_name == "sphere":
        run_sphere(device_id, output_path)
    elif test_name == "suboff":
        run_suboff(device_id, output_path)
    elif test_name == "naca0012":
        run_naca0012(device_id, output_path)
    else:
        print(f"Unknown test: {test_name}")
        print("  Available: cylinder | sphere | suboff | naca0012")
        sys.exit(1)


if __name__ == "__main__":
    main()
