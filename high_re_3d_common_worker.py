#!/usr/bin/env python3
"""3D high-Re benchmarks via the COMMON INTERFACE ONLY.

Pipeline (every benchmark):
  solid → get_near_wall_3d → SurfaceMesh.from_xxx → lbm_step_correct →
  drag_pressure_integration + drag_friction_integration → detect_strouhal

No custom force computation, no custom bounce-back — every step goes through
the verified common-interface modules.

BENCHMARK 1: Cylinder Re=3900 3D  (SDAA:24)
  - D=48, nx=200, ny=200, nz=200 (TRUE 3D!), u_in=0.08, Cs=0.15
  - 5000 steps, from_cylinder, far_field_bc
  - Reference: Cd=0.98
  - Key: Does 3D grid prevent divergence? (2D nz=4 diverged)

BENCHMARK 2: Cube Re=40000 3D  (SDAA:25)
  - D=24, nx=256, ny=128, nz=128, u_in=0.08, Cs=0.1
  - 10000 steps, from_gradient
  - Reference: Cd=1.1
  - Verify previous result (12.5%)

BENCHMARK 3: NACA 0012 Re=1000 3D  (SDAA:26)
  - chord=100, nx=400, ny=200, nz=4, u_in=0.05, Cs=0.05
  - 10000 steps, from_naca
  - Reference: Cd=0.05
  - Key: Does 3D help NACA?

BENCHMARK 4: Sphere Re=100 3D  (SDAA:27)
  - D=40, nx=180, ny=180, nz=180, u_in=0.08, Cs=0.05
  - 3000 steps, from_sphere
  - Reference: Cd=1.09
  - Verify previous result (3.4%)

Usage:
  python high_re_3d_common_worker.py <benchmark> <device_id> <output_path>
  benchmark: cyl_re3900_3d | cube_re40000_3d | naca_re1000_3d | sphere_re100_3d
"""
from __future__ import annotations

import functools
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

# ---- Common interface imports (ONLY these modules) ----
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.postprocess import detect_strouhal


# ========================================================================== #
# Geometry builders (mask construction — NOT force / BC computation)         #
# ========================================================================== #

def build_cylinder_solid(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along the z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def build_cube_mask(nx, ny, nz, cx, cz, D, device):
    """Wall-mounted cube: D×D×D cube on bottom wall (y=0)."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    in_cube = (
        (xx >= cx - D / 2) & (xx < cx + D / 2)
        & (yy >= 0) & (yy < D)
        & (zz >= cz - D / 2) & (zz < cz + D / 2)
    )
    # Bottom wall (2 layers for no-slip)
    wall = yy < 2
    return in_cube | wall


def build_cube_solid_only(nx, ny, nz, cx, cz, D, device):
    """Cube body only (for drag measurement, without bottom wall)."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (
        (xx >= cx - D / 2) & (xx < cx + D / 2)
        & (yy >= 0) & (yy < D)
        & (zz >= cz - D / 2) & (zz < cz + D / 2)
    )


def build_naca0012_mask(nx, ny, nz, x_le, y_c, chord, device):
    """NACA 0012 symmetric airfoil (2D extruded in z)."""
    t = 0.12
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xc = (xx - x_le) / chord
    xc = xc.clamp(min=1e-6, max=1.0)
    yt = (
        5.0 * t * (
            0.2969 * torch.sqrt(xc)
            - 0.1260 * xc
            - 0.3516 * xc ** 2
            + 0.2843 * xc ** 3
            - 0.1015 * xc ** 4
        ) * chord
    )
    in_chord = (xx >= x_le) & (xx <= x_le + chord)
    in_profile = (yy - y_c).abs() <= yt
    return in_chord & in_profile


def build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device):
    """Boolean solid mask for a sphere."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2


# ========================================================================== #
# BENCHMARK 1: Cylinder Re=3900 3D  (TRUE 3D grid, nz=200)                   #
# ========================================================================== #

def run_cyl_re3900_3d(device_id, output_path):
    """Cylinder Re=3900 on a TRUE 3D grid (nz=200).

    Key question: does a 3D grid prevent the divergence seen in 2D (nz=4)?
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[cyl_re3900_3d SDAA:{device_id}]"

    D = 48.0
    R = D / 2.0
    nx, ny, nz = 200, 200, 200
    u_in = 0.08
    Re = 3900.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 5000
    Cs = 0.15
    avg_window = 1500

    cx = nx * 0.25
    cy = ny * 0.5
    # Cylinder extruded along z → frontal area = D × nz
    A_frontal = D * nz
    dpS = 0.5 * u_in ** 2 * A_frontal

    print(f"{tag} D={D} nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f} MRT+Smag(Cs={Cs}) steps={n_steps}", flush=True)
    print(f"{tag} TRUE 3D grid (nz={nz}) — testing if 3D prevents divergence",
          flush=True)
    t0 = time.time()

    solid = build_cylinder_solid(nx, ny, nz, cx, cy, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells={n_solid}", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    print(f"{tag} SurfaceMesh.from_cylinder built", flush=True)

    # Far-field BC: y± free-stream, z± periodic (cylinder spans full z)
    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0),
                      torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    collide_fn = collide_smagorinsky_mrt3d
    collide_kwargs = {"C_s": Cs}

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid, u_in, far_field_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200, **collide_kwargs,
        )

        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap="none", p0_method="far_field", solid=solid)
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula="standard")
        cd_p, cd_f = float(fx_p), float(fx_f)
        cd_tot = cd_p + cd_f
        cl = float(fy_p + fy_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step > warmup and math.isfinite(cd_tot):
            cd_p_hist.append(cd_p)
            cd_f_hist.append(cd_f)
            cd_tot_hist.append(cd_tot)
            cl_hist.append(cl)
        if step % 500 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            ap = sum(cd_p_hist[-n_avg:]) / max(n_avg, 1)
            af = sum(cd_f_hist[-n_avg:]) / max(n_avg, 1)
            at = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            el = time.time() - t0
            print(f"{tag} step={step}/{n_steps} Cd_p={ap:.4f} Cd_f={af:.4f} "
                  f"Cd_tot={at:.4f} Cl={cl:.6f} ({el:.0f}s, {el/step:.3f}s/step)",
                  flush=True)

    elapsed = time.time() - t0
    nf = min(avg_window, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-nf:]) / max(nf, 1)
    cd_f_f = sum(cd_f_hist[-nf:]) / max(nf, 1)
    cd_tot_f = sum(cd_tot_hist[-nf:]) / max(nf, 1)
    cl_f = sum(cl_hist[-nf:]) / max(nf, 1)
    st = detect_strouhal(cl_hist, 1.0, u_in, D, min_cycles=3) if len(cl_hist) > 100 else None

    cd_ref = 0.98
    cd_err = abs(cd_tot_f - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    finite = bool(torch.isfinite(f).all().item())
    print(f"\n{tag} === FINAL ===  Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} "
          f"Cd_tot={cd_tot_f:.4f} (ref={cd_ref}) err={cd_err:.1f}% Cl={cl_f:.6f} "
          f"St={st} finite={finite} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "cyl_re3900_3d", "device": f"sdaa:{device_id}",
        "shape": "cylinder_3d", "lattice": "D3Q19",
        "collision": "MRT+Smag", "Cs": Cs,
        "boundary": "halfway_BB(f_pre)+farfield(y±)+periodic(z±)",
        "normal_method": "from_cylinder",
        "grid": f"{nx}x{ny}x{nz}", "D": D, "u_in": u_in, "Re": Re,
        "nu": nu, "tau": tau, "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cd_ref": cd_ref, "Cd_err_pct": cd_err,
        "Cl": cl_f, "St": st,
        "n_samples": len(cd_tot_hist),
        "finite": finite,
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
        "key_question": "Does 3D grid (nz=200) prevent divergence seen in 2D (nz=4)?",
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# BENCHMARK 2: Cube Re=40000 3D  (verify previous 12.5% result)             #
# ========================================================================== #

def run_cube_re40000_3d(device_id, output_path):
    """Wall-mounted cube Re=40000 on 3D grid."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[cube_re40000_3d SDAA:{device_id}]"

    D = 24.0
    nx, ny, nz = 256, 128, 128
    u_in = 0.08
    Re = 40000.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 10000
    Cs = 0.1
    avg_window = 2000

    cx = nx * 0.3
    cz = nz * 0.5
    A_frontal = D * D
    dpS = 0.5 * u_in ** 2 * A_frontal

    print(f"{tag} D={D} nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f} MRT+Smag(Cs={Cs}) steps={n_steps}", flush=True)
    t0 = time.time()

    solid_total = build_cube_mask(nx, ny, nz, cx, cz, D, device)  # cube + wall
    solid_cube = build_cube_solid_only(nx, ny, nz, cx, cz, D, device)  # cube only
    n_solid = int(solid_total.sum().item())
    print(f"{tag} solid_total={n_solid}", flush=True)

    near = get_near_wall_3d(solid_cube)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall (cube)={n_near}", flush=True)

    mesh = SurfaceMesh.from_gradient(solid_cube, near)
    print(f"{tag} SurfaceMesh.from_gradient built", flush=True)

    # Far-field BC: y+ top, z± far-field; y- is wall (solid)
    bc_config = {"far_field_faces": ["y+", "z-", "z+"], "periodic_faces": []}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid_total] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0),
                      torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    collide_fn = collide_smagorinsky_mrt3d
    collide_kwargs = {"C_s": Cs}

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid_total, u_in, far_field_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200, **collide_kwargs,
        )

        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap="quadratic", p0_method="far_field",
            solid=solid_cube)
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula="standard")
        cd_p, cd_f = float(fx_p), float(fx_f)
        cd_tot = cd_p + cd_f
        cl = float(fy_p + fy_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step > warmup and math.isfinite(cd_tot):
            cd_p_hist.append(cd_p)
            cd_f_hist.append(cd_f)
            cd_tot_hist.append(cd_tot)
            cl_hist.append(cl)
        if step % 1000 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            ap = sum(cd_p_hist[-n_avg:]) / max(n_avg, 1)
            af = sum(cd_f_hist[-n_avg:]) / max(n_avg, 1)
            at = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            el = time.time() - t0
            print(f"{tag} step={step}/{n_steps} Cd_p={ap:.4f} Cd_f={af:.4f} "
                  f"Cd_tot={at:.4f} Cl={cl:.6f} ({el:.0f}s, {el/step:.3f}s/step)",
                  flush=True)

    elapsed = time.time() - t0
    nf = min(avg_window, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-nf:]) / max(nf, 1)
    cd_f_f = sum(cd_f_hist[-nf:]) / max(nf, 1)
    cd_tot_f = sum(cd_tot_hist[-nf:]) / max(nf, 1)
    cl_f = sum(cl_hist[-nf:]) / max(nf, 1)
    st = detect_strouhal(cl_hist, 1.0, u_in, D, min_cycles=3) if len(cl_hist) > 100 else None

    cd_ref = 1.1
    cd_err = abs(cd_tot_f - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    finite = bool(torch.isfinite(f).all().item())
    print(f"\n{tag} === FINAL ===  Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} "
          f"Cd_tot={cd_tot_f:.4f} (ref={cd_ref}) err={cd_err:.1f}% Cl={cl_f:.6f} "
          f"St={st} finite={finite} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "cube_re40000_3d", "device": f"sdaa:{device_id}",
        "shape": "wall_mounted_cube", "lattice": "D3Q19",
        "collision": "MRT+Smag", "Cs": Cs,
        "boundary": "halfway_BB(f_pre)+farfield(y+top,z±)",
        "normal_method": "from_gradient",
        "grid": f"{nx}x{ny}x{nz}", "D": D, "u_in": u_in, "Re": Re,
        "nu": nu, "tau": tau, "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cd_ref": cd_ref, "Cd_err_pct": cd_err,
        "Cl": cl_f, "St": st,
        "n_samples": len(cd_tot_hist),
        "finite": finite,
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
        "previous_result": "Cd_tot=0.9628, err=12.47%",
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# BENCHMARK 3: NACA 0012 Re=1000 3D  (moderate Re, bounce-back)              #
# ========================================================================== #

def run_naca_re1000_3d(device_id, output_path):
    """NACA 0012 Re=1000 — moderate Re, uses bounce-back (not wall function).

    Key question: does 3D (nz=4 quasi-2D) help NACA at Re=1000?
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[naca_re1000_3d SDAA:{device_id}]"

    chord = 100.0
    nx, ny, nz = 400, 200, 4
    u_in = 0.05
    Re = 1000.0
    nu = u_in * chord / Re
    tau = 3.0 * nu + 0.5
    n_steps = 10000
    Cs = 0.05
    avg_window = 2000

    x_le = nx * 0.2  # 80
    y_c = ny * 0.5   # 100
    A_frontal = chord * nz  # 2D extruded
    dpS = 0.5 * u_in ** 2 * A_frontal

    print(f"{tag} chord={chord} nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f} MRT+Smag(Cs={Cs}) steps={n_steps}", flush=True)
    t0 = time.time()

    solid = build_naca0012_mask(nx, ny, nz, x_le, y_c, chord, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells={n_solid}", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # NACA 0012: m=0 (symmetric), p=0.4, t=0.12
    mesh = SurfaceMesh.from_naca(solid, near, x_le, y_c, chord, m=0.0, p=0.4, t=0.12)
    print(f"{tag} SurfaceMesh.from_naca built", flush=True)

    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0),
                      torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    collide_fn = collide_smagorinsky_mrt3d
    collide_kwargs = {"C_s": Cs}

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid, u_in, far_field_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200, **collide_kwargs,
        )

        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap="quadratic", p0_method="far_field", solid=solid)
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula="standard")
        cd_p, cd_f = float(fx_p), float(fx_f)
        cd_tot = cd_p + cd_f
        cl = float(fy_p + fy_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step > warmup and math.isfinite(cd_tot):
            cd_p_hist.append(cd_p)
            cd_f_hist.append(cd_f)
            cd_tot_hist.append(cd_tot)
            cl_hist.append(cl)
        if step % 1000 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            ap = sum(cd_p_hist[-n_avg:]) / max(n_avg, 1)
            af = sum(cd_f_hist[-n_avg:]) / max(n_avg, 1)
            at = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            al = sum(cl_hist[-n_avg:]) / max(n_avg, 1)
            el = time.time() - t0
            print(f"{tag} step={step}/{n_steps} Cd_p={ap:.6f} Cd_f={af:.6f} "
                  f"Cd_tot={at:.6f} Cl={al:.6f} ({el:.0f}s, {el/step:.3f}s/step)",
                  flush=True)

    elapsed = time.time() - t0
    nf = min(avg_window, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-nf:]) / max(nf, 1)
    cd_f_f = sum(cd_f_hist[-nf:]) / max(nf, 1)
    cd_tot_f = sum(cd_tot_hist[-nf:]) / max(nf, 1)
    cl_f = sum(cl_hist[-nf:]) / max(nf, 1)
    st = detect_strouhal(cl_hist, 1.0, u_in, chord, min_cycles=3) if len(cl_hist) > 100 else None

    cd_ref = 0.05
    cd_err = abs(cd_tot_f - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    finite = bool(torch.isfinite(f).all().item())
    print(f"\n{tag} === FINAL ===  Cd_p={cd_p_f:.6f} Cd_f={cd_f_f:.6f} "
          f"Cd_tot={cd_tot_f:.6f} (ref={cd_ref}) err={cd_err:.1f}% Cl={cl_f:.6f} "
          f"St={st} finite={finite} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "naca_re1000_3d", "device": f"sdaa:{device_id}",
        "shape": "NACA0012", "lattice": "D3Q19",
        "collision": "MRT+Smag", "Cs": Cs,
        "boundary": "halfway_BB(f_pre)+farfield(y±)+periodic(z±)",
        "normal_method": "from_naca(m=0,p=0.4,t=0.12)",
        "grid": f"{nx}x{ny}x{nz}", "chord": chord, "u_in": u_in, "Re": Re,
        "nu": nu, "tau": tau, "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cd_ref": cd_ref, "Cd_err_pct": cd_err,
        "Cl": cl_f, "St": st,
        "n_samples": len(cd_tot_hist),
        "finite": finite,
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
        "key_question": "Does 3D help NACA at Re=1000?",
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# BENCHMARK 4: Sphere Re=100 3D  (verify previous 3.4% result)               #
# ========================================================================== #

def run_sphere_re100_3d(device_id, output_path):
    """Sphere Re=100 on 3D grid — verify previous result."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[sphere_re100_3d SDAA:{device_id}]"

    D = 40.0
    R = D / 2.0
    nx, ny, nz = 180, 180, 180
    u_in = 0.08
    Re = 100.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 3000
    Cs = 0.05
    avg_window = 800

    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    A_frontal = math.pi * R ** 2
    dpS = 0.5 * u_in ** 2 * A_frontal

    print(f"{tag} D={D} nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f} MRT+Smag(Cs={Cs}) steps={n_steps}", flush=True)
    t0 = time.time()

    solid = build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells={n_solid}", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)
    print(f"{tag} SurfaceMesh.from_sphere built", flush=True)

    # All far-field faces (sphere is finite in all directions)
    bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0),
                      torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    collide_fn = collide_smagorinsky_mrt3d
    collide_kwargs = {"C_s": Cs}

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid, u_in, far_field_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200, **collide_kwargs,
        )

        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap="quadratic", p0_method="far_field", solid=solid)
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula="standard")
        cd_p, cd_f = float(fx_p), float(fx_f)
        cd_tot = cd_p + cd_f
        cl = float(fy_p + fy_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step > warmup and math.isfinite(cd_tot):
            cd_p_hist.append(cd_p)
            cd_f_hist.append(cd_f)
            cd_tot_hist.append(cd_tot)
            cl_hist.append(cl)
        if step % 200 == 0 or step == n_steps:
            n_avg = min(200, len(cd_tot_hist))
            ap = sum(cd_p_hist[-n_avg:]) / max(n_avg, 1)
            af = sum(cd_f_hist[-n_avg:]) / max(n_avg, 1)
            at = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            el = time.time() - t0
            print(f"{tag} step={step}/{n_steps} Cd_p={ap:.4f} Cd_f={af:.4f} "
                  f"Cd_tot={at:.4f} Cl={cl:.6f} ({el:.0f}s, {el/step:.3f}s/step)",
                  flush=True)

    elapsed = time.time() - t0
    nf = min(avg_window, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-nf:]) / max(nf, 1)
    cd_f_f = sum(cd_f_hist[-nf:]) / max(nf, 1)
    cd_tot_f = sum(cd_tot_hist[-nf:]) / max(nf, 1)
    cl_f = sum(cl_hist[-nf:]) / max(nf, 1)
    st = detect_strouhal(cl_hist, 1.0, u_in, D, min_cycles=3) if len(cl_hist) > 100 else None

    cd_ref = 1.09
    cd_err = abs(cd_tot_f - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    finite = bool(torch.isfinite(f).all().item())
    print(f"\n{tag} === FINAL ===  Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} "
          f"Cd_tot={cd_tot_f:.4f} (ref={cd_ref}) err={cd_err:.1f}% Cl={cl_f:.6f} "
          f"St={st} finite={finite} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "sphere_re100_3d", "device": f"sdaa:{device_id}",
        "shape": "sphere", "lattice": "D3Q19",
        "collision": "MRT+Smag", "Cs": Cs,
        "boundary": "halfway_BB(f_pre)+farfield(y±,z±)",
        "normal_method": "from_sphere",
        "grid": f"{nx}x{ny}x{nz}", "D": D, "u_in": u_in, "Re": Re,
        "nu": nu, "tau": tau, "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cd_ref": cd_ref, "Cd_err_pct": cd_err,
        "Cl": cl_f, "St": st,
        "n_samples": len(cd_tot_hist),
        "finite": finite,
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
        "previous_result": "Cd_tot=1.0793, err=0.98% (extrap=quadratic)",
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# Main                                                                        #
# ========================================================================== #

def main():
    if len(sys.argv) < 4:
        print("Usage: python high_re_3d_common_worker.py "
              "<benchmark> <device_id> <output_path>")
        print("  benchmark: cyl_re3900_3d | cube_re40000_3d | "
              "naca_re1000_3d | sphere_re100_3d")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    runners = {
        "cyl_re3900_3d": run_cyl_re3900_3d,
        "cube_re40000_3d": run_cube_re40000_3d,
        "naca_re1000_3d": run_naca_re1000_3d,
        "sphere_re100_3d": run_sphere_re100_3d,
    }
    if benchmark not in runners:
        print(f"Unknown benchmark: {benchmark}")
        print(f"Available: {', '.join(runners.keys())}")
        sys.exit(1)

    runners[benchmark](device_id, output_path)


if __name__ == "__main__":
    main()
