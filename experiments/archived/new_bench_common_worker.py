#!/usr/bin/env python3
"""New benchmarks via common interface — cube + delta wing + NACA high-Re + cylinder Re=40.

Uses ONLY common interface:
  - get_near_wall_3d, SurfaceMesh.from_gradient/from_cylinder/from_naca
  - drag_pressure_integration, drag_friction_integration
  - lbm_step_correct(), far_field_bc_3d, bounce_back_cells_3d(f_pre)
  - detect_strouhal(), wall_function_3d()

Benchmarks:
  1. cube_re40000  — Wall-mounted cube Re=40000      (SDAA:24)
  2. delta_wing    — Delta wing swept=70° Re=1000    (SDAA:25)
  3. naca0012_hr   — NACA 0012 Re=6e6                (SDAA:26)
  4. cyl_re40      — Cylinder Re=40 steady           (SDAA:27)

Usage:
  python new_bench_common_worker.py <benchmark> <device_id> <output_path>
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d, collide_mrt3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    get_near_wall_3d,
    get_near_wall_2d,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.postprocess import detect_strouhal


# ---------------------------------------------------------------------------
#  Geometry builders
# ---------------------------------------------------------------------------
def build_cube_mask(nx, ny, nz, cx, cy, cz, D, device):
    """Wall-mounted cube on bottom wall (y=0)."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Cube body
    in_cube = (
        (xx >= cx - D / 2)
        & (xx < cx + D / 2)
        & (yy >= 0)
        & (yy < D)
        & (zz >= cz - D / 2)
        & (zz < cz + D / 2)
    )
    # Bottom wall (2 layers for no-slip)
    wall = yy < 2
    return in_cube | wall


def build_delta_wing_mask(nx, ny, nz, x_le, cy, cz, chord, sweep_deg, thickness, device):
    """Delta wing with given sweep angle (thin flat plate)."""
    sweep_rad = math.radians(sweep_deg)
    half_span_max = chord / (2.0 * math.tan(sweep_rad))
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xc = (xx - x_le) / chord  # 0 at apex, 1 at trailing edge
    in_chord = (xc >= 0) & (xc <= 1)
    half_span = half_span_max * xc
    in_span = (zz - cz).abs() <= half_span
    in_thick = (yy - cy).abs() <= thickness / 2
    return in_chord & in_span & in_thick


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
        5.0
        * t
        * (
            0.2969 * torch.sqrt(xc)
            - 0.1260 * xc
            - 0.3516 * xc ** 2
            + 0.2843 * xc ** 3
            - 0.1015 * xc ** 4
        )
        * chord
    )
    in_chord = (xx >= x_le) & (xx <= x_le + chord)
    in_profile = (yy - y_c).abs() <= yt
    return in_chord & in_profile


def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Cylinder extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


# ---------------------------------------------------------------------------
#  Separation angle (cylinder)
# ---------------------------------------------------------------------------
def measure_separation_angle(ux, uy, near, cx, cy, R, mid_z):
    """Measure separation angle from rear stagnation point (upper half)."""
    near_2d = near[mid_z].cpu()
    ux_2d = ux[mid_z].cpu()
    uy_2d = uy[mid_z].cpu()
    ny, nx = near_2d.shape
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    dx = xx - cx
    dy = yy - cy
    phi = torch.atan2(dy, dx)
    phi_deg = torch.rad2deg(phi)
    upper = near_2d & (dy > 0)
    if upper.sum() < 4:
        return float("nan"), []
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    u_t = ux_2d * sin_phi - uy_2d * cos_phi
    idx = upper.nonzero(as_tuple=False).squeeze(1)
    phi_vals = phi_deg[idx[:, 0], idx[:, 1]].numpy()
    ut_vals = u_t[idx[:, 0], idx[:, 1]].numpy()
    order = np.argsort(-phi_vals)
    phi_sorted = phi_vals[order]
    ut_sorted = ut_vals[order]
    sep_angle = float("nan")
    for i in range(len(ut_sorted) - 1):
        if ut_sorted[i] > 0 and ut_sorted[i + 1] <= 0:
            t = ut_sorted[i] / (ut_sorted[i] - ut_sorted[i + 1])
            sep_angle = phi_sorted[i] + t * (phi_sorted[i + 1] - phi_sorted[i])
            break
    return sep_angle, list(zip(phi_sorted.tolist(), ut_sorted.tolist()))


# ---------------------------------------------------------------------------
#  BENCHMARK 1: Wall-mounted cube Re=40000
# ---------------------------------------------------------------------------
def run_cube_re40000(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[cube_re40000 SDAA:{device_id}]"

    D = 24.0
    nx, ny, nz = 256, 128, 128
    u_in = 0.08
    Re = 40000.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 10000
    Cs = 0.1

    print(f"{tag} D={D} nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f} MRT+Smag(Cs={Cs}) steps={n_steps}", flush=True)
    t0 = time.time()

    cx = nx * 0.3
    cz = nz * 0.5
    solid = build_cube_mask(nx, ny, nz, cx, 0, cz, D, device)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)
    A_frontal = D * D
    dpS = 0.5 * u_in ** 2 * A_frontal

    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"{tag} solid={n_solid} near={n_near} dpS={dpS:.6e}", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    # Far-field BC config: y+ top, z± far-field; y- is wall (solid)
    bc_config = {"far_field_faces": ["y+", "z-", "z+"], "periodic_faces": []}
    def far_field_fn(f_in, u):
        return far_field_bc_3d(f_in, u, bc_config=bc_config)

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
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p, cd_f = fx_p, fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

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
            cd_avg = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            cl_avg = sum(cl_hist[-n_avg:]) / max(n_avg, 1)
            el = time.time() - t0
            print(f"{tag} step={step}/{n_steps} Cd={cd_avg:.4f} Cl={cl_avg:.6f} "
                  f"({el:.0f}s, {el/step:.3f}s/step)", flush=True)

    elapsed = time.time() - t0
    n_final = min(max(n_steps // 10, 200), len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-n_final:]) / max(n_final, 1)
    cd_f_f = sum(cd_f_hist[-n_final:]) / max(n_final, 1)
    cd_tot_f = sum(cd_tot_hist[-n_final:]) / max(n_final, 1)
    cl_f = sum(cl_hist[-n_final:]) / max(n_final, 1)
    cl_rms = math.sqrt(sum((c - cl_f) ** 2 for c in cl_hist) / max(len(cl_hist), 1))
    st = detect_strouhal(cl_hist, 1.0, u_in, D) if len(cl_hist) > 100 else None

    cd_ref = 1.1
    cd_err = abs(cd_tot_f - cd_ref) / cd_ref * 100
    print(f"\n{tag} === FINAL ===")
    print(f"{tag} Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} Cd_tot={cd_tot_f:.4f} (ref={cd_ref}, err={cd_err:.1f}%)")
    print(f"{tag} Cl={cl_f:.6f} Cl_rms={cl_rms:.6f} St={st}")
    print(f"{tag} time={elapsed:.0f}s ({elapsed/n_steps:.3f}s/step)")

    result = {
        "case": "cube_re40000", "device": f"sdaa:{device_id}",
        "shape": "wall_mounted_cube", "lattice": "D3Q19",
        "collision": "MRT+Smag", "Cs": Cs,
        "boundary": "halfway_BB(f_pre)+farfield(y+top,z±)",
        "normal_method": "from_gradient",
        "grid": f"{nx}x{ny}x{nz}", "D": D, "u_in": u_in, "Re": Re,
        "nu": nu, "tau": tau, "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cd_ref": cd_ref, "Cd_err_pct": cd_err,
        "Cl": cl_f, "Cl_rms": cl_rms, "St": st,
        "n_samples": len(cd_tot_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
#  BENCHMARK 2: Delta wing swept=70° Re=1000
# ---------------------------------------------------------------------------
def run_delta_wing(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[delta_wing SDAA:{device_id}]"

    chord = 60.0
    sweep_deg = 70.0
    thickness = 3.0
    nx, ny, nz = 200, 32, 80
    u_in = 0.08
    Re = 1000.0
    nu = u_in * chord / Re
    tau = 3.0 * nu + 0.5
    n_steps = 10000
    Cs = 0.05

    half_span_max = chord / (2.0 * math.tan(math.radians(sweep_deg)))
    planform_area = 0.5 * chord * (2 * half_span_max)

    print(f"{tag} chord={chord} sweep={sweep_deg}° half_span={half_span_max:.1f} "
          f"nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f} MRT+Smag(Cs={Cs}) steps={n_steps}", flush=True)
    t0 = time.time()

    x_le = nx * 0.25
    cy = ny // 2
    cz = nz // 2
    solid = build_delta_wing_mask(nx, ny, nz, x_le, cy, cz, chord, sweep_deg, thickness, device)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)
    dpS = 0.5 * u_in ** 2 * planform_area

    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"{tag} solid={n_solid} near={n_near} planform={planform_area:.1f} dpS={dpS:.6e}", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    collide_fn = collide_smagorinsky_mrt3d
    collide_kwargs = {"C_s": Cs}

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid, u_in, far_field_bc_3d,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200, **collide_kwargs,
        )
        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap="quadratic", p0_method="far_field", solid=solid)
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p, cd_f = fx_p, fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

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
            cd_avg = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            cl_avg = sum(cl_hist[-n_avg:]) / max(n_avg, 1)
            el = time.time() - t0
            print(f"{tag} step={step}/{n_steps} Cd={cd_avg:.4f} Cl={cl_avg:.6f} "
                  f"({el:.0f}s, {el/step:.3f}s/step)", flush=True)

    elapsed = time.time() - t0
    n_final = min(max(n_steps // 10, 200), len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-n_final:]) / max(n_final, 1)
    cd_f_f = sum(cd_f_hist[-n_final:]) / max(n_final, 1)
    cd_tot_f = sum(cd_tot_hist[-n_final:]) / max(n_final, 1)
    cl_f = sum(cl_hist[-n_final:]) / max(n_final, 1)
    cl_rms = math.sqrt(sum((c - cl_f) ** 2 for c in cl_hist) / max(len(cl_hist), 1))
    st = detect_strouhal(cl_hist, 1.0, u_in, chord) if len(cl_hist) > 100 else None

    print(f"\n{tag} === FINAL ===")
    print(f"{tag} Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} Cd_tot={cd_tot_f:.4f}")
    print(f"{tag} Cl={cl_f:.6f} Cl_rms={cl_rms:.6f} St={st}")
    print(f"{tag} time={elapsed:.0f}s ({elapsed/n_steps:.3f}s/step)")

    result = {
        "case": "delta_wing", "device": f"sdaa:{device_id}",
        "shape": "delta_wing_70deg", "lattice": "D3Q19",
        "collision": "MRT+Smag", "Cs": Cs,
        "boundary": "halfway_BB(f_pre)+farfield",
        "normal_method": "from_gradient",
        "grid": f"{nx}x{ny}x{nz}", "chord": chord, "sweep_deg": sweep_deg,
        "half_span": half_span_max, "planform_area": planform_area,
        "u_in": u_in, "Re": Re, "nu": nu, "tau": tau,
        "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cl": cl_f, "Cl_rms": cl_rms, "St": st,
        "n_samples": len(cd_tot_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
#  BENCHMARK 3: NACA 0012 Re=6e6 (wall function)
# ---------------------------------------------------------------------------
def run_naca0012_hr(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[naca0012_hr SDAA:{device_id}]"

    chord = 1000.0
    nx, ny, nz = 1500, 400, 4
    u_in = 0.05
    tau = 0.50005
    nu = (tau - 0.5) / 3.0
    Re = u_in * chord / nu
    n_steps = 5000
    Cs = 0.01  # Small Smag for stability at extreme Re

    print(f"{tag} chord={chord} nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re:.3e} "
          f"nu={nu:.6e} tau={tau:.6f} wall_function_3d(log,y_val=1.0) steps={n_steps}", flush=True)
    t0 = time.time()

    x_le = nx * 0.2  # 300
    y_c = ny // 2     # 200
    solid = build_naca0012_mask(nx, ny, nz, x_le, y_c, chord, device)
    near = get_near_wall_3d(solid)
    # NACA 0012: m=0 (symmetric), p=0.5 (arbitrary since m=0), t=0.12
    mesh = SurfaceMesh.from_naca(solid, near, x_le, y_c, chord, m=0.0, p=0.5, t=0.12)
    dpS = 0.5 * u_in ** 2 * chord * nz

    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"{tag} solid={n_solid} near={n_near} dpS={dpS:.6e}", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    # Wall-function loop: collision → NoDynamics → stream → wall_function_3d → far_field → mass_corr
    sm = solid.unsqueeze(0).expand_as(f)
    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)
        # NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        # Stream (no bounce-back — wall function replaces it)
        f = stream3d(f)
        # Wall function (log-law, y_val=1.0)
        f, _, _ = wall_function_3d(f, solid, nu, y_val=1.0, wall_law="log", near_mask=near)
        # Far-field BC
        f = far_field_bc_3d(f, u_in)
        # Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Force measurement via common interface
        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap="quadratic", p0_method="far_field", solid=solid)
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p, cd_f = fx_p, fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

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
            cd_avg = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            cl_avg = sum(cl_hist[-n_avg:]) / max(n_avg, 1)
            el = time.time() - t0
            print(f"{tag} step={step}/{n_steps} Cd={cd_avg:.6f} Cl={cl_avg:.6f} "
                  f"({el:.0f}s, {el/step:.3f}s/step)", flush=True)

    elapsed = time.time() - t0
    n_final = min(max(n_steps // 10, 200), len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-n_final:]) / max(n_final, 1)
    cd_f_f = sum(cd_f_hist[-n_final:]) / max(n_final, 1)
    cd_tot_f = sum(cd_tot_hist[-n_final:]) / max(n_final, 1)
    cl_f = sum(cl_hist[-n_final:]) / max(n_final, 1)
    cl_rms = math.sqrt(sum((c - cl_f) ** 2 for c in cl_hist) / max(len(cl_hist), 1))
    st = detect_strouhal(cl_hist, 1.0, u_in, chord) if len(cl_hist) > 100 else None

    cd_ref = 0.008
    cd_err = abs(cd_tot_f - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    print(f"\n{tag} === FINAL ===")
    print(f"{tag} Cd_p={cd_p_f:.6f} Cd_f={cd_f_f:.6f} Cd_tot={cd_tot_f:.6f} (ref={cd_ref}, err={cd_err:.1f}%)")
    print(f"{tag} Cl={cl_f:.6f} Cl_rms={cl_rms:.6f} St={st}")
    print(f"{tag} time={elapsed:.0f}s ({elapsed/n_steps:.3f}s/step)")

    result = {
        "case": "naca0012_hr", "device": f"sdaa:{device_id}",
        "shape": "NACA0012", "lattice": "D3Q19",
        "collision": "MRT+Smag", "Cs": Cs,
        "boundary": "wall_function_3d(log,y_val=1.0)+farfield",
        "normal_method": "from_naca(m=0,p=0.5,t=0.12)",
        "grid": f"{nx}x{ny}x{nz}", "chord": chord, "u_in": u_in, "Re": Re,
        "nu": nu, "tau": tau, "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cd_ref": cd_ref, "Cd_err_pct": cd_err,
        "Cl": cl_f, "Cl_rms": cl_rms, "St": st,
        "n_samples": len(cd_tot_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
#  BENCHMARK 4: Cylinder Re=40 steady
# ---------------------------------------------------------------------------
def run_cyl_re40(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[cyl_re40 SDAA:{device_id}]"

    D = 48.0
    R = D / 2.0
    nx, ny, nz = 400, 160, 4
    u_in = 0.08
    Re = 40.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 10000

    print(f"{tag} D={D} nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f} MRT(no Smag) steps={n_steps}", flush=True)
    t0 = time.time()

    cx = nx * 0.25
    cy = ny * 0.5
    solid = build_cylinder_mask(nx, ny, nz, cx, cy, R, device)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    A_frontal = D * nz
    dpS = 0.5 * u_in ** 2 * A_frontal

    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"{tag} solid={n_solid} near={n_near} dpS={dpS:.6e}", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    collide_fn = collide_mrt3d
    collide_kwargs = {}

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid, u_in, far_field_bc_3d,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200, **collide_kwargs,
        )
        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap="quadratic", p0_method="far_field", solid=solid)
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p, cd_f = fx_p, fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

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
            cd_avg = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            cl_avg = sum(cl_hist[-n_avg:]) / max(n_avg, 1)
            el = time.time() - t0
            print(f"{tag} step={step}/{n_steps} Cd={cd_avg:.4f} Cl={cl_avg:.6f} "
                  f"({el:.0f}s, {el/step:.3f}s/step)", flush=True)

    elapsed = time.time() - t0
    n_final = min(max(n_steps // 10, 200), len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-n_final:]) / max(n_final, 1)
    cd_f_f = sum(cd_f_hist[-n_final:]) / max(n_final, 1)
    cd_tot_f = sum(cd_tot_hist[-n_final:]) / max(n_final, 1)
    cl_f = sum(cl_hist[-n_final:]) / max(n_final, 1)
    cl_rms = math.sqrt(sum((c - cl_f) ** 2 for c in cl_hist) / max(len(cl_hist), 1))
    st = detect_strouhal(cl_hist, 1.0, u_in, D) if len(cl_hist) > 100 else None

    # Separation angle
    sep_angle = float("nan")
    if len(cd_tot_hist) > 0:
        rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
        mid_z = nz // 2
        sep_angle, _ = measure_separation_angle(ux_f, uy_f, near, cx, cy, R, mid_z)

    cd_ref = 1.50
    cd_err = abs(cd_tot_f - cd_ref) / cd_ref * 100
    sep_ref = 53.0
    sep_err = abs(sep_angle - sep_ref) / sep_ref * 100 if math.isfinite(sep_angle) else float("nan")
    print(f"\n{tag} === FINAL ===")
    print(f"{tag} Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} Cd_tot={cd_tot_f:.4f} (ref={cd_ref}, err={cd_err:.1f}%)")
    print(f"{tag} Cl={cl_f:.6f} Cl_rms={cl_rms:.6f} St={st}")
    if not math.isnan(sep_angle):
        print(f"{tag} Sep_angle={sep_angle:.1f}° (ref={sep_ref}°, err={sep_err:.1f}%)")
    print(f"{tag} time={elapsed:.0f}s ({elapsed/n_steps:.3f}s/step)")

    result = {
        "case": "cyl_re40", "device": f"sdaa:{device_id}",
        "shape": "cylinder", "lattice": "D3Q19",
        "collision": "MRT", "Cs": 0.0,
        "boundary": "halfway_BB(f_pre)+farfield",
        "normal_method": "from_cylinder",
        "grid": f"{nx}x{ny}x{nz}", "D": D, "u_in": u_in, "Re": Re,
        "nu": nu, "tau": tau, "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cd_ref": cd_ref, "Cd_err_pct": cd_err,
        "Cl": cl_f, "Cl_rms": cl_rms, "St": st,
        "separation_angle_deg": sep_angle,
        "separation_angle_ref": sep_ref,
        "separation_angle_err_pct": sep_err,
        "n_samples": len(cd_tot_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print("Usage: python new_bench_common_worker.py <benchmark> <device_id> <output_path>")
        print("  benchmark: cube_re40000 | delta_wing | naca0012_hr | cyl_re40")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if benchmark == "cube_re40000":
        run_cube_re40000(device_id, output_path)
    elif benchmark == "delta_wing":
        run_delta_wing(device_id, output_path)
    elif benchmark == "naca0012_hr":
        run_naca0012_hr(device_id, output_path)
    elif benchmark == "cyl_re40":
        run_cyl_re40(device_id, output_path)
    else:
        print(f"Unknown benchmark: {benchmark}")
        sys.exit(1)


if __name__ == "__main__":
    main()
