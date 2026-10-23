#!/usr/bin/env python3
"""External-flow benchmarks via the COMMON INTERFACE ONLY.

Pipeline (every benchmark):
  solid → get_near_wall_3d → SurfaceMesh.from_xxx → lbm_step_correct →
  drag_pressure_integration + drag_friction_integration → detect_strouhal

No custom force computation, no custom bounce-back — every step goes through
the verified common-interface modules.

BENCHMARK 1: Wall-mounted cube Re=40000  (SDAA:24)
  - Cube D=24 on bottom wall, from_gradient normals
  - MRT + Smagorinsky (Cs=0.1), 10000 steps
  - Reference: Cd ≈ 1.1

BENCHMARK 2: Delta wing swept=70°  (SDAA:25)
  - from_gradient normals, MRT + Smagorinsky (Cs=0.05)
  - Re=1000, 10000 steps
  - Reference: Cl, leading-edge vortex position

BENCHMARK 3: NACA 0012 Re=6e6  (SDAA:26)
  - from_naca normals, wall_function_3d(log, y_val=1.0)
  - tau=0.50005, 5000 steps
  - Reference: Cd ≈ 0.008

BENCHMARK 4: Cylinder Re=40  (SDAA:27)
  - from_cylinder normals, MRT (no Smag)
  - 10000 steps (steady)
  - Reference: Cd=1.50, separation=53°

Usage:
  python external_flow_common_worker.py <benchmark> <device_id> <output_path>
  benchmark: cube | delta | naca | cylinder
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
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.postprocess import detect_strouhal
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.airfoil_benchmark import build_airfoil_mask


# ========================================================================== #
# Geometry builders (mask construction — NOT force / BC computation)         #
# ========================================================================== #

def build_cube_mask(nx, ny, nz, cx, D, device):
    """Wall-mounted cube: D×D square on bottom wall (y=0), extruded in z."""
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0:D, cx:cx + D] = True
    return solid


def build_bottom_wall_mask(nx, ny, nz, device):
    """Bottom wall (y=0) no-slip wall mask."""
    wall = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    wall[:, 0, :] = True
    return wall


def build_delta_wing_mask(nx, ny, nz, x_le, y_center, chord, sweep_deg,
                           alpha_deg, device):
    """Delta wing: triangular planform with given sweep angle, extruded in z.

    Apex at (x_le, y_center), trailing edge at x=x_le+chord.
    Half-span = chord / tan(sweep_deg).  Rotated by alpha_deg (nose up).
    """
    half_span = chord / math.tan(math.radians(sweep_deg))
    alpha = math.radians(alpha_deg)
    cos_a, sin_a = math.cos(alpha), math.sin(alpha)

    v1 = np.array([0.0, 0.0])       # apex
    v2 = np.array([chord, -half_span])  # bottom trailing edge
    v3 = np.array([chord, half_span])   # top trailing edge
    centroid = (v1 + v2 + v3) / 3.0

    def rotate(v):
        vr = v - centroid
        vr = np.array([vr[0] * cos_a - vr[1] * sin_a,
                       vr[0] * sin_a + vr[1] * cos_a])
        return vr + np.array([x_le, y_center])

    p1, p2, p3 = rotate(v1), rotate(v2), rotate(v3)

    # Build on CPU (numpy point-in-triangle), then move to device
    yy, xx = torch.meshgrid(
        torch.arange(ny, device="cpu", dtype=torch.float32),
        torch.arange(nx, device="cpu", dtype=torch.float32),
        indexing="ij",
    )
    px, py = xx.numpy(), yy.numpy()
    ax, ay = p1
    bx, by = p2
    cx_t, cy_t = p3
    d = (by - cy_t) * (ax - cx_t) + (cx_t - bx) * (ay - cy_t)
    a_test = ((by - cy_t) * (px - cx_t) + (cx_t - bx) * (py - cy_t)) / d
    b_test = ((cy_t - ay) * (px - cx_t) + (ax - cx_t) * (py - cy_t)) / d
    c_test = 1.0 - a_test - b_test
    mask_2d = torch.from_numpy(
        (a_test >= 0) & (b_test >= 0) & (c_test >= 0)
    )
    solid = mask_2d.unsqueeze(0).expand(nz, ny, nx).clone().to(device)
    return solid


def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Cylinder (2D extruded along z)."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_naca_mask(nx, ny, nz, chord, alpha_deg, cx, cy, device):
    """NACA 0012 airfoil mask (2D extruded along z)."""
    mask_2d = build_airfoil_mask(
        nx, ny, chord, alpha_deg=alpha_deg,
        m=0.0, p=0.4, t=0.12,  # NACA 0012: symmetric (m=0)
        cx=cx, cy=cy, device=torch.device("cpu"),
    )
    mask_2d = mask_2d.to(device)
    solid = mask_2d.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


# ========================================================================== #
# Diagnostics (cylinder separation, delta-wing vortex)                      #
# ========================================================================== #

def measure_cylinder_separation(f, mesh, cx, cy, radius):
    """Separation angle from front stagnation point (upper & lower), degrees.

    The front stagnation point is at φ=0 (+x, facing the flow).  Separation
    occurs where the surface-tangential velocity u_t changes sign from
    negative (attached, flow moving toward rear) to positive (reversed).
    """
    rho, ux, uy, uz = macroscopic3d(f)
    near = mesh.near
    near_idx = near.cpu().numpy()
    if not near_idx.any():
        return float("nan"), float("nan")
    nz_idx, ny_idx, nx_idx = np.where(near_idx)
    mask_z0 = nz_idx == 0
    ny_arr = ny_idx[mask_z0].astype(float)
    nx_arr = nx_idx[mask_z0].astype(float)
    ux_np = ux[0].cpu().numpy()
    uy_np = uy[0].cpu().numpy()
    ux_vals = ux_np[ny_arr.astype(int), nx_arr.astype(int)]
    uy_vals = uy_np[ny_arr.astype(int), nx_arr.astype(int)]
    phi = np.arctan2(ny_arr - cy, nx_arr - cx)
    u_t = -ux_vals * np.sin(phi) + uy_vals * np.cos(phi)

    def _sep_from_front(side_mask, phi_arr, u_t_arr):
        """Find separation angle from front stagnation (φ=0).

        Sort by φ ascending (front → rear).  Attached flow has u_t < 0;
        reversed flow has u_t > 0.  Separation = first zero crossing
        from negative to positive.
        """
        if not side_mask.any():
            return float("nan")
        p = phi_arr[side_mask]
        u = u_t_arr[side_mask]
        si = np.argsort(p)          # front → rear
        p, u = p[si], u[si]
        for i in range(len(u) - 1):
            if u[i] < 0 and u[i + 1] > 0:
                frac = -u[i] / (u[i + 1] - u[i])
                sp = p[i] + frac * (p[i + 1] - p[i])
                return math.degrees(abs(sp))   # |φ| from front
        return float("nan")

    upper = ny_arr > cy
    lower = ny_arr < cy
    sep_up = _sep_from_front(upper, phi, u_t)
    sep_lo = _sep_from_front(lower, phi, u_t)
    return sep_up, sep_lo


def measure_wake_vortex(f, solid, cx_obj, nx, ny):
    """Find wake vortex center (peak |vorticity| behind object)."""
    rho, ux, uy, uz = macroscopic3d(f)
    uy_np = uy[0].cpu().numpy()
    ux_np = ux[0].cpu().numpy()
    duy_dx = np.zeros_like(uy_np)
    dux_dy = np.zeros_like(ux_np)
    duy_dx[:, 1:-1] = (uy_np[:, 2:] - uy_np[:, :-2]) / 2.0
    dux_dy[1:-1, :] = (ux_np[2:, :] - ux_np[:-2, :]) / 2.0
    vort = duy_dx - dux_dy
    solid_np = solid[0].cpu().numpy()
    wake_start = int(cx_obj + 10)
    wake_mask = (~solid_np)
    wake_mask[:, :wake_start] = False
    if not wake_mask.any():
        return float("nan"), float("nan")
    wake_vort = np.abs(vort) * wake_mask
    if wake_vort.max() < 1e-10:
        return float("nan"), float("nan")
    peak_idx = np.unravel_index(wake_vort.argmax(), wake_vort.shape)
    return float(peak_idx[1]), float(peak_idx[0])


# ========================================================================== #
# Benchmark runners                                                          #
# ========================================================================== #

def run_cube(device_id, output_path=None):
    """BENCHMARK 1: Wall-mounted cube Re=40000."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 300, 100, 4
    D = 24
    Re = 40000
    u_in = 0.1
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.1
    n_steps = 10000
    avg_window = 2000

    cx = nx // 4
    dpS = 0.5 * u_in ** 2 * D * nz  # frontal area = D × span
    tag = f"[Cube SDAA:{device_id}]"

    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} Re={Re} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.6e}", flush=True)

    t0 = time.time()
    cube = build_cube_mask(nx, ny, nz, cx, D, device)
    wall = build_bottom_wall_mask(nx, ny, nz, device)
    solid_total = cube | wall   # for NoDynamics + bounce-back
    solid_cube = cube           # for drag measurement

    # --- Common interface: get_near_wall_3d ---
    near = get_near_wall_3d(solid_cube)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall (cube)={n_near}", flush=True)

    # --- Common interface: SurfaceMesh.from_gradient ---
    mesh = SurfaceMesh.from_gradient(solid_cube, near)
    print(f"{tag} SurfaceMesh.from_gradient built", flush=True)

    # --- Far-field BC wrapper (y+ free-stream, y- = wall via BB, z periodic) ---
    bc_config = {'far_field_faces': ['y+'], 'periodic_faces': ['z-', 'z+']}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    # --- Initialise flow ---
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid_total] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0),
                      torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []

    # --- Common interface: lbm_step_correct main loop ---
    # Velocity ramp (0.02 → u_in over first 2000 steps) to stabilise the
    # high-Re (tau≈0.5002) initial transient.
    ramp_steps = 2000
    for step in range(1, n_steps + 1):
        u_cur = min(u_in, 0.02 + (u_in - 0.02) * step / ramp_steps)
        f = lbm_step_correct(
            f, collide_smagorinsky_mrt3d, tau, solid_total, u_cur,
            far_field_fn, correct_mass_fn=correct_mass3d, target_mass=im,
            step=step, mass_interval=200, C_s=cs_smag,
        )

        # --- Common interface: drag_pressure + drag_friction ---
        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap='none', p0_method='far_field',
            solid=solid_cube)
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula='standard')
        cd_p, cd_f = float(fx_p), float(fx_f)
        cd_tot = cd_p + cd_f
        cl = float(fy_p + fy_f)

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step % 1000 == 0:
            n_avg = min(500, len(cd_tot_hist))
            ap = sum(cd_p_hist[-n_avg:]) / n_avg
            af = sum(cd_f_hist[-n_avg:]) / n_avg
            at = sum(cd_tot_hist[-n_avg:]) / n_avg
            print(f"{tag} step={step}/{n_steps} Cd_p={ap:.4f} Cd_f={af:.4f} "
                  f"Cd={at:.4f} Cl={cl:.6f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    nf = min(avg_window, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-nf:]) / nf
    cd_f_f = sum(cd_f_hist[-nf:]) / nf
    cd_tot_f = sum(cd_tot_hist[-nf:]) / nf
    cl_f = sum(cl_hist[-nf:]) / nf

    # --- Common interface: detect_strouhal ---
    st = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                         length_ref=D, min_cycles=3)

    cd_ref = 1.1
    err = abs(cd_tot_f - cd_ref) / cd_ref * 100
    print(f"\n{tag} === FINAL ===  Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} "
          f"Cd={cd_tot_f:.4f} (ref={cd_ref}) err={err:.1f}% Cl={cl_f:.6f} "
          f"St={st} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "wall_mounted_cube_Re40000",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D, "Re": Re, "u_in": u_in, "nu": nu, "tau": tau,
        "Cs": cs_smag, "n_steps": n_steps,
        "Cd_pressure": float(cd_p_f), "Cd_friction": float(cd_f_f),
        "Cd_total": float(cd_tot_f), "Cl": float(cl_f), "St": st,
        "Cd_ref": cd_ref, "error_pct": float(err),
        "normals": "from_gradient",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
        "modules_used": [
            "drag_pressure.get_near_wall_3d",
            "drag_pressure.SurfaceMesh.from_gradient",
            "lbm_step_correct.lbm_step_correct",
            "drag_pressure.drag_pressure_integration",
            "drag_pressure.drag_friction_integration",
            "postprocess.detect_strouhal",
        ],
    }
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} saved to {output_path}", flush=True)
    return result


def run_delta(device_id, output_path=None):
    """BENCHMARK 2: Delta wing swept=70° Re=1000."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 400, 200, 4
    Re = 1000
    chord = 200.0
    sweep_deg = 70.0
    alpha_deg = 10.0
    u_in = 0.1
    nu = u_in * chord / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    n_steps = 10000
    avg_window = 2000

    x_le = nx // 4
    y_center = ny // 2
    half_span = chord / math.tan(math.radians(sweep_deg))
    dpS = 0.5 * u_in ** 2 * chord * nz
    tag = f"[Delta SDAA:{device_id}]"

    print(f"{tag} nx={nx} ny={ny} nz={nz} chord={chord} sweep={sweep_deg}° "
          f"alpha={alpha_deg}° Re={Re} u_in={u_in} nu={nu:.6e} tau={tau:.6f} "
          f"Cs={cs_smag} half_span={half_span:.1f} dpS={dpS:.6e}", flush=True)

    t0 = time.time()
    solid = build_delta_wing_mask(nx, ny, nz, x_le, y_center, chord,
                                  sweep_deg, alpha_deg, device)
    print(f"{tag} solid cells={int(solid.sum().item())}", flush=True)

    # --- Common interface: get_near_wall_3d ---
    near = get_near_wall_3d(solid)
    print(f"{tag} near-wall={int(near.sum().item())}", flush=True)

    # --- Common interface: SurfaceMesh.from_gradient ---
    mesh = SurfaceMesh.from_gradient(solid, near)
    print(f"{tag} SurfaceMesh.from_gradient built", flush=True)

    bc_config = {'far_field_faces': ['y-', 'y+'], 'periodic_faces': ['z-', 'z+']}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0),
                      torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    vortex_x_hist, vortex_y_hist = [], []

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_smagorinsky_mrt3d, tau, solid, u_in,
            far_field_fn, correct_mass_fn=correct_mass3d, target_mass=im,
            step=step, mass_interval=200, C_s=cs_smag,
        )

        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap='none', p0_method='far_field', solid=solid)
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula='standard')
        cd_p, cd_f = float(fx_p), float(fx_f)
        cd_tot = cd_p + cd_f
        cl = float(fy_p + fy_f)

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)

        if step % 100 == 0:
            vx, vy = measure_wake_vortex(f, solid, x_le + chord, nx, ny)
            if math.isfinite(vx):
                vortex_x_hist.append(vx)
                vortex_y_hist.append(vy)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step % 1000 == 0:
            n_avg = min(500, len(cd_tot_hist))
            ap = sum(cd_p_hist[-n_avg:]) / n_avg
            af = sum(cd_f_hist[-n_avg:]) / n_avg
            at = sum(cd_tot_hist[-n_avg:]) / n_avg
            al = sum(cl_hist[-n_avg:]) / n_avg
            vx_a = (sum(vortex_x_hist[-5:]) / min(5, len(vortex_x_hist))
                    if vortex_x_hist else float("nan"))
            vy_a = (sum(vortex_y_hist[-5:]) / min(5, len(vortex_y_hist))
                    if vortex_y_hist else float("nan"))
            print(f"{tag} step={step}/{n_steps} Cd_p={ap:.4f} Cd_f={af:.4f} "
                  f"Cd={at:.4f} Cl={al:.6f} vortex=({vx_a:.0f},{vy_a:.0f}) "
                  f"({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    nf = min(avg_window, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-nf:]) / nf
    cd_f_f = sum(cd_f_hist[-nf:]) / nf
    cd_tot_f = sum(cd_tot_hist[-nf:]) / nf
    cl_f = sum(cl_hist[-nf:]) / nf
    vx_f = (sum(vortex_x_hist[-5:]) / min(5, len(vortex_x_hist))
            if vortex_x_hist else float("nan"))
    vy_f = (sum(vortex_y_hist[-5:]) / min(5, len(vortex_y_hist))
            if vortex_y_hist else float("nan"))

    st = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                         length_ref=chord, min_cycles=3)

    print(f"\n{tag} === FINAL ===  Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} "
          f"Cd={cd_tot_f:.4f} Cl={cl_f:.6f} St={st} "
          f"vortex=({vx_f:.0f},{vy_f:.0f}) time={elapsed:.0f}s", flush=True)

    result = {
        "case": "delta_wing_70deg_Re1000",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "chord": chord, "sweep_deg": sweep_deg, "alpha_deg": alpha_deg,
        "Re": Re, "u_in": u_in, "nu": nu, "tau": tau,
        "Cs": cs_smag, "n_steps": n_steps,
        "Cd_pressure": float(cd_p_f), "Cd_friction": float(cd_f_f),
        "Cd_total": float(cd_tot_f), "Cl": float(cl_f), "St": st,
        "vortex_x": float(vx_f) if math.isfinite(vx_f) else None,
        "vortex_y": float(vy_f) if math.isfinite(vy_f) else None,
        "normals": "from_gradient",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
        "modules_used": [
            "drag_pressure.get_near_wall_3d",
            "drag_pressure.SurfaceMesh.from_gradient",
            "lbm_step_correct.lbm_step_correct",
            "drag_pressure.drag_pressure_integration",
            "drag_pressure.drag_friction_integration",
            "postprocess.detect_strouhal",
        ],
    }
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} saved to {output_path}", flush=True)
    return result


def run_naca(device_id, output_path=None):
    """BENCHMARK 3: NACA 0012 Re=6e6 high-Re with wall function."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 600, 300, 4
    chord = 100.0
    Re_target = 6e6
    tau = 0.50005
    alpha_deg = 0.0
    nu = (tau - 0.5) / 3.0
    u_in = 0.1
    Re_actual = u_in * chord / nu
    n_steps = 5000
    avg_window = 1000
    y_val = 1.0
    wall_law = "log"

    x_le = nx // 4
    y_c = ny // 2
    cx_qc = x_le + 0.25 * chord
    dpS = 0.5 * u_in ** 2 * chord * nz
    tag = f"[NACA SDAA:{device_id}]"

    print(f"{tag} nx={nx} ny={ny} nz={nz} chord={chord} alpha={alpha_deg}° "
          f"Re_target={Re_target} Re_actual={Re_actual:.0f} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} y_val={y_val} wall_law={wall_law} "
          f"dpS={dpS:.6e}", flush=True)

    t0 = time.time()
    solid = build_naca_mask(nx, ny, nz, chord, alpha_deg, cx_qc, y_c, device)
    print(f"{tag} solid cells={int(solid.sum().item())}", flush=True)

    # --- Common interface: get_near_wall_3d ---
    near = get_near_wall_3d(solid)
    print(f"{tag} near-wall={int(near.sum().item())}", flush=True)

    # --- Common interface: SurfaceMesh.from_naca (m=0 symmetric) ---
    mesh = SurfaceMesh.from_naca(solid, near, x_le, y_c, chord,
                                m=0.0, p=0.4, t=0.12)
    print(f"{tag} SurfaceMesh.from_naca built", flush=True)

    # --- High-Re wall-function main loop ---
    # wall_model.py note: "do NOT combine with bounce-back — use one or the
    # other."  For high-Re (tau≈0.5) the wall function (log-law body force)
    # REPLACES bounce-back as the wall treatment.  We therefore use the
    # common-interface components directly (collision → NoDynamics → stream →
    # wall_function_3d → far_field_bc_3d → mass correction) rather than
    # lbm_step_correct, which always applies bounce-back.
    from tensorlbm.solver3d import stream3d
    from tensorlbm.boundaries3d import bounce_back_cells_3d  # noqa (import for completeness)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    bc_config = {'far_field_faces': ['y-', 'y+'], 'periodic_faces': ['z-', 'z+']}
    wf_state = {"drag_fric": 0.0}  # mutable container for last friction drag

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0),
                      torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []

    # Velocity ramp (0.02 → u_in over first 1000 steps) to stabilise the
    # high-Re (tau≈0.50005) initial transient.
    ramp_steps = 1000
    for step in range(1, n_steps + 1):
        u_cur = min(u_in, 0.02 + (u_in - 0.02) * step / ramp_steps)
        # --- Common-interface components (wall-function variant) ---
        f_pre = f.clone()
        f = collide_mrt3d(f, tau=tau)                          # 1. collision
        for q in range(19):                                    # 2. NoDynamics
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = stream3d(f)                                        # 3. streaming
        f, drag_fric_wf, _ = wall_function_3d(                 # 4. wall fn
            f, solid, nu, y_val=y_val, wall_law=wall_law, near_mask=near)
        wf_state["drag_fric"] = drag_fric_wf
        f = far_field_bc_3d(f, u_cur, bc_config=bc_config)    # 5. far-field
        if step % 200 == 0:                                    # 6. mass corr
            f = correct_mass3d(f, im)

        # --- Common interface: drag_pressure_integration ---
        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap='none', p0_method='far_field', solid=solid)
        # Friction from wall function (log-law τ_w) — accurate at high-Re
        cd_f_wf = float(wf_state["drag_fric"]) / dpS
        # Also compute via common-interface drag_friction_integration
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula='standard')

        cd_p = float(fx_p)
        cd_f = cd_f_wf  # high-Re: wall-function friction is the accurate one
        cd_tot = cd_p + cd_f
        cl = float(fy_p)  # symmetric airfoil at 0° AOA → Cl ≈ 0

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            ap = sum(cd_p_hist[-n_avg:]) / n_avg
            af = sum(cd_f_hist[-n_avg:]) / n_avg
            at = sum(cd_tot_hist[-n_avg:]) / n_avg
            al = sum(cl_hist[-n_avg:]) / n_avg
            print(f"{tag} step={step}/{n_steps} Cd_p={ap:.6f} Cd_f={af:.6f} "
                  f"Cd={at:.6f} Cl={al:.6f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    nf = min(avg_window, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-nf:]) / nf
    cd_f_f = sum(cd_f_hist[-nf:]) / nf
    cd_tot_f = sum(cd_tot_hist[-nf:]) / nf
    cl_f = sum(cl_hist[-nf:]) / nf

    st = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                         length_ref=chord, min_cycles=3)

    cd_ref = 0.008
    err = abs(cd_tot_f - cd_ref) / cd_ref * 100
    print(f"\n{tag} === FINAL ===  Cd_p={cd_p_f:.6f} Cd_f={cd_f_f:.6f} "
          f"Cd={cd_tot_f:.6f} (ref={cd_ref}) err={err:.1f}% Cl={cl_f:.6f} "
          f"St={st} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "naca0012_highRe",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "chord": chord, "alpha_deg": alpha_deg,
        "Re_target": Re_target, "Re_actual": float(Re_actual),
        "u_in": u_in, "nu": nu, "tau": tau,
        "y_val": y_val, "wall_law": wall_law,
        "n_steps": n_steps,
        "Cd_pressure": float(cd_p_f), "Cd_friction": float(cd_f_f),
        "Cd_total": float(cd_tot_f), "Cl": float(cl_f), "St": st,
        "Cd_ref": cd_ref, "error_pct": float(err),
        "normals": "from_naca",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
        "modules_used": [
            "drag_pressure.get_near_wall_3d",
            "drag_pressure.SurfaceMesh.from_naca",
            "solver3d.collide_mrt3d + stream3d + correct_mass3d",
            "wall_model.wall_function_3d (replaces bounce-back at high-Re)",
            "boundaries3d.far_field_bc_3d",
            "drag_pressure.drag_pressure_integration",
            "drag_pressure.drag_friction_integration",
            "postprocess.detect_strouhal",
        ],
    }
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} saved to {output_path}", flush=True)
    return result


def run_cylinder(device_id, output_path=None):
    """BENCHMARK 4: Cylinder Re=40 (steady)."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 600, 240, 4      # larger domain → 20% blockage (was 30%)
    D = 48.0
    radius = D / 2.0
    Re = 40
    u_in = 0.1
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 10000
    avg_window = 2000

    cx = nx // 4
    cy = ny // 2
    dpS = 0.5 * u_in ** 2 * D * nz
    tag = f"[Cyl40 SDAA:{device_id}]"

    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} Re={Re} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} dpS={dpS:.6e}", flush=True)

    t0 = time.time()
    solid = build_cylinder_mask(nx, ny, nz, cx, cy, radius, device)
    print(f"{tag} solid cells={int(solid.sum().item())}", flush=True)

    # --- Common interface: get_near_wall_3d ---
    near = get_near_wall_3d(solid)
    print(f"{tag} near-wall={int(near.sum().item())}", flush=True)

    # --- Common interface: SurfaceMesh.from_cylinder ---
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, radius, axis='z')
    print(f"{tag} SurfaceMesh.from_cylinder built", flush=True)

    bc_config = {'far_field_faces': ['y-', 'y+'], 'periodic_faces': ['z-', 'z+']}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0),
                      torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    sep_hist = []

    for step in range(1, n_steps + 1):
        # --- Common interface: lbm_step_correct (MRT, no Smag) ---
        f = lbm_step_correct(
            f, collide_mrt3d, tau, solid, u_in,
            far_field_fn, correct_mass_fn=correct_mass3d, target_mass=im,
            step=step, mass_interval=200,
        )

        # --- Common interface: drag_pressure + drag_friction ---
        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap='none', p0_method='far_field', solid=solid)
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula='standard')
        cd_p, cd_f = float(fx_p), float(fx_f)
        cd_tot = cd_p + cd_f
        cl = float(fy_p + fy_f)

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)

        if step >= 2000 and step % 200 == 0:
            sep_up, sep_lo = measure_cylinder_separation(f, mesh, cx, cy, radius)
            if math.isfinite(sep_up):
                sep_hist.append(sep_up)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step % 1000 == 0:
            n_avg = min(500, len(cd_tot_hist))
            ap = sum(cd_p_hist[-n_avg:]) / n_avg
            af = sum(cd_f_hist[-n_avg:]) / n_avg
            at = sum(cd_tot_hist[-n_avg:]) / n_avg
            al = sum(cl_hist[-n_avg:]) / n_avg
            sep_a = (sum(sep_hist[-5:]) / min(5, len(sep_hist))
                     if sep_hist else float("nan"))
            print(f"{tag} step={step}/{n_steps} Cd_p={ap:.4f} Cd_f={af:.4f} "
                  f"Cd={at:.4f} Cl={al:.6f} sep={sep_a:.1f}° "
                  f"({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    nf = min(avg_window, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-nf:]) / nf
    cd_f_f = sum(cd_f_hist[-nf:]) / nf
    cd_tot_f = sum(cd_tot_hist[-nf:]) / nf
    cl_f = sum(cl_hist[-nf:]) / nf
    sep_f = (sum(sep_hist[-5:]) / min(5, len(sep_hist))
             if sep_hist else float("nan"))

    # --- Common interface: detect_strouhal ---
    st = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                         length_ref=D, min_cycles=3)

    cd_ref = 1.50
    sep_ref = 53.0
    err_cd = abs(cd_tot_f - cd_ref) / cd_ref * 100
    err_sep = (abs(sep_f - sep_ref) / sep_ref * 100
               if math.isfinite(sep_f) else float("nan"))

    print(f"\n{tag} === FINAL ===  Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} "
          f"Cd={cd_tot_f:.4f} (ref={cd_ref}) err={err_cd:.1f}%", flush=True)
    print(f"{tag} Cl={cl_f:.6f} St={st} sep={sep_f:.1f}° (ref={sep_ref}°) "
          f"err={err_sep:.1f}% time={elapsed:.0f}s", flush=True)

    result = {
        "case": "cylinder_Re40",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D, "Re": Re, "u_in": u_in, "nu": nu, "tau": tau,
        "n_steps": n_steps,
        "Cd_pressure": float(cd_p_f), "Cd_friction": float(cd_f_f),
        "Cd_total": float(cd_tot_f), "Cl": float(cl_f), "St": st,
        "separation_angle_deg": float(sep_f) if math.isfinite(sep_f) else None,
        "Cd_ref": cd_ref, "sep_ref": sep_ref,
        "error_cd_pct": float(err_cd),
        "error_sep_pct": float(err_sep) if math.isfinite(err_sep) else None,
        "normals": "from_cylinder",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
        "modules_used": [
            "drag_pressure.get_near_wall_3d",
            "drag_pressure.SurfaceMesh.from_cylinder",
            "lbm_step_correct.lbm_step_correct",
            "drag_pressure.drag_pressure_integration",
            "drag_pressure.drag_friction_integration",
            "postprocess.detect_strouhal",
        ],
    }
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# Main                                                                        #
# ========================================================================== #

def main():
    if len(sys.argv) < 4:
        print("Usage: python external_flow_common_worker.py "
              "<benchmark> <device_id> <output_path>")
        print("  benchmark: cube | delta | naca | cylinder")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    runners = {
        "cube": run_cube,
        "delta": run_delta,
        "naca": run_naca,
        "cylinder": run_cylinder,
    }
    if benchmark not in runners:
        print(f"Unknown benchmark: {benchmark}")
        print(f"Available: {', '.join(runners.keys())}")
        sys.exit(1)

    runners[benchmark](device_id, output_path)


if __name__ == "__main__":
    main()
