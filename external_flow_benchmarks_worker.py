#!/usr/bin/env python3
"""External flow benchmarks: wall-mounted cube + delta wing + NACA high-Re + cylinder Re=40.

BENCHMARK 1: Wall-mounted cube (Re=40000, SDAA:24)
  - Cube D=24 on wall, nx=300, ny=100, nz=4
  - MRT+Smag(Cs=0.1), 10000 steps, from_gradient normals
  - Reference: Cd≈1.1

BENCHMARK 2: Delta wing (swept=70°, SDAA:25)
  - nx=400, ny=200, nz=4, Re=1000, MRT+Smag(Cs=0.05)
  - 10000 steps, from_gradient normals
  - Reference: vortex position, Cl

BENCHMARK 3: NACA 0012 Re=6e6 (high-Re, SDAA:26)
  - chord=100, nx=600, ny=300, nz=4, tau=0.50005
  - wall_function_3d (log law, y_val=1.0), 5000 steps
  - from_naca normals, Reference: Cd≈0.008

BENCHMARK 4: Cylinder Re=40 (steady, SDAA:27)
  - D=48, nx=400, ny=160, nz=4, MRT (no Smag)
  - 10000 steps, from_cylinder normals
  - Reference: Cd=1.50, separation angle=53°

Usage:
  python external_flow_benchmarks_worker.py <benchmark> <device_id> <output_path>
  benchmark: cube | delta | naca | cylinder
"""
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
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.drag_pressure import (
    get_near_wall_2d,
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.airfoil_benchmark import build_airfoil_mask


# ========================================================================== #
# Geometry builders
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


def build_delta_wing_mask(nx, ny, nz, x_le, y_center, chord, sweep_deg, alpha_deg, device):
    """Delta wing: triangular planform with given sweep angle, extruded in z.

    The triangle has apex at (x_le, y_center) and trailing edge at x=x_le+chord.
    Half-span = chord / tan(sweep_deg).
    Rotated by alpha_deg (nose up).
    """
    half_span = chord / math.tan(math.radians(sweep_deg))
    alpha = math.radians(alpha_deg)
    cos_a, sin_a = math.cos(alpha), math.sin(alpha)

    # Triangle vertices (centered at centroid for rotation)
    v1 = np.array([0.0, 0.0])  # apex
    v2 = np.array([chord, -half_span])  # bottom trailing edge
    v3 = np.array([chord, half_span])  # top trailing edge
    centroid = (v1 + v2 + v3) / 3.0

    # Rotate around centroid, then translate to (x_le, y_center)
    def rotate(v):
        vr = v - centroid
        vr = np.array([vr[0] * cos_a - vr[1] * sin_a,
                        vr[0] * sin_a + vr[1] * cos_a])
        return vr + np.array([x_le, y_center])

    p1 = rotate(v1)
    p2 = rotate(v2)
    p3 = rotate(v3)

    # Build 2D mask on CPU (numpy operations), then move to device
    yy_np, xx_np = np.meshgrid(
        np.arange(ny, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        indexing="ij",
    )
    # Barycentric coordinates for point-in-triangle test
    # Using vectorized cross-product method
    px, py = xx_np, yy_np
    # Triangle vertices as numpy arrays
    ax, ay = p1
    bx, by = p2
    cx_t, cy_t = p3

    # Compute barycentric coordinates
    d = (by - cy_t) * (ax - cx_t) + (cx_t - bx) * (ay - cy_t)
    a_test = ((by - cy_t) * (px - cx_t) + (cx_t - bx) * (py - cy_t)) / d
    b_test = ((cy_t - ay) * (px - cx_t) + (ax - cx_t) * (py - cy_t)) / d
    c_test = 1.0 - a_test - b_test

    mask_2d = torch.from_numpy(
        (a_test >= 0) & (b_test >= 0) & (c_test >= 0)
    ).to(device)

    # Extrude in z
    solid = mask_2d.unsqueeze(0).expand(nz, ny, nx).clone()
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
    # Build on CPU (scanline fill uses Python loops)
    mask_2d = build_airfoil_mask(
        nx, ny, chord, alpha_deg=alpha_deg,
        m=0.0, p=0.4, t=0.12,  # NACA 0012: symmetric (m=0)
        cx=cx, cy=cy, device=torch.device("cpu"),
    )
    # Move to target device and extrude
    mask_2d = mask_2d.to(device)
    solid = mask_2d.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


# ========================================================================== #
# Separation angle measurement for cylinder
# ========================================================================== #

def measure_cylinder_separation(f, mesh, cx, cy, radius, device):
    """Measure separation angle from front stagnation point.

    Finds where tangential velocity at near-wall cells changes sign.
    Returns separation angle in degrees (from front stagnation point).
    """
    rho, ux, uy, uz = macroscopic3d(f)
    near = mesh.near

    # Get near-wall cell coordinates and velocities
    near_idx = near.cpu().numpy()
    if not near_idx.any():
        return float("nan"), float("nan")

    # Get indices of near-wall cells (z=0 layer, 2D extruded)
    nz_idx, ny_idx, nx_idx = np.where(near_idx)
    # Use z=0 layer only (all z layers are identical for 2D extruded)
    mask_z0 = nz_idx == 0
    ny_arr = ny_idx[mask_z0].astype(float)
    nx_arr = nx_idx[mask_z0].astype(float)

    # Get velocities at these cells
    ux_np = ux[0].cpu().numpy()
    uy_np = uy[0].cpu().numpy()
    ux_vals = ux_np[ny_arr.astype(int), nx_arr.astype(int)]
    uy_vals = uy_np[ny_arr.astype(int), nx_arr.astype(int)]

    # Compute angle from cylinder center
    phi = np.arctan2(ny_arr - cy, nx_arr - cx)  # angle from +x axis

    # Tangential velocity (counterclockwise tangent: (-sin phi, cos phi))
    # u_t = -ux * sin(phi) + uy * cos(phi)
    u_t = -ux_vals * np.sin(phi) + uy_vals * np.cos(phi)

    # Upper half: y > cy → phi in (0, pi)
    upper = ny_arr > cy
    if not upper.any():
        return float("nan"), float("nan")

    phi_up = phi[upper]
    u_t_up = u_t[upper]

    # Sort by phi (from pi to 0, i.e., front to rear)
    sort_idx = np.argsort(-phi_up)
    phi_up = phi_up[sort_idx]
    u_t_up = u_t_up[sort_idx]

    # Separation angle from front stagnation point: theta = pi - phi
    theta_up = np.pi - phi_up

    # Find where u_t changes sign (from negative to positive, indicating flow reversal)
    # On upper surface: flow goes from front to rear (u_t < 0 for clockwise flow)
    # At separation, u_t changes from negative to positive (reversed flow)
    sep_angle_upper = float("nan")
    for i in range(len(u_t_up) - 1):
        if u_t_up[i] < 0 and u_t_up[i + 1] > 0:
            # Interpolate to find zero crossing
            frac = -u_t_up[i] / (u_t_up[i + 1] - u_t_up[i])
            sep_phi = phi_up[i] + frac * (phi_up[i + 1] - phi_up[i])
            sep_angle_upper = math.degrees(math.pi - sep_phi)
            break

    # Lower half: y < cy → phi in (-pi, 0)
    lower = ny_arr < cy
    sep_angle_lower = float("nan")
    if lower.any():
        phi_lo = phi[lower]
        u_t_lo = u_t[lower]
        sort_idx = np.argsort(phi_lo)  # from -pi to 0 (front to rear)
        phi_lo = phi_lo[sort_idx]
        u_t_lo = u_t_lo[sort_idx]
        theta_lo = np.pi + phi_lo  # angle from front stagnation (lower)
        for i in range(len(u_t_lo) - 1):
            if u_t_lo[i] < 0 and u_t_lo[i + 1] > 0:
                frac = -u_t_lo[i] / (u_t_lo[i + 1] - u_t_lo[i])
                sep_phi = phi_lo[i] + frac * (phi_lo[i + 1] - phi_lo[i])
                sep_angle_lower = math.degrees(math.pi + sep_phi)
                break

    return sep_angle_upper, sep_angle_lower


# ========================================================================== #
# Wake vortex center measurement (for delta wing)
# ========================================================================== #

def measure_wake_vortex(f, solid, cx_obj, nx, ny, nz, device):
    """Find wake vortex center (centroid of negative vorticity region)."""
    rho, ux, uy, uz = macroscopic3d(f)
    # Vorticity (z-component): du_y/dx - du_x/dy
    # Use central difference
    uy_np = uy[0].cpu().numpy()
    ux_np = ux[0].cpu().numpy()
    duy_dx = np.zeros_like(uy_np)
    dux_dy = np.zeros_like(ux_np)
    duy_dx[:, 1:-1] = (uy_np[:, 2:] - uy_np[:, :-2]) / 2.0
    dux_dy[1:-1, :] = (ux_np[2:, :] - ux_np[:-2, :]) / 2.0
    vort = duy_dx - dux_dy

    # Wake region: behind the object
    solid_np = solid[0].cpu().numpy()
    wake_start = int(cx_obj + 10)
    wake_mask = (~solid_np).copy()
    wake_mask[:, :wake_start] = False

    if not wake_mask.any():
        return float("nan"), float("nan")

    # Find peak vorticity location
    wake_vort = np.abs(vort) * wake_mask
    if wake_vort.max() < 1e-10:
        return float("nan"), float("nan")

    peak_idx = np.unravel_index(wake_vort.argmax(), wake_vort.shape)
    # peak_idx is (y, x) due to numpy indexing
    vortex_y = float(peak_idx[0])
    vortex_x = float(peak_idx[1])

    return vortex_x, vortex_y


# ========================================================================== #
# Benchmark runners
# ========================================================================== #

def run_cube(device_id, output_path=None):
    """BENCHMARK 1: Wall-mounted cube Re=40000."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 300, 100, 4
    D = 24
    Re = 40000
    u_in = 0.1  # lower Mach for stability
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.1
    n_steps = 10000
    use_wall_fn = True  # wall function for high-Re stabilization
    wall_law = "gradient"
    y_val_wf = 0.5

    cx = nx // 4  # cube position (quarter from inlet)
    dpS = 0.5 * u_in ** 2 * D * nz  # frontal area = D * nz

    tag = f"[Cube SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} Re={Re} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.6e}", flush=True)

    t0 = time.time()

    # Build masks
    cube = build_cube_mask(nx, ny, nz, cx, D, device)
    wall = build_bottom_wall_mask(nx, ny, nz, device)
    solid_total = cube | wall  # for NoDynamics + bounce_back
    solid_cube = cube  # for drag measurement

    n_solid = int(solid_total.sum().item())
    n_cube = int(solid_cube.sum().item())
    print(f"{tag} total solid={n_solid} cube cells={n_cube}", flush=True)

    # Near-wall mask (cube only, 2D extruded) — for drag measurement
    near_cube = get_near_wall_2d(solid_cube, axis='z')
    n_near_cube = int(near_cube.sum().item())
    print(f"{tag} near-wall cells (cube)={n_near_cube}", flush=True)

    # Near-wall mask (total solid) — for wall function
    near_total = get_near_wall_2d(solid_total, axis='z')
    n_near_total = int(near_total.sum().item())
    print(f"{tag} near-wall cells (total)={n_near_total}", flush=True)

    # Surface mesh with from_gradient normals (cube only — for drag)
    mesh = SurfaceMesh.from_gradient(solid_cube, near_cube)
    print(f"{tag} SurfaceMesh.from_gradient built", flush=True)

    # NoDynamics mask
    sm = solid_total.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize flow
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid_total] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # History
    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        # Wall function approach: stream → wall_function → far_field
        f = stream3d(f)
        f, drag_fric_raw, _ = wall_function_3d(
            f, solid_total, nu, y_val=y_val_wf, wall_law=wall_law,
            near_mask=near_total,
        )
        # Far-field: y+ only (y- is wall), z periodic
        bc_config = {'far_field_faces': ['y+'], 'periodic_faces': ['z-', 'z+']}
        f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)
        # Restore solid cells after BC (bottom wall extends to inlet/outlet)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # Drag (cube only — pressure + friction from surface mesh)
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, solid=solid_cube, p0_method='far_field')
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p, cd_f = fx_p, fx_f
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
            print(f"{tag} step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                  f"Cd={cd_tot_avg:.4f} Cl={cl_avg:.6f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(500, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_f = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_f = sum(cd_tot_hist[-n_final:]) / n_final
    cl_f = sum(cl_hist[-n_final:]) / n_final

    cd_ref = 1.1
    err = abs(cd_tot_f - cd_ref) / cd_ref * 100

    print(f"\n{tag} === FINAL ===", flush=True)
    print(f"{tag} Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} Cd={cd_tot_f:.4f} "
          f"(ref={cd_ref}) err={err:.1f}% Cl={cl_f:.6f} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "wall_mounted_cube_Re40000",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D, "Re": Re, "u_in": u_in, "nu": nu, "tau": tau,
        "Cs": cs_smag, "n_steps": n_steps,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cl": cl_f, "Cd_ref": cd_ref, "error_pct": err,
        "normals": "from_gradient",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
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
    alpha_deg = 10.0  # angle of attack for lift generation
    u_in = 0.1
    nu = u_in * chord / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    n_steps = 10000

    x_le = nx // 4  # leading edge position
    y_center = ny // 2
    half_span = chord / math.tan(math.radians(sweep_deg))
    dpS = 0.5 * u_in ** 2 * chord * nz  # reference area = chord * nz

    tag = f"[Delta SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} chord={chord} sweep={sweep_deg}° "
          f"alpha={alpha_deg}° Re={Re} u_in={u_in} nu={nu:.6e} tau={tau:.6f} "
          f"Cs={cs_smag} half_span={half_span:.1f} dpS={dpS:.6e}", flush=True)

    t0 = time.time()

    # Build delta wing mask
    solid = build_delta_wing_mask(nx, ny, nz, x_le, y_center, chord, sweep_deg, alpha_deg, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells={n_solid}", flush=True)

    # Near-wall mask (2D extruded)
    near = get_near_wall_2d(solid, axis='z')
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # Surface mesh with from_gradient normals
    mesh = SurfaceMesh.from_gradient(solid, near)
    print(f"{tag} SurfaceMesh.from_gradient built", flush=True)

    # NoDynamics mask
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize flow
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # History
    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    vortex_x_hist, vortex_y_hist = [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f = stream3d(f)
        bc_config = {'far_field_faces': ['y-', 'y+'], 'periodic_faces': ['z-', 'z+']}
        f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # Drag
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, solid=solid, p0_method='far_field')
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p, cd_f = fx_p, fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)

        # Vortex position (every 100 steps)
        if step % 100 == 0:
            vx, vy = measure_wake_vortex(f, solid, x_le + chord, nx, ny, nz, device)
            if math.isfinite(vx):
                vortex_x_hist.append(vx)
                vortex_y_hist.append(vy)

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
            vx_avg = sum(vortex_x_hist[-5:]) / min(5, len(vortex_x_hist)) if vortex_x_hist else float("nan")
            vy_avg = sum(vortex_y_hist[-5:]) / min(5, len(vortex_y_hist)) if vortex_y_hist else float("nan")
            print(f"{tag} step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                  f"Cd={cd_tot_avg:.4f} Cl={cl_avg:.6f} vortex=({vx_avg:.0f},{vy_avg:.0f}) "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(500, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_f = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_f = sum(cd_tot_hist[-n_final:]) / n_final
    cl_f = sum(cl_hist[-n_final:]) / n_final
    vx_f = sum(vortex_x_hist[-5:]) / min(5, len(vortex_x_hist)) if vortex_x_hist else float("nan")
    vy_f = sum(vortex_y_hist[-5:]) / min(5, len(vortex_y_hist)) if vortex_y_hist else float("nan")

    print(f"\n{tag} === FINAL ===", flush=True)
    print(f"{tag} Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} Cd={cd_tot_f:.4f} "
          f"Cl={cl_f:.6f} vortex=({vx_f:.0f},{vy_f:.0f}) time={elapsed:.0f}s", flush=True)

    result = {
        "case": "delta_wing_70deg_Re1000",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "chord": chord, "sweep_deg": sweep_deg, "alpha_deg": alpha_deg,
        "Re": Re, "u_in": u_in, "nu": nu, "tau": tau,
        "Cs": cs_smag, "n_steps": n_steps,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cl": cl_f,
        "vortex_x": vx_f, "vortex_y": vy_f,
        "normals": "from_gradient",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
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
    alpha_deg = 0.0  # symmetric airfoil at 0° AOA
    nu = (tau - 0.5) / 3.0
    u_in = 0.1  # lower Mach for stability; actual Re = u_in*chord/nu
    Re_actual = u_in * chord / nu
    n_steps = 5000
    y_val = 1.0
    wall_law = "gradient"  # more stable than log for very high Re

    x_le = nx // 4  # leading edge position
    y_c = ny // 2
    cx_qc = x_le + 0.25 * chord  # quarter-chord position
    dpS = 0.5 * u_in ** 2 * chord * nz  # reference area = chord * nz

    tag = f"[NACA SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} chord={chord} alpha={alpha_deg}° "
          f"Re_target={Re_target} Re_actual={Re_actual:.0f} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} y_val={y_val} wall_law={wall_law} "
          f"dpS={dpS:.6e}", flush=True)

    t0 = time.time()

    # Build NACA 0012 mask
    solid = build_naca_mask(nx, ny, nz, chord, alpha_deg, cx_qc, y_c, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells={n_solid}", flush=True)

    # Near-wall mask (2D extruded)
    near = get_near_wall_2d(solid, axis='z')
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # Surface mesh with from_naca normals (m=0 for symmetric NACA 0012)
    mesh = SurfaceMesh.from_naca(solid, near, x_le, y_c, chord, m=0.0, p=0.4, t=0.12)
    print(f"{tag} SurfaceMesh.from_naca built", flush=True)

    # NoDynamics mask
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize flow
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # History
    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        # MRT collision (no Smag for wall function case)
        f = collide_mrt3d(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        # Wall function approach: stream → wall_function → far_field
        f = stream3d(f)
        f, drag_fric_raw, drag_pres_wf = wall_function_3d(
            f, solid, nu, y_val=y_val, wall_law=wall_law, near_mask=near,
        )
        bc_config = {'far_field_faces': ['y-', 'y+'], 'periodic_faces': ['z-', 'z+']}
        f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)
        # Restore solid cells after BC
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # Drag: pressure from surface mesh, friction from wall function
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, solid=solid, p0_method='far_field')
        cd_f = drag_fric_raw / dpS
        cd_p = fx_p
        cd_tot = cd_p + cd_f
        cl = fy_p  # pressure lift (friction y-comp ≈ 0 for symmetric)

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
            print(f"{tag} step={step} Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                  f"Cd={cd_tot_avg:.6f} Cl={cl_avg:.6f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(500, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_f = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_f = sum(cd_tot_hist[-n_final:]) / n_final
    cl_f = sum(cl_hist[-n_final:]) / n_final

    cd_ref = 0.008
    err = abs(cd_tot_f - cd_ref) / cd_ref * 100

    print(f"\n{tag} === FINAL ===", flush=True)
    print(f"{tag} Cd_p={cd_p_f:.6f} Cd_f={cd_f_f:.6f} Cd={cd_tot_f:.6f} "
          f"(ref={cd_ref}) err={err:.1f}% Cl={cl_f:.6f} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "naca0012_highRe",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "chord": chord, "alpha_deg": alpha_deg,
        "Re_target": Re_target, "Re_actual": Re_actual,
        "u_in": u_in, "nu": nu, "tau": tau,
        "y_val": y_val, "wall_law": wall_law,
        "n_steps": n_steps,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cl": cl_f, "Cd_ref": cd_ref, "error_pct": err,
        "normals": "from_naca",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} saved to {output_path}", flush=True)
    return result


def run_cylinder(device_id, output_path=None):
    """BENCHMARK 4: Cylinder Re=40 (steady)."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 400, 160, 4
    D = 48.0
    radius = D / 2.0
    Re = 40
    u_in = 0.1
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 10000

    cx = nx // 4  # cylinder center x
    cy = ny // 2  # cylinder center y
    dpS = 0.5 * u_in ** 2 * D * nz  # frontal area = D * nz

    tag = f"[Cyl40 SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} Re={Re} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} dpS={dpS:.6e}", flush=True)

    t0 = time.time()

    # Build cylinder mask
    solid = build_cylinder_mask(nx, ny, nz, cx, cy, radius, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells={n_solid}", flush=True)

    # Near-wall mask (2D extruded)
    near = get_near_wall_2d(solid, axis='z')
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # Surface mesh with from_cylinder normals
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, radius, axis='z')
    print(f"{tag} SurfaceMesh.from_cylinder built", flush=True)

    # NoDynamics mask
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize flow
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # History
    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    sep_hist = []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        # MRT collision (no Smagorinsky — laminar Re=40)
        f = collide_mrt3d(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f = stream3d(f)
        bc_config = {'far_field_faces': ['y-', 'y+'], 'periodic_faces': ['z-', 'z+']}
        f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # Drag
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, solid=solid, p0_method='far_field')
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p, cd_f = fx_p, fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)

        # Separation angle (every 200 steps after step 2000)
        if step >= 2000 and step % 200 == 0:
            sep_up, sep_lo = measure_cylinder_separation(f, mesh, cx, cy, radius, device)
            if math.isfinite(sep_up):
                sep_hist.append(sep_up)

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
            sep_avg = sum(sep_hist[-5:]) / min(5, len(sep_hist)) if sep_hist else float("nan")
            print(f"{tag} step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                  f"Cd={cd_tot_avg:.4f} Cl={cl_avg:.6f} sep={sep_avg:.1f}° "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(500, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_f = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_f = sum(cd_tot_hist[-n_final:]) / n_final
    cl_f = sum(cl_hist[-n_final:]) / n_final
    sep_f = sum(sep_hist[-5:]) / min(5, len(sep_hist)) if sep_hist else float("nan")

    cd_ref = 1.50
    sep_ref = 53.0
    err_cd = abs(cd_tot_f - cd_ref) / cd_ref * 100
    err_sep = abs(sep_f - sep_ref) / sep_ref * 100 if math.isfinite(sep_f) else float("nan")

    print(f"\n{tag} === FINAL ===", flush=True)
    print(f"{tag} Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} Cd={cd_tot_f:.4f} "
          f"(ref={cd_ref}) err={err_cd:.1f}%", flush=True)
    print(f"{tag} Cl={cl_f:.6f} sep_angle={sep_f:.1f}° (ref={sep_ref}°) "
          f"err={err_sep:.1f}% time={elapsed:.0f}s", flush=True)

    result = {
        "case": "cylinder_Re40",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D, "Re": Re, "u_in": u_in, "nu": nu, "tau": tau,
        "n_steps": n_steps,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cl": cl_f,
        "separation_angle_deg": sep_f,
        "Cd_ref": cd_ref, "sep_ref": sep_ref,
        "error_cd_pct": err_cd, "error_sep_pct": err_sep,
        "normals": "from_cylinder",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# Main
# ========================================================================== #

def main():
    if len(sys.argv) < 4:
        print("Usage: python external_flow_benchmarks_worker.py <benchmark> <device_id> <output_path>")
        print("  benchmark: cube | delta | naca | cylinder")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if benchmark == "cube":
        run_cube(device_id, output_path)
    elif benchmark == "delta":
        run_delta(device_id, output_path)
    elif benchmark == "naca":
        run_naca(device_id, output_path)
    elif benchmark == "cylinder":
        run_cylinder(device_id, output_path)
    else:
        print(f"Unknown benchmark: {benchmark}")
        sys.exit(1)


if __name__ == "__main__":
    main()
