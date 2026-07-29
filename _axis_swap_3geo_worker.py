#!/usr/bin/env python3
"""y↔z swap verification on cylinder, sphere, and SUBOFF.

Tests direction-agnostic property of the verified modules:
  - drag_pressure.py: SurfaceMesh.from_cylinder (axis='z'/'y'),
    from_sphere, from_suboff
  - boundaries3d.py: far_field_bc_3d (bc_config)
  - get_near_wall_2d (axis parameter), get_near_wall_3d

BENCHMARK 1: Cylinder y↔z swap at D=96 (large domain)
  - axis='z': D=96, nx=1200, ny=400, nz=4  (SDAA:28)
  - axis='y': D=96, nx=1200, ny=4,  nz=400 (SDAA:29)
  - u_in=0.08, Re=200, tau=0.5152, 10000 steps
  - Both should give same Cd (within 1%)

BENCHMARK 2: Sphere y↔z swap
  - standard:  D=40, nx=180, ny=180, nz=180 (SDAA:30)
  - swapped:    swap y and z of solid mask   (SDAA:31)
  - u_in=0.08, Re=100, tau=0.596, 3000 steps
  - Tests true 3D direction-agnostic

BENCHMARK 3: SUBOFF y↔z swap
  - standard:  L=80, nx=200, ny=80, nz=80 (SDAA:30/31 after sphere)
  - swapped:   swap y and z of solid mask
  - u_in=0.06, Re=1000, tau=0.5144, 5000 steps
  - Tests axisymmetric body direction-agnostic

Usage:
  PYTHONPATH=src python _axis_swap_3geo_worker.py <geometry> <variant> <device_id> <output_json>
  geometry: cylinder | sphere | suboff
  variant:  z | y  (cylinder); standard | swapped (sphere/suboff)
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
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.solver3d import stream3d, correct_mass3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, drag_pressure_integration, drag_friction_integration,
    get_near_wall_2d, get_near_wall_3d,
)


# ---------------------------------------------------------------------------
#  Cylinder mask builder (axis-aware)
# ---------------------------------------------------------------------------
def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device, axis='z', cz=None):
    """Boolean solid mask for a cylinder extruded along *axis*."""
    if axis == 'z':
        yy, xx = torch.meshgrid(
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    elif axis == 'y':
        cz_c = cz if cz is not None else nz / 2.0
        zz, xx = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        circle = (xx - cx) ** 2 + (zz - cz_c) ** 2 <= radius ** 2
        solid = circle.unsqueeze(1).expand(nz, ny, nx).clone()
    else:
        raise ValueError(f"axis must be 'y' or 'z', got '{axis}'")
    return solid


# ---------------------------------------------------------------------------
#  Sphere mask builder
# ---------------------------------------------------------------------------
def build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device):
    """Vectorized sphere mask: (i-cx)^2+(j-cy)^2+(k-cz)^2 < R^2."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2


# ===========================================================================
#  BENCHMARK 1: Cylinder y↔z swap at D=96
# ===========================================================================
def run_cylinder(device_id, axis, output_path):
    """Cylinder flow with given axis orientation, pressure+friction drag."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # Parameters
    diameter = 96.0
    radius = diameter / 2.0
    u_in = 0.08
    Re = 200.0
    nu = u_in * diameter / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    n_steps = 10000
    warmup = 1000

    # Grid: 12.5D x ~4.17D in cross-flow plane, 4 layers along axis
    nx = 1200
    n_cross = 400
    n_axis = 4

    if axis == 'z':
        ny, nz = n_cross, n_axis
        cx_c = nx * 0.25
        cy_c = ny * 0.5
        cz_c = None
        bc_config = {
            'far_field_faces': ['y-', 'y+'],
            'periodic_faces': ['z-', 'z+'],
        }
        A_frontal = diameter * nz
    elif axis == 'y':
        ny, nz = n_axis, n_cross
        cx_c = nx * 0.25
        cy_c = None
        cz_c = nz * 0.5
        bc_config = {
            'far_field_faces': ['z-', 'z+'],
            'periodic_faces': ['y-', 'y+'],
        }
        A_frontal = diameter * ny
    else:
        raise ValueError(f"axis must be 'y' or 'z', got '{axis}'")

    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    tag = f"[cyl axis={axis} SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={diameter} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.4f}", flush=True)
    print(f"{tag} bc_config={bc_config}", flush=True)

    t0 = time.time()

    # Build cylinder mask
    solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device,
                                axis=axis, cz=cz_c)
    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Near-wall mask
    near = get_near_wall_2d(solid, axis=axis)
    n_near = int(near.sum().item())
    print(f"{tag} solid cells={n_solid} near-wall cells={n_near}", flush=True)

    # Surface mesh for pressure drag
    mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius,
                                     axis=axis, cz=cz_c)
    print(f"{tag} SurfaceMesh built (axis={axis})", flush=True)

    # Initialize
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    # Accumulators
    cd_p_hist = []
    cd_f_hist = []

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Stream
        f = stream3d(f)

        # 6. Far-field BC (direction-agnostic via bc_config)
        f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)

        # 7. Pressure + friction drag (post-stream, post-BC)
        fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)

        # 8. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Record post-warmup
        if step > warmup:
            cd_p_hist.append(fx_p)
            cd_f_hist.append(fx_f)

        if step % 500 == 0:
            _, ux, _, _ = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux).max().item())
            elapsed = time.time() - t0
            n_avg = min(500, len(cd_p_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg if cd_p_hist else 0
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg if cd_f_hist else 0
            print(f"{tag} step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                  f"Cd_tot={cd_p_avg+cd_f_avg:.4f} max|ux|={ms:.4f} "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0

    cd_p_mean = sum(cd_p_hist) / max(len(cd_p_hist), 1) if cd_p_hist else float("nan")
    cd_f_mean = sum(cd_f_hist) / max(len(cd_f_hist), 1) if cd_f_hist else float("nan")
    cd_tot_mean = cd_p_mean + cd_f_mean

    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

    cd_p_std = _std(cd_p_hist)
    cd_f_std = _std(cd_f_hist)

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cd_p = {cd_p_mean:.4f} ± {cd_p_std:.4f}", flush=True)
    print(f"{tag} Cd_f = {cd_f_mean:.4f} ± {cd_f_std:.4f}", flush=True)
    print(f"{tag} Cd_tot = {cd_tot_mean:.4f}", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
        "geometry": "cylinder",
        "axis": axis,
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "diameter": diameter,
        "Re": Re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "warmup": warmup,
        "bc_config": bc_config,
        "Cd_p_mean": cd_p_mean,
        "Cd_f_mean": cd_f_mean,
        "Cd_tot_mean": cd_tot_mean,
        "Cd_p_std": cd_p_std,
        "Cd_f_std": cd_f_std,
        "n_samples": len(cd_p_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result


# ===========================================================================
#  BENCHMARK 2: Sphere y↔z swap
# ===========================================================================
def run_sphere(device_id, variant, output_path):
    """Sphere drag — standard vs y↔z swapped solid mask."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # Parameters
    D = 40.0
    R = D / 2.0
    u_in = 0.08
    Re = 100.0
    nu = u_in * D / Re
    tau = 0.596
    cs_smag = 0.05
    n_steps = 3000
    warmup = 300

    nx, ny, nz = 180, 180, 180
    cx = nx * 0.25   # 45
    cy = ny * 0.5    # 90
    cz = nz * 0.5    # 90

    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2

    tag = f"[sphere {variant} SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} R={R} "
          f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
          f"dpS={dpS:.6e}", flush=True)

    t0 = time.time()

    # Build sphere solid mask
    solid = build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    if variant == "swapped":
        # Swap y and z axes of the solid mask: (nz, ny, nx) → (ny, nz, nx) → (nz, ny, nx)
        # i.e., transpose dims 0 and 1
        solid = solid.transpose(0, 1).contiguous()
        # After swap, the sphere center in y-z is swapped: cy↔cz
        # But since cy=cz=90, the center is the same.
        # The near-wall and mesh must use the swapped mask.
        cy_mesh, cz_mesh = cz, cy  # swapped
    else:
        cy_mesh, cz_mesh = cy, cz

    # Near-wall mask (3D — direction-agnostic)
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # Surface mesh (from_sphere is direction-agnostic by construction)
    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy_mesh, cz_mesh, R)

    # Normal statistics
    nx_n_vals = mesh.nx_n[near]
    ny_n_vals = mesh.ny_n[near]
    nz_n_vals = mesh.nz_n[near]
    print(f"{tag} normal stats: "
          f"nx=[{float(nx_n_vals.min()):.3f},{float(nx_n_vals.max()):.3f}] "
          f"ny=[{float(ny_n_vals.min()):.3f},{float(ny_n_vals.max()):.3f}] "
          f"nz=[{float(nz_n_vals.min()):.3f},{float(nz_n_vals.max()):.3f}]",
          flush=True)

    # Solid mask for NoDynamics
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    im = float(rho0.sum().item())

    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    cd_p_hist = []
    cd_f_hist = []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in)
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            cd_p_hist.append(fx_p)
            cd_f_hist.append(fx_f)

        if step % 200 == 0:
            n_avg = min(200, len(cd_p_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg if cd_p_hist else 0
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg if cd_f_hist else 0
            elapsed = time.time() - t0
            print(f"{tag} step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                  f"Cd_tot={cd_p_avg+cd_f_avg:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(500, len(cd_p_hist))
    cd_p_mean = sum(cd_p_hist[-n_final:]) / n_final if cd_p_hist else float("nan")
    cd_f_mean = sum(cd_f_hist[-n_final:]) / n_final if cd_f_hist else float("nan")
    cd_tot_mean = cd_p_mean + cd_f_mean

    # Reference: Henderson empirical Cd=1.09 at Re=100
    cd_ref = 1.09
    ref_name = "Henderson empirical"

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cd_p = {cd_p_mean:.4f}", flush=True)
    print(f"{tag} Cd_f = {cd_f_mean:.4f}", flush=True)
    print(f"{tag} Cd_tot = {cd_tot_mean:.4f} (ref={cd_ref})", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
        "geometry": "sphere",
        "variant": variant,
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "Cd_p_mean": cd_p_mean,
        "Cd_f_mean": cd_f_mean,
        "Cd_tot_mean": cd_tot_mean,
        "Cd_ref": cd_ref,
        "ref_name": ref_name,
        "n_samples": len(cd_p_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result


# ===========================================================================
#  BENCHMARK 3: SUBOFF y↔z swap
# ===========================================================================
def run_suboff(device_id, variant, output_path):
    """SUBOFF bare-hull drag — standard vs y↔z swapped solid mask."""
    from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # Parameters
    L = 80.0
    u_in = 0.06
    Re = 1000.0
    nu = u_in * L / Re
    tau = 0.5144
    cs_smag = 0.05
    n_steps = 5000
    warmup = 500

    nx, ny, nz = 200, 80, 80
    config = SuboffConfig()
    radius = config.r_over_l * L
    cx = nx * 0.30   # bow at 30%
    cy = ny * 0.5
    cz = nz * 0.5

    dpS = 0.5 * u_in ** 2 * math.pi * radius ** 2
    Cf_ittc = 0.075 / (np.log10(Re) - 2.0) ** 2

    tag = f"[suboff {variant} SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} L={L} R_max={radius:.3f} "
          f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
          f"dpS={dpS:.6e} Cf_ITTC={Cf_ittc:.6f}", flush=True)

    t0 = time.time()

    # Build SUBOFF solid mask
    solid, stats = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=L, radius=radius,
        config=config,
        device=device,
    )
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}  L/D={stats['L_D_ratio']}", flush=True)

    if variant == "swapped":
        # Swap y and z axes of the solid mask: transpose dims 0 and 1
        solid = solid.transpose(0, 1).contiguous()
        # After swap, center y↔z
        cy_mesh, cz_mesh = cz, cy
    else:
        cy_mesh, cz_mesh = cy, cz

    # Near-wall mask (3D — direction-agnostic)
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # Surface mesh (from_suboff is direction-agnostic by construction)
    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy_mesh, cz_mesh, L, radius,
                                   config)

    # Normal statistics
    nx_n_vals = mesh.nx_n[near]
    ny_n_vals = mesh.ny_n[near]
    nz_n_vals = mesh.nz_n[near]
    print(f"{tag} normal stats: "
          f"nx=[{float(nx_n_vals.min()):.3f},{float(nx_n_vals.max()):.3f}] "
          f"ny=[{float(ny_n_vals.min()):.3f},{float(ny_n_vals.max()):.3f}] "
          f"nz=[{float(nz_n_vals.min()):.3f},{float(nz_n_vals.max()):.3f}]",
          flush=True)

    # Solid mask for NoDynamics
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    im = float(rho0.sum().item())

    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    cd_p_hist = []
    cd_f_hist = []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in)
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            cd_p_hist.append(fx_p)
            cd_f_hist.append(fx_f)

        if step % 500 == 0:
            n_avg = min(500, len(cd_p_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg if cd_p_hist else 0
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg if cd_f_hist else 0
            elapsed = time.time() - t0
            print(f"{tag} step={step} Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                  f"Cd_tot={cd_p_avg+cd_f_avg:.6f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(500, len(cd_p_hist))
    cd_p_mean = sum(cd_p_hist[-n_final:]) / n_final if cd_p_hist else float("nan")
    cd_f_mean = sum(cd_f_hist[-n_final:]) / n_final if cd_f_hist else float("nan")
    cd_tot_mean = cd_p_mean + cd_f_mean

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cd_p = {cd_p_mean:.6f}", flush=True)
    print(f"{tag} Cd_f = {cd_f_mean:.6f}", flush=True)
    print(f"{tag} Cd_tot = {cd_tot_mean:.6f} (ITTC Cf={Cf_ittc:.6f})", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
        "geometry": "suboff",
        "variant": variant,
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "L": L,
        "R_max": radius,
        "L_D_ratio": stats["L_D_ratio"],
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "Cd_p_mean": cd_p_mean,
        "Cd_f_mean": cd_f_mean,
        "Cd_tot_mean": cd_tot_mean,
        "Cd_ref": float(Cf_ittc),
        "ref_name": "ITTC-1957 Cf=0.075/(log10(Re)-2)^2",
        "n_samples": len(cd_p_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result


# ===========================================================================
#  Main entry point
# ===========================================================================
if __name__ == "__main__":
    geometry = sys.argv[1].lower()
    variant = sys.argv[2].lower()
    device_id = int(sys.argv[3])
    output_path = sys.argv[4]

    if geometry == "cylinder":
        run_cylinder(device_id, variant, output_path)
    elif geometry == "sphere":
        run_sphere(device_id, variant, output_path)
    elif geometry == "suboff":
        run_suboff(device_id, variant, output_path)
    else:
        raise ValueError(f"geometry must be cylinder/sphere/suboff, got '{geometry}'")
