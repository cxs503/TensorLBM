#!/usr/bin/env python3
"""STL-based ship hull drag benchmark worker.

Runs LBM simulation on real ship hull STL geometry with STL-derived
surface normals and STL-based dA (surface area correction).

Tests:
  TEST 1 (SDAA:20): KVLCC2, Re=1e5, STL normals, target <50% (was 239%)
  TEST 2 (SDAA:21): DTMB5415, Re=1e5, STL normals, target <15% (was 21.6%)
  TEST 3 (SDAA:22): KCS, Re=1000, STL normals, target finite convergence
  TEST 4 (SDAA:23): Sphere dA comparison (dA=1.0 vs stl_area vs cos_theta)

Pipeline:
  1. read_stl() → mirror_stl() → voxelize_stl() → SurfaceMesh.from_stl()
  2. LBM: MRT+Smagorinsky, NoDynamics, half-way BB, far-field BC
  3. Drag: pressure + friction integration with STL normals + dA

Usage:
  PYTHONPATH=src python stl_ship_worker.py <test_id> <device_id> <output_path>
  test_id: 1=KVLCC2, 2=DTMB5415, 3=KCS, 4=sphere_dA
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
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.stl_geometry import (
    read_stl,
    voxelize_stl,
    mirror_stl,
    SurfaceMesh_from_stl,
    make_sphere_stl,
)

STL_DIR = Path(
    "/root/ship-performance-platform-incoming/ship-performance-platform/"
    "backend/data/geometry/ships"
)


# -----------------------------------------------------------------------
# Grid mapping: place STL hull on LBM grid with inflow/wake room
# -----------------------------------------------------------------------

def setup_ship_grid(stl_path, nx, ny, nz, L_lattice_target=None):
    """Read, mirror, and map an STL half-hull onto the LBM grid.

    Returns: solid, near, vertices_full, faces_full, normals_full,
             origin, spacing, L_lattice, hull_info
    """
    vertices, faces, face_normals = read_stl(stl_path)

    # Mirror half-hull to full hull
    vertices_full, faces_full, normals_full = mirror_stl(
        vertices, faces, face_normals, axis=1
    )

    # STL bounding box
    x_min, x_max = vertices_full[:, 0].min(), vertices_full[:, 0].max()
    y_min, y_max = vertices_full[:, 1].min(), vertices_full[:, 1].max()
    z_min, z_max = vertices_full[:, 2].min(), vertices_full[:, 2].max()
    L_stl = x_max - x_min
    B_stl = y_max - y_min  # full beam
    D_stl = z_max - z_min  # total depth

    # Determine spacing: map hull length to L_lattice_target cells
    if L_lattice_target is None:
        L_lattice_target = nx * 0.6  # 60% of domain for hull
    spacing = L_stl / L_lattice_target
    L_lattice = L_lattice_target

    # Place hull center at 35% of domain x (room for inflow + wake)
    hull_center_x_stl = (x_min + x_max) / 2.0
    # Grid origin: hull center maps to nx*0.35
    origin_x = hull_center_x_stl - (nx * 0.35) * spacing
    # Y: center the hull
    hull_center_y_stl = (y_min + y_max) / 2.0
    origin_y = hull_center_y_stl - (ny * 0.5) * spacing
    # Z: place waterline (z=0 in STL) at nz*0.5
    origin_z = 0.0 - (nz * 0.5) * spacing

    origin = (origin_x, origin_y, origin_z)
    sp = (spacing, spacing, spacing)

    # Voxelize
    solid = voxelize_stl(
        vertices_full, faces_full, (nx, ny, nz), origin, sp
    )
    near = get_near_wall_3d(solid)

    hull_info = {
        "stl_file": str(stl_path),
        "n_verts": len(vertices_full),
        "n_faces": len(faces_full),
        "L_stl": float(L_stl),
        "B_stl": float(B_stl),
        "D_stl": float(D_stl),
        "L_lattice": float(L_lattice),
        "spacing": float(spacing),
        "origin": (float(origin_x), float(origin_y), float(origin_z)),
        "n_solid": int(solid.sum().item()),
        "n_near": int(near.sum().item()),
    }

    return solid, near, vertices_full, faces_full, normals_full, origin, sp, L_lattice, hull_info


# -----------------------------------------------------------------------
# Ship hull benchmark
# -----------------------------------------------------------------------

def run_ship_benchmark(
    test_id, device_id, stl_path, ship_name,
    nx, ny, nz, Re, u_in, n_steps, warmup, cs_smag,
    dA_method="stl_area", normal_method="stl", output_path=None,
):
    """Run STL-based ship hull drag benchmark."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id} {ship_name}]"

    # ---- Grid setup ----
    L_lattice_target = int(nx * 0.6)
    (solid, near, vertices, faces, normals,
     origin, spacing, L_lattice, hull_info) = setup_ship_grid(
        stl_path, nx, ny, nz, L_lattice_target
    )

    solid = solid.to(device)
    near = near.to(device)

    # ---- Flow parameters ----
    nu = u_in * L_lattice / Re
    tau = 3.0 * nu + 0.5

    # ITTC-1957 reference
    cf_ittc = 0.075 / (math.log10(Re) - 2.0) ** 2

    print(f"{tag} STL ship benchmark: {ship_name}", flush=True)
    print(f"{tag} grid={nx}x{ny}x{nz} L_lat={L_lattice} spacing={spacing[0]:.4f}", flush=True)
    print(f"{tag} u_in={u_in} Re={Re:.0e} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag}", flush=True)
    print(f"{tag} dA_method={dA_method} normal={normal_method}", flush=True)
    print(f"{tag} solid={hull_info['n_solid']} near={hull_info['n_near']} "
          f"faces={hull_info['n_faces']}", flush=True)
    print(f"{tag} Cf_ITTC={cf_ittc:.6f}", flush=True)

    t0 = time.time()

    # ---- Build surface mesh with STL normals + dA ----
    if normal_method == "stl":
        mesh = SurfaceMesh_from_stl(
            solid, near, vertices, faces, normals.astype(np.float32),
            origin, spacing, dA_method=dA_method,
        )
    elif normal_method == "from_gradient":
        mesh = SurfaceMesh.from_gradient(solid, near)
    else:
        raise ValueError(f"normal_method must be 'stl' or 'from_gradient', got '{normal_method}'")

    # dA stats
    dA_vals = mesh.dA[near]
    print(f"{tag} dA stats: mean={float(dA_vals.mean()):.4f} "
          f"min={float(dA_vals.min()):.4f} max={float(dA_vals.max()):.4f} "
          f"sum={float(dA_vals.sum()):.1f}", flush=True)

    # ---- Wetted area normalization ----
    # dpS = 0.5 * u^2 * S_ref
    # S_ref = wetted surface area ≈ STL area (in lattice units)
    tri_areas = np.zeros(len(faces))
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    s2 = float(spacing[0] * spacing[1])
    S_wetted = float(tri_areas.sum()) / s2  # lattice units
    dpS = 0.5 * 1.0 * u_in ** 2 * S_wetted
    print(f"{tag} S_wetted(STL)={S_wetted:.1f} dpS={dpS:.6e}", flush=True)

    # Solid mask for NoDynamics
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # ---- Initialize flow field ----
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s) mass={initial_mass}", flush=True)

    # ---- BC config ----
    bc_config = {
        'far_field_faces': ['y-', 'y+', 'z-', 'z+'],
        'periodic_faces': [],
    }

    # ---- Time history ----
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup:
            fx_p, _, _ = drag_pressure_integration(f, mesh, dpS, extrap='quadratic')
            fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_p_hist.append(fx_p)
            cd_f_hist.append(fx_f)
            cd_tot_hist.append(fx_p + fx_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            if n_avg > 0:
                cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
                cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
                cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            else:
                cd_p_avg = cd_f_avg = cd_tot_avg = 0.0
            elapsed = time.time() - t0
            print(f"{tag} step={step} Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                  f"Cd_tot={cd_tot_avg:.6f} (ITTC={cf_ittc:.6f}) [{elapsed:.0f}s]",
                  flush=True)

    elapsed = time.time() - t0
    win = min(1000, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-win:]) / max(win, 1)
    cd_f_final = sum(cd_f_hist[-win:]) / max(win, 1)
    cd_tot_final = sum(cd_tot_hist[-win:]) / max(win, 1)

    err_ittc = abs(cd_tot_final - cf_ittc) / cf_ittc * 100 if cf_ittc > 0 else float('nan')

    result = {
        "benchmark": f"stl_ship_{ship_name}",
        "test_id": test_id,
        "device": f"sdaa:{device_id}",
        "ship": ship_name,
        "stl_file": str(stl_path),
        "lattice": "D3Q19",
        "collision": "MRT+Smagorinsky",
        "C_s": cs_smag,
        "normal_method": normal_method,
        "dA_method": dA_method,
        "nx": nx, "ny": ny, "nz": nz,
        "L_lattice": L_lattice,
        "spacing": float(spacing[0]),
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_samples": len(cd_tot_hist),
        "S_wetted": S_wetted,
        "dpS": dpS,
        "n_solid": hull_info["n_solid"],
        "n_near": hull_info["n_near"],
        "n_faces": hull_info["n_faces"],
        "Cf_ITTC": cf_ittc,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "error_vs_ITTC_pct": err_ittc,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
    }

    print(f"\n{'=' * 60}")
    print(f"{tag} FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Cd_pressure  = {cd_p_final:.6f}")
    print(f"  Cd_friction  = {cd_f_final:.6f}")
    print(f"  Cd_total     = {cd_tot_final:.6f}")
    print(f"  Cf_ITTC      = {cf_ittc:.6f}  (err={err_ittc:.1f}%)")
    print(f"  S_wetted     = {S_wetted:.1f}")
    print(f"  Wall time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Finite: {result['finite']}")

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"  Saved to {output_path}")

    return result


# -----------------------------------------------------------------------
# TEST 4: Sphere dA comparison
# -----------------------------------------------------------------------

def run_sphere_dA_comparison(test_id, device_id, output_path=None):
    """Compare dA=1.0 vs stl_area vs cos_theta on sphere drag."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id} Sphere-dA]"

    nx = ny = nz = 64
    cx = cy = cz = 32.0
    R = 12.0
    u_in = 0.05
    Re = 100
    nu = u_in * 2 * R / Re
    tau = 3.0 * nu + 0.5
    n_steps = 3000
    warmup = 500
    cs_smag = 0.1

    # Analytical reference (Stokes/Clift for Re=100):
    # Cd ≈ 1.09 (Clift table 4.2 for Re=100)
    cd_ref = 1.09

    print(f"{tag} Sphere dA comparison: R={R} Re={Re}", flush=True)
    print(f"{tag} grid={nx}x{ny}x{nz} u_in={u_in} nu={nu:.6e} tau={tau:.6f}", flush=True)

    # Build sphere STL
    verts, faces = make_sphere_stl((cx, cy, cz), R, n_lat=30, n_lon=60)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    fn = (cross / np.where(norms > 1e-10, norms, 1.0)).astype(np.float32)

    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 1.0)
    solid = voxelize_stl(verts, faces, (nx, ny, nz), origin, spacing)
    near = get_near_wall_3d(solid)
    solid = solid.to(device)
    near = near.to(device)

    # True surface area
    true_area = 4 * math.pi * R ** 2
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    stl_area = float(tri_areas.sum())

    print(f"{tag} true_area={true_area:.2f} stl_area={stl_area:.2f} "
          f"n_near={int(near.sum().item())}", flush=True)

    # dpS = 0.5 * u^2 * pi * R^2 (projected area)
    dpS = 0.5 * 1.0 * u_in ** 2 * math.pi * R ** 2

    sm = solid.unsqueeze(0).expand(19, nx, ny, nx)

    results = {}
    for dA_method in ["none", "stl_area", "cos_theta"]:
        tag_m = f"[SDAA:{device_id} Sphere-dA={dA_method}]"
        mesh = SurfaceMesh_from_stl(
            solid, near, verts, faces, fn, origin, spacing,
            dA_method=dA_method,
        )
        dA_vals = mesh.dA[near]
        sum_dA = float(dA_vals.sum().item())
        print(f"{tag_m} sum_dA={sum_dA:.2f} ratio_to_true={sum_dA / true_area:.4f}",
              flush=True)

        # Initialize flow
        rho0 = torch.ones((nx, ny, nx), device=device)
        ux0 = torch.full((nx, ny, nx), u_in, device=device)
        ux0[solid] = 0.0
        f = equilibrium3d(
            rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
        )
        initial_mass = float(rho0.sum().item())

        bc_config = {
            'far_field_faces': ['y-', 'y+', 'z-', 'z+'],
            'periodic_faces': [],
        }

        cd_p_hist = []
        cd_f_hist = []
        cd_tot_hist = []
        t0 = time.time()

        for step in range(1, n_steps + 1):
            f_pre = f.clone()
            f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
            for q in range(19):
                f[q] = torch.where(sm[q], f_pre[q], f[q])
            f = bounce_back_cells_3d(f, solid)
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)
            if step % 200 == 0:
                f = correct_mass3d(f, initial_mass)

            if step > warmup:
                fx_p, _, _ = drag_pressure_integration(f, mesh, dpS, extrap='quadratic')
                fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
                cd_p_hist.append(fx_p)
                cd_f_hist.append(fx_f)
                cd_tot_hist.append(fx_p + fx_f)

            if not torch.isfinite(f).all():
                print(f"{tag_m} DIVERGED at step {step}", flush=True)
                break

            if step % 500 == 0:
                n_avg = min(500, len(cd_tot_hist))
                if n_avg > 0:
                    cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
                else:
                    cd_tot_avg = 0.0
                print(f"{tag_m} step={step} Cd={cd_tot_avg:.6f} "
                      f"(ref={cd_ref:.4f}) [{time.time() - t0:.0f}s]", flush=True)

        elapsed = time.time() - t0
        win = min(1000, len(cd_tot_hist))
        cd_p_f = sum(cd_p_hist[-win:]) / max(win, 1)
        cd_f_f = sum(cd_f_hist[-win:]) / max(win, 1)
        cd_tot_f = sum(cd_tot_hist[-win:]) / max(win, 1)
        err = abs(cd_tot_f - cd_ref) / cd_ref * 100

        results[dA_method] = {
            "Cd_pressure": cd_p_f,
            "Cd_friction": cd_f_f,
            "Cd_total": cd_tot_f,
            "Cd_ref": cd_ref,
            "error_pct": err,
            "sum_dA": sum_dA,
            "ratio_to_true_area": sum_dA / true_area,
            "finite": bool(torch.isfinite(f).all().item()),
            "wall_time_s": elapsed,
        }

        print(f"{tag_m} FINAL: Cd_p={cd_p_f:.6f} Cd_f={cd_f_f:.6f} "
              f"Cd_tot={cd_tot_f:.6f} (ref={cd_ref:.4f}, err={err:.1f}%)",
              flush=True)

    result = {
        "benchmark": "sphere_dA_comparison",
        "test_id": test_id,
        "device": f"sdaa:{device_id}",
        "R": R,
        "Re": Re,
        "true_area": true_area,
        "stl_area": stl_area,
        "n_near": int(near.sum().item()),
        "results": results,
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"\n{tag} Saved to {output_path}")

    return result


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    did = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out = sys.argv[3] if len(sys.argv) > 3 else None

    if test_id == 1:
        # KVLCC2, Re=1e5
        run_ship_benchmark(
            test_id=1, device_id=did,
            stl_path=STL_DIR / "KVLCC2_Hull.stl",
            ship_name="KVLCC2",
            nx=200, ny=80, nz=80,
            Re=1e5, u_in=0.06,
            n_steps=5000, warmup=1000,
            cs_smag=0.05,
            dA_method="stl_area",
            output_path=out,
        )
    elif test_id == 2:
        # DTMB5415, Re=1e5
        run_ship_benchmark(
            test_id=2, device_id=did,
            stl_path=STL_DIR / "DTMB5415_Hull.stl",
            ship_name="DTMB5415",
            nx=200, ny=80, nz=80,
            Re=1e5, u_in=0.06,
            n_steps=5000, warmup=1000,
            cs_smag=0.05,
            dA_method="stl_area",
            output_path=out,
        )
    elif test_id == 3:
        # KCS, Re=1000
        run_ship_benchmark(
            test_id=3, device_id=did,
            stl_path=STL_DIR / "KCS_Hull.stl",
            ship_name="KCS",
            nx=200, ny=80, nz=80,
            Re=1000, u_in=0.06,
            n_steps=5000, warmup=500,
            cs_smag=0.05,
            dA_method="stl_area",
            output_path=out,
        )
    elif test_id == 4:
        # Sphere dA comparison
        run_sphere_dA_comparison(test_id=4, device_id=did, output_path=out)
    else:
        print(f"Unknown test_id: {test_id}. Use 1-4.")
        sys.exit(1)
