#!/usr/bin/env python3
"""STL ship hull benchmark via common interface — Bug 29 fix + Wigley + Series 60.

Tests (SDAA 20-23):
  TEST 1 (SDAA:20): KVLCC2 STL, Re=1e5, MRT+Smag, 5000 steps, STL normals
                    (Bug 29 centroid-direction check). Previous: Cd≈0 (cancelled).
  TEST 2 (SDAA:21): DTMB5415 STL, Re=1000, MRT+Smag, 5000 steps, STL normals.
                    Previous: Cd≈0.0008 (cancelled).
  TEST 3 (SDAA:22): Wigley analytical hull, Re=1000, L=80, from_gradient normals.
  TEST 4 (SDAA:23): Series 60 Cb=0.60, Re=1000, compare from_gradient vs from_stl.

CRITICAL: Uses common interface ONLY:
  - read_stl / voxelize_stl / SurfaceMesh_from_stl  (stl_geometry.py)
  - get_near_wall_3d, drag_pressure_integration, drag_friction_integration
  - lbm_step_correct(), far_field_bc_3d, bounce_back_cells_3d(f_pre)
  NO custom LBM code. Common modules only.

Usage:
  PYTHONPATH=src python bug29_ship_common_worker.py <test_id> <device_id> <output_path>
  test_id: 1=KVLCC2, 2=DTMB5415, 3=Wigley, 4=Series60
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
from tensorlbm.lbm_step_correct import lbm_step_correct
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
)
from tensorlbm.ship_cad import (
    build_hull_mask,
    ShipHullType,
    export_hull_stl,
    hull_block_coefficient,
)
from tensorlbm.obstacles import wigley_hull_mask

STL_DIR = Path(
    "/root/ship-performance-platform-incoming/ship-performance-platform/"
    "backend/data/geometry/ships"
)


# -----------------------------------------------------------------------
# Grid mapping for STL hulls
# -----------------------------------------------------------------------

def setup_stl_ship_grid(stl_path, nx, ny, nz, L_lattice_target=None):
    """Read, mirror, and map an STL half-hull onto the LBM grid.

    Uses common interface: read_stl → mirror_stl → voxelize_stl → get_near_wall_3d
    """
    vertices, faces, face_normals = read_stl(stl_path)

    vertices_full, faces_full, normals_full = mirror_stl(
        vertices, faces, face_normals, axis=1
    )

    x_min, x_max = vertices_full[:, 0].min(), vertices_full[:, 0].max()
    y_min, y_max = vertices_full[:, 1].min(), vertices_full[:, 1].max()
    z_min, z_max = vertices_full[:, 2].min(), vertices_full[:, 2].max()
    L_stl = x_max - x_min
    B_stl = y_max - y_min
    D_stl = z_max - z_min

    if L_lattice_target is None:
        L_lattice_target = nx * 0.6
    spacing = L_stl / L_lattice_target
    L_lattice = L_lattice_target

    hull_center_x_stl = (x_min + x_max) / 2.0
    origin_x = hull_center_x_stl - (nx * 0.35) * spacing
    hull_center_y_stl = (y_min + y_max) / 2.0
    origin_y = hull_center_y_stl - (ny * 0.5) * spacing
    origin_z = 0.0 - (nz * 0.5) * spacing

    origin = (origin_x, origin_y, origin_z)
    sp = (spacing, spacing, spacing)

    solid = voxelize_stl(vertices_full, faces_full, (nx, ny, nz), origin, sp)
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

    return (solid, near, vertices_full, faces_full, normals_full,
            origin, sp, L_lattice, hull_info)


# -----------------------------------------------------------------------
# Core LBM ship benchmark runner — uses lbm_step_correct() common interface
# -----------------------------------------------------------------------

def run_ship_lbm(
    test_id, device_id, ship_name,
    solid, near, mesh, dpS, S_wetted,
    nx, ny, nz, L_lattice, Re, u_in, n_steps, warmup, cs_smag,
    nu, tau, normal_method, dA_method, hull_info, output_path=None,
):
    """Run the LBM simulation loop using lbm_step_correct() and return drag."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id} {ship_name}]"

    cf_ittc = 0.075 / (math.log10(Re) - 2.0) ** 2

    print(f"{tag} Bug 29 retest via common interface (lbm_step_correct)", flush=True)
    print(f"{tag} grid={nx}x{ny}x{nz} L_lat={L_lattice} "
          f"spacing={hull_info.get('spacing', 1.0):.4f}", flush=True)
    print(f"{tag} u_in={u_in} Re={Re:.0e} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag}",
          flush=True)
    print(f"{tag} normal_method={normal_method} dA_method={dA_method}", flush=True)
    print(f"{tag} solid={hull_info.get('n_solid', '?')} "
          f"near={hull_info.get('n_near', '?')} "
          f"faces={hull_info.get('n_faces', '?')}", flush=True)
    print(f"{tag} Cf_ITTC={cf_ittc:.6f} S_wetted={S_wetted:.1f} dpS={dpS:.6e}",
          flush=True)

    # dA stats
    dA_vals = mesh.dA[near]
    print(f"{tag} dA stats: mean={float(dA_vals.mean()):.4f} "
          f"min={float(dA_vals.min()):.4f} max={float(dA_vals.max()):.4f} "
          f"sum={float(dA_vals.sum()):.1f}", flush=True)

    # Normal direction sanity check (Bug 29 verification)
    solid_cpu = solid.cpu()
    solid_coords = np.argwhere(solid_cpu.numpy())
    if len(solid_coords) > 0:
        solid_center = solid_coords.mean(axis=0)
        near_idx = near.cpu().nonzero(as_tuple=False).numpy()
        n_near = len(near_idx)
        nx_n = mesh.nx_n.cpu().numpy()
        ny_n = mesh.ny_n.cpu().numpy()
        nz_n = mesh.nz_n.cpu().numpy()
        sample = np.random.RandomState(42).choice(
            n_near, min(1000, n_near), replace=False
        )
        outward_count = 0
        for s in sample:
            iz, iy, ix = near_idx[s]
            cell_pos = np.array([iz, iy, ix], dtype=np.float64)
            to_center = solid_center - cell_pos
            to_center_norm = np.linalg.norm(to_center)
            if to_center_norm > 1e-10:
                to_center_dir = to_center / to_center_norm
            else:
                to_center_dir = np.array([0.0, 0.0, 0.0])
            normal = np.array([nx_n[iz, iy, ix], ny_n[iz, iy, ix],
                               nz_n[iz, iy, ix]])
            dot = np.dot(normal, to_center_dir)
            if dot < 0:
                outward_count += 1
        pct_outward = 100.0 * outward_count / len(sample)
        print(f"{tag} Normal orientation check: {pct_outward:.1f}% outward "
              f"({outward_count}/{len(sample)} sampled)", flush=True)

    t0 = time.time()

    solid = solid.to(device)
    near = near.to(device)
    # Move mesh tensors to device (built on CPU by SurfaceMesh_from_stl)
    mesh.near = mesh.near.to(device)
    mesh.nx_n = mesh.nx_n.to(device)
    mesh.ny_n = mesh.ny_n.to(device)
    mesh.nz_n = mesh.nz_n.to(device)
    mesh.dA = mesh.dA.to(device)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s) mass={initial_mass}",
          flush=True)

    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []

    for step in range(1, n_steps + 1):
        # ---- Common interface: lbm_step_correct() ----
        # Internally does: collision → NoDynamics → bounce_back_cells_3d(f_pre)
        # → stream3d → far_field_bc_3d → mass correction
        f = lbm_step_correct(
            f,
            collide_fn=collide_smagorinsky_mrt3d,
            tau=tau,
            solid=solid,
            u_in=u_in,
            far_field_bc_fn=far_field_bc_3d,
            correct_mass_fn=correct_mass3d,
            target_mass=initial_mass,
            step=step,
            mass_interval=200,
            C_s=cs_smag,
        )

        if step > warmup:
            fx_p, _, _ = drag_pressure_integration(
                f, mesh, dpS, extrap='quadratic',
                p0_method='far_field', solid=solid,
            )
            fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_p_hist.append(float(fx_p))
            cd_f_hist.append(float(fx_f))
            cd_tot_hist.append(float(fx_p + fx_f))

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

    err_ittc = (abs(cd_tot_final - cf_ittc) / cf_ittc * 100
                if cf_ittc > 0 else float('nan'))

    result = {
        "benchmark": f"bug29_common_{ship_name}",
        "test_id": test_id,
        "device": f"sdaa:{device_id}",
        "ship": ship_name,
        "lattice": "D3Q19",
        "collision": "MRT+Smagorinsky",
        "step_function": "lbm_step_correct",
        "C_s": cs_smag,
        "normal_method": normal_method,
        "dA_method": dA_method,
        "nx": nx, "ny": ny, "nz": nz,
        "L_lattice": L_lattice,
        "spacing": float(hull_info.get("spacing", 1.0)),
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_samples": len(cd_tot_hist),
        "S_wetted": S_wetted,
        "dpS": dpS,
        "n_solid": hull_info.get("n_solid", 0),
        "n_near": hull_info.get("n_near", 0),
        "n_faces": hull_info.get("n_faces", 0),
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
# TEST 1: KVLCC2 with Bug 29 fix (SDAA:20)
# -----------------------------------------------------------------------

def run_test1_kvlcc2(device_id, output_path=None):
    """KVLCC2 STL hull, Re=1e5, MRT+Smag, 5000 steps, STL normals."""
    nx, ny, nz = 200, 80, 80
    Re = 1e5
    u_in = 0.06
    n_steps = 5000
    warmup = 1000
    cs_smag = 0.05

    (solid, near, vertices, faces, normals,
     origin, spacing, L_lattice, hull_info) = setup_stl_ship_grid(
        STL_DIR / "KVLCC2_Hull.stl", nx, ny, nz
    )

    nu = u_in * L_lattice / Re
    tau = 3.0 * nu + 0.5

    # Build surface mesh with STL normals (Bug 29 centroid check)
    mesh = SurfaceMesh_from_stl(
        solid, near, vertices, faces, normals.astype(np.float32),
        origin, spacing, dA_method="stl_area",
    )

    # Wetted area from STL
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    s2 = float(spacing[0] * spacing[1])
    S_wetted = float(tri_areas.sum()) / s2
    dpS = 0.5 * 1.0 * u_in ** 2 * S_wetted

    return run_ship_lbm(
        1, device_id, "KVLCC2",
        solid, near, mesh, dpS, S_wetted,
        nx, ny, nz, L_lattice, Re, u_in, n_steps, warmup, cs_smag,
        nu, tau, "stl", "stl_area", hull_info, output_path,
    )


# -----------------------------------------------------------------------
# TEST 2: DTMB5415 with Bug 29 fix (SDAA:21)
# -----------------------------------------------------------------------

def run_test2_dtmb5415(device_id, output_path=None):
    """DTMB5415 STL hull, Re=1000, MRT+Smag, 5000 steps, STL normals."""
    nx, ny, nz = 200, 80, 80
    Re = 1000
    u_in = 0.06
    n_steps = 5000
    warmup = 1000
    cs_smag = 0.05

    (solid, near, vertices, faces, normals,
     origin, spacing, L_lattice, hull_info) = setup_stl_ship_grid(
        STL_DIR / "DTMB5415_Hull.stl", nx, ny, nz
    )

    nu = u_in * L_lattice / Re
    tau = 3.0 * nu + 0.5

    mesh = SurfaceMesh_from_stl(
        solid, near, vertices, faces, normals.astype(np.float32),
        origin, spacing, dA_method="stl_area",
    )

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    s2 = float(spacing[0] * spacing[1])
    S_wetted = float(tri_areas.sum()) / s2
    dpS = 0.5 * 1.0 * u_in ** 2 * S_wetted

    return run_ship_lbm(
        2, device_id, "DTMB5415",
        solid, near, mesh, dpS, S_wetted,
        nx, ny, nz, L_lattice, Re, u_in, n_steps, warmup, cs_smag,
        nu, tau, "stl", "stl_area", hull_info, output_path,
    )


# -----------------------------------------------------------------------
# TEST 3: Wigley hull (SDAA:22)
# -----------------------------------------------------------------------

def run_test3_wigley(device_id, output_path=None):
    """Wigley analytical hull, Re=1000, L=80, from_gradient normals."""
    nx, ny, nz = 200, 80, 80
    L = 80.0
    Re = 1000
    u_in = 0.06
    n_steps = 5000
    warmup = 1000
    cs_smag = 0.05

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # Wigley hull parameters
    cx = nx * 0.30
    cy = ny * 0.5
    cz_keel = nz * 0.5 - 10
    beam = ny * 0.20   # 16
    draft = nz * 0.15  # 12

    solid = wigley_hull_mask(
        nx, ny, nz, cx, cy, cz_keel, L, beam, draft, device
    )
    near = get_near_wall_3d(solid)

    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())

    cb = hull_block_coefficient(solid, beam=beam, draft=draft, length=L)

    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5

    # from_gradient normals (analytical hull)
    mesh = SurfaceMesh.from_gradient(solid, near)

    # Wetted area: approximate from voxel surface
    S_wetted = float(n_near)  # lattice units (each cell ~1 face)
    dpS = 0.5 * 1.0 * u_in ** 2 * S_wetted

    hull_info = {
        "n_solid": n_solid,
        "n_near": n_near,
        "n_faces": 0,
        "spacing": 1.0,
        "L_stl": L,
        "B_stl": beam,
        "D_stl": draft,
        "Cb": float(cb),
    }

    cf_ittc = 0.075 / (math.log10(Re) - 2.0) ** 2

    print(f"[SDAA:{device_id} Wigley] Analytical hull benchmark", flush=True)
    print(f"[SDAA:{device_id} Wigley] grid={nx}x{ny}x{nz} L={L} "
          f"B={beam} T={draft}", flush=True)
    print(f"[SDAA:{device_id} Wigley] u_in={u_in} Re={Re:.0e} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag}", flush=True)
    print(f"[SDAA:{device_id} Wigley] normal_method=from_gradient "
          f"dA_method=none", flush=True)
    print(f"[SDAA:{device_id} Wigley] solid={n_solid} near={n_near} "
          f"Cb={cb:.4f} (theoretical ~0.444)", flush=True)
    print(f"[SDAA:{device_id} Wigley] Cf_ITTC={cf_ittc:.6f} "
          f"S_wetted={S_wetted:.1f} dpS={dpS:.6e}", flush=True)

    return run_ship_lbm(
        3, device_id, "Wigley",
        solid, near, mesh, dpS, S_wetted,
        nx, ny, nz, L, Re, u_in, n_steps, warmup, cs_smag,
        nu, tau, "from_gradient", "none", hull_info, output_path,
    )


# -----------------------------------------------------------------------
# TEST 4: Series 60 Cb=0.60 (SDAA:23)
# -----------------------------------------------------------------------

def run_test4_series60(device_id, output_path=None):
    """Series 60 Cb=0.60, Re=1000, compare from_gradient vs from_stl normals."""
    nx, ny, nz = 200, 80, 80
    L = 80.0
    Re = 1000
    u_in = 0.06
    n_steps = 5000
    warmup = 1000
    cs_smag = 0.05

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # Build Series 60 hull mask
    cx = nx * 0.30
    cy = ny * 0.5
    cz_keel = nz * 0.5 - 10
    beam = ny * 0.20   # 16
    draft = nz * 0.15  # 12

    solid, stats = build_hull_mask(
        hull_type=ShipHullType.SERIES60,
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz_keel=cz_keel,
        length=L, beam=beam, draft=draft,
        device=str(device),
    )
    near = get_near_wall_3d(solid)

    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    cb = stats.get("Cb_numerical", 0)

    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5

    # Wetted area (voxel-based)
    S_wetted = float(n_near)
    dpS = 0.5 * 1.0 * u_in ** 2 * S_wetted

    hull_info = {
        "n_solid": n_solid,
        "n_near": n_near,
        "n_faces": 0,
        "spacing": 1.0,
        "L_stl": L,
        "B_stl": beam,
        "D_stl": draft,
        "Cb": float(cb),
    }

    cf_ittc = 0.075 / (math.log10(Re) - 2.0) ** 2

    print(f"[SDAA:{device_id} Series60] Cb=0.60 hull benchmark", flush=True)
    print(f"[SDAA:{device_id} Series60] grid={nx}x{ny}x{nz} L={L} "
          f"B={beam} T={draft}", flush=True)
    print(f"[SDAA:{device_id} Series60] u_in={u_in} Re={Re:.0e} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag}", flush=True)
    print(f"[SDAA:{device_id} Series60] solid={n_solid} near={n_near} "
          f"Cb={cb:.4f} (theoretical ~0.60)", flush=True)
    print(f"[SDAA:{device_id} Series60] Cf_ITTC={cf_ittc:.6f} "
          f"S_wetted={S_wetted:.1f} dpS={dpS:.6e}", flush=True)

    # ---- Run A: from_gradient normals ----
    print(f"\n[SDAA:{device_id} Series60] === RUN A: from_gradient normals ===",
          flush=True)
    mesh_grad = SurfaceMesh.from_gradient(solid, near)

    result_a = run_ship_lbm(
        4, device_id, "Series60_grad",
        solid, near, mesh_grad, dpS, S_wetted,
        nx, ny, nz, L, Re, u_in, n_steps, warmup, cs_smag,
        nu, tau, "from_gradient", "none", hull_info,
        output_path.replace(".json", "_grad.json") if output_path else None,
    )

    # ---- Run B: from_stl normals ----
    # Export Series 60 hull to STL, then voxelize and use from_stl
    print(f"\n[SDAA:{device_id} Series60] === RUN B: from_stl normals ===",
          flush=True)

    stl_path = Path(f"/tmp/series60_hull_{device_id}.stl")
    export_hull_stl(
        hull_type=ShipHullType.SERIES60,
        length=L, beam=beam, draft=draft,
        n_long=60, n_vert=30,
        output_path=stl_path,
    )
    print(f"[SDAA:{device_id} Series60] Exported STL: {stl_path}", flush=True)

    # Read STL and voxelize (the STL is a full hull, no mirror needed)
    vertices_stl, faces_stl, normals_stl = read_stl(stl_path)
    print(f"[SDAA:{device_id} Series60] STL: {len(vertices_stl)} verts, "
          f"{len(faces_stl)} faces", flush=True)

    # Map STL onto the same grid as the analytical hull
    origin_x = -(cx - L / 2.0)
    origin_y = -(cy - 0.0)  # STL y=0 → iy = cy
    origin_z = cz_keel       # STL z=0 → iz = cz_keel
    origin_stl = (origin_x, origin_y, origin_z)
    spacing_stl = (1.0, 1.0, 1.0)

    solid_stl = voxelize_stl(
        vertices_stl, faces_stl, (nx, ny, nz), origin_stl, spacing_stl
    )
    near_stl = get_near_wall_3d(solid_stl)

    n_solid_stl = int(solid_stl.sum().item())
    n_near_stl = int(near_stl.sum().item())
    print(f"[SDAA:{device_id} Series60] STL voxelized: "
          f"solid={n_solid_stl} near={n_near_stl} "
          f"(analytical: solid={n_solid} near={n_near})", flush=True)

    # Use the STL-voxelized solid for from_stl normals
    mesh_stl = SurfaceMesh_from_stl(
        solid_stl, near_stl, vertices_stl, faces_stl,
        normals_stl.astype(np.float32),
        origin_stl, spacing_stl, dA_method="stl_area",
    )

    # STL wetted area
    v0 = vertices_stl[faces_stl[:, 0]]
    v1 = vertices_stl[faces_stl[:, 1]]
    v2 = vertices_stl[faces_stl[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    S_wetted_stl = float(tri_areas.sum())
    dpS_stl = 0.5 * 1.0 * u_in ** 2 * S_wetted_stl

    hull_info_stl = {
        "n_solid": n_solid_stl,
        "n_near": n_near_stl,
        "n_faces": len(faces_stl),
        "spacing": 1.0,
        "L_stl": L,
        "B_stl": beam,
        "D_stl": draft,
        "Cb": float(cb),
    }

    result_b = run_ship_lbm(
        4, device_id, "Series60_stl",
        solid_stl, near_stl, mesh_stl, dpS_stl, S_wetted_stl,
        nx, ny, nz, L, Re, u_in, n_steps, warmup, cs_smag,
        nu, tau, "stl", "stl_area", hull_info_stl,
        output_path.replace(".json", "_stl.json") if output_path else None,
    )

    # ---- Comparison ----
    comparison = {
        "benchmark": "series60_normal_comparison_common",
        "test_id": 4,
        "device": f"sdaa:{device_id}",
        "step_function": "lbm_step_correct",
        "Re": Re,
        "L": L,
        "Cb_numerical": cb,
        "from_gradient": {
            "Cd_pressure": result_a["Cd_pressure"],
            "Cd_friction": result_a["Cd_friction"],
            "Cd_total": result_a["Cd_total"],
            "n_solid": result_a["n_solid"],
            "n_near": result_a["n_near"],
        },
        "from_stl": {
            "Cd_pressure": result_b["Cd_pressure"],
            "Cd_friction": result_b["Cd_friction"],
            "Cd_total": result_b["Cd_total"],
            "n_solid": result_b["n_solid"],
            "n_near": result_b["n_near"],
            "n_faces": result_b["n_faces"],
        },
        "delta_Cd_p": result_b["Cd_pressure"] - result_a["Cd_pressure"],
        "delta_Cd_f": result_b["Cd_friction"] - result_a["Cd_friction"],
        "delta_Cd_tot": result_b["Cd_total"] - result_a["Cd_total"],
    }

    print(f"\n{'=' * 60}")
    print(f"[SDAA:{device_id} Series60] NORMAL METHOD COMPARISON")
    print(f"{'=' * 60}")
    print(f"  from_gradient: Cd_p={result_a['Cd_pressure']:.6f} "
          f"Cd_f={result_a['Cd_friction']:.6f} "
          f"Cd_tot={result_a['Cd_total']:.6f}")
    print(f"  from_stl:      Cd_p={result_b['Cd_pressure']:.6f} "
          f"Cd_f={result_b['Cd_friction']:.6f} "
          f"Cd_tot={result_b['Cd_total']:.6f}")
    print(f"  delta:          dCd_p={comparison['delta_Cd_p']:.6f} "
          f"dCd_f={comparison['delta_Cd_f']:.6f} "
          f"dCd_tot={comparison['delta_Cd_tot']:.6f}")

    if output_path:
        comp_path = output_path.replace(".json", "_comparison.json")
        Path(comp_path).write_text(json.dumps(comparison, indent=2))
        print(f"  Comparison saved to {comp_path}")

    return comparison


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    did = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out = sys.argv[3] if len(sys.argv) > 3 else None

    if test_id == 1:
        run_test1_kvlcc2(did, out)
    elif test_id == 2:
        run_test2_dtmb5415(did, out)
    elif test_id == 3:
        run_test3_wigley(did, out)
    elif test_id == 4:
        run_test4_series60(did, out)
    else:
        print(f"Unknown test_id: {test_id}. Use 1-4.")
        sys.exit(1)
