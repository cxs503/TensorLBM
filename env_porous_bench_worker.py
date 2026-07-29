#!/usr/bin/env python3
"""Askervein hill + cylinder array (porous media) benchmarks.

Benchmark 1: Askervein hill (simplified Gaussian hill, Re=1000)
  - nx=400, ny=200, nz=4
  - Hill: h(x) = H*exp(-(x/L)^2), H=30, L=100
  - u_in=0.05, Re=1000, tau=0.515
  - MRT+Smagorinsky (Cs=0.05), 10000 steps
  - Measure: u/u_in at hilltop, velocity field

Benchmark 2: Channel with cylinder array (porous media, Re=100)
  - nx=300, ny=200, nz=4
  - 3x3 array of cylinders, D=20, spacing=40
  - u_in=0.05, Re=100, tau=0.515
  - MRT+Smagorinsky (Cs=0.05), 5000 steps
  - Measure: pressure drop across array, drag per cylinder

Usage:
  python env_porous_bench_worker.py <benchmark> <device_id> [output_path]
  benchmark: askervein | cylinder_array
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
from tensorlbm.solver3d import stream3d, correct_mass3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def cylinder_mask_3d(nx, ny, nz, cx, cy, R, device):
    """Boolean mask for a 2D circular cylinder extruded in z.

    Shape: (nz, ny, nx).  The cross-section is a circle in the x-y plane.
    """
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing='ij')
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2


def get_near_wall_for_solid(total_solid, target_solid):
    """Near-wall mask for a specific solid region.

    Finds fluid cells (not in total_solid) that are adjacent to target_solid.
    This allows per-cylinder near-wall extraction in a multi-obstacle domain.
    """
    fluid = ~total_solid
    near = torch.zeros_like(total_solid)
    near[:, :, 1:-1] |= (target_solid[:, :, 2:] | target_solid[:, :, :-2]) & fluid[:, :, 1:-1]
    near[:, 1:-1, :] |= (target_solid[:, 2:, :] | target_solid[:, :-2, :]) & fluid[:, 1:-1, :]
    near[1:-1, :, :] |= (target_solid[2:, :, :] | target_solid[:-2, :, :]) & fluid[1:-1, :, :]
    return near


# ---------------------------------------------------------------------------
# Benchmark 1: Askervein hill
# ---------------------------------------------------------------------------

def run_askervein(device_id, output_path=None):
    """Simplified Askervein hill: Gaussian hill on bottom wall, Re=1000."""
    tag = f"[SDAA:{device_id} Askervein]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # ---- Parameters ----
    nx, ny, nz = 400, 200, 4
    H, L_hill = 30, 100
    x_center = nx / 2.0          # 200
    u_in = 0.05
    tau = 0.515
    cs_smag = 0.05
    n_steps = 10000
    nu = (tau - 0.5) / 3.0       # 0.005
    Re = u_in * L_hill / nu       # 1000

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} H={H} L={L_hill} x_center={x_center} "
        f"u_in={u_in} nu={nu:.6e} tau={tau} Re={Re:.0f} Cs={cs_smag} "
        f"n_steps={n_steps}",
        flush=True,
    )

    # ---- Hill solid mask: h(x) = H*exp(-((x-xc)/L)^2) ----
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing='ij')
    h_field = H * torch.exp(-((xx - x_center) / L_hill) ** 2)
    solid = yy < h_field

    n_solid = int(solid.sum().item())
    print(f"{tag} hill solid cells: {n_solid}", flush=True)

    # NoDynamics solid mask (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # BC config: top far-field, bottom solid (hill), z periodic
    bc_config = {
        'far_field_faces': ['y+'],
        'periodic_faces': ['z-', 'z+'],
    }

    # ---- Initialise: uniform inflow, zero inside solid ----
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    t0 = time.time()
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    # Hilltop measurement: first fluid cell above crest
    y_hilltop = int(round(float(H)))   # 30
    x_hilltop = int(round(x_center))    # 200
    u_hilltop_hist = []

    step_done = 0
    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells to pre-collision
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (before streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC (top far-field, inlet free-stream, outlet zero-grad)
        f = far_field_bc_3d(f, u_in, bc_config=bc_config)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Velocity measurement at hilltop (every 100 steps)
        if step % 100 == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            u_ht = float(ux[:, y_hilltop, x_hilltop].mean().item())
            u_hilltop_hist.append(u_ht)

        step_done = step

        # Divergence guard
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(5, len(u_hilltop_hist))
            u_avg = sum(u_hilltop_hist[-n_avg:]) / n_avg
            print(
                f"{tag} step={step} u_hilltop={u_avg:.6f} "
                f"speedup={u_avg / u_in:.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # Final averages (last 5 measurements ≈ last 500 steps)
    n_final = min(5, len(u_hilltop_hist))
    u_hilltop_final = sum(u_hilltop_hist[-n_final:]) / n_final
    speedup = u_hilltop_final / u_in

    # Save velocity field (averaged over z)
    rho, ux, uy, uz = macroscopic3d(f)
    vel_field = torch.stack([ux.mean(dim=0), uy.mean(dim=0)]).cpu().numpy()
    vel_path = None
    if output_path:
        vel_path = str(Path(output_path).with_suffix('.npy'))
        np.save(vel_path, vel_field)

    # Reference: linear theory speedup ≈ 1 + 2H/L
    ref_speedup = 1.0 + 2.0 * H / L_hill

    result = {
        "case": "askervein_hill",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Re": Re,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "steps_completed": step_done,
        "H": H,
        "L": L_hill,
        "x_center": x_center,
        "n_solid": n_solid,
        "u_hilltop": u_hilltop_final,
        "u_speedup": speedup,
        "u_ref": u_in,
        "ref_name": "Linear theory speedup ~ 1 + 2H/L",
        "ref_speedup": ref_speedup,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
        "time_per_step_ms": elapsed / max(step_done, 1) * 1000.0,
        "avg_window": n_final,
        "velocity_field": vel_path,
    }

    print(
        f"{tag} DONE u_hilltop={u_hilltop_final:.6f} speedup={speedup:.4f} "
        f"(ref={ref_speedup:.4f}) time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Benchmark 2: Cylinder array (porous media)
# ---------------------------------------------------------------------------

def run_cylinder_array(device_id, output_path=None):
    """3x3 cylinder array in a channel, Re=100, porous media benchmark."""
    tag = f"[SDAA:{device_id} CylinderArray]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # ---- Parameters ----
    nx, ny, nz = 300, 200, 4
    D = 20
    R = D / 2.0
    spacing = 40
    u_in = 0.05
    tau = 0.515
    cs_smag = 0.05
    n_steps = 5000
    nu = (tau - 0.5) / 3.0       # 0.005
    Re = u_in * D / nu            # 200 (with tau=0.515)

    # Cylinder centers (3x3 array, centered in domain)
    cx_center, cy_center = nx / 2.0, ny / 2.0   # 150, 100
    offsets = [-spacing, 0, spacing]
    cylinder_centers = []
    for dx in offsets:
        for dy in offsets:
            cylinder_centers.append((cx_center + dx, cy_center + dy))

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} D={D} R={R} spacing={spacing} "
        f"u_in={u_in} nu={nu:.6e} tau={tau} Re={Re:.0f} Cs={cs_smag} "
        f"n_cylinders={len(cylinder_centers)} n_steps={n_steps}",
        flush=True,
    )

    # ---- Create solid mask (cylinders + channel walls) ----
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    cylinder_solids = []
    for cx, cy in cylinder_centers:
        cyl_mask = cylinder_mask_3d(nx, ny, nz, cx, cy, R, device)
        cylinder_solids.append(cyl_mask)
        solid |= cyl_mask
    # Add channel walls (top/bottom)
    solid[:, 0, :] = True    # bottom wall
    solid[:, -1, :] = True   # top wall

    n_solid = int(solid.sum().item())
    n_solid_cyl = sum(int(s.sum().item()) for s in cylinder_solids)
    print(
        f"{tag} solid cells: {n_solid} (cylinders: {n_solid_cyl}, "
        f"walls: {n_solid - n_solid_cyl})",
        flush=True,
    )

    # ---- Per-cylinder near-wall masks and surface meshes ----
    meshes = []
    for i, (cx, cy) in enumerate(cylinder_centers):
        cyl_solid = cylinder_solids[i]
        near_i = get_near_wall_for_solid(solid, cyl_solid)
        mesh_i = SurfaceMesh.from_cylinder(solid, near_i, cx, cy, R, axis='z')
        meshes.append(mesh_i)
        n_near_i = int(near_i.sum().item())
        print(
            f"{tag} cylinder {i + 1} center=({cx},{cy}) near_cells={n_near_i}",
            flush=True,
        )

    # dpS for each cylinder (2D extruded, reference area = D * nz)
    dpS = 0.5 * u_in ** 2 * D * nz

    # NoDynamics solid mask
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # BC config: no far-field faces (walls in solid), z periodic
    bc_config = {
        'far_field_faces': [],
        'periodic_faces': ['z-', 'z+'],
    }

    # ---- Initialise: uniform inflow, zero inside solid ----
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    t0 = time.time()
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    # Pressure drop measurement planes
    x_inlet = 50     # well before array (array spans x~100 to x~200)
    x_outlet = 250   # well after array

    # History buffers
    dp_hist = []
    drag_hist = [[] for _ in cylinder_centers]

    step_done = 0
    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC (inlet free-stream, outlet zero-grad, z periodic)
        f = far_field_bc_3d(f, u_in, bc_config=bc_config)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Pressure drop + drag measurement (every 100 steps)
        if step % 100 == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            p = (rho - 1.0) / 3.0
            p_inlet = float(p[:, :, x_inlet].mean().item())
            p_outlet = float(p[:, :, x_outlet].mean().item())
            dp_hist.append(p_inlet - p_outlet)

            for i, mesh in enumerate(meshes):
                cd_p, _, _ = drag_pressure_integration(f, mesh, dpS)
                cd_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
                drag_hist[i].append((cd_p, cd_f, cd_p + cd_f))

        step_done = step

        # Divergence guard
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(5, len(dp_hist))
            dp_avg = sum(dp_hist[-n_avg:]) / n_avg
            print(
                f"{tag} step={step} dp={dp_avg:.6e} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # Final averages (last 5 measurements ≈ last 500 steps)
    n_final = min(5, len(dp_hist))
    dp_final = sum(dp_hist[-n_final:]) / n_final

    drag_final = []
    for i in range(len(cylinder_centers)):
        n_d = min(5, len(drag_hist[i]))
        if n_d > 0:
            cd_p_avg = sum(d[0] for d in drag_hist[i][-n_d:]) / n_d
            cd_f_avg = sum(d[1] for d in drag_hist[i][-n_d:]) / n_d
            cd_tot_avg = sum(d[2] for d in drag_hist[i][-n_d:]) / n_d
        else:
            cd_p_avg = cd_f_avg = cd_tot_avg = float('nan')
        drag_final.append({
            "cylinder": i + 1,
            "center": list(cylinder_centers[i]),
            "Cd_pressure": cd_p_avg,
            "Cd_friction": cd_f_avg,
            "Cd_total": cd_tot_avg,
        })

    # Save velocity + pressure fields (averaged over z)
    rho, ux, uy, uz = macroscopic3d(f)
    p = (rho - 1.0) / 3.0
    fields = torch.stack([ux.mean(dim=0), uy.mean(dim=0), p.mean(dim=0)]).cpu().numpy()
    field_path = None
    if output_path:
        field_path = str(Path(output_path).with_suffix('.npy'))
        np.save(field_path, fields)

    result = {
        "case": "cylinder_array",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Re": Re,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "steps_completed": step_done,
        "D": D,
        "R": R,
        "spacing": spacing,
        "n_cylinders": len(cylinder_centers),
        "cylinder_centers": [list(c) for c in cylinder_centers],
        "n_solid": n_solid,
        "dpS": dpS,
        "dpS_formula": "0.5*u_in^2*D*nz (2D extruded cylinder)",
        "pressure_drop": dp_final,
        "p_inlet_x": x_inlet,
        "p_outlet_x": x_outlet,
        "drag_per_cylinder": drag_final,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
        "time_per_step_ms": elapsed / max(step_done, 1) * 1000.0,
        "avg_window": n_final,
        "field_file": field_path,
    }

    print(f"{tag} DONE dp={dp_final:.6e} time={elapsed:.0f}s", flush=True)
    for d in drag_final:
        print(
            f"{tag} cyl {d['cylinder']} Cd_p={d['Cd_pressure']:.6f} "
            f"Cd_f={d['Cd_friction']:.6f} Cd_tot={d['Cd_total']:.6f}",
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
    if len(sys.argv) < 3:
        print("Usage: python env_porous_bench_worker.py <benchmark> <device_id> [output_path]")
        print("  benchmark: askervein | cylinder_array")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    if benchmark == "askervein":
        run_askervein(device_id, output_path)
    elif benchmark == "cylinder_array":
        run_cylinder_array(device_id, output_path)
    else:
        print(f"Unknown benchmark: {benchmark}")
        sys.exit(1)


if __name__ == "__main__":
    main()
