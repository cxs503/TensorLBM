"""y↔z swap verification: direction-agnostic preprocessing.

Runs the same cylinder flow case with two axis orientations:
  axis='z': cylinder in x-y plane, z extruded, y±=far_field, z=periodic  (SDAA:8)
  axis='y': cylinder in x-z plane, y extruded, z±=far_field, y=periodic  (SDAA:9)

Both should give the SAME Cd (within 1% difference).
If different → z-direction bug in preprocessing.

Test: D=24, Re=200, 3000 steps, pressure integration (BB + pressure drag).

Usage:
    PYTHONPATH=src python _axis_swap_test.py <device_id> <axis> <output_json>
    axis: z or y
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.solver3d import stream3d, correct_mass3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, drag_pressure_integration, get_near_wall_2d,
)


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
    elif axis == 'x':
        cz_c = cz if cz is not None else nz / 2.0
        zz, yy = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(ny, device=device, dtype=torch.float32),
            indexing="ij",
        )
        circle = (yy - cy) ** 2 + (zz - cz_c) ** 2 <= radius ** 2
        solid = circle.unsqueeze(2).expand(nz, ny, nx).clone()
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
    return solid


def run_axis_test(device_id, axis, output_path):
    """Run cylinder flow with given axis orientation, pressure-integration drag."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # Parameters
    diameter = 24.0
    radius = diameter / 2.0
    u_in = 0.08
    Re = 200.0
    nu = u_in * diameter / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    n_steps = 3000
    warmup = 300

    # Grid: 10D x 4D in the cross-flow plane, 4 layers along axis
    n_cross = int(4 * diameter)   # 96
    nx = int(10 * diameter)        # 240
    n_axis = 4                     # 4 layers (2D)

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

    tag = f"[axis={axis} SDAA:{device_id}]"
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

        # 7. Pressure drag (post-stream, post-BC)
        cd_p, _ = drag_pressure_integration(f, mesh, dpS)

        # 8. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Record post-warmup
        if step > warmup:
            if math.isfinite(cd_p):
                cd_p_hist.append(cd_p)

        if step % 500 == 0:
            _, ux, _, _ = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux).max().item())
            elapsed = time.time() - t0
            cd_avg = sum(cd_p_hist) / max(len(cd_p_hist), 1)
            print(f"{tag} step={step} Cd_p={cd_avg:.4f} "
                  f"max|ux|={ms:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0

    cd_p_mean = sum(cd_p_hist) / max(len(cd_p_hist), 1) if cd_p_hist else float("nan")

    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

    cd_p_std = _std(cd_p_hist)

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cd_p = {cd_p_mean:.4f} ± {cd_p_std:.4f}", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
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
        "Cd_p_std": cd_p_std,
        "n_samples": len(cd_p_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    device_id = int(sys.argv[1])
    axis = sys.argv[2].lower()
    output_path = sys.argv[3]
    run_axis_test(device_id, axis, output_path)
