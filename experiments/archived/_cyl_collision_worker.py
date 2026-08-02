"""Cylinder Re=200 collision-model comparison worker (D3Q19).

Runs one collision model on one SDAA card and writes a JSON result.
Uses the verified NoDynamics + half-way bounce-back main loop
(lbm_step_correct pattern) with pressure-integration drag (analytical normal).

Usage:
    python _cyl_collision_worker.py <device_id> <collision_name> <output_path>
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

# ---- Collision dispatch ----------------------------------------------------
def make_collide(name):
    """Return a callable f -> f_post for the named collision model."""
    if name == "BGK":
        from tensorlbm.solver3d import collide_bgk3d
        return lambda f, tau: collide_bgk3d(f, tau=tau)
    if name == "MRT":
        from tensorlbm.solver3d import collide_mrt3d
        return lambda f, tau: collide_mrt3d(f, tau=tau)
    if name == "MRT+Smag":
        from tensorlbm.turbulence import collide_smagorinsky_mrt3d
        return lambda f, tau: collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.05)
    if name == "Cumulant":
        from tensorlbm.cumulant import collide_cumulant_d3q19
        return lambda f, tau: collide_cumulant_d3q19(f, tau)
    raise ValueError(f"Unknown collision: {name}")


def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2  # (ny, nx)
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()  # (nz, ny, nx)
    return solid


def main():
    did = int(sys.argv[1])
    collision_name = sys.argv[2]
    output_path = sys.argv[3]

    # ---- Problem parameters ------------------------------------------------
    nx, ny, nz = 400, 160, 4
    D = 48.0
    R = D / 2.0
    u_in = 0.08
    Re = 200.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 3000
    win = 300  # averaging window (last 300 steps)
    ref_cd = 1.30

    # Frontal area = D * nz (cylinder extruded along z, flow in x)
    A_frontal = D * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{did} {collision_name}]"
    print(
        f"{tag} nx={nx} ny={ny} nz={nz} D={D} u_in={u_in} "
        f"nu={nu:.6e} tau={tau:.6f} Re={Re}",
        flush=True,
    )

    t0 = time.time()

    # ---- Geometry ---------------------------------------------------------
    cx_cyl = nx * 0.25  # 100
    cy_cyl = ny * 0.5   # 80
    solid = build_cylinder_mask(nx, ny, nz, cx_cyl, cy_cyl, R, device)

    # ---- Precompute surface mesh for pressure integration -----------------
    from tensorlbm.drag_pressure import (
        get_near_wall_2d,
        SurfaceMesh,
        drag_pressure_integration,
    )
    near = get_near_wall_2d(solid, axis="z")
    mesh = SurfaceMesh.from_cylinder(solid, near, cx_cyl, cy_cyl, R, axis="z")
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)  # (19, nz, ny, nx)

    # ---- Initialise flow field --------------------------------------------
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.solver3d import stream3d, correct_mass3d
    from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    initial_mass = float(rho0.sum().item())

    collide_fn = make_collide(collision_name)

    print(f"{tag} init done ({time.time() - t0:.1f}s)", flush=True)

    # ---- Time the LBM step separately from drag measurement ---------------
    cd_hist = []
    cl_hist = []
    lbm_time = 0.0
    all_finite = True
    final_step = 0

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (all cells)
        ts = time.time()
        f = collide_fn(f, tau)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)
        lbm_time += time.time() - ts

        # 8. Drag / lift via pressure integration (analytical normal)
        cd, cl, _ = drag_pressure_integration(f, mesh, dpS)
        cd_hist.append(cd)
        cl_hist.append(cl)
        final_step = step

        # 9. Divergence check
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            all_finite = False
            break

        if step % 500 == 0 or step == n_steps:
            elapsed = time.time() - t0
            print(
                f"{tag} step={step} Cd={cd:.4f} Cl={cl:.4f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # ---- Final statistics: average over last `win` steps -------------------
    tail_cd = cd_hist[-win:] if len(cd_hist) >= win else cd_hist
    tail_cl = cl_hist[-win:] if len(cl_hist) >= win else cl_hist

    cd_mean = sum(tail_cd) / max(len(tail_cd), 1)
    cl_mean = sum(tail_cl) / max(len(tail_cl), 1)
    cl_max = max(tail_cl) if tail_cl else 0.0
    cl_min = min(tail_cl) if tail_cl else 0.0
    cl_amp = (cl_max - cl_min) / 2.0  # peak-to-peak amplitude
    # RMS amplitude (more robust for oscillating signal)
    cl_rms = math.sqrt(sum(c * c for c in tail_cl) / max(len(tail_cl), 1))

    err_pct = abs(cd_mean - ref_cd) / ref_cd * 100 if ref_cd > 0 else float("nan")
    time_per_step = lbm_time / max(final_step, 1)

    result = {
        "collision": collision_name,
        "device": f"sdaa:{did}",
        "lattice": "D3Q19",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D,
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "win": win,
        "Cd": cd_mean,
        "Cl_mean": cl_mean,
        "Cl_amp": cl_amp,
        "Cl_rms": cl_rms,
        "Cd_ref": ref_cd,
        "error_pct": err_pct,
        "time_per_step_s": time_per_step,
        "lbm_time_s": lbm_time,
        "total_elapsed_s": elapsed,
        "steps_run": final_step,
        "finite": all_finite,
    }
    print(
        f"{tag} DONE Cd={cd_mean:.4f} (ref={ref_cd}) err={err_pct:.1f}% "
        f"Cl_amp={cl_amp:.4f} t/step={time_per_step:.4f}s ({elapsed:.0f}s)",
        flush=True,
    )
    Path(output_path).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
