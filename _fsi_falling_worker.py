#!/usr/bin/env python3
"""FSI Test 4: Free falling sphere (neutrally buoyant).

SDAA:27  |  Re=100, D=40, m*=1.5 (neutrally buoyant)

Pipeline (common-interface only):
  solid → get_near_wall_3d → SurfaceMesh.from_sphere →
  lbm_step_correct → fsi_step_drag (drag_pressure + drag_friction → rigid body) →
  shift_solid_mask (moving boundary)

The sphere falls under gravity; fluid drag opposes gravity.  At terminal
velocity, drag = gravity force:
    F_drag = m * g  →  Cd = 2*m*g / (rho * U^2 * A)

Reference: Stokes drag Cd = 24/Re (laminar, Re=100 → Cd ≈ 0.24 from Stokes,
actual Cd ≈ 1.09 from experiments at Re=100).

Target: terminal velocity within 20% of reference.

Usage:
  python _fsi_falling_worker.py <device_id> [output_path]
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
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
)
from tensorlbm.fsi_common import (
    SpringMassDamper,
    SpringMassState,
    fsi_step_drag,
    shift_solid_mask,
)
from tensorlbm.sixdof_common import RigidBodyState
from tensorlbm.sixdof import SixDOFBody


def build_sphere_mask(nx, ny, nz, cx, cy, cz, R, device):
    """3D sphere mask."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    sphere = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= R ** 2
    return sphere


def run_falling(device_id, output_path=None):
    """Free falling sphere: Re=100, D=40, m*=1.5."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # --- Grid & flow parameters ---
    nx, ny, nz = 120, 120, 120
    D = 40                          # sphere diameter (lattice units)
    R = D / 2.0
    cx = nx // 2
    cy = ny // 2
    cz = nz // 2
    Re = 100
    u_in = 0.0                      # no inflow — sphere falls in quiescent fluid
    # We use a small initial velocity to start
    u_init = 0.01
    nu = u_init * D / Re            # use u_init for Re definition
    tau = 3.0 * nu + 0.5
    n_steps = 15000
    tag = f"[Falling SDAA:{device_id}]"

    # --- Structural parameters ---
    m_star = 1.5                    # mass ratio m* = m / (rho_f * D^3)
    rho_f = 1.0
    # Sphere mass in lattice units
    mass = m_star * rho_f * D ** 3
    # Gravity in lattice units: g_lbm = F_gravity / m
    # At terminal velocity: F_drag = m * g  →  g = F_drag / m
    # We want terminal velocity ~0.01-0.05 (low Mach)
    # Cd ≈ 1.09 at Re=100, A = pi*R^2, F_drag = 0.5*rho*U^2*A*Cd
    # g = F_drag / m = 0.5 * 1 * U^2 * pi*R^2 * Cd / (m_star * D^3)
    # For U=0.03: g = 0.5 * 0.03^2 * pi*20^2 * 1.09 / (1.5 * 40^3)
    #           = 0.5 * 9e-4 * 1256.6 * 1.09 / 96000 ≈ 7.6e-6
    # Let's set g to achieve U_terminal ≈ 0.03
    U_target = 0.03
    A_front = math.pi * R ** 2
    Cd_ref = 1.09  # experimental Cd at Re=100
    g_lbm = 0.5 * rho_f * U_target ** 2 * A_front * Cd_ref / mass

    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} R={R} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f}", flush=True)
    print(f"{tag} m*={m_star} mass={mass:.2f} g_lbm={g_lbm:.6e} "
          f"U_target={U_target}", flush=True)

    dpS = 0.5 * u_init ** 2 * A_front  # using u_init for dpS (will scale)

    t0 = time.time()
    solid = build_sphere_mask(nx, ny, nz, cx, cy, cz, R, device)

    # --- Common interface: get_near_wall_3d ---
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # --- Common interface: SurfaceMesh.from_sphere ---
    mesh = SurfaceMesh.from_sphere(solid, near, float(cx), float(cy), float(cz), float(R))
    print(f"{tag} SurfaceMesh.from_sphere built", flush=True)

    # --- Far-field BC (all walls = no-slip via bounce-back, z periodic) ---
    # For a falling sphere in a box, use no-slip walls (bounce-back)
    bc_config = {'far_field_faces': [], 'periodic_faces': ['z-', 'z+']}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    # --- Initialise flow (quiescent) ---
    rho0 = torch.ones((nz, ny, nx), device=device)
    zero_v = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, zero_v, zero_v, zero_v, device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # --- Structure state (rigid body with gravity) ---
    body = SixDOFBody(
        mass=mass,
        ixx=0.4 * mass * R ** 2,  # solid sphere inertia
        iyy=0.4 * mass * R ** 2,
        izz=0.4 * mass * R ** 2,
        gravity=(0.0, -g_lbm, 0.0),  # gravity in -y
        fix_surge=True,   # x fixed
        fix_heave=True,   # z fixed
        fix_roll=True, fix_pitch=True, fix_yaw=True,  # no rotation
    )
    rb_state = RigidBodyState(
        pos=torch.tensor([float(cx), float(cy), float(cz)], dtype=torch.float64),
        vel=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        omega_body=torch.zeros(3, dtype=torch.float64),
    )

    # --- Histories ---
    cd_p_hist, cd_f_hist, cd_tot_hist = [], [], []
    vel_hist, pos_hist = [], []

    # --- Main loop ---
    for step in range(1, n_steps + 1):
        # LBM step (no inflow, sphere falls)
        f = lbm_step_correct(
            f, collide_mrt3d, tau, solid, 0.0,
            far_field_fn, correct_mass_fn=correct_mass3d, target_mass=im,
            step=step, mass_interval=200,
        )

        # Current sphere velocity for dpS
        v_y = rb_state.vel[1].item()
        u_cur = abs(v_y) if abs(v_y) > 1e-6 else u_init
        dpS_cur = 0.5 * u_cur ** 2 * A_front

        # FSI step: drag integration → rigid body update
        result = fsi_step_drag(
            f, mesh, dpS_cur, nu, rb_state,
            body=body, dt=1.0, force_axis=1,
            extrap='none', p0_method='far_field', solid=solid,
            friction_formula='standard',
        )
        rb_state = result.structure_updated

        # Moving boundary: shift solid mask
        new_cy = rb_state.pos[1].item()
        dy_shift = int(round(new_cy - cy))
        if abs(dy_shift) >= 1:
            solid = shift_solid_mask(solid, dx=0, dy=dy_shift, dz=0)
            near = get_near_wall_3d(solid)
            mesh = SurfaceMesh.from_sphere(
                solid, near, float(cx), float(new_cy), float(cz), float(R))
            cy = new_cy

        # Record
        cd_p = result.cd_pressure[1]  # y-component (drag direction)
        cd_f = result.cd_friction[1]
        cd_tot = cd_p + cd_f
        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        vel_hist.append(v_y)
        pos_hist.append(new_cy)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step % 1000 == 0:
            n_avg = min(200, len(cd_tot_hist))
            cd_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            v_avg = sum(vel_hist[-n_avg:]) / n_avg
            print(f"{tag} step={step}/{n_steps} Cd={cd_avg:.4f} "
                  f"v_y={v_avg:.6f} pos_y={new_cy:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # --- Post-processing ---
    # Terminal velocity: average of last 20% of velocity (should be constant)
    n_term = max(1, len(vel_hist) // 5)
    vel_arr = np.array(vel_hist[-n_term:])
    v_terminal = float(np.mean(np.abs(vel_arr)))

    # Cd at terminal velocity
    cd_arr = np.array(cd_tot_hist[-n_term:])
    cd_terminal = float(np.mean(cd_arr))

    # Reference: Stokes drag Cd = 24/Re (laminar)
    # At Re=100: Stokes Cd = 0.24 (underestimate), experimental Cd ≈ 1.09
    cd_stokes = 24.0 / Re
    cd_exp = 1.09

    # Terminal velocity from force balance: m*g = 0.5*rho*U^2*A*Cd
    # U_terminal = sqrt(2*m*g / (rho*A*Cd))
    v_ref = math.sqrt(2 * mass * g_lbm / (rho_f * A_front * cd_exp)) if cd_exp > 0 else 0.0
    err = abs(v_terminal - v_ref) / v_ref * 100 if v_ref > 0 else float('inf')

    print(f"\n{tag} === FINAL ===  v_terminal={v_terminal:.6f} (ref={v_ref:.6f}) "
          f"err={err:.1f}% Cd_terminal={cd_terminal:.4f} "
          f"Cd_stokes={cd_stokes:.4f} Cd_exp={cd_exp:.4f} "
          f"time={elapsed:.0f}s", flush=True)

    result_dict = {
        "case": "free_falling_sphere_Re100",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D, "R": R, "Re": Re, "nu": nu, "tau": tau,
        "m_star": m_star, "mass": mass, "g_lbm": g_lbm,
        "n_steps": n_steps,
        "v_terminal": v_terminal,
        "v_ref": v_ref,
        "Cd_terminal": cd_terminal,
        "Cd_stokes": cd_stokes,
        "Cd_exp": cd_exp,
        "error_pct": float(err),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
        "modules_used": [
            "drag_pressure.get_near_wall_3d",
            "drag_pressure.SurfaceMesh.from_sphere",
            "lbm_step_correct.lbm_step_correct",
            "fsi_common.fsi_step_drag",
            "fsi_common.shift_solid_mask",
            "sixdof_common.RigidBodyState",
            "sixdof.SixDOFBody",
        ],
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result_dict, indent=2))
        print(f"{tag} results saved to {output_path}", flush=True)

    return result_dict


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 27
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_falling(dev, out)
