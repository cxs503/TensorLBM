#!/usr/bin/env python3
"""FSI falling sphere with IBM — SDAA:9.

Fix: device mismatch — IBM markers must be on same device as fluid.

Uses ibm_direct_forcing_3d_common with markers generated on device=device.
The ibm_step_correct function handles collision + IBM forcing + streaming + BC.

Sphere Re=100, D=20, 80³, m*=1.5
10000 steps
Target: Cd > 0.5, v_terminal < 0.1
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
import torch_sdaa  # noqa: F401

import functools
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d
from tensorlbm.ibm_common import (
    ibm_step_correct,
    generate_sphere_markers,
    compute_ibm_drag_from_markers,
    update_moving_markers,
)


def build_sphere_mask(nx, ny, nz, cx, cy, cz, R, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    sphere = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= R ** 2
    return sphere


def run_falling(device_id, output_path=None):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device_id)

    nx, ny, nz = 80, 80, 80
    D = 20
    R = D / 2.0
    cx = nx // 2
    cy = ny // 2
    cz = nz // 2
    Re = 100
    u_ref = 0.05  # expected terminal velocity scale
    nu = u_ref * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 10000
    tag = f"[Falling-IBM SDAA:{device_id}]"

    # Structural parameters
    m_star = 1.5
    rho_f = 1.0
    mass = m_star * rho_f * D ** 3
    A_front = math.pi * R ** 2
    Cd_ref = 1.09

    # Gravity tuned for U_terminal ≈ 0.05
    U_target = 0.05
    g_lbm = U_target ** 2 * rho_f * A_front * Cd_ref / (2.0 * mass)
    dpS_ref = 0.5 * rho_f * U_target ** 2 * A_front

    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} R={R} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f}", flush=True)
    print(f"{tag} m*={m_star} mass={mass:.2f} g_lbm={g_lbm:.6e} "
          f"U_target={U_target}", flush=True)

    t0 = time.time()
    solid = build_sphere_mask(nx, ny, nz, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells={n_solid}", flush=True)

    # --- FIX: Generate IBM markers on the SAME device as fluid ---
    markers = generate_sphere_markers(
        cx, cy, cz, R, n_theta=24, n_phi=12, device=device,
    )
    marker_x, marker_y, marker_z = markers
    n_markers = marker_x.shape[0]
    print(f"{tag} IBM markers={n_markers} on device={device}", flush=True)

    # Initialise flow (quiescent)
    rho0 = torch.ones((nz, ny, nx), device=device)
    zero_v = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, zero_v, zero_v, zero_v, device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # Far-field BC (all faces)
    bc_config = {'far_field_faces': ['x-', 'x+', 'y-', 'y+', 'z-', 'z+']}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    # Sphere state (falling in -y direction)
    vel_y = 0.0
    pos_y = float(cy)
    disp_y_total = 0.0  # accumulated displacement from initial position

    cd_hist = []
    vel_hist = []
    force_hist = []

    for step in range(1, n_steps + 1):
        # Target velocity: sphere velocity (0, vel_y, 0)
        # For falling sphere, vel_y is negative (downward in -y)
        # IBM enforces no-slip: u_target = u_sphere
        u_target = torch.tensor([0.0, vel_y, 0.0], device=device, dtype=f.dtype)

        def u_target_fn(step_num, _ut=u_target):
            return _ut[0].expand(n_markers), _ut[1].expand(n_markers), _ut[2].expand(n_markers)

        # IBM LBM step: collision + IBM forcing + streaming + BC
        f, marker_forces = ibm_step_correct(
            f, collide_mrt3d, tau, solid, u_ref,
            far_field_fn, markers, u_target_fn,
            correct_mass_fn=correct_mass3d, target_mass=im,
            step=step, mass_interval=200,
            ramp_steps=500, n_force_iter=4, force_clip=0.05,
        )

        # Compute drag from marker forces
        marker_fx, marker_fy, marker_fz = marker_forces
        # Force on body = -sum(marker_forces) (Newton's third law)
        # fy_body = -sum(marker_fy) — drag opposes motion
        fy_drag = -float(marker_fy.sum().item())

        # Advance sphere: m*a = F_gravity + F_drag
        # gravity in -y, drag in +y (opposing motion when falling)
        a_y = (fy_drag - mass * g_lbm) / mass
        vel_y += a_y * 1.0  # dt=1
        pos_y += vel_y * 1.0
        disp_y_total += vel_y * 1.0

        # Update marker positions (translate by velocity)
        marker_x, marker_y, marker_z = update_moving_markers(
            marker_x, marker_y, marker_z,
            cx, cy, cz,
            displacement=(0.0, vel_y, 0.0),
        )
        # Ensure markers stay on device
        marker_x = marker_x.to(device)
        marker_y = marker_y.to(device)
        marker_z = marker_z.to(device)
        markers = (marker_x, marker_y, marker_z)

        # Compute Cd
        v_y = abs(vel_y)
        u_cur = max(v_y, 1e-6)
        dpS_cur = 0.5 * rho_f * u_cur ** 2 * A_front
        cd_cur = fy_drag / dpS_cur if dpS_cur > 1e-12 else 0.0

        cd_hist.append(cd_cur)
        vel_hist.append(v_y)
        force_hist.append(fy_drag)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step % 1000 == 0:
            n_avg = min(200, len(cd_hist))
            cd_avg = sum(cd_hist[-n_avg:]) / n_avg
            v_avg = sum(vel_hist[-n_avg:]) / n_avg
            f_avg = sum(force_hist[-n_avg:]) / n_avg
            print(f"{tag} step={step}/{n_steps} Cd={cd_avg:.4f} "
                  f"v_y={v_avg:.6f} F_drag={f_avg:.4f} "
                  f"pos_y={pos_y:.2f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # Terminal velocity: average of last 20%
    n_term = max(1, len(vel_hist) // 5)
    vel_arr = np.array(vel_hist[-n_term:])
    cd_arr = np.array(cd_hist[-n_term:])

    v_terminal = float(np.mean(vel_arr))
    cd_terminal = float(np.mean(cd_arr))

    v_ref = math.sqrt(2 * mass * g_lbm / (rho_f * A_front * Cd_ref)) if Cd_ref > 0 else 0.0
    err = abs(v_terminal - v_ref) / v_ref * 100 if v_ref > 0 else float('inf')

    print(f"\n{tag} === FINAL ===  v_terminal={v_terminal:.6f} (ref={v_ref:.6f}) "
          f"err={err:.1f}% Cd_terminal={cd_terminal:.4f} "
          f"Cd_exp={Cd_ref:.4f} time={elapsed:.0f}s", flush=True)

    result_dict = {
        "case": "free_falling_sphere_Re100_ibm",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D, "R": R, "Re": Re, "nu": nu, "tau": tau,
        "m_star": m_star, "mass": mass, "g_lbm": g_lbm,
        "n_steps": n_steps,
        "n_markers": n_markers,
        "v_terminal": v_terminal,
        "v_ref": v_ref,
        "Cd_terminal": cd_terminal,
        "Cd_exp": Cd_ref,
        "error_pct": float(err),
        "target_Cd_gt_0.5": cd_terminal > 0.5,
        "target_v_lt_0.1": v_terminal < 0.1,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
        "method": "ibm_direct_forcing_3d_common",
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result_dict, indent=2))
        print(f"{tag} results saved to {output_path}", flush=True)

    return result_dict


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_falling(dev, out)
