#!/usr/bin/env python3
"""FSI falling sphere fix — SDAA:5.

Fix: use IBM direct forcing (ibm_common) instead of drag integration.
The original approach used bounce-back with shift_solid_mask, but standard
bounce-back doesn't account for wall velocity → Cd=0.

IBM direct forcing enforces no-slip at the moving boundary by applying
a body force to the fluid. The reaction force gives the drag.

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

from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, stream3d, correct_mass3d
from tensorlbm.ibm_common import ibm_direct_forcing_3d_common
from tensorlbm.sixdof_common import RigidBodyState
from tensorlbm.sixdof import SixDOFBody, step_sixdof, FluidForcesMoments


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
    n_steps = 8000
    tag = f"[Falling-fix SDAA:{device_id}]"

    # Structural parameters
    m_star = 1.5
    rho_f = 1.0
    mass = m_star * rho_f * D ** 3
    A_front = math.pi * R ** 2
    Cd_ref = 1.09

    # Gravity tuned for U_terminal ≈ 0.05
    # v_term = sqrt(2*m*g / (rho*A*Cd)) → g = v^2 * rho * A * Cd / (2*m)
    U_target = 0.05
    g_lbm = U_target ** 2 * rho_f * A_front * Cd_ref / (2.0 * mass)

    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} R={R} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f}", flush=True)
    print(f"{tag} m*={m_star} mass={mass:.2f} g_lbm={g_lbm:.6e} "
          f"U_target={U_target}", flush=True)

    dpS_ref = 0.5 * rho_f * U_target ** 2 * A_front

    t0 = time.time()
    solid = build_sphere_mask(nx, ny, nz, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells={n_solid}", flush=True)

    # BC: all walls no-slip (bounce-back), no far-field
    bc_config = {'far_field_faces': [], 'periodic_faces': []}
    far_field_fn = functools_partial = None  # not used

    # Initialise flow (quiescent)
    rho0 = torch.ones((nz, ny, nx), device=device)
    zero_v = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, zero_v, zero_v, zero_v, device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # Rigid body with gravity in -y
    body = SixDOFBody(
        mass=mass,
        ixx=0.4 * mass * R ** 2,
        iyy=0.4 * mass * R ** 2,
        izz=0.4 * mass * R ** 2,
        gravity=(0.0, -g_lbm, 0.0),
        fix_surge=True,
        fix_heave=True,
        fix_roll=True, fix_pitch=True, fix_yaw=True,
    )
    rb_state = RigidBodyState(
        pos=torch.tensor([float(cx), float(cy), float(cz)], dtype=torch.float64),
        vel=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        omega_body=torch.zeros(3, dtype=torch.float64),
    )

    cd_hist = []
    vel_hist = []
    force_hist = []

    for step in range(1, n_steps + 1):
        # 1. IBM direct forcing: enforce no-slip at sphere surface
        #    Target velocity = sphere velocity (body falling in -y)
        u_target = rb_state.vel.detach().to(f.dtype).clone()

        force_on_fluid, f_corrected = ibm_direct_forcing_3d_common(
            f, solid, u_target,
            lattice="D3Q19", kernel="4pt", tau=tau,
        )
        f = f_corrected

        # 2. Save pre-collision state for NoDynamics + half-way BB
        f_pre = f.clone()

        # 3. Collision
        f = collide_mrt3d(f, tau=tau)

        # 4. NoDynamics: restore solid cells to pre-collision
        sm = solid.unsqueeze(0).expand_as(f)
        for q in range(f.shape[0]):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 5. Bounce-back at solid (half-way, before streaming)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # 6. Streaming
        f = stream3d(f)

        # 7. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 7. Compute drag force = reaction force from IBM
        #    F_fluid_on_body = -sum(IBM force on fluid)
        fy_drag = -float(force_on_fluid[1].sum().item())
        fx_drag = -float(force_on_fluid[0].sum().item())
        fz_drag = -float(force_on_fluid[2].sum().item())

        # 8. Advance rigid body with drag + gravity
        fluid = FluidForcesMoments(
            fx=fx_drag, fy=fy_drag, fz=fz_drag,
            mx=0.0, my=0.0, mz=0.0,
        )
        pos_new, vel_new, quat_new, omega_new = step_sixdof(
            rb_state.pos, rb_state.vel, rb_state.quat, rb_state.omega_body,
            fluid, body, 1.0,
        )
        rb_state = RigidBodyState(
            pos=pos_new, vel=vel_new, quat=quat_new, omega_body=omega_new,
        )

        # 9. Compute Cd from drag force
        v_y = abs(rb_state.vel[1].item())
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
                  f"({time.time()-t0:.0f}s)", flush=True)

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
        "case": "free_falling_sphere_Re100_fixed",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D, "R": R, "Re": Re, "nu": nu, "tau": tau,
        "m_star": m_star, "mass": mass, "g_lbm": g_lbm,
        "n_steps": n_steps,
        "v_terminal": v_terminal,
        "v_ref": v_ref,
        "Cd_terminal": cd_terminal,
        "Cd_exp": Cd_ref,
        "error_pct": float(err),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
        "method": "IBM_direct_forcing",
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result_dict, indent=2))
        print(f"{tag} results saved to {output_path}", flush=True)

    return result_dict


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_falling(dev, out)
