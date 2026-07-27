#!/usr/bin/env python3
"""FSI falling sphere fix — SDAA:7.

Fix: use shift_solid_mask when displacement > 1 cell.
Rebuild near-wall mesh after mask shift.
Uses moving bounce-back (includes sphere velocity) + Galilean-invariant
momentum exchange for drag computation.

Sphere Re=100, D=40, 120^3
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

from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, W, OPPOSITE
from tensorlbm.solver3d import collide_mrt3d, stream3d, correct_mass3d
from tensorlbm.drag_pressure import get_near_wall_3d
from tensorlbm.fsi_common import shift_solid_mask
from tensorlbm.momentum_exchange import momentum_exchange_galilean


def build_sphere_mask(nx, ny, nz, cx, cy, cz, R, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    sphere = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= R ** 2
    return sphere


def moving_bounce_back_3d(f, mask, u_wall, f_pre=None):
    """Half-way bounce-back with wall velocity for moving boundaries.

    f_new[i] = f_pre[opp[i]] - 6*w[i]*(c[i]·u_wall)  at solid cells
    f_new[i] = f[i]                                     at fluid cells
    """
    device = f.device
    opp = OPPOSITE.to(device)
    src = f_pre if f_pre is not None else f
    c = C.to(device).float()  # (19, 3)
    w = W.to(device).float()  # (19,)

    # Wall velocity correction: 6*w[i]*(c[i]·u_wall)
    # u_wall is (3,) tensor
    cu = (c * u_wall.unsqueeze(0)).sum(dim=1)  # (19,) c·u
    correction = (6.0 * w * cu).view(19, 1, 1, 1)  # (19,1,1,1)

    # At solid cells: f_pre[opp] - correction
    bounced = src[opp] - correction
    return torch.where(mask.unsqueeze(0), bounced, f)


def run_falling(device_id, output_path=None):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device_id)

    nx, ny, nz = 120, 120, 120
    D = 40
    R = D / 2.0
    cx = nx // 2
    cy = ny // 2
    cz = nz // 2
    Re = 100
    u_ref = 0.05  # expected terminal velocity scale
    nu = u_ref * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 10000
    tag = f"[Falling-fix SDAA:{device_id}]"

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

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # Initialise flow (quiescent)
    rho0 = torch.ones((nz, ny, nx), device=device)
    zero_v = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, zero_v, zero_v, zero_v, device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # Sphere state (falling in -y direction)
    pos_y = float(cy)
    vel_y = 0.0
    disp_y = 0.0  # accumulated displacement from initial position

    cd_hist = []
    vel_hist = []
    force_hist = []

    for step in range(1, n_steps + 1):
        # Sphere velocity as tensor
        u_wall = torch.tensor([0.0, vel_y, 0.0], device=device, dtype=f.dtype)

        # 1. Save pre-collision state for bounce-back
        f_pre = f.clone()

        # 2. Collision (MRT for stability)
        f = collide_mrt3d(f, tau=tau)

        # 3. Moving bounce-back at sphere surface (pre-stream, with wall velocity)
        f = moving_bounce_back_3d(f, solid, u_wall, f_pre=f_pre)

        # 4. Streaming
        f = stream3d(f)

        # 5. Moving bounce-back at sphere surface (post-stream, with wall velocity)
        f = moving_bounce_back_3d(f, solid, u_wall)

        # 6. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 7. Compute drag force via Galilean-invariant momentum exchange
        fx_drag, fy_drag, fz_drag = momentum_exchange_galilean(
            f, solid, near, tau=tau,
        )

        # fy_drag is the force on the wall (drag-positive convention)
        # For a falling sphere (moving in -y), drag opposes motion → positive fy_drag
        # means drag in +y (upward), which is correct

        # 8. Advance sphere: m*a = F_gravity + F_drag
        #    gravity in -y, drag in +y (opposing motion)
        a_y = (fy_drag - mass * g_lbm) / mass
        vel_y += a_y * 1.0  # dt=1
        pos_y += vel_y * 1.0
        disp_y += vel_y * 1.0

        # 9. Shift solid mask when displacement > 1 cell
        dy_shift = int(round(disp_y))
        if abs(dy_shift) >= 1:
            solid = shift_solid_mask(solid, dx=0, dy=dy_shift, dz=0)
            near = get_near_wall_3d(solid)
            # Reset accumulated displacement (mask has been shifted)
            disp_y -= dy_shift

            # Fill newly freed cells with equilibrium
            rho_cur, ux_cur, uy_cur, uz_cur = macroscopic3d(f)
            feq_fill = equilibrium3d(
                rho_cur, ux_cur, uy_cur, uz_cur, device=device,
            )
            f = torch.where(solid.unsqueeze(0), f, feq_fill)

        # 10. Compute Cd
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
        "target_Cd_gt_0.5": cd_terminal > 0.5,
        "target_v_lt_0.1": v_terminal < 0.1,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
        "method": "moving_bounce_back_MEM",
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result_dict, indent=2))
        print(f"{tag} results saved to {output_path}", flush=True)

    return result_dict


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_falling(dev, out)
