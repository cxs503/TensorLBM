#!/usr/bin/env python3
"""BFS retest with Bug 28 fix + BB fix (SDAA:30).

Bug 28: detection scans y=step_h..step_h+5 (not y=1)
Bug 27 (BB fix): bounce_back uses pre-collision f (f_pre)

Parameters: nx=400, ny=20, step_h=10, x_step=100
Re=1000, MRT+Smag(Cs=0.05), 10000 steps, parabolic inlet

Previous: xr/H=0 (no separation detected, y=1 scan)
Target:   xr/H>0 (detection finds recirculation at y=3-7)

Usage:
  PYTHONPATH=src python bfs_bbfix_retest_worker.py <device_id> <output_json>
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d,
    zou_he_inlet_velocity_3d,
)
from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.backward_facing_step import make_bfs_solid_mask


# ---------------------------------------------------------------------------
# Parabolic inlet profile for BFS
# ---------------------------------------------------------------------------
def parabolic_inlet_profile_3d(nz, ny, step_h, u_bulk, device):
    """Parabolic (Poiseuille) inlet profile for BFS.

    u(y) = 6 * u_bulk * y' * (H - y') / H^2
    where y' = y - step_h, H = ny - 1 - step_h (fluid height at inlet).
    """
    H = float(ny - 1 - step_h)
    if H <= 0:
        raise ValueError(f"Invalid inlet height H={H} (ny={ny}, step_h={step_h})")
    y = torch.arange(ny, device=device, dtype=torch.float32)
    y_local = (y - step_h).to(torch.float32)
    u_y = 6.0 * u_bulk * y_local * (H - y_local) / (H * H)
    u_y = torch.clamp(u_y, min=0.0)
    u_profile = u_y.unsqueeze(0).expand(nz, ny).contiguous()
    return u_profile


# ---------------------------------------------------------------------------
# BFS channel BC with parabolic inlet (BB fix version)
# ---------------------------------------------------------------------------
def bfs_channel_bc_parabolic_3d_bbfix(f, u_profile, solid, f_pre):
    """Channel BC for BFS with parabolic inlet (3D, BB fix).

    1. Zou/He velocity inlet at x=0 with parabolic profile.
    2. Zero-gradient outlet at x=nx-1.
    3. Bounce-back on the full solid mask (using f_pre for BB fix).
    """
    f = zou_he_inlet_velocity_3d(f, u_profile)
    # Zero-gradient outlet
    f[:, :, :, -1] = f[:, :, :, -2]
    # BB fix: use f_pre for bounce-back
    f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
    return f


# ---------------------------------------------------------------------------
# BFS benchmark (parabolic inlet + BB fix + Bug 28 detection)
# ---------------------------------------------------------------------------
def run_bfs_bbfix(
    device, output_path, tag,
    nx=400, ny=20, nz=4, step_h=10, x_step=100,
    u_in=0.05, Re=1000.0, Cs=0.05, n_steps=10000,
):
    """BFS with parabolic inlet, BB fix, and Bug 28 detection fix.

    Reference: xr/H = 6.0 (ER=2, Re=1000).
    Previous: xr/H = 0.0 (uniform inlet, y=1 scan).
    Target:   xr/H > 0 (Bug 28 fix: scan y=step_h..step_h+5).
    """
    nu = u_in * step_h / Re
    tau = 3.0 * nu + 0.5
    ref_xr = 6.0
    ER = ny / (ny - step_h)

    print(
        f"{tag} [BFS-BBfix] nx={nx} ny={ny} nz={nz} step_h={step_h} "
        f"x_step={x_step} ER={ER:.1f} u_in={u_in} nu={nu:.6e} tau={tau:.6f} "
        f"Re={Re} Cs={Cs} n_steps={n_steps}", flush=True,
    )

    t0 = time.time()

    # Build 2D solid mask and extrude to 3D
    solid_2d = make_bfs_solid_mask(ny, nx, step_h, x_step, device)
    solid = solid_2d.unsqueeze(0).expand(nz, ny, nx).clone()
    # Add front/back z-walls
    solid[0, :, :] = True
    solid[-1, :, :] = True

    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    print(f"{tag} [BFS-BBfix] solid={n_solid} ({time.time()-t0:.1f}s)", flush=True)

    # Parabolic inlet profile
    u_profile = parabolic_inlet_profile_3d(nz, ny, step_h, u_in, device)
    u_max_profile = float(u_profile.max().item())
    u_mean_profile = float(u_profile[1:-1, step_h:ny-1].mean().item())
    print(
        f"{tag} [BFS-BBfix] inlet profile: u_max={u_max_profile:.5f} "
        f"u_mean(fluid)={u_mean_profile:.5f} (bulk={u_in})", flush=True,
    )

    # Initialize: parabolic flow above the step, rest in solid
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    for z in range(nz):
        for y in range(ny):
            ux0[z, y, :] = u_profile[z, y]
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.zeros_like(ux0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} [BFS-BBfix] init done ({time.time()-t0:.1f}s), "
          f"starting loop...", flush=True)

    xr_hist = []
    diverged = False
    last_step = 0

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision (for BB fix)
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)

        # 3. NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming) with BB fix
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # 5. Streaming
        f = stream3d(f)

        # 6. Channel BC (parabolic inlet + outlet + bounce-back with BB fix)
        f = bfs_channel_bc_parabolic_3d_bbfix(f, u_profile, solid, f_pre)

        # 7. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} [BFS-BBfix] DIVERGED at step {step}", flush=True)
            diverged = True
            last_step = step
            break
        last_step = step

        # Measure reattachment length (Bug 28 fix: scan y=1..step_h+5)
        if step % 100 == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            ux_zmid = ux[nz // 2]
            ux_zmid = ux_zmid.masked_fill(solid[nz // 2], 0.0)

            # Bug 28 fix: scan multiple y levels (not just y=1)
            # Recirculation is at y=step_h..step_h+5, not y=1 (bottom wall)
            xr_star = 0.0
            for y_check in range(1, min(step_h + 6, ny - 1)):
                cl = ux_zmid[y_check, x_step:].cpu()
                has_neg = any(v < 0 for v in cl.tolist()[:20])
                if has_neg:
                    for i, val in enumerate(cl.tolist()):
                        if val > 0.0:
                            xr_star = float(i) / max(step_h, 1)
                            break
                    break  # found recirculation at this y level
            xr_hist.append(xr_star)

            if step % 500 == 0 or step == n_steps:
                elapsed = time.time() - t0
                ms = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())
                print(
                    f"{tag} [BFS-BBfix] step={step} xr/H={xr_star:.3f} "
                    f"max|u|={ms:.4f} ({elapsed:.0f}s)", flush=True,
                )

    elapsed = time.time() - t0

    # Final measurements
    rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
    ux_zmid = ux_f[nz // 2].masked_fill(solid[nz // 2], 0.0)

    # Bug 28 fix: scan multiple y levels for final detection
    final_xr = 0.0
    detection_y = 0
    for y_check in range(1, min(step_h + 6, ny - 1)):
        cl = ux_zmid[y_check, x_step:].cpu()
        has_neg = any(v < 0 for v in cl.tolist()[:20])
        if has_neg:
            detection_y = y_check
            for i, val in enumerate(cl.tolist()):
                if val > 0.0:
                    final_xr = float(i) / max(step_h, 1)
                    break
            break

    # Average xr over last 20% of history
    tail_xr = xr_hist[-max(len(xr_hist) // 5, 1):] if xr_hist else [0.0]
    xr_mean = sum(tail_xr) / len(tail_xr)

    err_pct = abs(xr_mean - ref_xr) / ref_xr * 100 if ref_xr > 0 else float("nan")

    # Also check velocity at key points for diagnostics
    ux_diag = {}
    for y_check in [1, 3, 5, 7, 9]:
        if y_check < ny - 1:
            cl = ux_zmid[y_check, x_step:].cpu()
            ux_diag[f"y{y_check}"] = {
                "ux_at_xstep": float(cl[0].item()) if len(cl) > 0 else 0,
                "ux_at_xstep+10": float(cl[10].item()) if len(cl) > 10 else 0,
                "ux_at_xstep+30": float(cl[30].item()) if len(cl) > 30 else 0,
                "has_negative": bool(any(v < 0 for v in cl.tolist()[:20])),
            }

    result = {
        "benchmark": "backward_facing_step_bbfix",
        "device": str(device),
        "grid": f"{nx}x{ny}x{nz}",
        "step_h": step_h,
        "x_step": x_step,
        "expansion_ratio": ER,
        "u_in": u_in,
        "u_max_profile": u_max_profile,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "Cs": Cs,
        "n_steps": n_steps,
        "inlet_type": "parabolic_poiseuille",
        "bb_fix": True,
        "bug28_fix": True,
        "detection_y": detection_y,
        "xr_H_final": final_xr,
        "xr_H_mean": xr_mean,
        "xr_H_ref": ref_xr,
        "xr_error_pct": err_pct,
        "ux_diagnostics": ux_diag,
        "finite": not diverged,
        "diverged": diverged,
        "last_step": last_step,
        "elapsed_s": elapsed,
    }
    print(
        f"{tag} [BFS-BBfix] DONE xr/H={xr_mean:.3f} (ref={ref_xr}, "
        f"err={err_pct:.1f}%) detection_y={detection_y} ({elapsed:.0f}s)",
        flush=True,
    )
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python bfs_bbfix_retest_worker.py <device_id> <output_json>")
        sys.exit(1)

    device_id = int(sys.argv[1])
    output_path = sys.argv[2]
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{device_id} BFS-BBfix]"
    run_bfs_bbfix(device, output_path, tag)


if __name__ == "__main__":
    main()
