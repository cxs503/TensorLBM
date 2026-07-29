#!/usr/bin/env python3
"""Grid convergence (BB fix) + BFS retest via common interface ONLY.

ALL tests use the common interface:
  - Geometry:  get_near_wall_3d(solid)              from drag_pressure.py
  - Normals:   SurfaceMesh.from_suboff / from_gradient
  - Force:     drag_friction_integration(formula='standard'/'lagrange')
  - BC:        lbm_step_correct (NoDynamics + half-way BB with f_pre)

TEST 1: SUBOFF grid convergence with BB fix (SDAA:28-29)
  L=40/80/160, Re=1000, MRT+Smag(Cs=0.05)
  from_suboff normals, lbm_step_correct(), 5000 steps
  Both formula='standard' and formula='lagrange'
  Previous (no BB fix): 14.0%→8.1%→? (diverged)
  KEY: Does BB fix solve grid divergence?

TEST 2: BFS retest with Bug 28 fix (SDAA:30)
  Bug 28: detection scans y=step_h..step_h+5
  nx=400, ny=20, step_h=10, x_step=100
  Re=1000, MRT+Smag, 10000 steps, parabolic inlet
  Previous: xr/H=0; Target: xr/H>0

TEST 3: Couette grid convergence with BB fix (SDAA:31)
  ny=8/16/32, tau=1.0
  lbm_step_correct() with custom bounce_back_fn (moving wall)
  Previous: 0.0007%→0.11%→12.3% (diverged!)
  KEY: Does BB fix solve divergence?

Usage:
  python grid_conv_bbfix_worker.py suboff   <L> <formula> <device_id> <out>
  python grid_conv_bbfix_worker.py bfs      <device_id> <out>
  python grid_conv_bbfix_worker.py couette  <ny> <device_id> <out>
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
from tensorlbm.d3q19 import (
    C, W, OPPOSITE, equilibrium3d, macroscopic3d,
)
from tensorlbm.solver3d import (
    collide_bgk3d, collide_mrt3d, correct_mass3d, stream3d,
)
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d, far_field_bc_3d, zou_he_inlet_velocity_3d,
)
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_3d,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig
from tensorlbm.backward_facing_step import make_bfs_solid_mask


# ===================================================================
# TEST 1: SUBOFF grid convergence with BB fix via lbm_step_correct
# ===================================================================
def run_suboff(device_id, L, formula, output_path):
    """SUBOFF bare-hull drag at Re=1000, BB fix via lbm_step_correct."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    Re = 1000
    u_in = 0.06
    cs_smag = 0.05
    n_steps = 5000
    win = 1000

    config = SuboffConfig()
    radius = config.r_over_l * L
    D = 2.0 * radius

    if L == 40:
        nx, ny, nz = 100, 40, 40
    elif L == 80:
        nx, ny, nz = 200, 80, 80
    elif L == 160:
        nx, ny, nz = 300, 120, 120
        n_steps = 3000  # reduced for large grid
    else:
        raise ValueError(f"L must be 40/80/160, got {L}")

    cx = nx * 0.30
    cy = ny * 0.5
    cz = nz * 0.5
    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5
    dpS = 0.5 * u_in ** 2 * math.pi * D * L
    Cf_ref = 1.328 / math.sqrt(Re)

    tag = f"[SDAA:{device_id} SUBOFF-BBfix L={L} {formula}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} L={L} D={D:.3f} "
          f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
          f"dpS={dpS:.6e} Cf_ref={Cf_ref:.6f} n_steps={n_steps}", flush=True)

    t0 = time.time()
    solid, stats = build_suboff_mask(
        hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=L, radius=radius,
        config=config, device=device,
    )
    n_solid = int(solid.sum().item())
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)
    print(f"{tag} solid={n_solid} near={n_near} mesh=from_suboff "
          f"({time.time()-t0:.1f}s)", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # Far-field BC via common interface (3D external aero)
    bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"],
                 "periodic_faces": []}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []

    for step in range(1, n_steps + 1):
        # === Common interface: lbm_step_correct (BB fix) ===
        f = lbm_step_correct(
            f,
            collide_fn=collide_smagorinsky_mrt3d,
            tau=tau,
            solid=solid,
            u_in=u_in,
            far_field_bc_fn=far_field_fn,
            correct_mass_fn=correct_mass3d,
            target_mass=initial_mass,
            step=step,
            mass_interval=200,
            C_s=cs_smag,
        )

        # === Force via common interface ===
        fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu, formula=formula)
        cd_p_hist.append(fx_p)
        cd_f_hist.append(fx_f)
        cd_tot_hist.append(fx_p + fx_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_p_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            print(f"{tag} step={step} Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                  f"Cd_tot={cd_tot_avg:.6f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(win, len(cd_p_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    err_pct = abs(cd_tot_final - Cf_ref) / Cf_ref * 100 if Cf_ref > 0 else float("nan")

    result = {
        "case": "suboff_bb_fix",
        "device": f"sdaa:{device_id}",
        "Re": Re, "L": L, "D": D, "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in, "nu": nu, "tau": tau, "Cs": cs_smag,
        "n_steps": n_steps, "win": win,
        "n_solid": n_solid, "n_near": n_near, "dpS": dpS,
        "Cf_ref": Cf_ref,
        "mesh_type": "from_suboff",
        "bb_fix": True,
        "lbm_step": "lbm_step_correct",
        "friction_formula": formula,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "err_pct": err_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    print(f"\n{tag} === FINAL (Cf_ref={Cf_ref:.6f}) ===", flush=True)
    print(f"{tag} Cd_p  = {cd_p_final:.6f}", flush=True)
    print(f"{tag} Cd_f  = {cd_f_final:.6f}", flush=True)
    print(f"{tag} Cd_tot= {cd_tot_final:.6f}  err={err_pct:.1f}%", flush=True)
    print(f"{tag} time={elapsed:.0f}s", flush=True)
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ===================================================================
# TEST 2: BFS retest with Bug 28 fix + BB fix
# ===================================================================
def _parabolic_inlet_profile_3d(nz, ny, step_h, u_bulk, device):
    """Parabolic (Poiseuille) inlet profile for BFS."""
    H = float(ny - 1 - step_h)
    y = torch.arange(ny, device=device, dtype=torch.float32)
    y_local = (y - step_h).to(torch.float32)
    u_y = 6.0 * u_bulk * y_local * (H - y_local) / (H * H)
    u_y = torch.clamp(u_y, min=0.0)
    return u_y.unsqueeze(0).expand(nz, ny).contiguous()


def _bfs_channel_bc(f, u_profile, solid, f_pre):
    """Channel BC for BFS: parabolic inlet + outlet + bounce-back (BB fix)."""
    f = zou_he_inlet_velocity_3d(f, u_profile)
    f[:, :, :, -1] = f[:, :, :, -2]  # Zero-gradient outlet
    f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
    return f


def run_bfs(device_id, output_path):
    """BFS with parabolic inlet, BB fix, and Bug 28 detection fix."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 400, 20, 4
    step_h, x_step = 10, 100
    u_in = 0.05
    Re = 1000.0
    Cs = 0.05
    n_steps = 10000
    ref_xr = 6.0
    ER = ny / (ny - step_h)

    nu = u_in * step_h / Re
    tau = 3.0 * nu + 0.5

    tag = f"[SDAA:{device_id} BFS-BBfix]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} step_h={step_h} "
          f"x_step={x_step} ER={ER:.1f} u_in={u_in} nu={nu:.6e} tau={tau:.6f} "
          f"Re={Re} Cs={Cs} n_steps={n_steps}", flush=True)

    t0 = time.time()
    solid_2d = make_bfs_solid_mask(ny, nx, step_h, x_step, device)
    solid = solid_2d.unsqueeze(0).expand(nz, ny, nx).clone()
    solid[0, :, :] = True
    solid[-1, :, :] = True
    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    print(f"{tag} solid={n_solid} ({time.time()-t0:.1f}s)", flush=True)

    u_profile = _parabolic_inlet_profile_3d(nz, ny, step_h, u_in, device)
    u_max_profile = float(u_profile.max().item())
    print(f"{tag} inlet: u_max={u_max_profile:.5f} bulk={u_in}", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    for z in range(nz):
        for y in range(ny):
            ux0[z, y, :] = u_profile[z, y]
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    xr_hist = []
    diverged = False
    last_step = 0

    for step in range(1, n_steps + 1):
        # === BB fix via lbm_step_correct ===
        # For BFS we need channel BC, so we use lbm_step_correct with
        # a custom far_field_bc_fn that does parabolic inlet + outlet + BB
        far_field_fn = functools.partial(_bfs_channel_bc, u_profile=u_profile,
                                         solid=solid)
        # But lbm_step_correct calls far_field_bc_fn(f, u_in) — we need f_pre
        # which is internal. So we do the loop manually but using the same
        # BB-fix logic as lbm_step_correct.
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)
        # NoDynamics
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        # Half-way BB (BEFORE streaming) with f_pre
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        # Streaming
        f = stream3d(f)
        # Channel BC (parabolic inlet + outlet + BB fix)
        f = _bfs_channel_bc(f, u_profile, solid, f_pre)
        # Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            diverged = True
            last_step = step
            break
        last_step = step

        # Bug 28 fix: scan y=1..step_h+5 for reattachment
        if step % 100 == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            ux_zmid = ux[nz // 2].masked_fill(solid[nz // 2], 0.0)
            xr_star = 0.0
            for y_check in range(1, min(step_h + 6, ny - 1)):
                cl = ux_zmid[y_check, x_step:].cpu()
                has_neg = any(v < 0 for v in cl.tolist()[:20])
                if has_neg:
                    for i, val in enumerate(cl.tolist()):
                        if val > 0.0:
                            xr_star = float(i) / max(step_h, 1)
                            break
                    break
            xr_hist.append(xr_star)
            if step % 500 == 0 or step == n_steps:
                elapsed = time.time() - t0
                ms = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())
                print(f"{tag} step={step} xr/H={xr_star:.3f} "
                      f"max|u|={ms:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
    ux_zmid = ux_f[nz // 2].masked_fill(solid[nz // 2], 0.0)

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

    tail_xr = xr_hist[-max(len(xr_hist) // 5, 1):] if xr_hist else [0.0]
    xr_mean = sum(tail_xr) / len(tail_xr)
    err_pct = abs(xr_mean - ref_xr) / ref_xr * 100 if ref_xr > 0 else float("nan")

    ux_diag = {}
    for y_check in [1, 3, 5, 7, 9]:
        if y_check < ny - 1:
            cl = ux_zmid[y_check, x_step:].cpu()
            ux_diag[f"y{y_check}"] = {
                "ux_at_xstep": float(cl[0].item()) if len(cl) > 0 else 0,
                "ux_at_xstep+10": float(cl[10].item()) if len(cl) > 10 else 0,
                "has_negative": bool(any(v < 0 for v in cl.tolist()[:20])),
            }

    result = {
        "benchmark": "backward_facing_step_bbfix",
        "device": str(device),
        "grid": f"{nx}x{ny}x{nz}",
        "step_h": step_h, "x_step": x_step,
        "expansion_ratio": ER,
        "u_in": u_in, "u_max_profile": u_max_profile,
        "Re": Re, "nu": nu, "tau": tau, "Cs": Cs,
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
    print(f"\n{tag} === FINAL ===", flush=True)
    print(f"{tag} xr/H={xr_mean:.3f} (ref={ref_xr}, err={err_pct:.1f}%) "
          f"detection_y={detection_y} ({elapsed:.0f}s)", flush=True)
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ===================================================================
# TEST 3: Couette grid convergence with BB fix via lbm_step_correct
# ===================================================================
def _make_channel_solid(nz, ny, nx, device):
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    return solid


def _near_wall_bottom_only(near, ny):
    near_b = near.clone()
    near_b[:, ny - 2, :] = False
    return near_b


def _moving_wall_bounce_back(f, solid, f_pre, top_wall_mask, u_top, rho_w=1.0):
    """Moving-wall bounce-back with BB fix (uses f_pre for solid cells)."""
    opp = OPPOSITE.to(f.device)
    src = f_pre
    f = torch.where(solid.unsqueeze(0), src[opp], f)
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    correction = 6.0 * rho_w * u_top * w * c[:, 0]
    top_mask = top_wall_mask.unsqueeze(0).float()
    f = f + correction.view(19, 1, 1, 1) * top_mask
    return f


def _noop_bc(f, u_in):
    """No-op BC for Couette (x is periodic via stream3d)."""
    return f


def run_couette(device_id, ny, output_path):
    """3D Couette: moving top wall, BB fix via lbm_step_correct."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, nz = 80, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0  # = 1/6
    u_top = 0.05
    n_steps = 4000
    warmup = max(500, ny * 50)

    H = ny - 2
    Cf_exact = 2.0 * nu / (H * u_top)

    tag = f"[SDAA:{device_id} Couette-BBfix ny={ny}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f} u_top={u_top} "
          f"H={H} Cf_exact={Cf_exact:.6f} warmup={warmup}", flush=True)

    t0 = time.time()
    solid = _make_channel_solid(nz, ny, nx, device)
    top_wall_mask = torch.zeros_like(solid)
    top_wall_mask[:, -1, :] = True

    near = get_near_wall_3d(solid)
    near_bottom = _near_wall_bottom_only(near, ny)
    mesh_bottom = SurfaceMesh.from_gradient(solid, near_bottom)
    print(f"{tag} near(bottom)={int(near_bottom.sum().item())} "
          f"({time.time()-t0:.1f}s)", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # Custom bounce-back for moving wall (BB fix: uses f_pre)
    bb_fn = functools.partial(_moving_wall_bounce_back,
                               top_wall_mask=top_wall_mask,
                               u_top=u_top)

    cf_hist = []

    for step in range(1, n_steps + 1):
        # === Common interface: lbm_step_correct (BB fix) ===
        # Uses custom bounce_back_fn for moving wall
        f = lbm_step_correct(
            f,
            collide_fn=collide_bgk3d,
            tau=tau,
            solid=solid,
            u_in=u_top,
            far_field_bc_fn=_noop_bc,  # no far-field for Couette
            correct_mass_fn=None,       # mass conserved by BB
            step=step,
            mass_interval=999999,
            bounce_back_fn=bb_fn,
        )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            A_wall = nx * nz
            dpS_wall = 0.5 * 1.0 * u_top ** 2 * A_wall
            ffx, _, _ = drag_friction_integration(f, mesh_bottom, dpS_wall, nu,
                                                   formula='standard')
            cf_hist.append(ffx)

        if step % 1000 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_prof = ux.mean(dim=(0, 2))
            cf_avg = sum(cf_hist) / max(len(cf_hist), 1) if cf_hist else float('nan')
            print(f"{tag} step={step} Cf={cf_avg:.6f} u[1]={float(u_prof[1]):.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    _, ux_f, _, _ = macroscopic3d(f)
    u_prof = ux_f.mean(dim=(0, 2))
    u_exact_1 = u_top / H
    u_err_1 = abs(float(u_prof[1]) - u_exact_1) / u_exact_1 * 100

    cf_mean = sum(cf_hist) / max(len(cf_hist), 1) if cf_hist else float("nan")
    cf_err = abs(cf_mean - Cf_exact) / Cf_exact * 100 if Cf_exact > 0 and math.isfinite(cf_mean) else float("nan")

    result = {
        "case": "couette_bb_fix",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "tau": tau, "nu": float(nu), "u_top": u_top,
        "H": H, "n_steps": n_steps, "warmup": warmup,
        "Cf_exact": float(Cf_exact),
        "bb_fix": True,
        "lbm_step": "lbm_step_correct",
        "friction_formula": "standard",
        "Cf_mean": float(cf_mean),
        "Cf_err_pct": float(cf_err),
        "u_at_y1": float(u_prof[1]),
        "u_exact_y1": float(u_exact_1),
        "u_err_pct_y1": float(u_err_1),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    print(f"\n{tag} === FINAL (Cf_exact={Cf_exact:.6f}) ===", flush=True)
    print(f"{tag} Cf={cf_mean:.6f} err={cf_err:.2f}%", flush=True)
    print(f"{tag} u[1]={float(u_prof[1]):.6f} exact={u_exact_1:.6f} "
          f"u_err={u_err_1:.2f}%", flush=True)
    print(f"{tag} time={elapsed:.0f}s", flush=True)
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ===================================================================
# Main
# ===================================================================
def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python grid_conv_bbfix_worker.py suboff  <L> <formula> <device_id> <out>")
        print("  python grid_conv_bbfix_worker.py bfs     <device_id> <out>")
        print("  python grid_conv_bbfix_worker.py couette <ny> <device_id> <out>")
        sys.exit(1)

    case = sys.argv[1]

    if case == "suboff":
        if len(sys.argv) < 6:
            print("Usage: python grid_conv_bbfix_worker.py suboff <L> <formula> <device_id> <out>")
            sys.exit(1)
        L = int(sys.argv[2])
        formula = sys.argv[3]
        device_id = int(sys.argv[4])
        output_path = sys.argv[5]
        run_suboff(device_id, L, formula, output_path)
    elif case == "bfs":
        if len(sys.argv) < 4:
            print("Usage: python grid_conv_bbfix_worker.py bfs <device_id> <out>")
            sys.exit(1)
        device_id = int(sys.argv[2])
        output_path = sys.argv[3]
        run_bfs(device_id, output_path)
    elif case == "couette":
        if len(sys.argv) < 5:
            print("Usage: python grid_conv_bbfix_worker.py couette <ny> <device_id> <out>")
            sys.exit(1)
        ny = int(sys.argv[2])
        device_id = int(sys.argv[3])
        output_path = sys.argv[4]
        run_couette(device_id, ny, output_path)
    else:
        print(f"Unknown case: {case}")
        sys.exit(1)


if __name__ == "__main__":
    main()
