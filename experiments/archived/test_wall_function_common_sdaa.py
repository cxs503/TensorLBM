#!/usr/bin/env python3
"""Wall function common module — high Re stability tests (SDAA 16-19).

Tests the wall function as a proper common module that REPLACES bounce-back
(not additive).  Key finding: WF mode = collide→NoDynamics→stream→WF→BC.

Tests:
  SDAA:16 → Cylinder Re=3900  (WF, y_val=1.0, ramp 0.02→0.08, 5000 steps)
  SDAA:17 → NACA Re=1000      (WF, 6L domain, 10000 steps, vs BB)
  SDAA:18 → Channel Re_tau=180 (WF, 64³, 20000 steps, vs DNS Cf=0.00735)
  SDAA:19 → SUBOFF Re=10000    (WF, L=80, 5000 steps)

Usage:
    PYTHONPATH=src python test_wall_function_common_sdaa.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tensorlbm.d3q19 import C, OPPOSITE, W, equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_function_common import (
    apply_wall_function,
    compute_u_tau,
    compute_y_plus,
    velocity_ramp,
    _near_wall_mask,
)


# ─── D3Q19 opposite-pair list (skip rest=0) ───
_OPP_PAIRS = []
for _i in range(1, 19):
    _j = int(OPPOSITE[_i].item())
    if _j > _i:
        _OPP_PAIRS.append((_i, _j))


# ════════════════════════════════════════════════════════════════════
# GEOMETRY BUILDERS
# ════════════════════════════════════════════════════════════════════

def build_cylinder(nx, ny, nz, device, diameter=24.0):
    radius = diameter / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy = nx * 0.25, ny * 0.5
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def naca_half_thickness(x_over_c, t=0.12):
    a = x_over_c.clamp(min=1e-12)
    return (t / 0.2) * (
        0.2969 * torch.sqrt(a) - 0.1260 * a - 0.3516 * a * a
        + 0.2843 * a ** 3 - 0.1015 * a ** 4
    )


def build_naca_airfoil(nx, ny, nz, device, chord=200.0, t=0.12, m=0.0, p=0.4):
    cx_le = nx * 0.1  # LE at 10% from inlet (6L domain)
    cy_center = ny * 0.5
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    yy = yy.unsqueeze(0).expand(nz, ny, nx)
    xx = xx.unsqueeze(0).expand(nz, ny, nx)
    x_norm = (xx - cx_le) / chord
    half_t = chord * naca_half_thickness(x_norm, t)
    in_chord = (x_norm >= 0.0) & (x_norm <= 1.0)
    in_profile = (yy >= cy_center - half_t) & (yy <= cy_center + half_t)
    solid = in_chord & in_profile
    return solid


def build_suboff(nx, ny, nz, device, length=80.0):
    from tensorlbm.suboff_cad import suboff_hull_mask
    cx = nx * 0.35
    cy = ny * 0.5
    cz = nz * 0.5
    radius = length * 0.05834
    solid = suboff_hull_mask(nx, ny, nz, cx, cy, cz, length, radius, device)
    return solid


def build_channel_walls(nx, ny, nz, device):
    """Channel with walls at y=0 and y=ny-1."""
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True   # bottom wall
    solid[:, -1, :] = True  # top wall
    return solid


# ════════════════════════════════════════════════════════════════════
# DRAG COMPUTATION
# ════════════════════════════════════════════════════════════════════

def drag_ladd(f, near, solid, dpS):
    """Full Ladd (1994) momentum-exchange drag."""
    c = C.to(f.device).float()
    opp = OPPOSITE.to(f.device)
    cx_k = c[:, 0].view(19, 1, 1, 1)
    dfric = torch.zeros(1, device=f.device)
    for i in range(1, 19):
        opp_i = int(opp[i].item())
        ci = c[i]
        dk, dj, di = int(ci[2].item()), int(ci[1].item()), int(ci[0].item())
        solid_shifted = torch.roll(solid, (dk, dj, di), dims=(0, 1, 2))
        crossing = near & solid_shifted
        if crossing.any():
            f_opp_solid = torch.roll(f[opp_i], (dk, dj, di), dims=(0, 1, 2))
            dfric -= ((f[i] + f_opp_solid) * cx_k[i] * crossing.float()).sum()
    return float(dfric.item() / dpS)


def drag_wf(f, solid, near, nu, y_val, dpS):
    """Wall-function drag: friction from tau_w + pressure drag."""
    rho, ux, uy, uz = macroscopic3d(f)
    u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
    u_tau = compute_u_tau(u_mag, nu, y_val=y_val, wall_law="log")
    u_tau = torch.where(near, u_tau, torch.zeros_like(u_tau))
    tau_w = u_tau * u_tau
    # Friction drag: integrate tau_w in x-direction
    drag_fric = float((tau_w * near.to(f.dtype)).sum().item()) / dpS
    # Pressure drag
    p = (rho - 1.0) / 3.0
    fluid = ~solid
    sp = torch.roll(solid, 1, dims=2)
    sm = torch.roll(solid, -1, dims=2)
    drag_pres = float((-p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()) / dpS
    return drag_fric + drag_pres, drag_fric, drag_pres


# ════════════════════════════════════════════════════════════════════
# FAR-FIELD BC FOR 2D EXTRUDED
# ════════════════════════════════════════════════════════════════════

def far_field_bc_2d_extruded(f, u_in):
    rho1 = torch.ones((f.shape[1], f.shape[2], f.shape[3]),
                      dtype=f.dtype, device=f.device)
    feq = equilibrium3d(
        rho1, torch.full_like(rho1, u_in),
        torch.zeros_like(rho1), torch.zeros_like(rho1),
        device=f.device,
    )
    f[:, :, :, 0] = feq[:, :, :, 0]
    f[:, :, :, -1] = f[:, :, :, -2]
    f[:, :, 0, :] = feq[:, :, 0, :]
    f[:, :, -1, :] = feq[:, :, -1, :]
    return f


# ════════════════════════════════════════════════════════════════════
# CHANNEL FLOW BODY FORCE
# ════════════════════════════════════════════════════════════════════

def apply_body_force_channel(f, force_x):
    """Apply constant body force to drive channel flow (Guo forcing)."""
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    w_view = w.view(19, 1, 1, 1)
    cs2 = 1.0 / 3.0
    rho, ux, uy, uz = macroscopic3d(f)
    cu_u = cx * ux.unsqueeze(0)
    cu_f = cx * force_x
    forcing = w_view * (1.0 + cu_u / cs2) * cu_f / cs2
    return f + forcing


# ════════════════════════════════════════════════════════════════════
# TEST 1: CYLINDER Re=3900 (SDAA:16)
# ════════════════════════════════════════════════════════════════════

def test_cylinder_re3900(device_id=16):
    """Cylinder Re=3900 with WF, y_val=1.0, velocity ramp, MRT+Smag(0.15)."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id} CylinderRe3900]"

    nx, ny, nz = 200, 80, 4
    diameter = 24.0
    u_target = 0.08
    Re = 3900
    nu = u_target * diameter / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.15
    n_steps = 5000
    warmup = 1000
    y_val = 1.0

    print(f"{tag} nx={nx} ny={ny} nz={nz} u_target={u_target} nu={nu:.6e} "
          f"tau={tau:.8f} Re={Re} Cs={cs_smag} y_val={y_val}", flush=True)

    solid = build_cylinder(nx, ny, nz, device, diameter=diameter)
    near = _near_wall_mask(solid)
    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"{tag} solid={n_solid} near={n_near}", flush=True)

    # Reference: frontal area = D * span
    S_ref = diameter * nz
    dpS = 0.5 * 1.0 * u_target ** 2 * S_ref

    t0 = time.time()
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), 0.02, device=device)  # start at ramp start
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      device=device)
    initial_mass = float(f.sum().item())

    cd_window = deque(maxlen=300)
    diverged = False
    yp_stats = []

    for step in range(1, n_steps + 1):
        # Velocity ramp
        u_in = velocity_ramp(step, u_target, u_start=0.02, ramp_steps=1000)

        # 1. Collision
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 2. NoDynamics
        f_pre = f.clone()
        sm = solid.unsqueeze(0).expand_as(f)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 3. Streaming (skip BB in WF mode)
        f = stream3d(f)

        # 4. Wall function (replaces BB, after streaming)
        f, diag = apply_wall_function(f, solid, near, nu=nu, y_val=y_val,
                                       lattice="D3Q19")

        # 5. Far-field BC
        f = far_field_bc_2d_extruded(f, u_in)

        # 6. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # 7. Drag
        cd_val = drag_ladd(f, near, solid, dpS)
        if step > warmup and math.isfinite(cd_val):
            cd_window.append(cd_val)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            diverged = True
            break

        if step % 500 == 0:
            cd_avg = sum(cd_window) / max(len(cd_window), 1) if cd_window else float("nan")
            elapsed = time.time() - t0
            print(f"{tag} step={step} u_in={u_in:.4f} Cd={cd_val:.4f} "
                  f"avg={cd_avg:.4f} yp_mean={diag['y_plus_mean']:.2f} "
                  f"yp_max={diag['y_plus_max']:.2f} bb={diag['n_bb_cells']} "
                  f"wf={diag['n_wf_cells']} ({elapsed:.0f}s)", flush=True)
            yp_stats.append({
                "step": step, "y_plus_mean": diag["y_plus_mean"],
                "y_plus_max": diag["y_plus_max"],
                "n_bb": diag["n_bb_cells"], "n_wf": diag["n_wf_cells"],
            })

    elapsed = time.time() - t0
    if diverged:
        cd_mean, cd_std, status = float("nan"), float("nan"), "DIV"
    else:
        cd_mean = sum(cd_window) / max(len(cd_window), 1) if cd_window else float("nan")
        cd_std = (sum((c - cd_mean) ** 2 for c in cd_window) /
                  max(len(cd_window) - 1, 1)) ** 0.5 if len(cd_window) > 1 else 0.0
        status = "OK" if math.isfinite(cd_mean) else "DIV"

    # Reference Cd for cylinder Re=3900 ≈ 0.98 (experimental)
    cd_ref = 0.98
    err_pct = abs(cd_mean - cd_ref) / cd_ref * 100 if cd_ref > 0 and math.isfinite(cd_mean) else float("nan")

    result = {
        "test": "cylinder_Re3900",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "Re": Re, "nu": nu, "tau": tau, "u_target": u_target,
        "cs_smag": cs_smag, "y_val": y_val,
        "n_steps": n_steps, "warmup": warmup,
        "Cd_mean": cd_mean, "Cd_std": cd_std, "Cd_ref": cd_ref,
        "error_pct": err_pct, "status": status,
        "elapsed_s": elapsed,
        "y_plus_stats": yp_stats[-5:] if yp_stats else [],
    }
    print(f"{tag} DONE Cd={cd_mean:.4f} (ref={cd_ref}) err={err_pct:.1f}% "
          f"status={status} time={elapsed:.0f}s", flush=True)
    return result


# ════════════════════════════════════════════════════════════════════
# TEST 2: NACA Re=1000 (SDAA:17)
# ════════════════════════════════════════════════════════════════════

def test_naca_re1000(device_id=17):
    """NACA 0012 Re=1000 with WF, 6L domain, 10000 steps, vs BB."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id} NACARe1000]"

    chord = 100.0
    nx = int(6 * chord)  # 6L domain = 600
    ny = 100
    nz = 4
    u_target = 0.06
    Re = 1000
    nu = u_target * chord / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.15
    n_steps = 10000
    warmup = 2000
    y_val = 1.0

    print(f"{tag} nx={nx} ny={ny} nz={nz} u_target={u_target} nu={nu:.6e} "
          f"tau={tau:.8f} Re={Re} Cs={cs_smag} y_val={y_val}", flush=True)

    solid = build_naca_airfoil(nx, ny, nz, device, chord=chord, t=0.12, m=0.0, p=0.4)
    near = _near_wall_mask(solid)
    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"{tag} solid={n_solid} near={n_near}", flush=True)

    S_ref = chord * nz
    dpS = 0.5 * 1.0 * u_target ** 2 * S_ref

    t0 = time.time()
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), 0.02, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      device=device)
    initial_mass = float(f.sum().item())

    cd_window_wf = deque(maxlen=500)
    cd_window_bb = deque(maxlen=500)
    diverged = False

    # Run WF mode
    f_wf = f.clone()
    for step in range(1, n_steps + 1):
        u_in = velocity_ramp(step, u_target, u_start=0.02, ramp_steps=1000)

        f_wf = collide_smagorinsky_mrt3d(f_wf, tau=tau, C_s=cs_smag)
        f_pre = f_wf.clone()
        sm = solid.unsqueeze(0).expand_as(f_wf)
        for q in range(19):
            f_wf[q] = torch.where(sm[q], f_pre[q], f_wf[q])
        f_wf = stream3d(f_wf)
        f_wf, diag = apply_wall_function(f_wf, solid, near, nu=nu, y_val=y_val,
                                          lattice="D3Q19")
        f_wf = far_field_bc_2d_extruded(f_wf, u_in)
        if step % 200 == 0:
            f_wf = correct_mass3d(f_wf, initial_mass)

        cd_val = drag_ladd(f_wf, near, solid, dpS)
        if step > warmup and math.isfinite(cd_val):
            cd_window_wf.append(cd_val)

        if not torch.isfinite(f_wf).all():
            print(f"{tag} WF DIVERGED at step {step}", flush=True)
            diverged = True
            break

        if step % 1000 == 0:
            cd_avg = sum(cd_window_wf) / max(len(cd_window_wf), 1) if cd_window_wf else float("nan")
            elapsed = time.time() - t0
            print(f"{tag} WF step={step} Cd={cd_val:.6f} avg={cd_avg:.6f} "
                  f"yp_mean={diag['y_plus_mean']:.2f} ({elapsed:.0f}s)", flush=True)

    # Run BB mode for comparison (shorter, 5000 steps)
    print(f"{tag} Starting BB comparison run (5000 steps)...", flush=True)
    f_bb = f.clone()
    t_bb = time.time()
    for step in range(1, 5001):
        u_in = velocity_ramp(step, u_target, u_start=0.02, ramp_steps=1000)
        f_bb = collide_smagorinsky_mrt3d(f_bb, tau=tau, C_s=cs_smag)
        f_pre = f_bb.clone()
        sm = solid.unsqueeze(0).expand_as(f_bb)
        for q in range(19):
            f_bb[q] = torch.where(sm[q], f_pre[q], f_bb[q])
        f_bb = bounce_back_cells_3d(f_bb, solid, f_pre=f_pre)
        f_bb = stream3d(f_bb)
        f_bb = far_field_bc_2d_extruded(f_bb, u_in)
        if step % 200 == 0:
            f_bb = correct_mass3d(f_bb, initial_mass)

        cd_val = drag_ladd(f_bb, near, solid, dpS)
        if step > 1000 and math.isfinite(cd_val):
            cd_window_bb.append(cd_val)

        if not torch.isfinite(f_bb).all():
            print(f"{tag} BB DIVERGED at step {step}", flush=True)
            break

        if step % 1000 == 0:
            cd_avg = sum(cd_window_bb) / max(len(cd_window_bb), 1) if cd_window_bb else float("nan")
            elapsed = time.time() - t_bb
            print(f"{tag} BB step={step} Cd={cd_val:.6f} avg={cd_avg:.6f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    cd_wf = sum(cd_window_wf) / max(len(cd_window_wf), 1) if cd_window_wf else float("nan")
    cd_bb = sum(cd_window_bb) / max(len(cd_window_bb), 1) if cd_window_bb else float("nan")

    # NACA 0012 Re=1000 Cd ≈ 0.011 (experimental/XFoil)
    cd_ref = 0.011
    err_wf = abs(cd_wf - cd_ref) / cd_ref * 100 if cd_ref > 0 and math.isfinite(cd_wf) else float("nan")
    err_bb = abs(cd_bb - cd_ref) / cd_ref * 100 if cd_ref > 0 and math.isfinite(cd_bb) else float("nan")

    result = {
        "test": "naca_Re1000",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "Re": Re, "nu": nu, "tau": tau, "u_target": u_target,
        "cs_smag": cs_smag, "y_val": y_val,
        "n_steps_wf": n_steps, "n_steps_bb": 5000,
        "Cd_wf": cd_wf, "Cd_bb": cd_bb, "Cd_ref": cd_ref,
        "error_wf_pct": err_wf, "error_bb_pct": err_bb,
        "status": "DIV" if diverged else "OK",
        "elapsed_s": elapsed,
    }
    print(f"{tag} DONE WF Cd={cd_wf:.6f} (err={err_wf:.1f}%) "
          f"BB Cd={cd_bb:.6f} (err={err_bb:.1f}%) time={elapsed:.0f}s", flush=True)
    return result


# ════════════════════════════════════════════════════════════════════
# TEST 3: CHANNEL Re_tau=180 (SDAA:18)
# ════════════════════════════════════════════════════════════════════

def test_channel_retau180(device_id=18):
    """Channel flow Re_tau=180 with WF, 64³, 20000 steps, vs DNS Cf=0.00735."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id} ChannelReTau180]"

    nx = ny = nz = 64
    Re_tau = 180
    delta = ny // 2  # half-channel height
    u_bulk = 0.1
    # Cf = 2*(u_tau/u_bulk)^2, DNS Cf=0.00735
    # u_tau = u_bulk * sqrt(Cf/2) = u_bulk * sqrt(0.003675)
    u_tau_target = u_bulk * math.sqrt(0.00735 / 2.0)
    nu = u_tau_target * delta / Re_tau
    tau = 3.0 * nu + 0.5
    cs_smag = 0.15
    n_steps = 10000
    warmup = 3000
    y_val = 0.5  # half-cell distance for channel walls

    # Body force to drive flow: F = u_tau^2 / delta (per unit mass)
    force_x = u_tau_target ** 2 / delta

    print(f"{tag} nx={nx} ny={ny} nz={nz} u_bulk={u_bulk} nu={nu:.6e} "
          f"tau={tau:.8f} Re_tau={Re_tau} Cs={cs_smag} y_val={y_val} "
          f"u_tau_target={u_tau_target:.6f} force={force_x:.6e}", flush=True)

    solid = build_channel_walls(nx, ny, nz, device)
    near = _near_wall_mask(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near={n_near}", flush=True)

    t0 = time.time()
    rho0 = torch.ones((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=device)
    initial_mass = float(f.sum().item())

    cf_window = deque(maxlen=500)
    diverged = False
    yp_stats = []

    for step in range(1, n_steps + 1):
        # 1. Collision
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 2. NoDynamics
        f_pre = f.clone()
        sm = solid.unsqueeze(0).expand_as(f)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 3. Streaming (periodic in all directions for channel)
        f = stream3d(f)

        # 4. Wall function (replaces BB at channel walls)
        f, diag = apply_wall_function(f, solid, near, nu=nu, y_val=y_val,
                                       lattice="D3Q19")

        # 5. Body force (drive the flow)
        f = apply_body_force_channel(f, force_x)

        # 6. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # 7. Compute Cf
        if step % 100 == 0:
            rho, ux, uy, uz = macroscopic3d(f)
            u_bulk_actual = float(ux[~solid].mean().item())
            # Compute u_tau from wall function
            u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
            u_tau_field = compute_u_tau(u_mag, nu, y_val=y_val, wall_law="log")
            u_tau_field = torch.where(near, u_tau_field, torch.zeros_like(u_tau_field))
            u_tau_actual = float(u_tau_field[near].mean().item()) if n_near > 0 else 0.0
            cf = 2.0 * (u_tau_actual / max(u_bulk_actual, 1e-8)) ** 2
            if step > warmup and math.isfinite(cf):
                cf_window.append(cf)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            diverged = True
            break

        if step % 1000 == 0:
            cf_avg = sum(cf_window) / max(len(cf_window), 1) if cf_window else float("nan")
            elapsed = time.time() - t0
            print(f"{tag} step={step} u_bulk={u_bulk_actual:.4f} "
                  f"u_tau={u_tau_actual:.6f} Cf={cf:.6f} avg={cf_avg:.6f} "
                  f"yp_mean={diag['y_plus_mean']:.2f} ({elapsed:.0f}s)", flush=True)
            yp_stats.append({
                "step": step, "y_plus_mean": diag["y_plus_mean"],
                "u_bulk": u_bulk_actual, "u_tau": u_tau_actual, "Cf": cf,
            })

    elapsed = time.time() - t0
    cf_mean = sum(cf_window) / max(len(cf_window), 1) if cf_window else float("nan")
    cf_ref = 0.00735  # DNS (Moser et al.)
    err_pct = abs(cf_mean - cf_ref) / cf_ref * 100 if cf_ref > 0 and math.isfinite(cf_mean) else float("nan")

    result = {
        "test": "channel_retau180",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "Re_tau": Re_tau, "nu": nu, "tau": tau,
        "u_bulk_target": u_bulk, "u_tau_target": u_tau_target,
        "cs_smag": cs_smag, "y_val": y_val,
        "n_steps": n_steps, "warmup": warmup,
        "Cf_mean": cf_mean, "Cf_ref": cf_ref,
        "error_pct": err_pct,
        "status": "DIV" if diverged else "OK",
        "elapsed_s": elapsed,
        "y_plus_stats": yp_stats[-5:] if yp_stats else [],
    }
    print(f"{tag} DONE Cf={cf_mean:.6f} (ref={cf_ref}) err={err_pct:.1f}% "
          f"status={'DIV' if diverged else 'OK'} time={elapsed:.0f}s", flush=True)
    return result


# ════════════════════════════════════════════════════════════════════
# TEST 4: SUBOFF Re=10000 (SDAA:19)
# ════════════════════════════════════════════════════════════════════

def test_suboff_re10000(device_id=19):
    """SUBOFF Re=10000 with WF, L=80, 5000 steps."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id} SUBOFFRe10000]"

    length = 80.0
    nx, ny, nz = 160, 80, 80
    u_target = 0.06
    Re = 10000
    nu = u_target * length / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.15
    n_steps = 5000
    warmup = 1000
    y_val = 1.0

    print(f"{tag} nx={nx} ny={ny} nz={nz} u_target={u_target} nu={nu:.6e} "
          f"tau={tau:.8f} Re={Re} Cs={cs_smag} y_val={y_val}", flush=True)

    solid = build_suboff(nx, ny, nz, device, length=length)
    near = _near_wall_mask(solid)
    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"{tag} solid={n_solid} near={n_near}", flush=True)

    # Wetted surface area ≈ near-wall cells
    S_ref = float(n_near)
    dpS = 0.5 * 1.0 * u_target ** 2 * S_ref

    t0 = time.time()
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), 0.02, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      device=device)
    initial_mass = float(f.sum().item())

    cd_window = deque(maxlen=300)
    diverged = False
    yp_stats = []

    for step in range(1, n_steps + 1):
        u_in = velocity_ramp(step, u_target, u_start=0.02, ramp_steps=1000)

        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f_pre = f.clone()
        sm = solid.unsqueeze(0).expand_as(f)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = stream3d(f)
        f, diag = apply_wall_function(f, solid, near, nu=nu, y_val=y_val,
                                       lattice="D3Q19")
        f = far_field_bc_3d(f, u_in)
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Drag: use WF drag (friction + pressure)
        cd_val, cd_fric, cd_pres = drag_wf(f, solid, near, nu, y_val, dpS)
        if step > warmup and math.isfinite(cd_val):
            cd_window.append(cd_val)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            diverged = True
            break

        if step % 500 == 0:
            cd_avg = sum(cd_window) / max(len(cd_window), 1) if cd_window else float("nan")
            elapsed = time.time() - t0
            print(f"{tag} step={step} Cd={cd_val:.6f} avg={cd_avg:.6f} "
                  f"fric={cd_fric:.6f} pres={cd_pres:.6f} "
                  f"yp_mean={diag['y_plus_mean']:.2f} "
                  f"yp_max={diag['y_plus_max']:.2f} ({elapsed:.0f}s)", flush=True)
            yp_stats.append({
                "step": step, "y_plus_mean": diag["y_plus_mean"],
                "y_plus_max": diag["y_plus_max"],
                "n_bb": diag["n_bb_cells"], "n_wf": diag["n_wf_cells"],
            })

    elapsed = time.time() - t0
    if diverged:
        cd_mean, status = float("nan"), "DIV"
    else:
        cd_mean = sum(cd_window) / max(len(cd_window), 1) if cd_window else float("nan")
        status = "OK" if math.isfinite(cd_mean) else "DIV"

    result = {
        "test": "suboff_Re10000",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "Re": Re, "nu": nu, "tau": tau, "u_target": u_target,
        "cs_smag": cs_smag, "y_val": y_val, "length": length,
        "n_steps": n_steps, "warmup": warmup,
        "Cd_mean": cd_mean, "status": status,
        "elapsed_s": elapsed,
        "y_plus_stats": yp_stats[-5:] if yp_stats else [],
    }
    print(f"{tag} DONE Cd={cd_mean:.6f} status={status} time={elapsed:.0f}s", flush=True)
    return result


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Wall Function Common Module — High Re Stability Tests (SDAA 16-19)")
    print("=" * 70)
    print()

    results = {}

    # Run tests sequentially (each on its own SDAA card)
    # Test 1: Cylinder Re=3900 (SDAA:16)
    try:
        results["cylinder_Re3900"] = test_cylinder_re3900(device_id=16)
    except Exception as e:
        print(f"Cylinder Re=3900 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["cylinder_Re3900"] = {"test": "cylinder_Re3900", "status": "ERROR", "error": str(e)}

    print()

    # Test 2: NACA Re=1000 (SDAA:17)
    try:
        results["naca_Re1000"] = test_naca_re1000(device_id=17)
    except Exception as e:
        print(f"NACA Re=1000 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["naca_Re1000"] = {"test": "naca_Re1000", "status": "ERROR", "error": str(e)}

    print()

    # Test 3: Channel Re_tau=180 (SDAA:18)
    try:
        results["channel_retau180"] = test_channel_retau180(device_id=18)
    except Exception as e:
        print(f"Channel Re_tau=180 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["channel_retau180"] = {"test": "channel_retau180", "status": "ERROR", "error": str(e)}

    print()

    # Test 4: SUBOFF Re=10000 (SDAA:19)
    try:
        results["suboff_Re10000"] = test_suboff_re10000(device_id=19)
    except Exception as e:
        print(f"SUBOFF Re=10000 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["suboff_Re10000"] = {"test": "suboff_Re10000", "status": "ERROR", "error": str(e)}

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        status = r.get("status", "UNKNOWN")
        if name == "cylinder_Re3900":
            cd = r.get("Cd_mean", float("nan"))
            err = r.get("error_pct", float("nan"))
            print(f"  {name}: status={status} Cd={cd:.4f} err={err:.1f}%")
        elif name == "naca_Re1000":
            cd_wf = r.get("Cd_wf", float("nan"))
            cd_bb = r.get("Cd_bb", float("nan"))
            err_wf = r.get("error_wf_pct", float("nan"))
            err_bb = r.get("error_bb_pct", float("nan"))
            print(f"  {name}: status={status} Cd_wf={cd_wf:.6f}(err={err_wf:.1f}%) "
                  f"Cd_bb={cd_bb:.6f}(err={err_bb:.1f}%)")
        elif name == "channel_retau180":
            cf = r.get("Cf_mean", float("nan"))
            err = r.get("error_pct", float("nan"))
            print(f"  {name}: status={status} Cf={cf:.6f} err={err:.1f}%")
        elif name == "suboff_Re10000":
            cd = r.get("Cd_mean", float("nan"))
            print(f"  {name}: status={status} Cd={cd:.6f}")

    # Save results
    out_path = Path("/tmp/wall_function_common_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
