#!/usr/bin/env python3
"""Quick comparison of wall function configurations on Poiseuille flow.

Tries multiple configurations to find the one that gives 0% error.
"""
from __future__ import annotations
import sys, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
import torch
from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d, correct_mass3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.ibm import ibm_apply_body_force_3d

DTYPE = torch.float32

def guo_body_force_3d(f, fx, fy, fz, tau):
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    factor = (1.0 - 1.0 / (2.0 * tau))
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
    wv = w.view(19, 1, 1, 1)
    forcing = factor * wv * 3.0 * (
        cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0) + cz * fz.unsqueeze(0))
    return f + forcing

def run_poiseuille(device, config_name, nsteps=3000,
                   use_bb=False, use_nodynamics=True, set_solid_eq=False,
                   use_guo_driving=True, y_val=0.5):
    """Run Poiseuille with given configuration."""
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    u_max_target = 0.05
    H = ny - 2
    G = 8.0 * nu * u_max_target / (H * H)

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    rho0 = torch.ones((nz, ny, nx), dtype=DTYPE, device=device)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=device)
    target_mass = f.sum().item()

    for step in range(nsteps):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau)

        if use_nodynamics:
            sm = solid.unsqueeze(0).expand_as(f)
            f = torch.where(sm, f_pre, f)

        if set_solid_eq:
            rho_w = torch.ones((nz, ny, nx), dtype=DTYPE, device=device)
            f_eq = equilibrium3d(rho_w, torch.zeros_like(rho_w),
                                  torch.zeros_like(rho_w), torch.zeros_like(rho_w),
                                  device=device)
            sm = solid.unsqueeze(0).expand_as(f)
            f = torch.where(sm, f_eq, f)

        if use_bb:
            f = bounce_back_cells_3d(f, solid)

        f, _, _ = wall_function_3d(f, solid, nu, y_val=y_val, wall_law="gradient")
        f = stream3d(f)

        fx = torch.full((nz, ny, nx), G, dtype=DTYPE, device=device)
        fy = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
        fz = torch.zeros((nz, ny, nx), dtype=DTYPE, device=device)
        if use_guo_driving:
            f = guo_body_force_3d(f, fx, fy, fz, tau)
        else:
            f = ibm_apply_body_force_3d(f, fx, fy, fz)

        f = correct_mass3d(f, target_mass)
        if not torch.isfinite(f).all():
            return None, float('inf')

    _, ux, _, _ = macroscopic3d(f)
    u_prof = ux.mean(dim=(0, 2)).cpu().numpy()
    u_exact = np.zeros(ny)
    for y in range(1, ny - 1):
        y_eff = y - 0.5
        u_exact[y] = (G / (2.0 * nu)) * y_eff * (H - y_eff)

    errs = []
    for y in range(1, ny - 1):
        if abs(u_exact[y]) > 1e-10:
            errs.append(abs(u_prof[y] - u_exact[y]) / abs(u_exact[y]) * 100)
    err_max = max(errs) if errs else 0
    err_mean = sum(errs) / len(errs) if errs else 0
    u_max_sim = float(u_prof[ny // 2])
    u_max_err = abs(u_max_sim - u_max_target) / u_max_target * 100
    return u_prof, u_max_err, err_max, err_mean


if __name__ == "__main__":
    device = torch.device("sdaa:4")
    torch.sdaa.set_device(device)

    configs = [
        # (name, use_bb, use_nodynamics, set_solid_eq, use_guo_driving, y_val)
        ("BB only (ref)",            True,  True,  False, True,  0.5),
        ("WF no_ND no_eq no_guo",    False, False, False, False, 0.5),
        ("WF no_ND no_eq guo",       False, False, False, True,  0.5),
        ("WF ND no_eq no_guo",       False, True,  False, False, 0.5),
        ("WF ND no_eq guo",          False, True,  False, True,  0.5),
        ("WF ND eq no_guo",          False, True,  True,  False, 0.5),
        ("WF ND eq guo",             False, True,  True,  True,  0.5),
        ("WF ND eq guo y1.0",        False, True,  True,  True,  1.0),
        ("WF ND no_eq guo y1.0",     False, True,  False, True,  1.0),
        ("BB+WF (bug12)",            True,  True,  False, True,  0.5),
    ]

    print(f"{'Config':<30s} {'u_max_err':>10s} {'err_max':>10s} {'err_mean':>10s}")
    print("-" * 65)
    for name, bb, nd, eq, guo, yv in configs:
        try:
            res = run_poiseuille(device, name, nsteps=2000,
                                use_bb=bb, use_nodynamics=nd,
                                set_solid_eq=eq, use_guo_driving=guo, y_val=yv)
            if res is None:
                print(f"{name:<30s} {'DIVERGED':>10s}")
            else:
                _, ue, em, ea = res
                print(f"{name:<30s} {ue:>10.2f}% {em:>10.2f}% {ea:>10.2f}%")
        except Exception as e:
            print(f"{name:<30s} ERROR: {e}")
