#!/usr/bin/env python3
"""Debug DG-LBM NaN: run a single step and check where NaN appears."""
from __future__ import annotations
import json, os, sys, time, traceback, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import shutil

import torch
import torch.nn.functional as F

from tensorlbm.dg_lbm import (
    DGLBMSuboffConfig, build_suboff_mask, build_dg_hull_band_mask,
)
from tensorlbm.dg_band import (
    build_band_topology, hybrid_step, project_band_to_lbm,
    compute_dg_solid_force, dg_lbm_step_band, write_back_exports,
)
from tensorlbm.dg_advection import get_ops
from tensorlbm.boundaries3d import (
    apply_simple_channel_boundaries_3d, make_channel_wall_mask_3d,
)
from tensorlbm.d3q19 import C, OPPOSITE, W, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d, collide_bgk3d, correct_mass3d

def check_nan(name, *tensors):
    for i, t in enumerate(tensors):
        if t.isnan().any():
            print(f"  ❌ NaN in {name}[{i}]: shape={t.shape}, n_nan={t.isnan().sum().item()}")
            return True
    return False

def run_debug_one_step():
    nx, ny, nz = 32, 32, 32
    hl = 16.0
    re = 50.0
    u_in = 0.06
    dg_band = 3.0
    dg_substeps = 4
    device = "cpu"
    ftype = torch.float32

    nu = u_in * hl / re
    tau = 3.0 * nu + 0.5
    tau_dg = tau - 0.5
    print(f"τ={tau:.4f}, τ_dg={tau_dg:.4f}, ν={nu:.4e}")

    cx, cy, cz = nx * 0.35, ny * 0.5, nz * 0.5
    obstacle, _ = build_suboff_mask("bare_hull", nx, ny, nz, cx, cy, cz, hl, device=device)
    band_mask = build_dg_hull_band_mask(obstacle, dg_band)
    print(f"  obstacle cells: {obstacle.sum().item()}, band cells: {band_mask.sum().item()}")
    topo = build_band_topology(band_mask, solid_mask=obstacle, periodic=False).to(device)
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, obstacle, device=device)
    print(f"  wall mask: {wall_mask.sum().item()}")

    # Initial field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.zeros_like(ux0)
    ux0[obstacle] = 0.0
    f_lbm = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    check_nan("f_lbm init", f_lbm)

    # DG field
    cb = topo.band_coords
    f_dg = f_lbm[:, cb[:, 0], cb[:, 1], cb[:, 2]]
    nn = 2  # P1 → 2 nodes per dim
    f_dg = f_dg.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, nn, nn, nn).contiguous().clone()
    check_nan("f_dg init", f_dg)

    ops = get_ops(degree=1, dx=1.0, dtype=ftype, device=device)
    C_d = C.to(ftype).to(device)
    W_d = W.to(ftype).to(device)
    opp = OPPOSITE.to(device)

    # ---- Manual step-by-step ----
    print("\n=== Step 1 ===")

    # Sub-step A: Collide exterior
    f_collided = collide_bgk3d(f_lbm.clone(), tau)
    check_nan("after collide_bgk3d", f_collided)
    print(f"  collide_bgk3d: f range [{f_collided.min():.6f}, {f_collided.max():.6f}]")

    # Sub-step B: Band MoL
    Q, *shape = f_collided.shape
    ext_field = f_collided.reshape(Q, int(torch.tensor(shape).prod().item()))
    check_nan("ext_field", ext_field)

    f_dg_new = dg_lbm_step_band(
        f_dg, C_d, W_d, tau_dg, ops, topo, ext_field,
        dt=1.0, n_substeps=dg_substeps, scheme="rk3", opposite=opp,
    )
    check_nan("after dg_lbm_step_band", f_dg_new)
    print(f"  dg_lbm_step_band: f_dg range [{f_dg_new.min():.6f}, {f_dg_new.max():.6f}]")

    # Sub-step C: Stream exterior
    f_streamed = stream3d(f_collided)
    check_nan("after stream3d", f_streamed)
    print(f"  stream3d: f range [{f_streamed.min():.6f}, {f_streamed.max():.6f}]")

    # Sub-step D: Write-back DG traces
    f_wb = write_back_exports(f_streamed, f_dg_new, C_d, ops, topo)
    check_nan("after write_back_exports", f_wb)

    # Sub-step E: Project band to LBM
    f_proj = project_band_to_lbm(f_wb, f_dg_new, topo)
    check_nan("after project_band_to_lbm", f_proj)
    print(f"  project_band_to_lbm: f range [{f_proj.min():.6f}, {f_proj.max():.6f}]")

    # Sub-step F: Boundary conditions
    f_bc = apply_simple_channel_boundaries_3d(f_proj, u_in, wall_mask, obstacle)
    check_nan("after apply_simple_channel_boundaries_3d", f_bc)
    print(f"  BC: f range [{f_bc.min():.6f}, {f_bc.max():.6f}]")

    # Check macroscopic
    rho, ux, uy, uz = macroscopic3d(f_bc)
    print(f"  Macro: rho range [{rho.min():.6f}, {rho.max():.6f}], "
          f"ux range [{ux.min():.6f}, {ux.max():.6f}]")
    check_nan("macro", rho, ux, uy, uz)

    # Run a few more steps to see instability develop
    f_lbm = f_bc
    f_dg = f_dg_new
    for step in range(2, 21):
        f_lbm, f_dg = hybrid_step(
            f_lbm, f_dg, C_d, W_d, ops, topo, tau_lbm=tau,
            dt=1.0, n_substeps=dg_substeps, opposite=opp,
        )
        if check_nan(f"step {step} after hybrid_step", f_lbm, f_dg):
            break
        f_lbm = project_band_to_lbm(f_lbm, f_dg, topo)
        f_lbm = apply_simple_channel_boundaries_3d(f_lbm, u_in, wall_mask, obstacle)
        if check_nan(f"step {step} after BC", f_lbm):
            break
        rho, ux, uy, uz = macroscopic3d(f_lbm)
        print(f"  Step {step}: rho [{rho.min():.6f}, {rho.max():.6f}], "
              f"ux [{ux.min():.6f}, {ux.max():.6f}], "
              f"nan={rho.isnan().any().item()}")
        if rho.isnan().any():
            print(f"  ❌ NaN at step {step}!")
            break

if __name__ == "__main__":
    run_debug_one_step()
