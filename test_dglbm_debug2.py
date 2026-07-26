#!/usr/bin/env python3
"""Debug DG-LBM NaN: isolate which term in dg_lbm_step_band causes NaN."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch

from tensorlbm.dg_lbm import build_suboff_mask, build_dg_hull_band_mask
from tensorlbm.dg_band import (
    build_band_topology, dg_rhs_band, dg_lbm_rhs_band,
    dg_advect_band, dg_lbm_step_band, write_back_exports, project_band_to_lbm,
)
from tensorlbm.dg_advection import get_ops, macroscopic_dg, equilibrium_dg
from tensorlbm.d3q19 import C, OPPOSITE, W, equilibrium3d

def check(name, t):
    n = t.isnan().sum().item()
    if n > 0:
        print(f"  ❌ NaN in {name}: {n}/{t.numel()}")
        # Find where NaN occurs
        nan_idx = torch.nonzero(t.isnan().flatten(), as_tuple=False)
        if len(nan_idx) > 0:
            print(f"     first NaN at flattened idx {nan_idx[0].item()}")
            # Decode shape
            idx = torch.nonzero(t.isnan())
            for i in range(min(3, idx.shape[0])):
                print(f"     nan at {tuple(idx[i].tolist())}")
        return True
    print(f"  ✅ {name}: range [{t.min():.6f}, {t.max():.6f}]")
    return False

def main():
    nx, ny, nz = 32, 32, 32
    hl = 16.0
    re = 50.0
    u_in = 0.06
    dg_band = 3.0
    device = "cpu"
    ftype = torch.float32
    dg_sub = 1  # single substep for debugging

    nu = u_in * hl / re
    tau = 3.0 * nu + 0.5
    tau_dg = tau - 0.5
    print(f"τ={tau:.4f}, τ_dg={tau_dg:.4f}, ν={nu:.4e}")

    cx, cy, cz = nx * 0.35, ny * 0.5, nz * 0.5
    obstacle, _ = build_suboff_mask("bare_hull", nx, ny, nz, cx, cy, cz, hl, device=device)
    band_mask = build_dg_hull_band_mask(obstacle, dg_band)
    topo = build_band_topology(band_mask, solid_mask=obstacle, periodic=False).to(device)
    print(f"  obstacle: {obstacle.sum().item()}, band: {band_mask.sum().item()}")

    # Check neighbor types
    for g in range(3):
        for suffix, arr in [("minus", topo.nbr_type_minus), ("plus", topo.nbr_type_plus)]:
            for typ, name in [(0, "band"), (1, "exterior"), (2, "solid")]:
                cnt = (arr[g] == typ).sum().item()
                if cnt > 0:
                    print(f"  axis {g} {suffix}: {cnt} {name} neighbors")

    # Build initial field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.zeros_like(ux0)
    ux0[obstacle] = 0.0
    f_lbm = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    cb = topo.band_coords
    f_dg = f_lbm[:, cb[:, 0], cb[:, 1], cb[:, 2]]
    nn = 2  # P1
    f_dg = f_dg.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, nn, nn, nn).contiguous().clone()

    C_d = C.to(ftype).to(device)
    W_d = W.to(ftype).to(device)
    opp = OPPOSITE.to(device)
    ops = get_ops(degree=1, dx=1.0, dtype=ftype, device=device)

    # --- Test 1: Advection only ---
    print("\n--- Test 1: dg_rhs_band (advection only) ---")
    adv = dg_rhs_band(f_dg, C_d, ops, topo, ext_field=None, opposite=opp)
    check("dg_rhs_band", adv)

    # Test 1b: With exterior field
    Q = f_lbm.shape[0]
    N = int(torch.tensor(f_lbm.shape[1:]).prod().item())
    ext_field = f_lbm.reshape(Q, N)
    adv_ext = dg_rhs_band(f_dg, C_d, ops, topo, ext_field=ext_field, opposite=opp)
    check("dg_rhs_band (with ext)", adv_ext)

    # --- Test 2: Collision RHS ---
    print("\n--- Test 2: dg_lbm_rhs_band (advection + collision) ---")
    rhs = dg_lbm_rhs_band(f_dg, C_d, W_d, tau_dg, ops, topo, ext_field=ext_field, opposite=opp)
    check("dg_lbm_rhs_band", rhs)

    # --- Test 3: Advection only sub-step ---
    print("\n--- Test 3: dg_advect_band (1 sub-step) ---")
    adv1 = dg_advect_band(f_dg, C_d, ops, topo, ext_field=ext_field, dt=1.0, n_substeps=1, scheme="rk3", opposite=opp)
    check("dg_advect_band rk3", adv1)

    # --- Test 4: Full MoL step ---
    print("\n--- Test 4: dg_lbm_step_band (1 sub-step) ---")
    step1 = dg_lbm_step_band(f_dg, C_d, W_d, tau_dg, ops, topo, ext_field=ext_field, dt=1.0, n_substeps=1, scheme="euler", opposite=opp)
    check("dg_lbm_step_band euler", step1)

    # Also test RK3
    print("\n--- Test 5: dg_lbm_step_band (1 sub-step, RK3) ---")
    step_rk3 = dg_lbm_step_band(f_dg, C_d, W_d, tau_dg, ops, topo, ext_field=ext_field, dt=1.0, n_substeps=1, scheme="rk3", opposite=opp)
    check("dg_lbm_step_band rk3", step_rk3)

    # --- Test 6: Check band topology sanity ---
    print("\n--- Test 6: Band topology sanity ---")
    # For each band cell, verify neighbors
    for g in range(3):
        coords = topo.band_coords
        # Check that solid neighbors actually correspond to obstacle cells
        solid_m = (topo.nbr_type_minus[g] == 2)
        solid_p = (topo.nbr_type_plus[g] == 2)
        if solid_m.any():
            solid_idx = torch.nonzero(solid_m, as_tuple=False).squeeze(-1)
            for idx in solid_idx[:3]:
                b_pos = coords[idx]
                nbr_pos = b_pos.clone()
                nbr_pos[g] -= 1
                nbr_pos = nbr_pos.clamp(0, torch.tensor([nz-1, ny-1, nx-1]))
                is_obstacle = obstacle[tuple(nbr_pos)].item()
                print(f"  solid minus g={g}: band_cell={b_pos.tolist()} → nbr={nbr_pos.tolist()} obstacle={is_obstacle}")

        if solid_p.any():
            solid_idx = torch.nonzero(solid_p, as_tuple=False).squeeze(-1)
            for idx in solid_idx[:3]:
                b_pos = coords[idx]
                nbr_pos = b_pos.clone()
                nbr_pos[g] += 1
                nbr_pos = nbr_pos.clamp(0, torch.tensor([nz-1, ny-1, nx-1]))
                is_obstacle = obstacle[tuple(nbr_pos)].item()
                print(f"  solid plus g={g}: band_cell={b_pos.tolist()} → nbr={nbr_pos.tolist()} obstacle={is_obstacle}")

    # Check if any solid neighbors point to cells that are NOT obstacles
    for g in range(3):
        for suffix, arr in [("minus", topo.nbr_type_minus), ("plus", topo.nbr_type_plus)]:
            solid = (arr[g] == 2)
            if not solid.any():
                continue
            coords = topo.band_coords[solid]
            nbr_coords = coords.clone()
            if suffix == "minus":
                nbr_coords[:, g] -= 1
            else:
                nbr_coords[:, g] += 1
            nbr_coords[:, g] = nbr_coords[:, g].clamp(0, torch.tensor([nz-1, ny-1, nx-1])[g].item())
            for i in range(min(3, nbr_coords.shape[0])):
                nc = nbr_coords[i]
                obs = obstacle[tuple(nc)].item()
                if not obs:
                    print(f"  ⚠️  MISMATCH: solid {suffix} g={g}, band at {coords[i].tolist()}, nbr at {nc.tolist()} is NOT obstacle!")

if __name__ == "__main__":
    main()
