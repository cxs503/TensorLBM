#!/usr/bin/env python3
"""DG-LBM stability: test different substep counts to find threshold."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch

from tensorlbm.dg_lbm import build_suboff_mask, build_dg_hull_band_mask
from tensorlbm.dg_band import build_band_topology, dg_lbm_step_band
from tensorlbm.dg_advection import get_ops
from tensorlbm.d3q19 import C, OPPOSITE, W, equilibrium3d

def main():
    nx, ny, nz = 32, 32, 32
    hl = 16.0
    u_in = 0.06
    re = 50.0
    device = "cpu"
    ftype = torch.float32

    nu = u_in * hl / re
    tau = 3.0 * nu + 0.5
    tau_dg = tau - 0.5
    print(f"τ_lbm={tau:.4f}, τ_dg={tau_dg:.4f}")

    cx, cy, cz = nx * 0.35, ny * 0.5, nz * 0.5
    obstacle, _ = build_suboff_mask("bare_hull", nx, ny, nz, cx, cy, cz, hl, device=device)
    band_mask = build_dg_hull_band_mask(obstacle, 3.0)
    topo = build_band_topology(band_mask, solid_mask=obstacle, periodic=False).to(device)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.zeros_like(ux0)
    ux0[obstacle] = 0.0
    f_lbm = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    cb = topo.band_coords
    f_dg = f_lbm[:, cb[:, 0], cb[:, 1], cb[:, 2]]
    nn = 2
    f_dg = f_dg.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, nn, nn, nn).contiguous().clone()

    C_d = C.to(ftype).to(device)
    W_d = W.to(ftype).to(device)
    opp = OPPOSITE.to(device)
    ops = get_ops(degree=1, dx=1.0, dtype=ftype, device=device)

    Q = f_lbm.shape[0]
    N = int(torch.tensor(f_lbm.shape[1:]).prod().item())
    ext_field = f_lbm.reshape(Q, N)

    # Test stability: tau_dg=0.0576, need dt_sub < 2*tau_dg=0.115 for Euler stability
    # But RK3 is more stable than Euler. Let's check.
    print(f"\nRK3 stability bound for collision: dt_sub < ~2.5*tau_dg = {2.5*tau_dg:.4f}")
    print(f"dt_sub = 1/n_substeps:")

    for n_sub in [1, 2, 4, 8, 10, 12, 14, 16, 18, 20, 24, 32, 40, 50, 64, 80, 100]:
        dt_sub = 1.0 / n_sub
        try:
            result = dg_lbm_step_band(
                f_dg, C_d, W_d, tau_dg, ops, topo, ext_field=ext_field,
                dt=1.0, n_substeps=n_sub, scheme="rk3", opposite=opp,
            )
            has_nan = result.isnan().any().item()
            rng = (result.min().item(), result.max().item()) if not has_nan else (float('nan'), float('nan'))
            if has_nan:
                print(f"  n_sub={n_sub:3d} dt_sub={dt_sub:.4f} → ❌ NaN!")
            elif abs(rng[0]) > 1e3 or abs(rng[1]) > 1e3:
                print(f"  n_sub={n_sub:3d} dt_sub={dt_sub:.4f} → ⚠️  range [{rng[0]:.1f}, {rng[1]:.1f}] (large)")
            else:
                print(f"  n_sub={n_sub:3d} dt_sub={dt_sub:.4f} → ✅ range [{rng[0]:.6f}, {rng[1]:.6f}]")
        except Exception as e:
            print(f"  n_sub={n_sub:3d} dt_sub={dt_sub:.4f} → ❌ {e}")

    # Now test with 10 steps to see if NaN develops over time
    print(f"\n--- Multi-step test with n_sub=50 (dt_sub={1/50:.4f}) ---")
    f_dg_cur = f_dg.clone()
    for step in range(1, 11):
        f_dg_cur = dg_lbm_step_band(
            f_dg_cur, C_d, W_d, tau_dg, ops, topo, ext_field=ext_field,
            dt=1.0, n_substeps=50, scheme="rk3", opposite=opp,
        )
        has_nan = f_dg_cur.isnan().any().item()
        rng = (f_dg_cur.min().item(), f_dg_cur.max().item())
        print(f"  step {step}: NaN={has_nan}, range [{rng[0]:.6f}, {rng[1]:.6f}]")
        if has_nan:
            break

if __name__ == "__main__":
    main()
