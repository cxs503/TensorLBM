#!/usr/bin/env python3
"""Quick test for sphere Re=100 on SDAA - minimal steps to verify setup."""

import sys
import math
import time
import torch
import torch_sdaa

sys.path.insert(0, "src")

from tensorlbm.bfl_common import bfl_boundary_link_indices, bfl_force_ledger_sparse, bfl_bounce_back_sparse
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.interpolated_bc import compute_q_sphere
from tensorlbm.solver3d import collide_mrt3d_low_memory, correct_mass3d, stream3d_roll

def schiller_naumann_cd(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re**0.687)

def quick_test(D, steps, device="sdaa:0"):
    print(f"\n{'='*60}")
    print(f"Quick test: D={D}, steps={steps}")
    print(f"{'='*60}")
    
    tau = 0.53
    nu_lb = (tau - 0.5) / 3.0
    u_lb = 100.0 * nu_lb / D
    re_eff = u_lb * D / nu_lb
    R_lb = D / 2.0
    
    lat_pad = int(3.0 * D)
    up_pad = int(3.0 * D)
    down_pad = int(4.0 * D)
    nx = D + 2 * lat_pad
    ny = D + 2 * lat_pad
    nz = D + up_pad + down_pad
    
    cx_lb = ny / 2.0
    cy_lb = nx / 2.0
    cz_lb = up_pad + R_lb
    
    cd_ref = schiller_naumann_cd(re_eff)
    dpS = 0.5 * u_lb**2 * math.pi * R_lb**2
    
    dev = torch.device(device)
    torch.sdaa.reset_peak_memory_stats()
    
    # Solid mask
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    r2 = (xx - cx_lb)**2 + (yy - cy_lb)**2 + (zz - cz_lb)**2
    solid = r2 <= R_lb**2
    
    # D3Q19 setup
    w = torch.tensor([1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
                      1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36],
                     device=dev, dtype=torch.float32)
    cq = torch.tensor([
        [0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
        [1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0],
        [1,0,1],[-1,0,-1],[1,0,-1],[-1,0,1],
        [0,1,1],[0,-1,-1],[0,1,-1],[0,-1,1],
    ], device=dev, dtype=torch.float32)
    
    # Init f
    u_dot_cq = u_lb * cq[:, 0]
    u_sq = u_lb**2
    feq_flat = w * (1.0 + 3.0 * u_dot_cq + 4.5 * u_dot_cq**2 - 1.5 * u_sq)
    f = feq_flat[:, None, None, None].expand(19, nz, ny, nx).contiguous().clone()
    solid_exp = solid.unsqueeze(0).expand_as(f)
    f = torch.where(solid_exp, torch.zeros_like(f), f)
    del solid_exp
    
    # BFL
    bfl_mask, bfl_q = compute_q_sphere(nx, ny, nz, cx_lb, cy_lb, cz_lb, R_lb, dev)
    links = bfl_boundary_link_indices(bfl_mask, bfl_q, lattice="D3Q19")
    del bfl_mask, bfl_q
    
    bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
    target_mass = float(f.sum().item())
    
    t0 = time.time()
    cd_samples = []
    
    for step in range(1, steps + 1):
        f_pre = f.clone()
        f = collide_mrt3d_low_memory(f, tau=tau)
        f = torch.where(solid.unsqueeze(0).expand_as(f), f_pre, f)
        del f_pre
        f = bounce_back_cells_3d(f, solid)
        f_pre_stream = f.clone()
        f = stream3d_roll(f)
        f = far_field_bc_3d(f, bc_config=bc_config, u_in=u_lb)
        force3, _ = bfl_force_ledger_sparse(f, f_pre_stream, links)
        f = bfl_bounce_back_sparse(f, f_pre_stream, links)
        del f_pre_stream
        
        if step % 20 == 0:
            f = correct_mass3d(f, target_mass)
        
        if step % 10 == 0:
            cd = float(force3[0].item()) / dpS
            cd_samples.append(cd)
    
    torch.sdaa.synchronize()
    wall_time = time.time() - t0
    peak_mem = torch.sdaa.max_memory_allocated() / (1024**3)
    
    # Average last 50% of samples
    if cd_samples:
        tail = cd_samples[len(cd_samples)//2:]
        cd_avg = sum(tail) / len(tail)
    else:
        cd_avg = float("nan")
    
    err_pct = (cd_avg - cd_ref) / cd_ref * 100.0 if not math.isnan(cd_avg) else float("nan")
    
    print(f"Domain: {nz}x{ny}x{nx}")
    print(f"Re_eff: {re_eff:.2f}")
    print(f"Cd_ref: {cd_ref:.6f}")
    print(f"Cd_avg: {cd_avg:.6f}")
    print(f"Error: {err_pct:+.3f}%")
    print(f"Peak memory: {peak_mem:.3f} GiB")
    print(f"Wall time: {wall_time:.1f}s ({wall_time/steps*1000:.0f} ms/step)")
    
    return {
        "D": D,
        "steps": steps,
        "re_eff": re_eff,
        "cd_ref": cd_ref,
        "cd_avg": cd_avg,
        "err_pct": err_pct,
        "peak_mem_gib": peak_mem,
        "wall_time_s": wall_time,
        "ms_per_step": wall_time / steps * 1000,
    }

if __name__ == "__main__":
    print("SDAA device:", torch.sdaa.get_device_name(0))
    
    # Quick tests with minimal steps
    res_d10 = quick_test(D=10, steps=100)
    torch.sdaa.empty_cache()
    
    res_d20 = quick_test(D=20, steps=100)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"D=10: Cd_err={res_d10['err_pct']:+.2f}%, mem={res_d10['peak_mem_gib']:.3f}GiB")
    print(f"D=20: Cd_err={res_d20['err_pct']:+.2f}%, mem={res_d20['peak_mem_gib']:.3f}GiB")
