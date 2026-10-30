#!/usr/bin/env python3
"""M8 · 3D球体阻力收敛（Re=100，D≤20）on SDAA

Execute sphere_re100 case on SDAA card with mesh D=10/20 two tiers.
Record Cd vs reference error, peak VRAM, wall time.

Criteria:
① Cd error ≤ 3%
② Peak VRAM ≤ 14.94 GiB (m75 measured single card capacity)
③ If OOM, must report 'OOM at which tensor operation'

Output: artifacts/sdaa_sphere_re100_envelope.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch_sdaa  # noqa: F401

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensorlbm.bfl_common import (
    bfl_bounce_back_sparse,
    bfl_boundary_link_indices,
    bfl_force_ledger_sparse,
)
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.interpolated_bc import compute_q_sphere
from tensorlbm.solver3d import collide_mrt3d_low_memory, correct_mass3d, stream3d_roll


def schiller_naumann_cd(re: float) -> float:
    """Schiller-Naumann correlation: Cd = 24/Re * (1 + 0.15 * Re^0.687)"""
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def run_sphere_re100_sdaa(D: int, steps: int, device: str = "sdaa:0"):
    """Run sphere Re=100 case on SDAA with given D (diameter in lattice cells).
    
    For Re=100: u_lb * D / nu_lb = 100
    nu_lb = (tau - 0.5) / 3
    
    Choose tau=0.53 → nu_lb=0.01
    Then u_lb = 100 * nu_lb / D = 1.0 / D
    
    D=10 → u_lb=0.1 (Ma ≈ 0.17, acceptable)
    D=20 → u_lb=0.05 (Ma ≈ 0.087, good)
    """
    print(f"\n{'='*60}")
    print(f"Running sphere Re=100, D={D}, steps={steps}, device={device}")
    print(f"{'='*60}")
    
    # Physics parameters for Re=100
    tau = 0.53
    nu_lb = (tau - 0.5) / 3.0  # 0.01
    u_lb = 100.0 * nu_lb / D  # Re = u_lb * D / nu_lb = 100
    re_eff = u_lb * D / nu_lb
    
    print(f"tau={tau}, nu_lb={nu_lb:.6f}, u_lb={u_lb:.6f}, Re_eff={re_eff:.2f}")
    
    # Domain setup: sphere at center, padding 3D in each direction
    R_lb = D / 2.0
    lat_pad = int(3.0 * D)  # lateral padding
    up_pad = int(3.0 * D)   # upstream padding
    down_pad = int(4.0 * D) # downstream padding
    
    nx = D + 2 * lat_pad
    ny = D + 2 * lat_pad
    nz = D + up_pad + down_pad
    
    print(f"Domain: {nz} x {ny} x {nx} (nz x ny x nx)")
    
    # Center in lattice units
    cx_lb = ny / 2.0
    cy_lb = nx / 2.0
    cz_lb = up_pad + R_lb
    print(f"Sphere R_lb={R_lb}, center=({cz_lb}, {cy_lb}, {cx_lb})")
    
    # Reference Cd
    cd_ref = schiller_naumann_cd(re_eff)
    print(f"Reference Cd (Schiller-Naumann at Re={re_eff:.1f}): {cd_ref:.6f}")
    
    # Dynamic pressure * area
    dpS = 0.5 * u_lb**2 * math.pi * R_lb**2
    
    dev = torch.device(device)
    
    # Reset memory stats
    torch.sdaa.reset_peak_memory_stats()
    
    t_start = time.time()
    
    # Create solid mask (sphere)
    print("Creating solid mask...")
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    r2 = (xx - cx_lb) ** 2 + (yy - cy_lb) ** 2 + (zz - cz_lb) ** 2
    solid = r2 <= R_lb**2
    n_solid = int(solid.sum().item())
    print(f"Solid cells: {n_solid}")
    
    # D3Q19 weights and velocities
    w = torch.tensor([
        1.0/3.0,  # 0: rest
        1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,  # 1-6: face
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,  # 7-12: edge
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,  # 13-18: edge
    ], device=dev, dtype=torch.float32)
    
    # D3Q19 velocities (qx, qy, qz)
    cq = torch.tensor([
        [0, 0, 0],  # 0
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],  # 1-6
        [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],  # 7-10
        [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],  # 11-14
        [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],  # 15-18
    ], device=dev, dtype=torch.float32)
    
    # Initialize distribution function (equilibrium at u_in)
    print("Initializing f...")
    rho = 1.0
    u_x, u_y, u_z = u_lb, 0.0, 0.0  # flow in x-direction (which is nx dimension)
    
    # Equilibrium distribution - uniform across domain
    u_dot_cq = u_x * cq[:, 0] + u_y * cq[:, 1] + u_z * cq[:, 2]  # shape (19,)
    u_sq = u_x**2 + u_y**2 + u_z**2
    feq_flat = rho * w * (1.0 + 3.0 * u_dot_cq + 4.5 * u_dot_cq**2 - 1.5 * u_sq)  # shape (19,)
    
    # Expand to full domain
    f = feq_flat[:, None, None, None].expand(19, nz, ny, nx).contiguous().clone()
    
    # Set solid cells to zero
    solid_expanded = solid.unsqueeze(0).expand_as(f)
    f = torch.where(solid_expanded, torch.zeros_like(f), f)
    del solid_expanded
    
    # Compute BFL (Bouzidi-Firdaouss-Lallemand) boundary links
    print("Computing BFL boundary links...")
    bfl_mask, bfl_q = compute_q_sphere(nx, ny, nz, cx_lb, cy_lb, cz_lb, R_lb, dev)
    n_links = int(bfl_mask[1:].sum().item())
    print(f"BFL links: {n_links}")
    
    # Precompute sparse boundary link indices
    links = bfl_boundary_link_indices(bfl_mask, bfl_q, lattice="D3Q19")
    print(f"Sparse links: {links.n_links}, wrapped: {links.n_wrapped}")
    
    del bfl_mask, bfl_q
    torch.sdaa.empty_cache()
    
    mem_after_setup = torch.sdaa.max_memory_allocated() / (1024**3)
    print(f"Memory after setup: {mem_after_setup:.3f} GiB")
    
    # Reset peak for loop
    torch.sdaa.reset_peak_memory_stats()
    
    # BC config
    bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
    
    # Simulation loop
    print(f"\nStarting simulation loop ({steps} steps)...")
    hist = []
    sample_interval = max(1, steps // 50)  # Sample 50 times
    target_mass = float(f.sum().item())
    initial_mass = target_mass
    
    t_loop_start = time.time()
    
    for step in range(1, steps + 1):
        # Collision
        f_pre = f.clone()
        f = collide_mrt3d_low_memory(f, tau=tau)
        
        # Restore solid cells
        f = torch.where(solid.unsqueeze(0).expand_as(f), f_pre, f)
        del f_pre
        
        # Bounce-back on solid cells
        f = bounce_back_cells_3d(f, solid)
        
        # Streaming
        f_pre_stream = f.clone()
        f = stream3d_roll(f)
        
        # Far-field BC
        f = far_field_bc_3d(f, bc_config=bc_config, u_in=u_lb)
        
        # BFL force and bounce-back
        force3, _ = bfl_force_ledger_sparse(f, f_pre_stream, links)
        f = bfl_bounce_back_sparse(f, f_pre_stream, links)
        
        del f_pre_stream
        
        # Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, target_mass)
        
        # Sample
        if step % sample_interval == 0:
            cd = float(force3[0].item()) / dpS
            cl = float(force3[1].item()) / dpS
            cs = float(force3[2].item()) / dpS
            hist.append({
                "step": step,
                "cd": cd,
                "cl": cl,
                "cs": cs,
                "mass": float(f.sum().item()),
            })
        
        # Progress
        if step % max(1, steps // 10) == 0:
            el = time.time() - t_loop_start
            mem_now = torch.sdaa.max_memory_allocated() / (1024**3)
            print(f"  step {step}/{steps} ({el:.1f}s, {el/step*1000:.0f} ms/step, mem={mem_now:.3f} GiB)")
    
    t_loop_end = time.time()
    
    # Peak memory
    peak_mem_gib = torch.sdaa.max_memory_allocated() / (1024**3)
    
    # Results
    t_total = time.time() - t_start
    wall_time = t_loop_end - t_loop_start
    
    # Compute average Cd from last 20% of samples
    if hist:
        tail_start = max(0, len(hist) - len(hist) // 5)
        tail = hist[tail_start:]
        cd_avg = sum(h["cd"] for h in tail) / len(tail)
        cd_last = hist[-1]["cd"]
    else:
        cd_avg = float("nan")
        cd_last = float("nan")
    
    err_pct = (cd_avg - cd_ref) / cd_ref * 100.0 if not math.isnan(cd_avg) else float("nan")
    
    mass_drift_ppm = (float(f.sum().item()) - initial_mass) / initial_mass * 1e6
    
    print(f"\n{'='*60}")
    print(f"Results for D={D}:")
    print(f"  Cd_avg (last 20%): {cd_avg:.6f}")
    print(f"  Cd_ref: {cd_ref:.6f}")
    print(f"  Error: {err_pct:+.3f}%")
    print(f"  Peak memory: {peak_mem_gib:.3f} GiB")
    print(f"  Wall time: {wall_time:.1f} s ({wall_time/steps*1000:.1f} ms/step)")
    print(f"  Mass drift: {mass_drift_ppm:.2f} ppm")
    print(f"{'='*60}\n")
    
    return {
        "D": D,
        "steps": steps,
        "re_eff": re_eff,
        "tau": tau,
        "u_lb": u_lb,
        "nu_lb": nu_lb,
        "domain_lu": [nz, ny, nx],
        "n_solid": n_solid,
        "n_bfl_links": n_links,
        "cd_ref": cd_ref,
        "cd_avg": cd_avg,
        "cd_last": cd_last,
        "err_pct": err_pct,
        "peak_mem_gib": peak_mem_gib,
        "wall_time_s": wall_time,
        "ms_per_step": wall_time / steps * 1000,
        "mass_drift_ppm": mass_drift_ppm,
        "history": hist,
    }


def main():
    device = "sdaa:0"
    print(f"Using device: {device}")
    print(f"SDAA device name: {torch.sdaa.get_device_name(0)}")
    print(f"SDAA device count: {torch.sdaa.device_count()}")
    
    results = {}
    
    # D=10 tier (500 steps for quick test)
    try:
        res_d10 = run_sphere_re100_sdaa(D=10, steps=500, device=device)
        results["D10"] = res_d10
    except torch.sdaa.OutOfMemoryError as e:
        print(f"OOM at D=10: {e}")
        traceback.print_exc()
        results["D10"] = {
            "D": 10,
            "status": "OOM",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    except Exception as e:
        print(f"Error at D=10: {e}")
        traceback.print_exc()
        results["D10"] = {
            "D": 10,
            "status": "ERROR",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    
    # Clear memory between runs
    torch.sdaa.empty_cache()
    import gc
    gc.collect()
    
    # D=20 tier (1000 steps)
    try:
        res_d20 = run_sphere_re100_sdaa(D=20, steps=1000, device=device)
        results["D20"] = res_d20
    except torch.sdaa.OutOfMemoryError as e:
        print(f"OOM at D=20: {e}")
        traceback.print_exc()
        results["D20"] = {
            "D": 20,
            "status": "OOM",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    except Exception as e:
        print(f"Error at D=20: {e}")
        traceback.print_exc()
        results["D20"] = {
            "D": 20,
            "status": "ERROR",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    
    # Build envelope JSON
    envelope = {
        "benchmark": "sphere_re100",
        "device": device,
        "device_name": torch.sdaa.get_device_name(0),
        "reference": {
            "cd_ref_formula": "Schiller-Naumann: 24/Re * (1 + 0.15 * Re^0.687)",
            "re_target": 100,
        },
        "tiers": {},
    }
    
    for tier_name, res in results.items():
        if res.get("status") in ["OOM", "ERROR"]:
            envelope["tiers"][tier_name] = {
                "D": res["D"],
                "status": res["status"],
                "error": res.get("error"),
                "traceback": res.get("traceback"),
            }
        else:
            envelope["tiers"][tier_name] = {
                "D": res["D"],
                "status": "SUCCESS",
                "re_eff": res["re_eff"],
                "tau": res["tau"],
                "u_lb": res["u_lb"],
                "domain_lu": res["domain_lu"],
                "n_solid": res["n_solid"],
                "n_bfl_links": res["n_bfl_links"],
                "steps": res["steps"],
                "cd_ref": res["cd_ref"],
                "cd_avg": res["cd_avg"],
                "cd_error_pct": res["err_pct"],
                "peak_mem_gib": res["peak_mem_gib"],
                "wall_time_s": res["wall_time_s"],
                "ms_per_step": res["ms_per_step"],
                "mass_drift_ppm": res["mass_drift_ppm"],
            }
    
    # Write JSON
    out_path = Path("artifacts/sdaa_sphere_re100_envelope.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(envelope, f, indent=2)
    print(f"\nWritten: {out_path}")
    print(json.dumps(envelope, indent=2))
    
    # Check criteria
    print("\n" + "="*60)
    print("CRITERIA CHECK:")
    print("="*60)
    
    all_pass = True
    for tier_name, tier in envelope["tiers"].items():
        print(f"\n{tier_name} (D={tier['D']}):")
        if tier["status"] != "SUCCESS":
            print(f"  Status: {tier['status']}")
            all_pass = False
            continue
        
        cd_err = tier["cd_error_pct"]
        mem_peak = tier["peak_mem_gib"]
        
        cd_pass = abs(cd_err) <= 3.0
        mem_pass = mem_peak <= 14.94
        
        print(f"  Cd error: {cd_err:+.3f}% (≤3%: {'PASS' if cd_pass else 'FAIL'})")
        print(f"  Peak memory: {mem_peak:.3f} GiB (≤14.94: {'PASS' if mem_pass else 'FAIL'})")
        
        if not cd_pass or not mem_pass:
            all_pass = False
    
    print("\n" + "="*60)
    if all_pass:
        print("ALL CRITERIA PASSED ✅")
    else:
        print("SOME CRITERIA FAILED ❌")
        print("\nHONEST-FAIL: D≤16为当前基线，3D大网格需分块单独立项")
    print("="*60)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
