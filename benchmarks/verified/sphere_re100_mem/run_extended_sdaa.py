#!/usr/bin/env python3
"""Extended resolution study for sphere_re100_mem: D=80 and D=160.

This script runs the two additional resolution tiers required by m243:
- T3: D=80 with 48k steps
- T4: D=160 with 96k steps

Modified to work with SDAA devices.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import threading
import time

import torch

# Add tensorlbm to path
sys.path.insert(0, '/root/tp-agent/work/m46/src')

from tensorlbm.bfl_common import (
    bfl_bounce_back_sparse,
    bfl_boundary_link_indices,
    bfl_force_ledger_sparse,
)
from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d,
    far_field_bc_3d,
)
from tensorlbm.general_sim import (
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    OutputFormat,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)
from tensorlbm.interpolated_bc import compute_q_sphere
from tensorlbm.solver3d import (
    collide_mrt3d_low_memory,
    correct_mass3d,
    stream3d_roll,
)

# Use SDAA device
DEV = "sdaa:0"

def build_engine_config(D, lat, up, down):
    pad = (up, down, lat, lat, lat, lat)
    return GeneralSimConfig(
        name=f"sphere_D{D}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=1.0e-6,
            inlet_velocity=1.0e-4,
            reference_length=1.0,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=D,
            domain_padding=pad,
            max_steps=10,
            snapshot_interval=10**9,
            force_sample_interval=50,
            device=DEV,
            wall_treatment=WallTreatment.BOUNCE_BACK,
            mass_correction=True,
            mass_correction_interval=200,
            force_method=ForceMethod.MOMENTUM_EXCHANGE,
            mem_variant="wet_node",
        ),
        output=OutputConfig(
            directory="out",
            formats=[OutputFormat.NPY],
            save_macroscopic=False,
            save_forces=True,
        ),
    )

def cd_ref_family(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re**0.687)

def run_case(out_path, D, steps, lat, up, down, tau, ulb, sample):
    """Run a single case with given parameters."""
    print(f"Setting up D={D}, steps={steps}...")
    
    cfg = build_engine_config(D, lat, up, down)
    engine = GeneralSimEngine(cfg)
    engine.setup()
    
    nu_lb = (tau - 0.5) / 3.0
    re_eff = ulb * D / nu_lb
    R_lb = cfg.geometry.sphere_radius / (cfg.physics.reference_length / cfg.solver.resolution)
    dpS = 0.5 * ulb**2 * math.pi * R_lb**2
    
    solid = engine.solid
    nz, ny, nx = solid.shape
    dx = cfg.physics.reference_length / cfg.solver.resolution
    cx_lb = (cfg.geometry.sphere_center[0] - engine.domain_phys[0]) / dx
    cy_lb = (cfg.geometry.sphere_center[1] - engine.domain_phys[2]) / dx
    cz_lb = (cfg.geometry.sphere_center[2] - engine.domain_phys[4]) / dx
    dev = torch.device(DEV)
    
    # Setup BFL
    bfl_mask, bfl_q = compute_q_sphere(nx, ny, nz, cx_lb, cy_lb, cz_lb, R_lb, dev)
    n_links = int(bfl_mask[1:].sum().item())
    
    links = bfl_boundary_link_indices(bfl_mask, bfl_q, lattice="D3Q19")
    if links.n_links != n_links:
        raise RuntimeError(f"link precompute mismatch: {links.n_links} vs {n_links}")
    
    q_min = float(links.link_q.min().item())
    q_max = float(links.link_q.max().item())
    del bfl_mask, bfl_q
    torch.sdaa.empty_cache()
    
    # Memory stats
    torch.sdaa.reset_peak_memory_stats()
    
    # Setup simulation
    bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)
    collide = lambda f: collide_mrt3d_low_memory(f, tau=tau)
    stream_fn = stream3d_roll
    
    f = engine.f.clone()
    engine.f = None
    target_mass = float(f.sum().item())
    initial_mass = target_mass
    
    hist = []
    finite = True
    diverged_at = None
    sample_set = set(range(sample, steps + 1, sample))
    solid_idx = solid.reshape(-1).nonzero(as_tuple=True)[0]
    
    t0 = time.time()
    print(f"Starting simulation loop...")
    
    for step in range(1, steps + 1):
        # Sparse kernel path
        f_solid_pre = f.reshape(19, -1)[:, solid_idx]
        f = collide(f)
        f.reshape(19, -1)[:, solid_idx] = f_solid_pre
        del f_solid_pre
        
        f = bounce_back_cells_3d(f, solid)
        f_pre_stream = f.clone()
        f = stream_fn(f)
        f = far_field_fn(f, ulb)
        
        force3, _ = bfl_force_ledger_sparse(f, f_pre_stream, links)
        f = bfl_bounce_back_sparse(f, f_pre_stream, links)
        
        if step in sample_set:
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
        
        del f_pre_stream
        
        if step % 200 == 0:
            f = correct_mass3d(f, target_mass)
        
        if step % 500 == 0 and not bool(torch.isfinite(f).all()):
            finite = False
            diverged_at = step
            break
        
        if step % 2000 == 0:
            el = time.time() - t0
            print(f"step {step}/{steps} ({el:.0f}s, {el / step * 1000:.0f} ms/step)", flush=True)
    
    # Memory stats
    peak_loop = float(torch.sdaa.max_memory_allocated()) / 2**30
    
    meta = {
        "kernel": "sparse",
        "route": "bfl",
        "collision": "mrt",
        "treat": "hard",
        "D": D,
        "lat": lat,
        "up": up,
        "down": down,
        "tau": tau,
        "u_lb": ulb,
        "nu_lb": nu_lb,
        "re_eff": re_eff,
        "steps": steps if finite else diverged_at,
        "diverged": not finite,
        "domain_lu": [nz, ny, nx],
        "dpS": dpS,
        "R_lb": R_lb,
        "center_lu": [cz_lb, cy_lb, cx_lb],
        "blockage": math.pi * R_lb**2 / (ny * nz),
        "cd_ref_at_re_eff": cd_ref_family(re_eff),
        "mass_drift_ppm": (
            (float(f.sum().item()) - initial_mass) / initial_mass * 1e6 if finite else None
        ),
        "bfl_links": n_links,
        "q_min": q_min,
        "q_max": q_max,
        "stream": "roll",
        "wall_s": time.time() - t0,
        "mem_peak_loop_gib": peak_loop,
        "device": DEV,
        "device_name": torch.sdaa.get_device_name(0),
    }
    
    with open(out_path, "w") as fh:
        json.dump({"meta": meta, "history": hist}, fh)
    
    print(f"Written: {out_path}")
    if hist:
        tail = [h["cd"] for h in hist[-8:]]
        print(f"Last-8 cd mean: {sum(tail) / len(tail):.6f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["T3", "T4", "all"], default="T3")
    args = ap.parse_args()
    
    # Configuration
    configs = {
        "T3": {
            "D": 80,
            "tau": 0.62,
            "steps": 48000,
            "sample": 50,
            "lat": 3.0,
            "up": 3.0,
            "down": 4.0,
            "out": "t3_d80big_48k.json"
        },
        "T4": {
            "D": 160,
            "tau": 0.68,
            "steps": 96000,
            "sample": 50,
            "lat": 3.0,
            "up": 3.0,
            "down": 4.0,
            "out": "t4_d160big_96k.json"
        }
    }
    
    tiers_to_run = ["T3", "T4"] if args.tier == "all" else [args.tier]
    
    for tier in tiers_to_run:
        cfg = configs[tier]
        print(f"\n{'='*60}")
        print(f"Running {tier}: D={cfg['D']}, steps={cfg['steps']}")
        print(f"{'='*60}\n")
        
        run_case(
            out_path=cfg["out"],
            D=cfg["D"],
            steps=cfg["steps"],
            lat=cfg["lat"],
            up=cfg["up"],
            down=cfg["down"],
            tau=cfg["tau"],
            ulb=0.05,
            sample=cfg["sample"]
        )

if __name__ == "__main__":
    main()
