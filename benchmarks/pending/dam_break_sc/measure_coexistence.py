#!/usr/bin/env python3
"""Measure coexistence densities for SC EOS models.

Run a 2D droplet in equilibrium and measure rho_l (droplet core) and
rho_v (far field) for different EOS / T_r combinations.
"""
import argparse
import json
import math
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from tensorlbm.multiphase import (
    collide_sc_single_component,
    psi_eos_carnahan_starling,
    psi_eos_peng_robinson,
    make_psi_eos,
)
from tensorlbm.solver import equilibrium, stream
from tensorlbm.boundaries import bounce_back_cells
# D2Q9 constants not needed for this measurement


def init_droplet(nx, ny, r_drop, rho_l, rho_v, device):
    """Initialize a liquid droplet of radius r_drop in gas."""
    ys = torch.arange(ny, dtype=torch.float32, device=device)
    xs = torch.arange(nx, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cx, cy = nx / 2, ny / 2
    rr = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    base = torch.zeros((ny, nx), dtype=torch.float32, device=device)
    rho = torch.where(rr < r_drop, torch.full_like(base, rho_l), torch.full_like(base, rho_v))
    return rho


def measure_coexistence(eos, T_r, G, tau, nx=128, ny=128, r_drop=20, steps=5000, device="cpu"):
    """Run droplet equilibration and measure rho_l, rho_v."""
    dev = torch.device(device)
    
    # Build psi function
    if eos == "cs":
        psi_fn = lambda rho: psi_eos_carnahan_starling(rho, T_r, a=1.0, b=4.0, R=1.0)
    elif eos == "pr":
        psi_fn = lambda rho: psi_eos_peng_robinson(rho, T_r, a=2.0/49.0, b=2.0/21.0, R=1.0)
    else:
        raise ValueError(f"Unknown EOS: {eos}")
    
    # Initial guess for coexistence densities (from Maxwell construction approx)
    # For CS: T_c = 1/(a*R*b) * (some factor), rho_c ~ 1/b
    # Rough estimate: rho_l ~ 0.5-2.0, rho_v ~ 0.01-0.2 depending on T_r
    if T_r < 0.5:
        rho_l_init, rho_v_init = 1.5, 0.05
    elif T_r < 0.7:
        rho_l_init, rho_v_init = 1.2, 0.1
    elif T_r < 0.9:
        rho_l_init, rho_v_init = 0.8, 0.2
    else:
        rho_l_init, rho_v_init = 0.5, 0.3
    
    rho = init_droplet(nx, ny, r_drop, rho_l_init, rho_v_init, dev)
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)
    
    # Time stepping
    for step in range(steps):
        f = collide_sc_single_component(
            f, G=G, tau=tau, psi_fn=psi_fn,
            gy=0.0, solid_mask=None, wall_psi=None,
            forcing="velocity_shift", scheme="standard"
        )
        f = stream(f)
        
        if step % 500 == 0:
            rho_cur = f.sum(dim=0)
            rho_min = float(rho_cur.min())
            rho_max = float(rho_cur.max())
            if not math.isfinite(rho_min) or not math.isfinite(rho_max):
                print(f"  EOS={eos} T_r={T_r}: NaN at step {step}")
                return None
    
    # Final measurement
    rho_final = f.sum(dim=0).cpu().numpy()
    
    # Sample center (liquid) and corner (gas)
    cx, cy = nx // 2, ny // 2
    rho_l = float(rho_final[cy, cx])
    rho_v = float(rho_final[5, 5])  # corner, away from droplet
    
    density_ratio = rho_l / rho_v if rho_v > 1e-10 else float('inf')
    
    return {
        "eos": eos,
        "T_r": T_r,
        "G": G,
        "tau": tau,
        "rho_l": rho_l,
        "rho_v": rho_v,
        "density_ratio": density_ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eos", choices=["cs", "pr"], default="cs")
    ap.add_argument("--T_r", type=float, default=0.6)
    ap.add_argument("--G", type=float, default=1.0)  # EOS convention: G=+1 attractive
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    
    print(f"Measuring coexistence: EOS={args.eos} T_r={args.T_r} G={args.G} tau={args.tau}")
    result = measure_coexistence(
        args.eos, args.T_r, args.G, args.tau,
        steps=args.steps, device=args.device
    )
    
    if result:
        print(f"  rho_l={result['rho_l']:.4f}  rho_v={result['rho_v']:.4f}  "
              f"ratio={result['density_ratio']:.2f}")
        print(json.dumps(result, indent=2))
    else:
        print("  FAILED (NaN)")


if __name__ == "__main__":
    main()
