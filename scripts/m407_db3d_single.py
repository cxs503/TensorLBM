#!/usr/bin/env python3
"""M12 · Dambreak3D single-device runner. Saves height fields to npz."""
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensorlbm import free_surface_step, init_flags_from_fill, init_mass_from_fill, init_population_from_fill

LIQUID = 1; INTERFACE = 2

def build_domain(a, g, device, rho_gas=1.0):
    nx, ny, nz = 5*a, 2*a+a//2, 2*a; ch = 2*a
    fill = torch.zeros((nz,ny,nx), dtype=torch.float32, device=device)
    fill[1:a+1, :ch, 1:a+1] = 1.0
    solid = torch.zeros((nz,ny,nx), dtype=torch.bool, device=device)
    solid[0,:,:]=True; solid[-1,:,:]=True; solid[:,0,:]=True; solid[:,-1,:]=True; solid[:,:,0]=True; solid[:,:,-1]=True
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags, rho_liquid=1.0)
    f = init_population_from_fill(fill, flags, rho_liquid=1.0, rho_gas=rho_gas)
    return f, fill, flags, mass, solid, (nx, ny, nz)

def height_field(fill, flags):
    wet = (flags == LIQUID) | ((flags == INTERFACE) & (fill >= 0.5))
    ny = wet.shape[1]
    y_idx = torch.arange(ny, device=wet.device, dtype=torch.float32).view(1, -1, 1)
    return (wet.float() * y_idx).max(dim=1).values

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--a", type=int, default=8)
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--outdir", default="artifacts/db3d_intermediate")
    args = ap.parse_args()

    a=args.a; g=1e-4; tau=0.8; rho_gas=1.0
    dev = torch.device(args.device)
    f, fill, flags, mass, solid, (nx,ny,nz) = build_domain(a, g, dev, rho_gas)
    m0 = float(mass.sum().item())
    t_scale = math.sqrt(2.0*g/a)
    series = []; heights_dict = {}
    t0 = time.time()
    for step in range(1, args.steps+1):
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=tau, gy=-g,
            rho_liquid=1.0, rho_gas=rho_gas, surface_tension=0.0,
            paired_liquid_interface_debit=True)
        if step % args.interval == 0 or step == args.steps:
            m_now = float(mass.sum().item())
            drift = (m_now - m0) / m0
            hf = height_field(fill, flags).cpu().numpy()
            heights_dict[step] = hf
            series.append({"step":step,"T":step*t_scale,"mass":m_now,
                          "mass_drift_rel":drift,"mass_drift_pct":drift*100.0})
            print(f"  [{args.device}] st={step:4d} drift={drift:+.6%} t={time.time()-t0:.1f}s",flush=True)
    elapsed = time.time()-t0
    total_drift = series[-1]["mass_drift_rel"] if series else 0.0
    total_steps = series[-1]["step"] if series else 1
    drift_per_1e4 = total_drift/total_steps*10000.0

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    tag = args.device.replace(":","_")
    # Save heights as npz
    np.savez(outdir/f"heights_{tag}.npz", **{f"step_{k}":v for k,v in heights_dict.items()})
    # Save series
    result = {"device":args.device,"a":a,"g":g,"tau":tau,"rho_gas":rho_gas,
              "steps":args.steps,"domain":{"nx":nx,"ny":ny,"nz":nz},
              "m0":m0,"elapsed_s":elapsed,
              "final_mass_drift_rel":total_drift,
              "drift_per_1e4_steps":drift_per_1e4,
              "series":series}
    (outdir/f"series_{tag}.json").write_text(json.dumps(result,indent=2))
    print(f"\nSaved to {outdir}/{tag}")
    print(f"drift={total_drift:+.6%} per_1e4={drift_per_1e4:+.6%} elapsed={elapsed:.1f}s")

if __name__ == "__main__":
    main()
