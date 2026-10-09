#!/usr/bin/env python3
"""Step-1 forensics: envelope structure, receiver/neighbour statistics."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID, SOLID, free_surface_step
from tensorlbm.core.d3q19_stencil import all_moving_neighbor_masks
import probe

def stats(a=8, g=0.0, rho_gas=1.0):
    f, fill, flags, mass, solid, (nx, ny, nz) = probe.build_domain(a, torch.device("cpu"))
    n_iface = int((flags == INTERFACE).sum())
    # liquid-neighbour count per interface cell
    nbf = torch.stack(all_moving_neighbor_masks(flags == LIQUID))
    n_liq = nbf.sum(0)
    iface = flags == INTERFACE
    print(f"a={a} iface={n_iface} liquid={int((flags==LIQUID).sum())} "
          f"envelope mass={(mass[iface]).sum().item():.6f} min={mass[iface].min().item():.3f} max={mass[iface].max().item():.3f}")
    print("liquid-neighbour histogram over interface cells:",
          {int(k): int((n_liq[iface] == k).sum()) for k in range(0, 7)})
    # run 1 step
    f1, fill1, flags1, mass1, df = free_surface_step(
        f, fill, flags, solid, mass=mass, tau=0.8, gy=-g, rho_liquid=1.0,
        rho_gas=rho_gas, paired_liquid_interface_debit=True)
    mg = mass1.clamp(min=0.0)
    donor = ((flags1 == INTERFACE) | (flags1 == LIQUID)) & (mg <= 0.01) if False else None
    print(f"after 1 step: mass={mass1.sum().item():.6f} iface={int((flags1==INTERFACE).sum())} "
          f"liquid={int((flags1==LIQUID).sum())} gas={int((flags1==GAS).sum())}")
    iface1 = flags1 == INTERFACE
    print(f"  interface mass: sum={mass1[iface1].sum().item():.4f} min={mass1[iface1].min().item():.4f} "
          f"max={mass1[iface1].max().item():.4f} neg_count={int((mass1[iface1]<0).sum())}")

if __name__ == "__main__":
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    g = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    stats(a, g)