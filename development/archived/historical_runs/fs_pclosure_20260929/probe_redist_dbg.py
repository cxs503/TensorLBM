#!/usr/bin/env python3
"""Debug TL_FS_GAS_CHANNEL=redist mass non-conservation via the mass_ledger."""
from __future__ import annotations
import math, os
import torch
from tensorlbm import (equilibrium3d, free_surface_step, init_flags_from_fill, init_mass_from_fill)

LIQUID, INTERFACE, GAS = 1, 2, 0


def build_domain(a, device):
    nx, ny, nz = 5 * a, 2 * a + a // 2, 2 * a
    ch = 2 * a
    fill = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    fill[1 : a + 1, :ch, 1 : a + 1] = 1.0
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[0, :, :] = True; solid[-1, :, :] = True
    solid[:, 0, :] = True; solid[:, -1, :] = True
    solid[:, :, 0] = True; solid[:, :, -1] = True
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags, rho_liquid=1.0)
    active = (flags == LIQUID) | (flags == INTERFACE)
    z = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(torch.where(active, torch.ones_like(z), z), z, z, z)
    return f, fill, flags, mass, solid, ch


a, g, tau, rho_gas = 8, 1e-4, 0.8, 0.99
dev = torch.device("cpu")
f, fill, flags, mass, solid, ch = build_domain(a, dev)
print("env:", {k: v for k, v in os.environ.items() if k.startswith("TL_FS_")})
for step in range(1, 6):
    led = {}
    f, fill, flags, mass, df = free_surface_step(
        f, fill, flags, solid, mass=mass, tau=tau, gy=-g, rho_liquid=1.0,
        rho_gas=rho_gas, surface_tension=0.0, paired_liquid_interface_debit=True,
        mass_ledger=led)
    print(f"step {step}: start={led.get('start'):.6f} exchange={led.get('exchange'):.6f} "
          f"redist={led.get('redistribution'):.6f} clamp={led.get('clamp'):.6f} "
          f"conv={led.get('conversion'):.6f} iso={led.get('isolation'):.6f} end={led.get('boundary'):.6f}")
    print(f"        dbg: exch_liq={led.get('exchange_liquid_delta')} exch_if={led.get('exchange_interface_delta')} "
          f"bulk_debit={led.get('exchange_bulk_debit')}")
    print(f"        mass_delta_pre={led.get('dbg_mass_delta_preclamp_sum')} post={led.get('dbg_mass_delta_postclamp_sum')} "
          f"redist_inc={led.get('dbg_redist_inc_sum')} excess={led.get('dbg_excess_sum')}")