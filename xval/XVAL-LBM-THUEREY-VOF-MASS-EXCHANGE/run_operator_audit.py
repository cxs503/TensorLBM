#!/usr/bin/env python3
"""Operator audit of the I/I selector of the Korner/Thuerey mass exchange.

Drives the *shipped* solver (``tensorlbm.free_surface_lbm_2d.free_surface_step_2d``)
on deterministic synthetic configurations and reads the observer-only ledger
channels added for this XVAL (``exchange_interface_delta``, ``ii_dead/live``).
The quantity of interest is the conservation residual of the INTERFACE/INTERFACE
stage: a paired exchange must commit +dm on one end and -dm on the other, so
``|sum(mass_delta_II)| / sum(|mass_delta_II|)`` must be round-off.

Output: artifacts/operator_audit.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.free_surface_lbm_2d import (  # noqa: E402
    GAS, INTERFACE, LIQUID, SOLID, free_surface_step_2d, total_liquid_inventory_2d,
)

RULES = ("repo", "pair_symmetric", "koerner", "oss")


def _base(ny, nx):
    flags = torch.full((ny, nx), GAS, dtype=torch.int8)
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    flags[solid] = SOLID
    return flags, solid


def configs(ny=24, nx=32):
    out = {}
    # C1: 3-cell-thick interface slab in gas.  Interior cells have neither a
    # LIQUID nor a GAS neighbour, boundary cells have a GAS neighbour: the
    # asymmetric repo/oss predicates can fire on one end of an I/I link only.
    flags, solid = _base(ny, nx)
    flags[8:11, 5:20] = INTERFACE
    out["thick_interface_slab"] = (flags, solid)
    # C2: interface layer over a liquid block (dam-break like surface band).
    flags, solid = _base(ny, nx)
    flags[1:10, 1:18] = LIQUID
    flags[10, 1:19] = INTERFACE
    out["surface_band"] = (flags, solid)
    # C3: 2-cell-thick interface ring (isolated droplet shell, no liquid).
    flags, solid = _base(ny, nx)
    yy = torch.arange(ny).view(-1, 1).expand(ny, nx).float()
    xx = torch.arange(nx).view(1, -1).expand(ny, nx).float()
    r = ((yy - 12.0) ** 2 + (xx - 16.0) ** 2).sqrt()
    flags[(r >= 2.0) & (r < 4.0)] = INTERFACE
    out["droplet_shell"] = (flags, solid)
    # C0 hydrostatic control: liquid block, flat free surface, gravity on.
    flags, solid = _base(ny, nx)
    flags[1:10, 1:20] = LIQUID
    flags[10, 1:20] = INTERFACE
    out["hydrostatic_column"] = (flags, solid)
    return out


def perturbed_state(flags):
    ny, nx = flags.shape
    yy = torch.arange(ny).view(-1, 1).expand(ny, nx).float()
    xx = torch.arange(nx).view(1, -1).expand(ny, nx).float()
    ux = 0.05 * torch.sin(2 * math.pi * xx / (nx - 1)) * torch.cos(math.pi * yy / ny)
    uy = 0.03 * torch.cos(2 * math.pi * yy / (ny - 1)) * torch.sin(math.pi * xx / nx)
    f = equilibrium(torch.ones_like(ux), ux, uy)
    fill = torch.where(flags == LIQUID, torch.ones_like(ux),
                       torch.where(flags == INTERFACE,
                                   0.5 + 0.3 * torch.sin(xx) * torch.cos(yy),
                                   torch.zeros_like(ux)))
    return f, fill


def audit(name, flags, solid, steps=20, gy=0.0, relax=True):
    torch.set_num_threads(1)
    f0, fill = perturbed_state(flags)
    if not relax:  # hydrostatic control: no perturbation, gravity on
        f0 = equilibrium(torch.ones_like(fill), torch.zeros_like(fill), torch.zeros_like(fill))
    res = {"steps": steps, "gy": gy, "rules": {}}
    for rule in RULES:
        f, fl, mass = f0.clone(), fill.clone(), fill.clone()
        flg = flags.clone()
        if not relax:
            mass = torch.where(flg == LIQUID, torch.ones_like(mass),
                               torch.where(flg == INTERFACE, 0.5 * torch.ones_like(mass),
                                           torch.zeros_like(mass)))
        inv0 = total_liquid_inventory_2d(f, mass, flg)
        net = absx = 0.0
        dead = live = 0
        ledger = {}
        for _ in range(steps):
            fl = torch.where(flg == LIQUID, torch.ones_like(mass),
                             torch.where(flg == INTERFACE, mass.clamp(0.0, 2.0), torch.zeros_like(mass)))
            ledger.clear()
            f, mass, flg = free_surface_step_2d(
                f, fl, flg, solid, tau=1.0, gy=-gy, mass=mass, return_mass=True,
                ledger=ledger, interface_exchange_rule=rule)
            net += ledger["exchange_interface_delta"]
            absx += abs(ledger["exchange_interface_delta"])
            dead += ledger["exchange_ii_dead_links"]
            live += ledger["exchange_ii_live_links"]
        inv1 = total_liquid_inventory_2d(f, mass, flg)
        res["rules"][rule] = {
            "net_dm_II": net,
            "sum_abs_dm_II": absx,
            "net_relative": abs(net) / absx if absx > 0 else 0.0,
            "ii_dead_links_summed": dead,
            "ii_live_links_summed": live,
            "inventory_drift_pct": 100.0 * (inv1 - inv0) / inv0,
            "finite": bool(torch.isfinite(f).all()),
        }
    return res


def main():
    payload = {"anchor": "XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE", "kind": "operator_audit",
               "solver": "tensorlbm.free_surface_lbm_2d.free_surface_step_2d (float32, CPU)",
               "metric": "net_relative = |sum dm_II| / sum |dm_II| over the run", "configs": {}}
    for name, (flags, solid) in configs().items():
        hydro = name == "hydrostatic_column"
        payload["configs"][name] = {
            "cells": {"LIQUID": int((flags == LIQUID).sum()), "INTERFACE": int((flags == INTERFACE).sum()),
                      "GAS": int((flags == GAS).sum())},
            "setup": audit(name, flags, solid, steps=20, gy=5e-5 if hydro else 0.0, relax=not hydro),
        }
    out = HERE / "artifacts" / "operator_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    for cfg, body in payload["configs"].items():
        print(f"\n[{cfg}]  {body['cells']}")
        for rule, m in body["setup"]["rules"].items():
            print(f"  {rule:15s} live={m['ii_live_links_summed']:7d} dead={m['ii_dead_links_summed']:7d} "
                  f"net={m['net_dm_II']:+.6e} sum|dm|={m['sum_abs_dm_II']:.6e} "
                  f"net_rel={m['net_relative']:.3e} inv_drift={m['inventory_drift_pct']:+.5f}%")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
