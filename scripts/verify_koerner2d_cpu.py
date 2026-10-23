#!/usr/bin/env python3
"""CPU verification for the rewritten Körner 2D free-surface module.

Checks:
  A) hydrostatic column mass conservation (dual: ledger sum + physical inventory)
  B) dam-break wave front X(T) vs Martin & Moyce (1952), a = 20 / 40
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.free_surface_lbm_2d import (  # noqa: E402
    GAS,
    INTERFACE,
    LIQUID,
    SOLID,
    free_surface_step_2d,
    init_fill_rectangular_2d,
    init_flags_from_fill_2d,
    init_mass_from_fill_2d,
    koerner_step_2d,
    total_liquid_inventory_2d,
    count_direct_liquid_gas_links_2d,
)

torch.set_num_threads(8)


def build_ic(ny, nx, cw, ch, rho_liquid=1.0, rho_gas=1.0):
    fill, solid = init_fill_rectangular_2d(ny, nx, cw, ch, "cpu")
    flags = init_flags_from_fill_2d(fill, solid)
    mass = init_mass_from_fill_2d(fill, flags, rho_liquid)
    rho0 = torch.where(
        flags == LIQUID, torch.full_like(fill, rho_liquid), torch.full_like(fill, rho_gas)
    )
    zero = torch.zeros_like(fill)
    feq = equilibrium(rho0, zero, zero)
    active = (flags == LIQUID) | (flags == INTERFACE)
    f = torch.where(active.unsqueeze(0), feq, torch.zeros_like(feq))
    return f, mass, flags, solid


def measure_front_x(flags, a):
    active = (flags == LIQUID) | (flags == INTERFACE)
    active = active[1:-1, 1:-1]
    cols = active.any(dim=0).nonzero(as_tuple=True)[0]
    x_front = float(cols.max().item()) + 1.0 if cols.numel() else 0.0
    return x_front / a, x_front


def hydrostatic(ny=64, nx=64, cw=30, ch=40, g=1.0e-4, tau=1.0, steps=800, rho_liquid=1.0):
    f, mass, flags, solid = build_ic(ny, nx, cw, ch, rho_liquid)
    inv0 = total_liquid_inventory_2d(f, mass, flags, rho_liquid)
    m0 = float(mass.sum().item())
    print(f"[hydro] init: L={int((flags==LIQUID).sum())} I={int((flags==INTERFACE).sum())} "
          f"G={int((flags==GAS).sum())} S={int((flags==SOLID).sum())}  "
          f"inventory={inv0:.3f} mass={m0:.3f}")
    worst_inv = 0.0
    worst_mass = 0.0
    t0 = time.time()
    for step in range(1, steps + 1):
        f, mass, flags = koerner_step_2d(
            f, mass, flags, solid, tau=tau, gy=-g, rho_liquid=rho_liquid, rho_gas=rho_liquid
        )
        if step % 100 == 0 or step == steps:
            inv = total_liquid_inventory_2d(f, mass, flags, rho_liquid)
            md = abs(float(mass.sum().item()) - m0) / abs(m0)
            idi = abs(inv - inv0) / abs(inv0)
            worst_inv = max(worst_inv, idi)
            worst_mass = max(worst_mass, md)
            na = count_direct_liquid_gas_links_2d(flags)
            print(f"[hydro] step={step:5d} mass_drift={md:.2e} inv_drift={idi:.2e} "
                  f"directLG={na} iface={int((flags==INTERFACE).sum())}")
        if torch.isnan(f).any():
            print("[hydro] NaN!")
            break
    print(f"[hydro] DONE worst inv_drift={worst_inv:.3e} worst mass_drift={worst_mass:.3e} "
          f"({time.time()-t0:.0f}s)")
    return worst_inv, worst_mass


def dam_break(a=20.0, g=2.0e-4, tau=1.0, steps=None, sample=50, rho_liquid=1.0):
    nx = int(round(6.4 * a))
    ny = int(round(3.2 * a))
    if steps is None:
        steps = int(math.ceil(2.2 / math.sqrt(g / a)))
    f, mass, flags, solid = build_ic(ny, nx, a, 2.0 * a, rho_liquid)
    inv0 = total_liquid_inventory_2d(f, mass, flags, rho_liquid)
    m0 = float(mass.sum().item())
    sq = math.sqrt(g / a)
    print(f"[dam a={a:.0f}] nx={nx} ny={ny} steps={steps} sqrt(g/a)={sq:.4e}")
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        f, mass, flags = koerner_step_2d(
            f, mass, flags, solid, tau=tau, gy=-g, rho_liquid=rho_liquid, rho_gas=rho_liquid
        )
        if torch.isnan(f).any():
            print(f"[dam] NaN at step {step}")
            break
        if step % sample == 0:
            X, xf = measure_front_x(flags, a)
            T = step * sq
            inv = total_liquid_inventory_2d(f, mass, flags, rho_liquid)
            md = abs(float(mass.sum().item()) - m0) / abs(m0)
            idi = abs(inv - inv0) / abs(inv0)
            hist.append((T, X, md, idi, int((flags == INTERFACE).sum())))
            print(f"[dam a={a:.0f}] step={step:5d} T={T:.4f} X={X:.4f} xf={xf:.0f} "
                  f"md={md:.1e} invd={idi:.1e} I={int((flags==INTERFACE).sum())}")
    print(f"[dam a={a:.0f}] DONE ({time.time()-t0:.0f}s)")
    return hist


def interp(hist, Tq):
    Ts = [h[0] for h in hist]
    Xs = [h[1] for h in hist]
    if Tq <= Ts[0]:
        return Xs[0]
    if Tq >= Ts[-1]:
        return Xs[-1]
    for i in range(1, len(Ts)):
        if Ts[i] >= Tq:
            t0, t1, x0, x1 = Ts[i - 1], Ts[i], Xs[i - 1], Xs[i]
            return x0 + (x1 - x0) * (Tq - t0) / (t1 - t0)
    return Xs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["hydro", "dam", "all"])
    ap.add_argument("--a", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()
    if args.which in ("hydro", "all"):
        hydrostatic()
    if args.which in ("dam", "all"):
        for a in ([20.0, 40.0] if args.which == "all" else [args.a]):
            steps = args.steps if args.steps > 0 else None
            hist = dam_break(a=a, steps=steps)
            if hist:
                for Tq, Xref in ((1.0, 1.5), (2.0, 2.7)):
                    Xs = interp(hist, Tq)
                    print(f"  --> a={a:.0f} T={Tq} X={Xs:.4f} ref={Xref} "
                          f"err={100*(Xs-Xref)/Xref:+.2f}%")


if __name__ == "__main__":
    main()