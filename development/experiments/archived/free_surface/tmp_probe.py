"""Quick probe: does the Thürey-table-4.1 I/I selector break pair antisymmetry?"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
import torch

from tensorlbm.d2q9 import equilibrium
from tensorlbm.free_surface_lbm_2d import (
    INTERFACE, LIQUID, free_surface_step_2d, init_fill_rectangular_2d,
    init_flags_from_fill_2d, total_liquid_inventory_2d,
)


def run(a=16, g=5e-5, steps=600, tau=1.0):
    torch.set_num_threads(1)
    h0, nx = 2 * a, 5 * a
    ny = h0 + a
    fill, solid = init_fill_rectangular_2d(ny, nx, float(a + 1), float(h0 + 1), "cpu")
    flags = init_flags_from_fill_2d(fill, solid)
    f = equilibrium(torch.ones_like(fill), torch.zeros_like(fill), torch.zeros_like(fill))
    mass = fill.clone()
    inv0 = total_liquid_inventory_2d(f, mass, flags)
    cum_if = 0.0
    abs_if = 0.0
    dead = 0
    live = 0
    worst = 0.0
    ledger = {}
    for step in range(steps):
        fill = torch.where(flags == LIQUID, torch.ones_like(fill),
                           torch.where(flags == INTERFACE, mass.clamp(0, 1), torch.zeros_like(fill)))
        ledger.clear()
        f, mass, flags = free_surface_step_2d(
            f, fill, flags, solid, tau=tau, gy=-g, mass=mass, return_mass=True, ledger=ledger)
        if "exchange_interface_delta" not in ledger:
            print("no ledger channel"); return
        d = ledger["exchange_interface_delta"]
        cum_if += d
        worst = max(worst, abs(d))
        dead += ledger["exchange_ii_dead_links"]
        live += ledger["exchange_ii_live_links"]
    inv1 = total_liquid_inventory_2d(f, mass, flags)
    print(f"a={a} steps={steps}  sum(dm_II)={cum_if:+.6e}  max|dm_II| step={worst:.3e}  "
          f"sum|.|={abs_if:.3e}  dead_links={dead} live_links={live}  "
          f"inventory drift={100*(inv1-inv0)/inv0:+.4f}%  mass_sum={float(mass.sum()):.8f}")


if __name__ == "__main__":
    run(a=int(sys.argv[1]) if len(sys.argv) > 1 else 16,
        steps=int(sys.argv[2]) if len(sys.argv) > 2 else 600)