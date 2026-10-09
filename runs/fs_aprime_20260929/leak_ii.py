#!/usr/bin/env python3
"""A-prime leak isolation: per-operator exchange-channel budget.

Runs one (or N) steps of the graded receive gate and reports each exchange
sub-channel sum (L/I credit, bulk debit, I/I half-weight) plus the count of
asymmetric I/I links (receiver gate passed at one link endpoint but not the
other).  The I/I channel only nets to zero when the gate is symmetric across
every link; asymmetric gating is the observed leak.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402

from tensorlbm.free_surface_lbm import free_surface_step  # noqa: E402
import probe  # noqa: E402


def main(steps: int = 3, a: int = 8, g: float = 0.0, rho_gas: float = 1.0) -> None:
    os.environ.setdefault("TL_FS_DBG", "1")
    dev = torch.device("cpu")
    f, fill, flags, mass, solid, _ = probe.build_domain(a, dev)
    m0 = float(mass.sum())
    print(f"a={a} g={g} steps={steps} m0={m0:.6f} envigraded="
          f"{os.environ.get('TL_FS_RECV_GRADED')} nmin={os.environ.get('TL_FS_RECV_NMIN')}")
    hdr = ("st  mass_end      drift      exc_LI      exc_bulk   exc_II      "
           "ii_asym   to_liq to_gas recv_new")
    print(hdr)
    for step in range(1, steps + 1):
        led: dict = {}
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=0.8, gy=-g, rho_liquid=1.0,
            rho_gas=rho_gas, paired_liquid_interface_debit=True, mass_ledger=led,
        )
        me = float(mass.sum())
        print(f"{step:3d} {me:12.6f} {me-m0:+11.6f} "
              f"{led['dbg_mass_delta_liquid_sum']:+11.6f} "
              f"{led['dbg_mass_delta_bulk_debit_sum']:+11.6f} "
              f"{led['dbg_mass_delta_interface_sum']:+11.6f} "
              f"{led['dbg_ii_asym_links']:8.0f} "
              f"{led['dbg_to_liq_sum']:6.0f} {led['dbg_to_gas_sum']:6.0f} "
              f"{led['dbg_recv_new_sum']:8.0f}")
    print(f"total drift {float(mass.sum())-m0:+.6f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--a", type=int, default=8)
    ap.add_argument("--g", type=float, default=0.0)
    ap.add_argument("--rho_gas", type=float, default=1.0)
    args = ap.parse_args()
    main(args.steps, args.a, args.g, args.rho_gas)