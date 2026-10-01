#!/usr/bin/env python3
"""Probe the I->G ownership builder residual on donor/receiver fields with
division-rounding (multiple receivers per donor) and negative donor mass."""
from __future__ import annotations
import torch
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID, SOLID
from tensorlbm.free_surface_topology_transaction import (
    build_i_to_g_ownership_transaction, TopologyTransactionError,
)
from tensorlbm.core.d3q19_stencil import D3Q19_MOVING_Q, roll_from_pull_source


def try_build(flags, mass, donor, label):
    solid = torch.zeros_like(flags, dtype=torch.bool)
    try:
        t = build_i_to_g_ownership_transaction(
            flags, mass, to_gas=donor, to_liq=torch.zeros_like(donor),
            solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID,
            interface_flag=INTERFACE, rho_liquid=1.0,
        )
        print(f"  {label}: OK debit={float(t.donor_debit):.9g} credit={float(t.receiver_credit):.9g} resid={float(t.residual):.3g}")
        return True
    except TopologyTransactionError as e:
        print(f"  {label}: RAISED {e}")
        return False


def random_case(seed, shape=(8, 8, 8), n_donor=76, neg_frac=0.5):
    g = torch.Generator().manual_seed(seed)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    mass = torch.zeros(shape)
    # random interface cells
    idx = torch.randperm(shape[0]*shape[1]*shape[2], generator=g)[: n_donor * 3]
    cells = [(int(i)//(shape[1]*shape[2]), (int(i)//shape[2])%shape[1], int(i)%shape[2]) for i in idx]
    for c in cells:
        flags[c] = INTERFACE
        mass[c] = float(torch.rand(1, generator=g)) * 0.8 - (0.3 * neg_frac)
    donor_cells = set(cells[:n_donor])
    donor = torch.zeros(shape, dtype=torch.bool)
    for c in donor_cells:
        donor[c] = True
    return flags, mass, donor


def main():
    print("== synthetic donor/receiver fields ==")
    for seed in range(6):
        flags, mass, donor = random_case(seed)
        try_build(flags, mass, donor, f"seed{seed} n_donor={int(donor.sum())} nrecv={int(((flags==INTERFACE)&~donor).sum())} minmass={float(mass[donor].min()):.4g}")


if __name__ == "__main__":
    main()