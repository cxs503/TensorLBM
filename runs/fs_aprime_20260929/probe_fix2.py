#!/usr/bin/env python3
"""Test candidate exact constructions (reconciliation) over many random cases."""
from __future__ import annotations
import torch
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID
from tensorlbm.core.d3q19_stencil import (
    D3Q19_MOVING_Q, roll_from_pull_source, roll_to_neighbor,
)


def random_case(seed, shape=(8, 8, 8), n_donor=76, neg_frac=0.5):
    g = torch.Generator().manual_seed(seed)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    mass = torch.zeros(shape)
    idx = torch.randperm(shape[0] * shape[1] * shape[2], generator=g)[: n_donor * 3]
    cells = [(int(i) // (shape[1] * shape[2]), (int(i) // shape[2]) % shape[1], int(i) % shape[2]) for i in idx]
    for c in cells:
        flags[c] = INTERFACE
        mass[c] = float(torch.rand(1, generator=g)) * 0.8 - (0.3 * neg_frac)
    donor_cells = set(cells[:n_donor])
    donor = torch.zeros(shape, dtype=torch.bool)
    for c in donor_cells:
        donor[c] = True
    return flags, mass, donor


def core(flags, mass, donor):
    receiver_mask = (flags == INTERFACE) & ~donor
    link_fields = [roll_to_neighbor(torch.where(donor, mass, torch.zeros_like(mass)) / torch.stack([roll_from_pull_source(receiver_mask, q) for q in D3Q19_MOVING_Q]).sum(dim=0).clamp(min=1).to(mass.dtype), q) * receiver_mask for q in D3Q19_MOVING_Q]
    increment = torch.stack(link_fields).sum(dim=0)
    return receiver_mask, increment


def mode_neg_increment(mass, donor, increment):
    donor_debit_records = -increment
    donor_debit = donor_debit_records.sum()
    receiver_credit_records = increment
    receiver_credit = receiver_credit_records.sum()
    return donor_debit, donor_debit_records, receiver_credit, receiver_credit_records


def mode_emit_reconcile(mass, donor, increment, receiver_mask):
    receiver_credit = increment.sum()
    emit = torch.zeros_like(mass)
    for q in D3Q19_MOVING_Q:
        emit = emit + (torch.where(donor, mass, torch.zeros_like(mass)) / torch.stack([roll_from_pull_source(receiver_mask, qq) for qq in D3Q19_MOVING_Q]).sum(dim=0).clamp(min=1).to(mass.dtype)) * roll_from_pull_source(receiver_mask, q).to(mass.dtype)
    donor_debit_records = -emit
    donor_debit = -receiver_credit
    delta = donor_debit - donor_debit_records.sum()
    if delta != 0:
        flat = donor_debit_records.reshape(-1)
        p = int(torch.argmax(flat.abs()))
        flat[p] = flat[p] + delta
    # iterate if needed
    for _ in range(3):
        cur = donor_debit_records.sum()
        if cur == donor_debit:
            break
        d = donor_debit - cur
        flat = donor_debit_records.reshape(-1)
        p = int(torch.argmax(flat.abs()))
        flat[p] = flat[p] + d
    return donor_debit, donor_debit_records, receiver_credit, increment


def check(mass, donor, dd, ddr, rc, rcr):
    return (bool(dd == ddr.sum()) and bool(rc == rcr.sum()) and bool(dd + rc == 0.0))


def main():
    for mode in ("neg_increment", "emit_reconcile"):
        bad = 0
        for s in range(40):
            flags, mass, donor = random_case(s)
            receiver_mask, increment = core(flags, mass, donor)
            if mode == "neg_increment":
                dd, ddr, rc, rcr = mode_neg_increment(mass, donor, increment)
            else:
                dd, ddr, rc, rcr = mode_emit_reconcile(mass, donor, increment, receiver_mask)
            if not check(mass, donor, dd, ddr, rc, rcr):
                bad += 1
                if bad <= 3:
                    print(f"  {mode} seed{s} FAIL dd={float(dd):.9g} ddr={float(ddr.sum()):.9g} rc={float(rc):.9g} rcr={float(rcr.sum()):.9g}")
        print(f"{mode}: bad={bad}/40")


if __name__ == "__main__":
    main()