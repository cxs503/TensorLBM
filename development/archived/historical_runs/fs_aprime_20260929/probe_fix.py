#!/usr/bin/env python3
"""Experiment: candidate exact constructions for the I->G ownership debit/credit."""
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


def build_candidate(flags, mass, donor, *, mode):
    receiver_mask = (flags == INTERFACE) & ~donor
    receiver_by_q = torch.stack([roll_from_pull_source(receiver_mask, q) for q in D3Q19_MOVING_Q])
    receiver_count = receiver_by_q.sum(dim=0)
    debit_field = torch.where(donor, mass, torch.zeros_like(mass))
    credit_per_link = debit_field / receiver_count.clamp(min=1).to(mass.dtype)
    increment = torch.stack(
        [roll_to_neighbor(credit_per_link, q) * receiver_mask for q in D3Q19_MOVING_Q]
    ).sum(dim=0)

    if mode == "current":
        donor_debit = -debit_field.sum()
        donor_debit_records = -debit_field
    elif mode == "emit_total":
        emit = torch.zeros_like(mass)
        for q in D3Q19_MOVING_Q:
            emit = emit + credit_per_link * roll_from_pull_source(receiver_mask, q).to(mass.dtype)
        donor_debit = -emit.sum()
        donor_debit_records = -emit
    elif mode == "force_neg_credit":
        donor_debit = -increment.sum()
        donor_debit_records = -debit_field
    elif mode == "emit_and_force":
        emit = torch.zeros_like(mass)
        for q in D3Q19_MOVING_Q:
            emit = emit + credit_per_link * roll_from_pull_source(receiver_mask, q).to(mass.dtype)
        donor_debit = -increment.sum()
        # scale emit to sum exactly to donor_debit via a residual correction on the max element
        resid = donor_debit - (-emit).sum()  # want sum(donor_debit_records) == donor_debit
        donor_debit_records = -emit
    receiver_credit = increment.sum()
    residual = donor_debit + receiver_credit
    return dict(donor_debit=float(donor_debit), receiver_credit=float(receiver_credit),
                residual=float(residual), records_ok=bool(donor_debit == donor_debit_records.sum()))


def main():
    print("seed1 n_donor synthetic check")
    for mode in ("current", "emit_total", "force_neg_credit"):
        flags, mass, donor = random_case(1)
        try:
            r = build_candidate(flags, mass, donor, mode=mode)
            print(f"  {mode}: resid={r['residual']:.3g} records_ok={r['records_ok']}")
        except Exception as e:
            print(f"  {mode}: EXC {e}")
    print("== 12 seeds, emit_total ==")
    bad = 0
    for s in range(12):
        flags, mass, donor = random_case(s)
        try:
            r = build_candidate(flags, mass, donor, mode="emit_total")
            if r['residual'] != 0.0 or not r['records_ok']:
                bad += 1
            print(f"  seed{s}: resid={r['residual']:.3g} records_ok={r['records_ok']}")
        except Exception as e:
            print(f"  seed{s}: EXC {e}")
    print("bad:", bad)


if __name__ == "__main__":
    main()