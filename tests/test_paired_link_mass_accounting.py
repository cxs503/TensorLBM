"""Unit tests for the pure LIQUID↔INTERFACE paired-link reference ledger."""
from __future__ import annotations

import torch

from tensorlbm.d3q19 import C, OPPOSITE
from tensorlbm.free_surface_lbm import INTERFACE, LIQUID
from tensorlbm.paired_link_mass_accounting import paired_liquid_interface_transfers


def _source_flags(flags: torch.Tensor) -> torch.Tensor:
    """Flags at x-c_q, consistent with D3Q19 pull streaming."""
    return torch.stack([
        flags.roll((int(C[q, 2]), int(C[q, 1]), int(C[q, 0])), (0, 1, 2))
        for q in range(19)
    ])


def test_paired_link_ledger_is_exactly_conservative_for_arbitrary_populations() -> None:
    """Every accepted L/I link has one transfer and equal/opposite endpoints."""
    generator = torch.Generator().manual_seed(9173)
    nz, ny, nx = 3, 4, 7
    # Deliberately arbitrary non-equilibrium populations.  Their cell sums
    # therefore include strong, unrelated density fluctuations.
    # Double precision makes the closed-domain aggregate check resolve the
    # algebraic pair, rather than a float32 reduction-order roundoff.
    f_post = torch.randn((19, nz, ny, nx), generator=generator, dtype=torch.float64)
    flags = torch.randint(0, 3, (nz, ny, nx), generator=generator, dtype=torch.int8)
    # Ensure there are several selected links independent of random outcome.
    flags[:, :, 2] = INTERFACE
    flags[:, :, 3] = LIQUID

    ledger = paired_liquid_interface_transfers(f_post, flags)
    expected_mask = (flags == INTERFACE).unsqueeze(0) & (_source_flags(flags) == LIQUID)

    assert torch.equal(ledger.link_mask, expected_mask)
    # Endpoints differ spatially, but each individual link transfer is credited
    # once and debited once; hence the closed-domain sum is exactly paired.
    assert torch.equal(ledger.total_delta, ledger.interface_delta + ledger.bulk_delta)
    aggregate = ledger.bulk_delta.sum() + ledger.interface_delta.sum()
    assert torch.allclose(aggregate, torch.zeros((), dtype=aggregate.dtype), atol=1.0e-12, rtol=0.0)


def test_paired_link_ledger_places_bulk_counterpart_at_liquid_source() -> None:
    """The bulk counterpart is at x-c_q, rather than at the interface cell."""
    flags = torch.zeros((1, 1, 5), dtype=torch.int8)
    flags[0, 0, 2] = INTERFACE
    flags[0, 0, 3] = LIQUID
    f_post = torch.zeros((19, 1, 1, 5))
    q = 1  # +x pull at x=2 originates from the liquid x=3 only if periodic? no
    # For pull q=+x, source of x=2 is x=1. Select the q=-x direction instead.
    q = int(OPPOSITE[q])
    # At target x=2: f_q^*(x-c_q) - f_bar(q)^*(x) = 1.25 - 0.5.
    f_post[q, 0, 0, 3] = 1.25
    f_post[int(OPPOSITE[q]), 0, 0, 2] = 0.5

    ledger = paired_liquid_interface_transfers(f_post, flags)

    assert ledger.link_mask[q, 0, 0, 2]
    assert ledger.transfer[q, 0, 0, 2].item() == 0.75
    assert ledger.interface_delta[0, 0, 2].item() == 0.75
    assert ledger.bulk_delta[0, 0, 3].item() == -0.75
    assert ledger.bulk_delta.sum().item() == -0.75


def test_reference_transfer_matches_current_f_post_fill_formula_on_interface_side() -> None:
    """The current mass stencil contains the interface half of this ledger."""
    generator = torch.Generator().manual_seed(44)
    f_post = torch.randn((19, 2, 3, 6), generator=generator)
    flags = torch.zeros((2, 3, 6), dtype=torch.int8)
    flags[:, :, 2] = INTERFACE
    flags[:, :, 3:] = LIQUID

    ledger = paired_liquid_interface_transfers(f_post, flags)
    pulled = torch.stack([
        f_post[q].roll((int(C[q, 2]), int(C[q, 1]), int(C[q, 0])), (0, 1, 2))
        for q in range(19)
    ])
    current_interface_delta = torch.where(
        (flags == INTERFACE).unsqueeze(0) & (_source_flags(flags) == LIQUID),
        pulled - f_post[OPPOSITE],
        torch.zeros_like(f_post),
    ).sum(0)

    assert torch.allclose(ledger.interface_delta, current_interface_delta, atol=0.0, rtol=0.0)
