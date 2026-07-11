"""TDD specification for the pure free-surface inventory reference ledger."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.free_surface_inventory import (
    InventoryState,
    InventoryTopologyError,
    SolverMappingGap,
    apply_frozen_step,
    solver_mapping_gaps,
)
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID


def _state() -> InventoryState:
    # The liquid densities are intentionally neither equal nor tied to fill.
    flags = torch.tensor([LIQUID, INTERFACE, INTERFACE, GAS], dtype=torch.int8)
    return InventoryState.from_fields(
        flags,
        bulk_liquid=torch.tensor([1.37, 0.0, 0.0, 0.0]),
        interface_mass=torch.tensor([0.0, 0.42, 0.18, 0.0]),
        rho_liquid=1.0,
    )


def test_frozen_closed_step_pairs_liquid_interface_links_at_arbitrary_density() -> None:
    state = _state()
    initial = state.total
    # Positive is owned by the liquid endpoint and transferred to its interface
    # endpoint.  The inverse link proves that no global rescaling is involved.
    next_state, ledger = apply_frozen_step(
        state,
        link_exchanges=[(0, 1, 0.31), (0, 2, -0.07)],
    )

    assert next_state.bulk_liquid.tolist() == pytest.approx([1.13, 0.0, 0.0, 0.0])
    assert next_state.interface_mass.tolist() == pytest.approx([0.0, 0.73, 0.11, 0.0])
    assert next_state.total == pytest.approx(initial)
    assert ledger.delta("liquid_interface_exchange") == pytest.approx(0.0)
    assert ledger.total_delta == pytest.approx(0.0)


def test_conversion_overflow_is_owned_then_redistributed_without_loss() -> None:
    flags = torch.tensor([LIQUID, INTERFACE, INTERFACE, GAS], dtype=torch.int8)
    state = InventoryState.from_fields(
        flags,
        bulk_liquid=torch.tensor([0.83, 0.0, 0.0, 0.0]),
        interface_mass=torch.tensor([0.0, 1.35, 0.20, 0.0]),
    )
    next_state, ledger = apply_frozen_step(
        state,
        conversions=[(1, LIQUID)],
        redistributions=[(1, ((2, 0.35),))],
    )

    assert next_state.flags.tolist() == [LIQUID, LIQUID, INTERFACE, GAS]
    assert next_state.bulk_liquid.tolist() == pytest.approx([0.83, 1.0, 0.0, 0.0])
    assert next_state.interface_mass.tolist() == pytest.approx([0.0, 0.0, 0.55, 0.0])
    assert next_state.total == pytest.approx(state.total)
    # The conversion reserves its owned overflow; the following redistribution
    # transfers exactly that reservation to the receiving interface site.
    assert ledger.delta("interface_to_liquid") == pytest.approx(-0.35)
    assert ledger.delta("redistribution") == pytest.approx(0.35)
    assert ledger.total_delta == pytest.approx(0.0, abs=1.0e-6)


def test_invalid_overflow_or_topology_is_rejected_instead_of_clamped_or_rescaled() -> None:
    state = InventoryState.from_fields(
        torch.tensor([LIQUID, INTERFACE], dtype=torch.int8),
        bulk_liquid=torch.tensor([1.0, 0.0]),
        interface_mass=torch.tensor([0.0, 1.1]),
    )
    with pytest.raises(InventoryTopologyError, match="overflow"):
        apply_frozen_step(state)
    state = _state()
    with pytest.raises(InventoryTopologyError, match="LIQUID.*INTERFACE"):
        apply_frozen_step(state, link_exchanges=[(1, 2, 0.1)])


def test_current_solver_ledger_has_explicit_unmappable_ownership_gaps() -> None:
    gaps = solver_mapping_gaps(
        {
            "exchange_liquid_delta": -0.2,
            "abb_population_delta": 0.3,
            "conversion": 4.0,
            "redistribution": 4.0,
        }
    )
    assert {gap.code for gap in gaps} == {
        "unpaired_liquid_interface_exchange",
        "abb_population_inventory_ownership",
        "conversion_transaction_ownership",
        "redistribution_transaction_ownership",
    }
    assert all(isinstance(gap, SolverMappingGap) for gap in gaps)
