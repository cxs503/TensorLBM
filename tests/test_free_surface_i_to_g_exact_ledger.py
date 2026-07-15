"""Cold float32 D3Q19 I→G exact-ledger feasibility contracts."""
from __future__ import annotations

from typing import cast

import pytest
import torch

from tensorlbm.d3q19 import C
from tensorlbm.free_surface_i_to_g_exact_ledger import (
    DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION,
    WITHHELD_NOT_REPRESENTABLE,
    diagnose_i_to_g_exact_ledger,
)
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID


def _single_link_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (3, 3, 5)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    flags[1, 1, 2] = INTERFACE
    flags[1, 1, 3] = INTERFACE
    mass = torch.zeros(shape, dtype=torch.float32)
    mass[1, 1, 2] = 0.25
    donor = torch.zeros(shape, dtype=torch.bool)
    donor[1, 1, 2] = True
    return flags, mass, donor, torch.zeros_like(donor), torch.zeros_like(donor)


def _multi_link_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (5, 5, 5)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    donor_cell = (2, 2, 2)
    flags[donor_cell] = INTERFACE
    for q in (1, 2, 3):
        dz, dy, dx = (int(C[q, 2]), int(C[q, 1]), int(C[q, 0]))
        flags[(donor_cell[0] - dz) % 5, (donor_cell[1] - dy) % 5, (donor_cell[2] - dx) % 5] = INTERFACE
    mass = torch.zeros(shape, dtype=torch.float32)
    mass[donor_cell] = 0.1
    donor = torch.zeros(shape, dtype=torch.bool)
    donor[donor_cell] = True
    return flags, mass, donor, torch.zeros_like(donor), torch.zeros_like(donor)


def _diagnose(state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    flags, mass, to_gas, to_liq, solid = state
    return diagnose_i_to_g_exact_ledger(
        flags, mass, to_gas=to_gas, to_liq=to_liq, solid_mask=solid,
        gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE, rho_liquid=1.0,
    )


def test_single_link_is_withheld_without_complete_topology_replay() -> None:
    flags, mass, to_gas, to_liq, solid = _single_link_state()
    before = tuple(item.clone() for item in (flags, mass, to_gas, to_liq, solid))

    report = _diagnose((flags, mass, to_gas, to_liq, solid))

    assert report.status == WITHHELD_NOT_REPRESENTABLE
    assert report.physical_closure_claim is False
    assert report.mutates_solver_state is False
    assert report.method_a.status == DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION
    assert report.method_b.status == DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION
    assert report.method_c.status == WITHHELD_NOT_REPRESENTABLE
    assert report.method_c.reason == "complete same-order topology mutation is unavailable to this cold observer"
    assert report.actual_float32_operation_residual == 0.0
    assert report.donor_vs_rounded_link_residual_quanta == 0
    assert report.receiver_aggregation_nonzero_count == 0
    assert report.population_owner_status == "WITHHELD_NO_POPULATION_TRANSFER"
    for actual, expected in zip((flags, mass, to_gas, to_liq, solid), before):
        assert torch.equal(actual, expected)


def test_three_link_float32_division_is_withheld_when_actual_receiver_state_addition_rounds_away_residual() -> None:
    flags, mass, to_gas, to_liq, solid = _multi_link_state()
    # This nonzero receiver state is part of the actual solver mutation:
    # it demonstrates why an increment-only C check would be a false positive.
    mass[flags == INTERFACE] = torch.where(
        mass[flags == INTERFACE] == 0.0,
        torch.full_like(mass[flags == INTERFACE], 0.5),
        mass[flags == INTERFACE],
    )
    report = _diagnose((flags, mass, to_gas, to_liq, solid))

    assert report.status == WITHHELD_NOT_REPRESENTABLE
    assert report.method_a.status == DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION
    assert report.method_b.status == DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION
    assert report.method_a.residual_quanta == 0
    assert report.method_b.residual_quanta == 0
    assert report.donor_vs_rounded_link_residual_quanta != 0
    assert report.method_c.status == WITHHELD_NOT_REPRESENTABLE
    assert report.method_c.reason == "per-donor residual is rounded away or altered by actual receiver float32 state addition"


@pytest.mark.parametrize(
    ("case_id", "requested_steps"),
    (
        ("B_forced_conversion_deterministic", 3),
        ("C_dam_break_style_tiny_dynamic_topology", 10),
    ),
)
def test_real_b_and_c_candidates_are_withheld_by_receiver_float32_aggregation(
    monkeypatch, case_id: str, requested_steps: int,
) -> None:
    import tensorlbm.free_surface_lbm as solver
    from tensorlbm.free_surface_closure_experiment import _conversion_state, _run_case

    captured: list[dict[str, object]] = []
    original = solver.build_i_to_g_ownership_transaction

    def capture(*args, **kwargs):
        captured.append({"flags": args[0].clone(), "mass": args[1].clone(), **kwargs})
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "build_i_to_g_ownership_transaction", capture)
    f, fill, flags, solid = _conversion_state()
    case = _run_case(
        case_id, f, fill, flags, solid, requested_steps, False, True,
        enable_i_to_g_ownership_closure=True,
    )

    assert case.case_id == case_id
    assert case.status == "FAILED_DIAGNOSTIC"
    assert case.physical_closure_claim is False
    assert len(captured) == 1
    candidate = captured[0]
    exact = diagnose_i_to_g_exact_ledger(
        cast(torch.Tensor, candidate["flags"]), cast(torch.Tensor, candidate["mass"]),
        to_gas=cast(torch.Tensor, candidate["to_gas"]),
        to_liq=cast(torch.Tensor, candidate["to_liq"]),
        solid_mask=cast(torch.Tensor, candidate["solid_mask"]),
        gas_flag=cast(int, candidate["gas_flag"]),
        liquid_flag=cast(int, candidate["liquid_flag"]),
        interface_flag=cast(int, candidate["interface_flag"]),
        rho_liquid=cast(float, candidate["rho_liquid"]),
    )
    assert exact.status == WITHHELD_NOT_REPRESENTABLE
    assert exact.physical_closure_claim is False
    assert exact.mutates_solver_state is False
    assert exact.method_a.status == DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION
    assert exact.method_b.status == DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION
    assert exact.donor_count == 76
    assert exact.receiver_count == 110
    assert exact.link_count == 620
    assert exact.receiver_aggregation_nonzero_count > 0
    assert exact.receiver_aggregation_nonzero_count == 77
    assert exact.receiver_aggregation_max_abs_quanta > 0
    assert exact.method_c.status == WITHHELD_NOT_REPRESENTABLE
    assert exact.method_c.reason == "per-donor residual is rounded away or altered by receiver float32 increment aggregation"


def test_no_legal_receiver_and_liquid_donor_are_withheld() -> None:
    flags, mass, to_gas, to_liq, solid = _single_link_state()
    flags[1, 1, 3] = GAS
    missing_receiver = _diagnose((flags, mass, to_gas, to_liq, solid))
    assert missing_receiver.status == WITHHELD_NOT_REPRESENTABLE
    assert missing_receiver.legal_receivers is False

    flags, mass, to_gas, to_liq, solid = _single_link_state()
    flags[1, 1, 2] = LIQUID
    liquid_donor = _diagnose((flags, mass, to_gas, to_liq, solid))
    assert liquid_donor.status == WITHHELD_NOT_REPRESENTABLE
    assert liquid_donor.legal_receivers is False
