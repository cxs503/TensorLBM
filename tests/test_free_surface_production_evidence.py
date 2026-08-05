"""R1 production topology observer tests: real step, no invented f transfer."""
from __future__ import annotations

import torch

from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.free_surface_lbm import GAS, INTERFACE
from tensorlbm.free_surface_production_evidence import (
    extract_runtime_korner_evidence,
    observe_korner_runtime_evidence,
    run_free_surface_step_with_observer,
)
from tensorlbm.free_surface_transaction_contract import WITHHELD_NO_POPULATION_TRANSFER


def _state():
    shape = (5, 6, 7)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    fill = torch.zeros(shape)
    centre = (2, 3, 3)
    flags[centre] = INTERFACE
    fill[centre] = 1.0
    for q in range(1, 19):
        dz, dy, dx = int(C[q, 2]), int(C[q, 1]), int(C[q, 0])
        source = tuple((index - delta) % extent for index, delta, extent in zip(centre, (dz, dy, dx), shape))
        flags[source] = INTERFACE
        fill[source] = 0.5
    zero = torch.zeros(shape)
    return equilibrium3d(torch.ones(shape), zero, zero, zero), fill, flags, torch.zeros(shape, dtype=torch.bool), fill.clone()


def test_real_production_step_is_observed_and_withheld_without_f_transfer() -> None:
    f, fill, flags, solid, mass = _state()
    outcome, report = run_free_surface_step_with_observer(f, fill, flags, solid, mass=mass)

    assert len(outcome) == 5
    assert report.status == WITHHELD_NO_POPULATION_TRANSFER
    assert report.provenance == "production_free_surface_step_runtime_ledger"
    assert "runtime_ledger.steps" in report.available_keys
    assert "runtime_ledger.steps.conversion_evidence.i_to_g_population_owner_status" in report.available_keys
    assert report.contract_report is None


def test_shaped_mapping_is_not_claimed_production_and_does_not_promote_snapshots() -> None:
    evidence = extract_runtime_korner_evidence({
        "runtime_ledger": {"steps": [{"conversion_evidence": {
            "conversion_cells": (),
            "i_to_g_population_owner_status": WITHHELD_NO_POPULATION_TRANSFER,
            "f_before": (1.0,), "f_after": (0.0,),
        }}]},
    })
    report = observe_korner_runtime_evidence(evidence)

    assert evidence.provenance == "shaped_result_mapping_not_claimed_production"
    assert evidence.actual_f_population_transfer is False
    assert report.status == WITHHELD_NO_POPULATION_TRANSFER
    assert "runtime_ledger.steps.conversion_evidence.f_before" in report.available_keys
