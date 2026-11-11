"""Reachability of the free-surface I->G conversion evidence path.

Background (see ``docs/CI_QUARANTINE_ADJUDICATION_R2.md``): sixteen free-surface
ledger tests are quarantined in ``tests/ci_quarantine.txt`` because they ask for
an ``I_TO_G`` cell in ``run_free_surface_closure_experiment()`` cases B/C at step
index 2 or 3.  Those cases never produce one, so the ledger payloads are empty
and the assertions fail with ``assert 0 > 0``.

The adjudication question is whether that is (a) a broken evidence path, or
(b) fixtures that no longer trigger the conversion rule.  These tests answer it
by driving the rule directly: ``free_surface_step`` converts an INTERFACE cell
to GAS when the cell has no LIQUID neighbour and its fill is below
``cell_conversion_force_threshold`` (``free_surface_lbm.py`` conversion marking,
Körner step 5).  An isolated low-fill INTERFACE cell therefore *must* emit an
``I_TO_G`` record in ``conversion_evidence`` and the cold audit must classify it
as such -- with zero LIQUID cells anywhere in the state.

These are capability-reachability tests, not physical validation: no analytic
solution exists for a one-cell state.  They exist so that "the I->G ledger is
unverified" stops being an unanswerable claim: either the path emits evidence or
it does not.
"""

from __future__ import annotations

import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_closure_experiment import run_free_surface_closure_experiment
from tensorlbm.free_surface_conversion_density_audit import (
    DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE,
    build_conversion_density_audit,
)
from tensorlbm.free_surface_lbm import GAS, INTERFACE, free_surface_step


def _as_mapping(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, tuple) and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value
    ):
        return {key: item for key, item in value}
    return value


def _isolated_low_fill_interface_state(fill_value: float = 0.05):
    shape = (6, 6, 6)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    fill = torch.zeros(shape)
    centre = (3, 3, 3)
    flags[centre] = INTERFACE
    fill[centre] = fill_value
    zero = torch.zeros(shape)
    f = equilibrium3d(fill.clone(), zero, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)
    return f, fill, flags, solid, centre


def test_isolated_low_fill_interface_converts_to_gas_with_evidence() -> None:
    f, fill, flags, solid, centre = _isolated_low_fill_interface_state()
    runtime: dict[str, object] = {}

    f, fill, flags, mass, _ = free_surface_step(
        f,
        fill,
        flags,
        solid,
        mass=fill.clone(),
        tau=1.0,
        rho_gas=1.0,
        runtime_ledger=runtime,
    )

    assert not (flags == 1).any(), "probe state must contain no LIQUID cell"
    assert int(flags[centre]) == GAS, "isolated low-fill INTERFACE cell must convert to GAS"
    assert float(mass[centre]) == 0.0

    record = _as_mapping(runtime["steps"][-1])
    evidence = _as_mapping(record["conversion_evidence"])
    cells = [_as_mapping(cell) for cell in evidence["conversion_cells"]]
    assert len(cells) == 1, f"expected exactly one conversion record, got {cells}"
    cell = cells[0]
    assert tuple(cell["cell"]) == centre
    assert int(cell["flag_before"]) == INTERFACE
    assert int(cell["flag_after"]) == GAS

    audit = build_conversion_density_audit(evidence, rho_liquid=1.0)
    assert audit.status == DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE
    assert audit.i_to_g_count == 1
    assert audit.i_to_l_count == 0
    assert audit.cells[0].classification == "I_TO_G"


def test_closure_experiment_b_and_c_emit_no_i_to_g_conversion_events() -> None:
    """Characterisation, not a target: the quarantined fixtures never reach I->G.

    Records the root cause of the batch-5 quarantine.  When this changes (an
    I_TO_G event appears), the free-surface ledger tests must be re-adjudicated
    and ``tests/ci_quarantine.txt`` re-examined before anything else.
    """
    report = run_free_surface_closure_experiment()
    seen = {"I_TO_L": 0, "I_TO_G": 0, "L_TO_G": 0, "G_TO_I": 0}
    for case in report.cases:
        if case.case_id not in {"B_forced_conversion_deterministic", "C_dam_break_style_tiny_dynamic_topology"}:
            continue
        for step in case.steps:
            record = _as_mapping(step.runtime_ledger)
            evidence = record.get("conversion_evidence")
            if not isinstance(evidence, (dict, tuple)):
                continue
            for cell in _as_mapping(evidence)["conversion_cells"]:
                cell = _as_mapping(cell)
                key = f"{ {0: 'G', 1: 'L', 2: 'I'}[int(cell['flag_before'])] }_TO_" + str(
                    {0: "G", 1: "L", 2: "I"}[int(cell["flag_after"])]
                )
                seen[key] = seen.get(key, 0) + 1

    assert seen["I_TO_G"] == 0
    assert seen["I_TO_L"] >= 1, "the fixture family should still exercise I->L conversion"