"""TDD contract for the detached D3Q19 free-surface topology transaction."""
from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch

from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID, SOLID, free_surface_step
from tensorlbm.free_surface_topology_transaction import (
    TopologyTransactionError,
    build_topology_transaction,
    commit_topology_transaction,
)


def _state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    solid = torch.zeros(shape, dtype=torch.bool)
    zero = torch.zeros(shape)
    return equilibrium3d(torch.ones(shape), zero, zero, zero), fill, flags, fill.clone(), solid


def _masks(flags: torch.Tensor, fill: torch.Tensor, solid: torch.Tensor):
    gas = flags == GAS
    interface = flags == INTERFACE
    liquid = flags == LIQUID
    to_iface = gas & (fill > 0.01) & ~solid
    to_liq = interface & (fill >= 0.999) & ~solid
    to_gas = (interface | liquid) & (fill <= 0.01) & ~solid
    recv_new = torch.zeros_like(solid)
    return to_iface, to_liq, to_gas, recv_new


def _build(f, fill, flags, mass, solid):
    to_iface, to_liq, to_gas, recv_new = _masks(flags, fill, solid)
    zero = torch.zeros_like(fill)
    return build_topology_transaction(
        f, fill, flags, mass, to_iface=to_iface, to_liq=to_liq, to_gas=to_gas,
        recv_new=recv_new, redistribution_increment=zero, rho_liquid=1.0, rho_gas=1.0,
        solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE,
        solid_flag=SOLID,
        ux=zero, uy=zero, uz=zero,
    )


def test_candidate_is_detached_and_commit_does_not_mutate_inputs() -> None:
    f, fill, flags, mass, solid = _state()
    before = tuple(value.clone() for value in (f, fill, flags, mass))

    plan = _build(f, fill, flags, mass, solid)
    out = commit_topology_transaction(plan)

    for actual, expected in zip((f, fill, flags, mass), before):
        assert torch.equal(actual, expected)
    for actual, source in zip(out, (f, fill, flags, mass)):
        assert actual.data_ptr() != source.data_ptr()


@pytest.mark.parametrize("kind", ["direct-lg", "nonfinite", "solid-mismatch"])
def test_invalid_candidate_fails_closed_without_input_pollution(kind: str) -> None:
    f, fill, flags, mass, solid = _state()
    before = tuple(value.clone() for value in (f, fill, flags, mass))
    plan = _build(f, fill, flags, mass, solid)
    candidate = list(commit_topology_transaction(plan))
    if kind == "direct-lg":
        candidate[2].fill_(GAS)
        candidate[2][2, 3, 3] = LIQUID
    elif kind == "nonfinite":
        candidate[0][0, 0, 0, 0] = float("nan")
    else:
        solid[0, 0, 0] = True
        candidate[2][0, 0, 0] = GAS

    with pytest.raises(TopologyTransactionError):
        commit_topology_transaction(plan, candidate=tuple(candidate), solid_mask=solid)
    for actual, expected in zip((f, fill, flags, mass), before):
        assert torch.equal(actual, expected)


def _direct_lg(flags: torch.Tensor) -> int:
    liquid = flags == LIQUID
    return sum(int((liquid & (flags.roll((int(C[q, 2]), int(C[q, 1]), int(C[q, 0])), dims=(0, 1, 2)) == GAS)).sum()) for q in range(1, 19))


def test_actual_step_builds_once_and_returns_builder_candidate(monkeypatch) -> None:
    import tensorlbm.free_surface_lbm as module

    calls = 0
    original = module.build_topology_transaction

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        plan = original(*args, **kwargs)
        return replace(plan, fill=torch.full_like(plan.fill, 0.314159))

    monkeypatch.setattr(module, "build_topology_transaction", tracked)
    f, fill, flags, mass, solid = _state()
    _, returned_fill, after, _, _ = free_surface_step(f, fill, flags, solid, mass=mass)
    assert calls == 1
    assert torch.equal(returned_fill, torch.full_like(returned_fill, 0.314159))
    assert _direct_lg(after) == 0


def test_freeze_topology_does_not_invoke_transaction(monkeypatch) -> None:
    import tensorlbm.free_surface_lbm as module

    monkeypatch.setattr(module, "build_topology_transaction", lambda *args, **kwargs: pytest.fail("called"))
    f, fill, flags, mass, solid = _state()
    free_surface_step(f, fill, flags, solid, mass=mass, freeze_topology=True)


def test_builder_failure_is_atomic_and_does_not_append_runtime_step(monkeypatch) -> None:
    import tensorlbm.free_surface_lbm as module

    def fail(*args, **kwargs):
        raise TopologyTransactionError("injected builder failure")

    monkeypatch.setattr(module, "build_topology_transaction", fail)
    f, fill, flags, mass, solid = _state()
    before = tuple(value.clone() for value in (f, fill, flags, mass))
    runtime_ledger = {"steps": [{"step": 0}]}
    mass_ledger = {"sentinel": {"unchanged": True}}
    expected_mass_ledger = {"sentinel": {"unchanged": True}}

    with pytest.raises(TopologyTransactionError, match="injected builder failure"):
        free_surface_step(
            f, fill, flags, solid, mass=mass,
            runtime_ledger=runtime_ledger, mass_ledger=mass_ledger,
        )

    for actual, expected in zip((f, fill, flags, mass), before):
        assert torch.equal(actual, expected)
    assert runtime_ledger["steps"] == [{"step": 0}]
    assert mass_ledger == expected_mass_ledger


def test_runtime_ledger_checkpoints_and_evidence_come_from_built_plan(monkeypatch) -> None:
    import tensorlbm.free_surface_lbm as module

    original = module.build_topology_transaction
    sentinel_evidence = {"source": "built-plan"}

    def tracked(*args, **kwargs):
        plan = original(*args, **kwargs)
        return replace(
            plan,
            mass_after_redistribution=101.0,
            mass_after_clamp=102.0,
            mass_after_conversion=103.0,
            mass_after_isolation=104.0,
            conversion_evidence=sentinel_evidence,
        )

    monkeypatch.setattr(module, "build_topology_transaction", tracked)
    f, fill, flags, mass, solid = _state()
    runtime_ledger = {}
    free_surface_step(f, fill, flags, solid, mass=mass, runtime_ledger=runtime_ledger)

    step = runtime_ledger["steps"][-1]
    assert step["mass_after_redistribution"] == 101.0
    assert step["mass_after_clamp"] == 102.0
    assert step["mass_after_conversion"] == 103.0
    assert step["mass_after_isolation"] == 104.0
    assert step["conversion_evidence"] is sentinel_evidence


def test_real_topology_campaign_matches_baseline_conversion_contract() -> None:
    f, fill, flags, mass, solid = _state()
    for _ in range(10):
        f, fill, flags, mass, _ = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=1.0, rho_gas=1.0e-3,
            paired_liquid_interface_debit=True,
        )
        assert _direct_lg(flags) == 0
        assert bool(torch.isfinite(f).all())
        assert bool(torch.isfinite(fill).all())
        assert bool(torch.isfinite(mass).all())


def test_step_imports_transaction_cold_path() -> None:
    source = inspect.getsource(__import__("tensorlbm.free_surface_lbm", fromlist=["*"]))
    assert "free_surface_topology_transaction" in source
