"""Strict local I→G independent-mass ownership closure contracts."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID, SOLID
from tensorlbm.free_surface_topology_transaction import (
    IToGOwnershipTransaction,
    TopologyTransactionError,
    build_i_to_g_ownership_transaction,
    build_topology_transaction,
)


def _state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (3, 3, 5)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    donor, receiver = (1, 1, 2), (1, 1, 3)
    flags[donor] = INTERFACE
    flags[receiver] = INTERFACE
    fill = torch.zeros(shape)
    fill[donor], fill[receiver] = 0.25, 0.50
    mass = fill.clone()
    zero = torch.zeros(shape)
    f = equilibrium3d(torch.ones(shape), zero, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)
    return f, fill, flags, mass, solid


def test_i_to_g_ownership_transaction_has_independent_signed_debit_and_credit_records() -> None:
    _, _, flags, mass, solid = _state()
    donor = torch.zeros_like(solid)
    donor[1, 1, 2] = True

    transaction = build_i_to_g_ownership_transaction(
        flags, mass, to_gas=donor, to_liq=torch.zeros_like(donor),
        solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE,
        rho_liquid=1.0,
    )

    assert transaction.donor_debit.dtype == mass.dtype
    assert transaction.receiver_credit.dtype == mass.dtype
    assert transaction.donor_debit.item() == pytest.approx(-0.25)
    assert transaction.receiver_credit.item() == pytest.approx(0.25)
    assert transaction.residual.item() == 0.0
    assert transaction.receiver_increment[1, 1, 3] == pytest.approx(0.25)
    assert len(transaction.links) == 1
    link = transaction.links[0]
    assert link["donor"] == (1, 1, 2)
    assert link["receiver"] == (1, 1, 3)
    assert link["q"] in range(1, 19)
    assert link["debit"] == pytest.approx(-0.25)
    assert link["credit"] == pytest.approx(0.25)


def test_i_to_g_ownership_transaction_rejects_tampered_debit_credit_evidence() -> None:
    _, _, flags, mass, solid = _state()
    donor = torch.zeros_like(solid)
    donor[1, 1, 2] = True
    transaction = build_i_to_g_ownership_transaction(
        flags, mass, to_gas=donor, to_liq=torch.zeros_like(donor),
        solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE,
        rho_liquid=1.0,
    )
    tampered = replace(transaction, receiver_credit=transaction.receiver_credit + torch.tensor(0.125))
    with pytest.raises(TopologyTransactionError, match="WITHHELD.*debit/credit"):
        tampered.validate()


def test_i_to_g_ownership_transaction_rejects_tampered_receiver_increment() -> None:
    _, _, flags, mass, solid = _state()
    donor = torch.zeros_like(solid)
    donor[1, 1, 2] = True
    transaction = build_i_to_g_ownership_transaction(
        flags, mass, to_gas=donor, to_liq=torch.zeros_like(donor),
        solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE,
        rho_liquid=1.0,
    )
    with pytest.raises(TopologyTransactionError, match="WITHHELD.*receiver credit aggregation"):
        replace(transaction, receiver_increment=torch.zeros_like(transaction.receiver_increment)).validate()


def test_topology_transaction_rejects_i_to_g_increment_detached_from_ownership_evidence() -> None:
    f, fill, flags, mass, solid = _state()
    donor = torch.zeros_like(solid)
    donor[1, 1, 2] = True
    transaction = build_i_to_g_ownership_transaction(
        flags, mass, to_gas=donor, to_liq=torch.zeros_like(donor),
        solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE,
        rho_liquid=1.0,
    )
    zero = torch.zeros_like(solid)
    velocity = torch.zeros_like(fill)
    with pytest.raises(TopologyTransactionError, match="WITHHELD.*tampered I→G increment evidence"):
        build_topology_transaction(
            f, fill, flags, mass, to_iface=zero, to_liq=zero, to_gas=donor, recv_new=zero,
            redistribution_increment=torch.zeros_like(mass), i_to_g_increment=torch.zeros_like(mass),
            i_to_g_ownership=transaction, rho_liquid=1.0, rho_gas=0.001,
            solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE, solid_flag=SOLID,
            ux=velocity, uy=velocity, uz=velocity,
        )


def test_i_to_g_mixed_legacy_increment_rejects_overflow_before_clamp() -> None:
    f, fill, flags, mass, solid = _state()
    donor = torch.zeros_like(solid)
    donor[1, 1, 2] = True
    transaction = build_i_to_g_ownership_transaction(
        flags, mass, to_gas=donor, to_liq=torch.zeros_like(donor),
        solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE,
        rho_liquid=1.0,
    )
    legacy = torch.zeros_like(mass)
    legacy[1, 1, 3] = 0.30
    zero = torch.zeros_like(solid)
    velocity = torch.zeros_like(fill)
    with pytest.raises(TopologyTransactionError, match="WITHHELD.*combined receiver capacity"):
        build_topology_transaction(
            f, fill, flags, mass, to_iface=zero, to_liq=zero, to_gas=donor, recv_new=zero,
            redistribution_increment=legacy, i_to_g_increment=transaction.receiver_increment,
            i_to_g_ownership=transaction, rho_liquid=1.0, rho_gas=0.001,
            solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE, solid_flag=SOLID,
            ux=velocity, uy=velocity, uz=velocity,
        )


def test_i_to_g_multi_donor_same_receiver_uses_combined_capacity_and_exact_records() -> None:
    _, _, flags, mass, solid = _state()
    donors = torch.zeros_like(solid)
    flags[1, 1, 1] = INTERFACE
    flags[1, 1, 2] = INTERFACE
    flags[1, 1, 3] = INTERFACE
    mass[1, 1, 1] = 0.125
    mass[1, 1, 2] = 0.50
    mass[1, 1, 3] = 0.125
    donors[1, 1, 1] = True
    donors[1, 1, 3] = True
    transaction = build_i_to_g_ownership_transaction(
        flags, mass, to_gas=donors, to_liq=torch.zeros_like(donors),
        solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE,
        rho_liquid=1.0,
    )
    assert transaction.donor_debit.item() + transaction.receiver_credit.item() == 0.0
    assert transaction.receiver_increment[1, 1, 2] == pytest.approx(0.25)


def test_i_to_g_ownership_transaction_fails_closed_when_no_legal_interface_receiver() -> None:
    _, _, flags, mass, solid = _state()
    donor = torch.zeros_like(solid)
    donor[1, 1, 2] = True
    flags[1, 1, 3] = GAS

    with pytest.raises(TopologyTransactionError, match="WITHHELD.*no legal INTERFACE receiver"):
        build_i_to_g_ownership_transaction(
            flags, mass, to_gas=donor, to_liq=torch.zeros_like(donor),
            solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE,
            rho_liquid=1.0,
        )


def test_i_to_g_transaction_rejects_liquid_to_gas_and_receiver_capacity_overflow() -> None:
    _, _, flags, mass, solid = _state()
    donor = torch.zeros_like(solid)
    donor[1, 1, 2] = True
    flags[1, 1, 2] = LIQUID
    mass[1, 1, 2] = 1.0
    with pytest.raises(TopologyTransactionError, match="WITHHELD.*LIQUID→GAS"):
        build_i_to_g_ownership_transaction(
            flags, mass, to_gas=donor, to_liq=torch.zeros_like(donor),
            solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE,
            rho_liquid=1.0,
        )

    flags[1, 1, 2] = INTERFACE
    flags[1, 1, 3] = INTERFACE
    mass[1, 1, 2], mass[1, 1, 3] = 0.25, 0.90
    with pytest.raises(TopologyTransactionError, match="WITHHELD.*capacity"):
        build_i_to_g_ownership_transaction(
            flags, mass, to_gas=donor, to_liq=torch.zeros_like(donor),
            solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE,
            rho_liquid=1.0,
        )


@pytest.mark.parametrize("failure", ("zero_receiver", "combined_capacity"))
def test_free_surface_step_i_to_g_failure_is_withheld_and_leaves_state_and_ledgers_unchanged(
    failure: str,
) -> None:
    from tensorlbm.free_surface_lbm import free_surface_step

    f, fill, flags, mass, solid = _state()
    donor, receiver = (1, 1, 2), (1, 1, 3)
    fill[donor] = 0.0
    mass[donor] = 0.0
    if failure == "zero_receiver":
        flags[receiver] = GAS
    else:
        fill[donor] = mass[donor] = 0.01
        fill[receiver] = mass[receiver] = 0.995
    before = tuple(field.clone() for field in (f, fill, flags, mass))
    runtime = {"nested": {"steps": ["unchanged"]}}
    ownership = {"nested": {"latest": ["unchanged"]}}
    audit = {"nested": {"audit": ["unchanged"]}}
    with pytest.raises(TopologyTransactionError, match="WITHHELD.*entire free_surface_step topology candidate"):
        free_surface_step(
            f, fill, flags, solid, mass=mass, runtime_ledger=runtime,
            ownership_ledger=ownership, conversion_density_audit_ledger=audit,
            enable_i_to_g_ownership_closure=True,
        )
    for actual, expected in zip((f, fill, flags, mass), before):
        assert torch.equal(actual, expected)
    assert runtime == {"nested": {"steps": ["unchanged"]}}
    assert ownership == {"nested": {"latest": ["unchanged"]}}
    assert audit == {"nested": {"audit": ["unchanged"]}}


def test_default_step_does_not_construct_i_to_g_proposal_and_matches_explicit_disabled(monkeypatch) -> None:
    import tensorlbm.free_surface_lbm as module
    from tensorlbm.free_surface_closure_experiment import _conversion_state

    f, fill, flags, solid = _conversion_state()
    monkeypatch.setattr(
        module, "build_i_to_g_ownership_transaction",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("default must not build I→G proposal")),
    )
    default = module.free_surface_step(f, fill, flags, solid, mass=fill.clone(), rho_gas=1.0e-3)
    disabled = module.free_surface_step(
        f, fill, flags, solid, mass=fill.clone(), rho_gas=1.0e-3,
        enable_i_to_g_ownership_closure=False,
    )
    for observed, expected in zip(default, disabled):
        assert torch.equal(observed, expected)



@pytest.mark.parametrize("invalid", [None, 0, 1, "false", "true"])
def test_i_to_g_ownership_opt_in_requires_strict_bool(invalid: object) -> None:
    from tensorlbm.free_surface_lbm import free_surface_step
    from tensorlbm.free_surface_closure_experiment import _conversion_state

    f, fill, flags, solid = _conversion_state()
    with pytest.raises(ValueError, match="enable_i_to_g_ownership_closure must be bool"):
        free_surface_step(
            f, fill, flags, solid, mass=fill.clone(),
            enable_i_to_g_ownership_closure=invalid,  # type: ignore[arg-type]
        )


def test_opt_in_closure_experiment_reports_b_and_c_as_failed_diagnostics() -> None:
    from tensorlbm.free_surface_closure_experiment import run_free_surface_closure_experiment

    report = run_free_surface_closure_experiment(enable_i_to_g_ownership_closure=True)
    cases = {case.case_id: case for case in report.cases}
    for case_id in ("B_forced_conversion_deterministic", "C_dam_break_style_tiny_dynamic_topology"):
        case = cases[case_id]
        assert case.status == "FAILED_DIAGNOSTIC"
        assert case.failure_reason is not None
        assert "WITHHELD: entire free_surface_step topology candidate has non-exact I→G debit/credit closure" in case.failure_reason


def test_topology_candidate_updates_receiver_fill_but_never_transfers_populations() -> None:
    f, fill, flags, mass, solid = _state()
    donor = torch.zeros_like(solid)
    donor[1, 1, 2] = True
    transaction = build_i_to_g_ownership_transaction(
        flags, mass, to_gas=donor, to_liq=torch.zeros_like(donor),
        solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE,
        rho_liquid=1.0,
    )
    zero = torch.zeros_like(solid)
    velocity = torch.zeros_like(fill)
    plan = build_topology_transaction(
        f, fill, flags, mass, to_iface=zero, to_liq=zero, to_gas=donor, recv_new=zero,
        redistribution_increment=torch.zeros_like(mass), i_to_g_increment=transaction.receiver_increment,
        i_to_g_ownership=transaction, rho_liquid=1.0, rho_gas=0.001,
        solid_mask=solid, gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE, solid_flag=SOLID,
        ux=velocity, uy=velocity, uz=velocity,

    )

    # The existing halo contract may immediately re-promote the emptied cell
    # to INTERFACE, but it owns zero mass/fill and no old donor population.
    assert plan.flags[1, 1, 2] == INTERFACE
    assert plan.mass[1, 1, 2] == pytest.approx(0.0)
    assert plan.fill[1, 1,2] == pytest.approx(0.0)
    # The halo may initialize a new empty interface at rho_gas, but no donor
    # population is copied into it.
    assert plan.f[:, 1, 1, 2].sum() == pytest.approx(0.001)
    assert plan.mass[1, 1, 3] == pytest.approx(0.75)
    assert plan.fill[1, 1, 3] == pytest.approx(0.75)
    assert torch.equal(plan.f[:, 1, 1, 3], f[:, 1, 1, 3])
