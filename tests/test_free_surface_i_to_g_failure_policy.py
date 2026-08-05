"""Cold I→G strict-failure campaign policy contracts."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID, free_surface_step
from tensorlbm.free_surface_topology_transaction import TopologyTransactionError


@dataclass(frozen=True)
class _State:
    f: torch.Tensor
    fill: torch.Tensor
    flags: torch.Tensor
    mass: torch.Tensor


def _failure_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (3, 3, 5)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    flags[1, 1, 2] = INTERFACE
    fill = torch.zeros(shape)
    fill[1, 1, 2] = 0.01
    zero = torch.zeros(shape)
    return (
        equilibrium3d(torch.ones(shape), zero, zero, zero), fill, flags, fill.clone(),
        torch.zeros(shape, dtype=torch.bool),
    )


def _legacy_step(state: _State) -> _State:
    f, fill, flags, mass, solid = _failure_inputs()
    del f, fill, flags, mass
    result = free_surface_step(
        state.f, state.fill, state.flags, solid, mass=state.mass,
        enable_i_to_g_ownership_closure=False,
    )
    return _State(*result[:4])


def _strict_step(state: _State, capture: dict[str, object]) -> _State:
    f, fill, flags, mass, solid = _failure_inputs()
    del f, fill, flags, mass
    result = free_surface_step(
        state.f, state.fill, state.flags, solid, mass=state.mass,
        enable_i_to_g_ownership_closure=True, capture_replay_stages=True,
        replay_capture=capture,
    )
    return _State(*result[:4])


def _equal(left: _State, right: _State) -> bool:
    return all(torch.equal(a, b) for a, b in zip(left.__dict__.values(), right.__dict__.values()))


def _snapshot(state: _State) -> _State:
    return _State(*(field.clone() for field in state.__dict__.values()))


def _fingerprint(state: _State) -> tuple[tuple[str, tuple[int, ...], bytes], ...]:
    return tuple(
        (str(field.dtype), tuple(field.shape), field.detach().cpu().contiguous().numpy().tobytes())
        for field in state.__dict__.values()
    )


def test_policy_module_is_cold_and_not_package_default_import() -> None:
    import tensorlbm
    import tensorlbm.free_surface_i_to_g_failure_policy as module

    source = inspect.getsource(module)
    assert "hull_free_surface" not in source
    assert "dam_break" not in source
    assert "free_surface_step" not in source
    assert "free_surface_i_to_g_failure_policy" not in inspect.getsource(tensorlbm)


def test_raise_propagates_original_error_without_evidence_and_preserves_mutating_callback_prestate() -> None:
    from tensorlbm.free_surface_i_to_g_failure_policy import run_i_to_g_policy_campaign

    f, fill, flags, mass, _ = _failure_inputs()
    initial = _State(f, fill, flags, mass)
    before = _State(*(field.clone() for field in initial.__dict__.values()))
    expected = TopologyTransactionError("WITHHELD: original strict rejection")

    def mutating_without_evidence(state: _State, _: dict[str, object]) -> _State:
        state.mass.fill_(99.0)
        raise expected

    with pytest.raises(TopologyTransactionError) as caught:
        run_i_to_g_policy_campaign(
            initial, 1, mutating_without_evidence,
            snapshot_state=_snapshot, states_equal=_equal, fingerprint_state=_fingerprint,
        )
    assert caught.value is expected
    assert _equal(initial, before)


def test_non_raising_policy_rejects_missing_exact_failure_evidence_after_preserving_prestate() -> None:
    from tensorlbm.free_surface_i_to_g_failure_policy import (
        IToGStrictFailurePolicy,
        run_i_to_g_policy_campaign,
    )

    f, fill, flags, mass, _ = _failure_inputs()
    initial = _State(f, fill, flags, mass)
    before = _snapshot(initial)

    def mutating_without_capture(state: _State, _: dict[str, object]) -> _State:
        state.mass.fill_(99.0)
        raise TopologyTransactionError("WITHHELD: synthetic strict rejection")

    with pytest.raises(RuntimeError, match="strict failure evidence is unavailable"):
        run_i_to_g_policy_campaign(
            initial, 1, mutating_without_capture,
            policy=IToGStrictFailurePolicy.STOP_AND_REPORT,
            snapshot_state=_snapshot, states_equal=_equal, fingerprint_state=_fingerprint,
        )
    assert _equal(initial, before)


def test_shallow_snapshot_tensor_sharing_is_deepcopied_before_strict_callback() -> None:
    from tensorlbm.free_surface_i_to_g_failure_policy import run_i_to_g_policy_campaign

    f, fill, flags, mass, _ = _failure_inputs()
    initial = _State(f, fill, flags, mass)
    before = _snapshot(initial)

    def shallow_snapshot(state: _State) -> _State:
        return _State(*state.__dict__.values())

    def mutating_failure(state: _State, _: dict[str, object]) -> _State:
        state.mass.fill_(99.0)
        raise TopologyTransactionError("WITHHELD: shallow snapshot mutation")

    with pytest.raises(TopologyTransactionError, match="shallow snapshot mutation"):
        run_i_to_g_policy_campaign(
            initial, 1, mutating_failure,
            snapshot_state=shallow_snapshot, states_equal=_equal, fingerprint_state=_fingerprint,
        )
    assert _equal(initial, before)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({}, "all policies require snapshot_state, states_equal, and fingerprint_state"),
        ({"snapshot_state": _snapshot, "states_equal": _equal}, "all policies require snapshot_state, states_equal, and fingerprint_state"),
        ({"snapshot_state": _snapshot, "fingerprint_state": _fingerprint}, "all policies require snapshot_state, states_equal, and fingerprint_state"),
        ({"states_equal": _equal, "fingerprint_state": _fingerprint}, "all policies require snapshot_state, states_equal, and fingerprint_state"),
    ),
)
def test_all_policies_reject_missing_state_isolation_callbacks(kwargs: dict[str, object], message: str) -> None:
    from tensorlbm.free_surface_i_to_g_failure_policy import run_i_to_g_policy_campaign

    f, fill, flags, mass, _ = _failure_inputs()
    with pytest.raises(ValueError, match=message):
        run_i_to_g_policy_campaign(_State(f, fill, flags, mass), 1, _strict_step, **kwargs)


def test_fallback_deepcopies_adapter_identity_snapshot() -> None:
    from tensorlbm.free_surface_i_to_g_failure_policy import (
        IToGStrictFailurePolicy,
        run_i_to_g_policy_campaign,
    )

    f, fill, flags, mass, _ = _failure_inputs()
    initial = _State(f, fill, flags, mass)
    report = run_i_to_g_policy_campaign(
        initial, 1, _strict_step,
        policy=IToGStrictFailurePolicy.SKIP_EXPERIMENTAL_PROPOSAL,
        allow_experimental_fallback=True,
        legacy_step=_legacy_step,
        snapshot_state=lambda state: state,
        states_equal=_equal,
        fingerprint_state=_fingerprint,
    )
    assert report.committed_steps == 1


def test_stop_and_report_returns_last_committed_state_and_exact_failure_evidence() -> None:
    from tensorlbm.free_surface_i_to_g_failure_policy import (
        IToGStrictFailurePolicy,
        STOPPED_AND_REPORTED,
        run_i_to_g_policy_campaign,
    )

    f, fill, flags, mass, _ = _failure_inputs()
    initial = _State(f, fill, flags, mass)
    committed = _State(*(field.clone() for field in initial.__dict__.values()))
    attempts = 0

    def step(state: _State, capture: dict[str, object]) -> _State:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            return committed
        return _strict_step(state, capture)

    report = run_i_to_g_policy_campaign(
        initial, 3, step, policy=IToGStrictFailurePolicy.STOP_AND_REPORT,
        snapshot_state=_snapshot, states_equal=_equal, fingerprint_state=_fingerprint,
    )
    assert report.status == STOPPED_AND_REPORTED
    assert report.physical_closure_claim is False
    assert report.committed_steps == 2
    assert report.attempted_steps == 3
    assert _equal(report.state, committed)
    assert report.failure is not None
    assert report.failure.strict_replay.status == "STRICT_FAILURE_REPLAYED_EXACT"
    assert report.failure.residual_audit.status == "WITHHELD_NOT_REPRESENTABLE"
    assert report.ledger.committed_steps == (1, 2)
    assert report.ledger.attempted_steps == (1, 2, 3)
    assert report.ledger.fallback_steps == ()


def test_explicit_fallback_restarts_same_prestate_with_legacy_path_and_is_withheld() -> None:
    from tensorlbm.free_surface_i_to_g_failure_policy import (
        FALLBACK_NOT_PHYSICAL,
        IToGStrictFailurePolicy,
        WITHHELD,
        run_i_to_g_policy_campaign,
    )

    f, fill, flags, mass, _ = _failure_inputs()
    initial = _State(f, fill, flags, mass)
    strict_prestate: _State | None = None
    legacy_prestate: _State | None = None

    def strict(state: _State, capture: dict[str, object]) -> _State:
        nonlocal strict_prestate
        strict_prestate = state
        return _strict_step(state, capture)

    def legacy(state: _State) -> _State:
        nonlocal legacy_prestate
        legacy_prestate = state
        return _legacy_step(state)

    report = run_i_to_g_policy_campaign(
        initial, 1, strict,
        policy=IToGStrictFailurePolicy.SKIP_EXPERIMENTAL_PROPOSAL,
        allow_experimental_fallback=True,
        legacy_step=legacy,
        snapshot_state=_snapshot,
        states_equal=_equal,
        fingerprint_state=_fingerprint,
    )
    assert report.status == WITHHELD
    assert report.fallback_status == FALLBACK_NOT_PHYSICAL
    assert report.physical_closure_claim is False
    assert strict_prestate is not None and legacy_prestate is not None
    assert _equal(strict_prestate, legacy_prestate)
    assert report.committed_steps == 1
    assert report.attempted_steps == 1
    assert report.ledger.committed_steps == (1,)
    assert report.ledger.attempted_steps == (1,)
    assert report.ledger.fallback_steps == (1,)
    assert report.failure is not None


def test_fallback_deepcopies_a_shallow_wrapper_before_legacy_mutation() -> None:
    from tensorlbm.free_surface_i_to_g_failure_policy import (
        IToGStrictFailurePolicy,
        run_i_to_g_policy_campaign,
    )

    f, fill, flags, mass, _ = _failure_inputs()
    initial = _State(f, fill, flags, mass)
    original_prestate = _snapshot(initial)
    original_committed = _snapshot(initial)
    original_baseline = _snapshot(initial)

    def shallow_wrapper_snapshot(state: _State) -> _State:
        return _State(*state.__dict__.values())

    def mutating_legacy(state: _State) -> _State:
        state.mass.fill_(99.0)
        return state

    with pytest.raises(RuntimeError, match="legacy callback caused fallback input fingerprint mutation"):
        run_i_to_g_policy_campaign(
            initial,
            1,
            _strict_step,
            policy=IToGStrictFailurePolicy.SKIP_EXPERIMENTAL_PROPOSAL,
            allow_experimental_fallback=True,
            legacy_step=mutating_legacy,
            snapshot_state=shallow_wrapper_snapshot,
            states_equal=_equal,
            fingerprint_state=_fingerprint,
        )

    assert _equal(initial, original_prestate)
    assert _equal(initial, original_committed)
    assert _equal(initial, original_baseline)


def test_fallback_revalidates_isolation_after_legacy_exception() -> None:
    from tensorlbm.free_surface_i_to_g_failure_policy import (
        IToGStrictFailurePolicy,
        run_i_to_g_policy_campaign,
    )

    f, fill, flags, mass, _ = _failure_inputs()
    initial = _State(f, fill, flags, mass)
    original = _snapshot(initial)

    def shallow_wrapper_snapshot(state: _State) -> _State:
        return _State(*state.__dict__.values())

    def mutating_legacy_then_raises(state: _State) -> _State:
        state.mass.fill_(99.0)
        raise ValueError("legacy callback failure")

    with pytest.raises(RuntimeError, match="legacy callback caused fallback input fingerprint mutation"):
        run_i_to_g_policy_campaign(
            initial,
            1,
            _strict_step,
            policy=IToGStrictFailurePolicy.SKIP_EXPERIMENTAL_PROPOSAL,
            allow_experimental_fallback=True,
            legacy_step=mutating_legacy_then_raises,
            snapshot_state=shallow_wrapper_snapshot,
            states_equal=_equal,
            fingerprint_state=_fingerprint,
        )

    assert _equal(initial, original)


@pytest.mark.parametrize("case_id", ("B_forced_conversion_deterministic", "C_dam_break_style_tiny_dynamic_topology"))
def test_real_b_c_stop_reports_two_commits_then_step_three_strict_failure(case_id: str) -> None:
    from tensorlbm.free_surface_closure_experiment import _conversion_state
    from tensorlbm.free_surface_i_to_g_failure_policy import (
        IToGStrictFailurePolicy,
        STOPPED_AND_REPORTED,
        run_i_to_g_policy_campaign,
    )

    f, fill, flags, solid = _conversion_state()
    initial = _State(f, fill, flags, fill.clone())

    def strict(state: _State, capture: dict[str, object]) -> _State:
        result = free_surface_step(
            state.f, state.fill, state.flags, solid, mass=state.mass,
            tau=1.0, rho_gas=1.0e-3, paired_liquid_interface_debit=True,
            enable_i_to_g_ownership_closure=True, capture_replay_stages=True,
            replay_capture=capture,
        )
        return _State(*result[:4])

    report = run_i_to_g_policy_campaign(
        initial, 3 if case_id.startswith("B_") else 10, strict,
        policy=IToGStrictFailurePolicy.STOP_AND_REPORT,
        snapshot_state=_snapshot, states_equal=_equal, fingerprint_state=_fingerprint,
    )
    assert report.status == STOPPED_AND_REPORTED
    assert report.committed_steps == 2
    assert report.attempted_steps == 3
    assert report.failure is not None
    assert report.failure.strict_replay.status == "STRICT_FAILURE_REPLAYED_EXACT"


@pytest.mark.parametrize("case_id", ("B_forced_conversion_deterministic", "C_dam_break_style_tiny_dynamic_topology"))
def test_real_b_c_explicit_fallback_continues_legacy_but_remains_not_physical(case_id: str) -> None:
    from tensorlbm.free_surface_closure_experiment import _conversion_state
    from tensorlbm.free_surface_i_to_g_failure_policy import (
        FALLBACK_NOT_PHYSICAL,
        IToGStrictFailurePolicy,
        WITHHELD,
        run_i_to_g_policy_campaign,
    )

    f, fill, flags, solid = _conversion_state()
    initial = _State(f, fill, flags, fill.clone())

    def strict(state: _State, capture: dict[str, object]) -> _State:
        result = free_surface_step(
            state.f, state.fill, state.flags, solid, mass=state.mass,
            tau=1.0, rho_gas=1.0e-3, paired_liquid_interface_debit=True,
            enable_i_to_g_ownership_closure=True, capture_replay_stages=True,
            replay_capture=capture,
        )
        return _State(*result[:4])

    def legacy(state: _State) -> _State:
        result = free_surface_step(
            state.f, state.fill, state.flags, solid, mass=state.mass,
            tau=1.0, rho_gas=1.0e-3, paired_liquid_interface_debit=True,
            enable_i_to_g_ownership_closure=False,
        )
        return _State(*result[:4])

    report = run_i_to_g_policy_campaign(
        initial, 3 if case_id.startswith("B_") else 10, strict,
        policy=IToGStrictFailurePolicy.SKIP_EXPERIMENTAL_PROPOSAL,
        allow_experimental_fallback=True,
        legacy_step=legacy,
        snapshot_state=_snapshot,
        states_equal=_equal,
        fingerprint_state=_fingerprint,
    )
    assert report.status == WITHHELD
    assert report.fallback_status == FALLBACK_NOT_PHYSICAL
    assert report.committed_steps == (3 if case_id.startswith("B_") else 10)
    assert report.attempted_steps == report.committed_steps
    assert report.ledger.fallback_steps
    assert report.ledger.fallback_steps[0] == 3
