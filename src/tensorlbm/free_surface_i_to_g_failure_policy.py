"""Cold campaign policy for strict I→G experimental ownership failures.

This module is intentionally not imported by :mod:`tensorlbm` or any default
solver/hull/dam-break caller.  It accepts caller-owned immutable state adapters,
replays a rejected experimental proposal only through the caller's explicit
legacy callback, and reports the pre-existing strict failure evidence without
altering the numerical gate or solver state.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, TypeVar

from .free_surface_i_to_g_failed_residual_audit import (
    FailedIToGResidualAuditReport,
    audit_failed_i_to_g_residual,
)
from .free_surface_topology_mutation_replay_contract import (
    StrictFailureReplayReport,
    audit_strict_failure_replay,
)
from .free_surface_topology_transaction import TopologyTransactionError


WITHHELD = "WITHHELD"
STOPPED_AND_REPORTED = "STOPPED_AND_REPORTED"
FALLBACK_NOT_PHYSICAL = "FALLBACK_NOT_PHYSICAL"

StateT = TypeVar("StateT")
StrictStep = Callable[[StateT, dict[str, object]], StateT]
LegacyStep = Callable[[StateT], StateT]
StateComparator = Callable[[StateT, StateT], bool]
StateSnapshot = Callable[[StateT], StateT]
StateFingerprint = Callable[[StateT], object]


class IToGStrictFailurePolicy(str, Enum):
    """Caller-selected response to a strict experimental I→G rejection."""

    RAISE = "RAISE"
    STOP_AND_REPORT = "STOP_AND_REPORT"
    SKIP_EXPERIMENTAL_PROPOSAL = "SKIP_EXPERIMENTAL_PROPOSAL"


@dataclass(frozen=True)
class IToGFailureDiagnostic:
    step: int
    error_type: str
    error_message: str
    strict_replay: StrictFailureReplayReport
    residual_audit: FailedIToGResidualAuditReport


@dataclass(frozen=True)
class IToGCampaignLedger:
    """Step identities, keeping failed attempts distinct from commits."""

    attempted_steps: tuple[int, ...]
    committed_steps: tuple[int, ...]
    fallback_steps: tuple[int, ...]


@dataclass(frozen=True)
class IToGPolicyCampaignReport(Generic[StateT]):
    """Cold-policy result; never a physical closure result."""

    status: str
    fallback_status: str | None
    physical_closure_claim: bool
    requested_steps: int
    attempted_steps: int
    committed_steps: int
    state: StateT
    ledger: IToGCampaignLedger
    failure: IToGFailureDiagnostic | None


def _validate_inputs(
    requested_steps: int,
    strict_step: StrictStep[StateT],
    policy: IToGStrictFailurePolicy,
    allow_experimental_fallback: bool,
    legacy_step: LegacyStep[StateT] | None,
    snapshot_state: StateSnapshot[StateT] | None,
    states_equal: StateComparator[StateT] | None,
    fingerprint_state: StateFingerprint[StateT] | None,
) -> None:
    if isinstance(requested_steps, bool) or not isinstance(requested_steps, int) or requested_steps <= 0:
        raise ValueError("requested_steps must be a positive integer")
    if not isinstance(policy, IToGStrictFailurePolicy):
        raise ValueError("policy must be an IToGStrictFailurePolicy")
    if not isinstance(allow_experimental_fallback, bool):
        raise ValueError("allow_experimental_fallback must be bool")
    fallback = policy is IToGStrictFailurePolicy.SKIP_EXPERIMENTAL_PROPOSAL
    if fallback != allow_experimental_fallback:
        raise ValueError("experimental fallback requires both explicit policy and allow_experimental_fallback=True")
    if fallback and legacy_step is None:
        raise ValueError("experimental fallback requires legacy_step")
    if not callable(strict_step):
        raise ValueError("strict_step must be callable")
    if snapshot_state is None or states_equal is None or fingerprint_state is None:
        raise ValueError("all policies require snapshot_state, states_equal, and fingerprint_state")
    if not fallback and legacy_step is not None:
        raise ValueError("legacy_step is permitted only for explicit experimental fallback")
    if not callable(snapshot_state):
        raise ValueError("snapshot_state must be callable")
    if not callable(states_equal):
        raise ValueError("states_equal must be callable")
    if not callable(fingerprint_state):
        raise ValueError("fingerprint_state must be callable")
    if legacy_step is not None and not callable(legacy_step):
        raise ValueError("legacy_step must be callable")


def _require_unchanged_prestate(
    prestate: StateT,
    committed_snapshot: StateT,
    immutable_baseline: StateT,
    expected_fingerprints: tuple[object, object, object],
    states_equal: StateComparator[StateT],
    fingerprint_state: StateFingerprint[StateT],
    *,
    fallback_input: StateT | None = None,
    expected_fallback_fingerprint: object | None = None,
    callback_name: str = "strict",
) -> None:
    """Fail closed unless retained states prove callback isolation."""
    try:
        comparisons = (
            states_equal(prestate, immutable_baseline),
            states_equal(committed_snapshot, immutable_baseline),
        )
    except Exception as error:
        raise RuntimeError(f"states_equal failed while verifying {callback_name} callback isolation") from error
    if not all(type(result) is bool for result in comparisons):
        raise RuntimeError(f"states_equal must return bool while verifying {callback_name} callback isolation")
    if not all(comparisons):
        raise RuntimeError(f"{callback_name} callback mutated campaign prestate")
    try:
        fingerprints = (
            fingerprint_state(prestate),
            fingerprint_state(committed_snapshot),
            fingerprint_state(immutable_baseline),
        )
    except Exception as error:
        raise RuntimeError(f"fingerprint_state failed while verifying {callback_name} callback isolation") from error
    if fingerprints != expected_fingerprints:
        raise RuntimeError(f"{callback_name} callback caused state fingerprint mutation")
    if fallback_input is not None:
        if expected_fallback_fingerprint is None:
            raise RuntimeError("fallback fingerprint is required while verifying legacy callback isolation")
        try:
            fallback_fingerprint = fingerprint_state(fallback_input)
        except Exception as error:
            raise RuntimeError("fingerprint_state failed while verifying legacy callback isolation") from error
        if fallback_fingerprint != expected_fallback_fingerprint:
            raise RuntimeError("legacy callback caused fallback input fingerprint mutation")


def _isolated_snapshot(
    source: StateT,
    snapshot_state: StateSnapshot[StateT],
    description: str,
) -> StateT:
    """Apply the caller adapter, then enforce wrapper-level deep isolation."""
    try:
        snapshot = snapshot_state(source)
    except Exception as error:
        raise RuntimeError(f"snapshot_state failed while creating {description}") from error
    try:
        isolated = copy.deepcopy(snapshot)
    except Exception as error:
        raise RuntimeError(f"{description} must support deepcopy isolation") from error
    if isolated is snapshot:
        raise RuntimeError(f"{description} deepcopy did not return an independent state")
    return isolated


def _diagnostic(step: int, error: TopologyTransactionError, capture: dict[str, object]) -> IToGFailureDiagnostic:
    evidence = capture.get("strict_failure_evidence")
    diagnostic = IToGFailureDiagnostic(
        step=step,
        error_type=type(error).__name__,
        error_message=str(error),
        strict_replay=audit_strict_failure_replay(evidence),
        residual_audit=audit_failed_i_to_g_residual(evidence),
    )
    if diagnostic.strict_replay.status != "STRICT_FAILURE_REPLAYED_EXACT":
        raise RuntimeError("strict failure evidence is unavailable or does not replay exactly") from error
    return diagnostic


def run_i_to_g_policy_campaign(
    initial_state: StateT,
    requested_steps: int,
    strict_step: StrictStep[StateT],
    *,
    policy: IToGStrictFailurePolicy = IToGStrictFailurePolicy.RAISE,
    allow_experimental_fallback: bool = False,
    legacy_step: LegacyStep[StateT] | None = None,
    snapshot_state: StateSnapshot[StateT] | None = None,
    states_equal: StateComparator[StateT] | None = None,
    fingerprint_state: StateFingerprint[StateT] | None = None,
) -> IToGPolicyCampaignReport[StateT]:
    """Run caller-provided cold campaign steps with an explicit failure policy.

    All policies require ``snapshot_state``, ``states_equal``, and
    ``fingerprint_state``. ``snapshot_state`` must return a new state wrapper
    and the complete state must support ``copy.deepcopy``; the wrapper deep-copies
    every adapter snapshot before any callback. ``states_equal`` must compare every state tensor exactly, and
    ``fingerprint_state`` must return an immutable value representing the whole
    state. The wrapper invokes ``strict_step`` only on a detached proposal
    snapshot and verifies its retained prestate and two independent snapshots
    against a pre-attempt fingerprint. On a strict failure, it never retries
    the rejected proposal.
    Explicit fallback invokes ``legacy_step`` once on a fresh deep-isolated
    snapshot of that preserved prestate. It verifies the retained original,
    committed snapshot, immutable baseline, and fallback input fingerprints both
    after a legacy return and after a legacy exception, before publishing a
    deep-copied candidate. This wrapper itself neither imports nor invokes the solver.
    """
    _validate_inputs(
        requested_steps, strict_step, policy, allow_experimental_fallback, legacy_step, snapshot_state,
        states_equal, fingerprint_state,
    )
    assert snapshot_state is not None
    assert states_equal is not None
    assert fingerprint_state is not None
    state = initial_state
    attempted: list[int] = []
    committed: list[int] = []
    fallback_steps: list[int] = []
    last_failure: IToGFailureDiagnostic | None = None
    for step in range(1, requested_steps + 1):
        prestate = state
        committed_snapshot = _isolated_snapshot(prestate, snapshot_state, "committed snapshot")
        immutable_baseline = _isolated_snapshot(prestate, snapshot_state, "immutable baseline")
        strict_input = _isolated_snapshot(committed_snapshot, snapshot_state, "strict proposal state")
        try:
            expected_fingerprints = (
                fingerprint_state(prestate),
                fingerprint_state(committed_snapshot),
                fingerprint_state(immutable_baseline),
            )
        except Exception as error:
            raise RuntimeError("fingerprint_state failed while recording pre-attempt state") from error
        if expected_fingerprints[1:] != (expected_fingerprints[0], expected_fingerprints[0]):
            raise RuntimeError("snapshot isolation changed the pre-attempt state fingerprint")
        _require_unchanged_prestate(
            prestate, committed_snapshot, immutable_baseline, expected_fingerprints, states_equal, fingerprint_state,
        )
        capture: dict[str, object] = {}
        attempted.append(step)
        try:
            candidate = strict_step(strict_input, capture)
        except TopologyTransactionError as error:
            _require_unchanged_prestate(
                prestate, committed_snapshot, immutable_baseline, expected_fingerprints, states_equal, fingerprint_state,
            )
            if policy is IToGStrictFailurePolicy.RAISE:
                raise
            diagnostic = _diagnostic(step, error, capture)
            last_failure = diagnostic
            if policy is IToGStrictFailurePolicy.STOP_AND_REPORT:
                return IToGPolicyCampaignReport(
                    STOPPED_AND_REPORTED, None, False, requested_steps, len(attempted), len(committed),
                    state, IToGCampaignLedger(tuple(attempted), tuple(committed), tuple(fallback_steps)), diagnostic,
                )
            assert legacy_step is not None
            fallback_input = _isolated_snapshot(committed_snapshot, snapshot_state, "fallback prestate")
            expected_fallback_fingerprint = fingerprint_state(fallback_input)
            _require_unchanged_prestate(
                prestate, committed_snapshot, immutable_baseline, expected_fingerprints, states_equal, fingerprint_state,
                fallback_input=fallback_input, expected_fallback_fingerprint=expected_fallback_fingerprint,
                callback_name="legacy",
            )
            # This is intentionally a legacy re-run of the exact prestate, not
            # a repair, relaxation, or continuation of a partial strict result.
            try:
                fallback_candidate = legacy_step(fallback_input)
            except BaseException:
                _require_unchanged_prestate(
                    prestate, committed_snapshot, immutable_baseline, expected_fingerprints, states_equal, fingerprint_state,
                    fallback_input=fallback_input, expected_fallback_fingerprint=expected_fallback_fingerprint,
                    callback_name="legacy",
                )
                raise
            _require_unchanged_prestate(
                prestate, committed_snapshot, immutable_baseline, expected_fingerprints, states_equal, fingerprint_state,
                fallback_input=fallback_input, expected_fallback_fingerprint=expected_fallback_fingerprint,
                callback_name="legacy",
            )
            try:
                state = copy.deepcopy(fallback_candidate)
            except Exception as deepcopy_error:
                raise RuntimeError("legacy candidate must support deepcopy isolation before publication") from deepcopy_error
            if state is fallback_candidate:
                raise RuntimeError("legacy candidate deepcopy did not return an independent state")
            committed.append(step)
            fallback_steps.append(step)
            continue
        except BaseException:
            _require_unchanged_prestate(
                prestate, committed_snapshot, immutable_baseline, expected_fingerprints, states_equal, fingerprint_state,
            )
            raise
        _require_unchanged_prestate(
            prestate, committed_snapshot, immutable_baseline, expected_fingerprints, states_equal, fingerprint_state,
        )
        state = candidate
        committed.append(step)

    fallback_status = FALLBACK_NOT_PHYSICAL if fallback_steps else None
    return IToGPolicyCampaignReport(
        WITHHELD, fallback_status, False, requested_steps, len(attempted), len(committed), state,
        IToGCampaignLedger(tuple(attempted), tuple(committed), tuple(fallback_steps)),
        last_failure if fallback_steps else None,
    )
