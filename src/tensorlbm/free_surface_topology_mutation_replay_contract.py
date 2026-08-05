"""Cold, fail-closed audit of hash-bound D3Q19 topology replay evidence.

This module never imports or calls the runtime solver.  It only reconstructs a
captured production transaction invocation through the detached transaction
builder after verifying immutable serialized evidence bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .free_surface_topology_transaction import (
    IToGOwnershipTransaction,
    ReplayEvidence,
    StrictFailureReplayEvidence,
    TopologyTransactionError,
    build_topology_transaction,
    is_trusted_replay_evidence,
    is_trusted_strict_failure_evidence,
    restore_i_to_g_ownership,
    restore_replay_payload,
    restore_strict_failure_invocation,
    _replay_tensor_records,
)

AVAILABLE_REPLAYED_EXACT = "AVAILABLE_REPLAYED_EXACT"
MISSING_INPUT_WITHHELD = "MISSING_INPUT_WITHHELD"
ORDER_UNAVAILABLE_WITHHELD = "ORDER_UNAVAILABLE_WITHHELD"
WITHHELD = "WITHHELD"
STRICT_FAILURE_REPLAYED_EXACT = "STRICT_FAILURE_REPLAYED_EXACT"

_FIELDS = ("f", "fill", "flags", "mass")
_PHASES = (
    "to_iface_initialization",
    "legacy_redistribution_and_i_to_g_increment",
    "clamp",
    "to_liq_to_gas_conversion",
    "halo_boundary",
    "isolated_interface",
    "solid_enforcement",
)
_REQUIRED_INPUTS = (
    "f", "fill", "flags", "mass", "to_iface", "to_liq", "to_gas", "recv_new",
    "redistribution_increment", "rho_liquid", "rho_gas", "solid_mask", "gas_flag",
    "liquid_flag", "interface_flag", "solid_flag", "ux", "uy", "uz",
    "i_to_g_increment", "i_to_g_ownership",
)
_ALLOWED_INPUTS = frozenset(_REQUIRED_INPUTS)


@dataclass(frozen=True)
class ReplayPhaseReport:
    name: str
    status: str
    required_inputs: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    compared_tensors: tuple[str, ...]
    reason: str | None


@dataclass(frozen=True)
class TopologyMutationReplayReport:
    """Exact replay contract only; it is never a physical closure claim."""

    status: str
    mutates_solver_state: bool
    physical_claim: bool
    final_candidate_exact: bool
    final_compared_tensors: tuple[str, ...]
    phases: tuple[ReplayPhaseReport, ...]


@dataclass(frozen=True)
class StrictFailureReplayReport:
    """Exact reproduction of a rejection, never a candidate or closure claim."""

    status: str
    mutates_solver_state: bool
    physical_claim: bool
    error_type: str | None
    error_message: str | None
    reason: str | None


def _tensor_state(value: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if isinstance(value, Mapping) and all(isinstance(value.get(name), torch.Tensor) for name in _FIELDS):
        return tuple(value[name] for name in _FIELDS)  # type: ignore[return-value]
    if isinstance(value, tuple) and len(value) == len(_FIELDS) and all(isinstance(item, torch.Tensor) for item in value):
        return value  # type: ignore[return-value]
    return None


def _exact_fields(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[str, ...]:
    return tuple(name for name, left, right in zip(_FIELDS, actual, expected) if torch.equal(left, right))


def _withheld(reason: str, missing: tuple[str, ...] = ()) -> TopologyMutationReplayReport:
    return TopologyMutationReplayReport(
        status=WITHHELD,
        mutates_solver_state=False,
        physical_claim=False,
        final_candidate_exact=False,
        final_compared_tensors=(),
        phases=tuple(
            ReplayPhaseReport(name, MISSING_INPUT_WITHHELD, _REQUIRED_INPUTS, missing, (), reason)
            for name in _PHASES
        ),
    )


def _load_evidence(evidence: ReplayEvidence) -> tuple[dict[str, object], dict[str, object], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    invocation = restore_replay_payload(evidence.invocation_payload, evidence.invocation_sha256)
    phases = restore_replay_payload(evidence.phase_payload, evidence.phase_sha256)
    candidate = restore_replay_payload(evidence.candidate_payload, evidence.candidate_sha256)
    if not isinstance(invocation, dict) or not isinstance(phases, dict) or _tensor_state(candidate) is None:
        raise TopologyTransactionError("WITHHELD: replay evidence schema is invalid")
    records = (
        _replay_tensor_records(invocation, "invocation")
        + _replay_tensor_records(phases, "phases")
        + _replay_tensor_records(candidate, "candidate")
    )
    if records != evidence.tensor_records:
        raise TopologyTransactionError("WITHHELD: replay tensor records do not match payload tensors")
    return invocation, phases, _tensor_state(candidate)  # type: ignore[return-value]


def audit_topology_mutation_replay(evidence: object) -> TopologyMutationReplayReport:
    """Verify and replay a trusted in-process capture, failing closed on gaps.

    This is capture integrity, not provenance or attestation: publicly
    constructed records (even with valid recomputable hashes) are withheld.
    """
    if not isinstance(evidence, ReplayEvidence):
        return _withheld("immutable ReplayEvidence is required")
    if not is_trusted_replay_evidence(evidence):
        return _withheld("ReplayEvidence is not a trusted in-process capture")
    try:
        captured_inputs, expected_phases, expected_candidate = _load_evidence(evidence)
    except (TopologyTransactionError, RuntimeError, ValueError) as error:
        return _withheld(str(error))

    unknown = tuple(sorted(set(captured_inputs) - _ALLOWED_INPUTS))
    missing = tuple(name for name in _REQUIRED_INPUTS if name not in captured_inputs)
    pair_present = (captured_inputs.get("i_to_g_increment") is not None, captured_inputs.get("i_to_g_ownership") is not None)
    if unknown or missing or pair_present[0] != pair_present[1]:
        reason = "captured invocation has unknown inputs" if unknown else "captured pre-state transaction inputs are incomplete or inconsistent"
        return _withheld(reason, missing)
    if pair_present[1] and not isinstance(captured_inputs["i_to_g_ownership"], IToGOwnershipTransaction):
        return _withheld("captured I→G ownership has invalid schema")

    try:
        captured_inputs["i_to_g_ownership"] = restore_i_to_g_ownership(captured_inputs["i_to_g_ownership"])
        plan = build_topology_transaction(**captured_inputs, capture_replay_stages=True)  # type: ignore[arg-type]
        replayed_final = (plan.f, plan.fill, plan.flags, plan.mass)
        replayed_phases = plan.replay_stages
    except (TopologyTransactionError, RuntimeError, ValueError) as error:
        return _withheld(str(error))
    if replayed_phases is None:
        return _withheld("production builder did not emit replay phases")

    final_compared = _exact_fields(replayed_final, expected_candidate)
    reports: list[ReplayPhaseReport] = []
    for name in _PHASES:
        expected = _tensor_state(expected_phases.get(name))
        actual = replayed_phases.get(name)
        if expected is None or actual is None:
            reports.append(ReplayPhaseReport(name, ORDER_UNAVAILABLE_WITHHELD, _REQUIRED_INPUTS, (), (), "phase evidence schema is incomplete"))
            continue
        compared = _exact_fields(actual, expected)
        reports.append(ReplayPhaseReport(
            name,
            AVAILABLE_REPLAYED_EXACT if compared == _FIELDS else ORDER_UNAVAILABLE_WITHHELD,
            _REQUIRED_INPUTS, (), compared,
            None if compared == _FIELDS else "captured phase-boundary tensors differ from production-builder replay",
        ))
    complete = final_compared == _FIELDS and all(item.status == AVAILABLE_REPLAYED_EXACT for item in reports)
    return TopologyMutationReplayReport(
        status=AVAILABLE_REPLAYED_EXACT if complete else WITHHELD,
        mutates_solver_state=False,
        physical_claim=False,
        final_candidate_exact=final_compared == _FIELDS,
        final_compared_tensors=final_compared,
        phases=tuple(reports),
    )


def audit_strict_failure_replay(evidence: object) -> StrictFailureReplayReport:
    """Replay a trusted pre-invocation capture and require its rejection exactly.

    An unexpected candidate, a different exception, or malformed evidence is
    withheld. This does not produce phase evidence or a final candidate.
    """
    if not isinstance(evidence, StrictFailureReplayEvidence):
        return StrictFailureReplayReport(WITHHELD, False, False, None, None, "immutable strict failure evidence is required")
    if not is_trusted_strict_failure_evidence(evidence):
        return StrictFailureReplayReport(WITHHELD, False, False, None, None, "strict failure evidence is not a trusted in-process capture")
    try:
        invocation = restore_strict_failure_invocation(evidence)
        from .free_surface_topology_transaction import build_i_to_g_ownership_transaction
        build_i_to_g_ownership_transaction(**invocation)  # type: ignore[arg-type]
    except TopologyTransactionError as error:
        if type(error).__name__ == evidence.error_type and str(error) == evidence.error_message:
            return StrictFailureReplayReport(
                STRICT_FAILURE_REPLAYED_EXACT, False, False,
                evidence.error_type, evidence.error_message, None,
            )
        return StrictFailureReplayReport(
            WITHHELD, False, False, type(error).__name__, str(error),
            "detached builder rejection differs from captured strict failure",
        )
    except (RuntimeError, ValueError) as error:
        return StrictFailureReplayReport(
            WITHHELD, False, False, type(error).__name__, str(error),
            "detached builder raised an unexpected error type",
        )
    return StrictFailureReplayReport(
        WITHHELD, False, False, None, None,
        "detached builder unexpectedly produced a candidate",
    )
