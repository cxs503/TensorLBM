"""Cold audit of a real strict I→G failure invocation.

This observer restores only trusted, pre-invocation diagnostic evidence.  It
never imports the runtime solver, changes the strict gate, or publishes a
partial solver ledger.  Candidate results are feasibility screens, not an
integration path or physical-closure claim.
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import cast

import torch

from .free_surface_i_to_g_exact_ledger import (
    WITHHELD_NOT_REPRESENTABLE,
    diagnose_i_to_g_exact_ledger,
)
from .free_surface_topology_transaction import (
    StrictFailureReplayEvidence,
    TopologyTransactionError,
    is_trusted_strict_failure_evidence,
    restore_strict_failure_invocation,
)

PROPOSAL_FEASIBLE_NOT_INTEGRATED = "PROPOSAL_FEASIBLE_NOT_INTEGRATED"


@dataclass(frozen=True)
class FailedResidualCandidateReport:
    name: str
    status: str
    same_float32_state_mutation_exact: bool
    donor_state_delta_matches: bool
    combined_capacity_ok: bool
    no_clamp_loss: bool
    full_phase_replay_compatible: bool
    reason: str


@dataclass(frozen=True)
class FailedIToGResidualAuditReport:
    status: str
    mutates_solver_state: bool
    physical_claim: bool
    builder: str | None
    strict_error: str | None
    phase_replay_context: str
    donor_count: int
    receiver_count: int
    candidates: tuple[FailedResidualCandidateReport, ...]


def _withheld(reason: str) -> FailedIToGResidualAuditReport:
    return FailedIToGResidualAuditReport(
        WITHHELD_NOT_REPRESENTABLE, False, False, None, None, "UNAVAILABLE", 0, 0,
        tuple(
            FailedResidualCandidateReport(name, WITHHELD_NOT_REPRESENTABLE, False, False, False, False, False, reason)
            for name in ("record_only", "local_receiver_residual", "alternative_exact_split")
        ),
    )


def audit_failed_i_to_g_residual(evidence: object) -> FailedIToGResidualAuditReport:
    """Screen all three non-mutating candidates from one real failed invocation.

    Current B/C failures occur in the I→G transaction builder before topology
    phases exist. Thus no candidate may claim full phase compatibility; this is
    explicit rather than silently treating an absent candidate as a replay.
    """
    if not isinstance(evidence, StrictFailureReplayEvidence):
        return _withheld("immutable strict failure evidence is required")
    if not is_trusted_strict_failure_evidence(evidence):
        return _withheld("strict failure evidence is not a trusted in-process capture")
    if evidence.builder != "i_to_g_ownership":
        return _withheld("failure was not captured at the I→G ownership builder")
    try:
        invocation = restore_strict_failure_invocation(evidence)
        fields = {name: invocation[name] for name in ("flags", "mass", "to_gas", "to_liq", "solid_mask")}
        if not all(isinstance(value, torch.Tensor) for value in fields.values()):
            return _withheld("failed I→G invocation has invalid tensor fields")
        flags = cast(torch.Tensor, fields["flags"])
        mass = cast(torch.Tensor, fields["mass"])
        to_gas = cast(torch.Tensor, fields["to_gas"])
        to_liq = cast(torch.Tensor, fields["to_liq"])
        solid_mask = cast(torch.Tensor, fields["solid_mask"])
        gas_flag = cast(int, invocation["gas_flag"])
        liquid_flag = cast(int, invocation["liquid_flag"])
        interface_flag = cast(int, invocation["interface_flag"])
        rho_liquid = cast(float, invocation["rho_liquid"])
        ledger = diagnose_i_to_g_exact_ledger(
            flags, mass, to_gas=to_gas, to_liq=to_liq, solid_mask=solid_mask,
            gas_flag=gas_flag, liquid_flag=liquid_flag, interface_flag=interface_flag,
            rho_liquid=rho_liquid,
        )
    except (TopologyTransactionError, RuntimeError, ValueError, KeyError, TypeError) as error:
        return _withheld(str(error))

    phase_context = "BUILDER_REJECTED_BEFORE_TOPOLOGY_PHASE_EVIDENCE"
    common = dict(combined_capacity_ok=ledger.capacity_ok, no_clamp_loss=ledger.capacity_ok, full_phase_replay_compatible=False)
    record_only = FailedResidualCandidateReport(
        "record_only", WITHHELD_NOT_REPRESENTABLE, False, False, **common,
        reason="record-only changes no float32 state, so it cannot match the converting donor delta",
    )
    local = FailedResidualCandidateReport(
        "local_receiver_residual", WITHHELD_NOT_REPRESENTABLE, False, False, **common,
        reason=ledger.method_c.reason or "local residual assignment is not representable",
    )
    alternative = FailedResidualCandidateReport(
        "alternative_exact_split", WITHHELD_NOT_REPRESENTABLE, False, False, **common,
        reason="any alternative split changes the actual legacy float32 link increments; no complete phase candidate exists to prove same-order compatibility",
    )
    return FailedIToGResidualAuditReport(
        WITHHELD_NOT_REPRESENTABLE, False, False, evidence.builder, evidence.error_message,
        phase_context, ledger.donor_count, ledger.receiver_count, (record_only, local, alternative),
    )
