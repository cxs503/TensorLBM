"""Pure R1 evidence decision gate for Körner I→G ``f`` ownership policies.

This cold-path module selects no policy and performs no population mutation.
It only names the minimum production evidence a later writer must publish for
one of the three mutually exclusive ownership models.  Every R1 outcome is
withheld, including a complete future-shaped evidence record.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

WITHHELD_MISSING_POLICY_EVIDENCE = "WITHHELD_MISSING_POLICY_EVIDENCE"
PRODUCTION_PROVENANCE = "production_free_surface_step_runtime_ledger"


class IToGPopulationPolicy(str, Enum):
    EXPLICIT_BOUNDARY_RECONSTRUCTION = "explicit_boundary_reconstruction"
    CONSERVATIVE_PARTITION_TRANSFER = "conservative_partition_transfer"
    GAS_BOUNDARY_RESERVOIR = "gas_boundary_reservoir"


@dataclass(frozen=True)
class IToGPolicyEvidenceReport:
    """Fail-closed feasibility result; ``feasible`` is always false in R1."""

    policy: IToGPopulationPolicy
    status: str
    feasible: bool
    missing_evidence: tuple[str, ...]
    reason: str
    production_provenance: str | None


_POLICY_REQUIREMENTS: dict[IToGPopulationPolicy, tuple[str, ...]] = {
    IToGPopulationPolicy.EXPLICIT_BOUNDARY_RECONSTRUCTION: (
        "operator_id", "source_cells", "reconstructed_cells",
        "qwise_reconstruction", "boundary_state", "replay_reference",
    ),
    IToGPopulationPolicy.CONSERVATIVE_PARTITION_TRANSFER: (
        "operator_id", "source_cells", "destination_cells",
        "qwise_transfer_map", "partition_weights", "momentum_treatment",
        "replay_reference",
    ),
    IToGPopulationPolicy.GAS_BOUNDARY_RESERVOIR: (
        "operator_id", "source_cells", "reservoir_id",
        "qwise_reservoir_debit", "reservoir_accounting", "boundary_state",
        "replay_reference",
    ),
}


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _cells(value: object) -> bool:
    if not isinstance(value, tuple) or not value:
        return False
    return all(
        isinstance(cell, tuple) and len(cell) == 3
        and all(isinstance(part, int) and not isinstance(part, bool) for part in cell)
        for cell in value
    )


def _published(value: object, name: str) -> bool:
    if name in {"source_cells", "destination_cells", "reconstructed_cells"}:
        return _cells(value)
    return isinstance(value, str) and bool(value.strip())


def _fields(evidence: object) -> tuple[str | None, bool, Mapping[str, object] | None]:
    """Read published facts only; no snapshots, ledger, or topology inference."""
    if isinstance(evidence, Mapping):
        source = evidence
        provenance = source.get("provenance")
        actual = source.get("actual_f_population_transfer")
        policies = _mapping(source.get("policy_evidence"))
    else:
        provenance = getattr(evidence, "provenance", None)
        actual = getattr(evidence, "actual_f_population_transfer", None)
        policies = _mapping(getattr(evidence, "policy_evidence", None))
    return (provenance if isinstance(provenance, str) else None, actual is True, policies)


def evaluate_i_to_g_policy_evidence(
    policy: IToGPopulationPolicy | object, evidence: object,
) -> IToGPolicyEvidenceReport:
    """Withhold one policy unless its own minimum *production* evidence exists.

    Mass debit/credit, ``f_before``/``f_after``, and arithmetic residuals are
    intentionally absent from the requirements: none establishes ownership or
    validates an ownership operator.
    """
    if not isinstance(policy, IToGPopulationPolicy):
        raise TypeError("policy must be an IToGPopulationPolicy")
    provenance, actual_transfer, policies = _fields(evidence)
    missing: list[str] = []
    if provenance != PRODUCTION_PROVENANCE:
        missing.append("production_provenance")
    if not actual_transfer:
        missing.append("actual_f_population_transfer")
    payload = _mapping(policies.get(policy.value)) if policies is not None else None
    for requirement in _POLICY_REQUIREMENTS[policy]:
        if payload is None or not _published(payload.get(requirement), requirement):
            missing.append(requirement)
    missing_tuple = tuple(missing)
    if missing_tuple:
        reason = f"{policy.value} lacks minimum production policy evidence"
    else:
        reason = "R1 never authorizes or implements an f ownership policy"
    return IToGPolicyEvidenceReport(
        policy, WITHHELD_MISSING_POLICY_EVIDENCE, False, missing_tuple, reason, provenance,
    )


def evaluate_production_i_to_g_policy_evidence(
    production_result: object,
) -> dict[IToGPopulationPolicy, IToGPolicyEvidenceReport]:
    """Report the three R1 options independently against one production result."""
    return {
        policy: evaluate_i_to_g_policy_evidence(policy, production_result)
        for policy in IToGPopulationPolicy
    }
