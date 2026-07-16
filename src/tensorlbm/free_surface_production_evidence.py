"""Fail-closed R1 observer adapter for real Körner topology outputs.

This is an additive cold-path adapter.  It observes the detached evidence
already published by ``free_surface_step`` / its topology transaction; it does
not reconstruct, copy, or mutate populations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .free_surface_transaction_contract import (
    CellConversion,
    CellState,
    TransactionContractReport,
    TransactionInput,
    WITHHELD_NO_POPULATION_TRANSFER,
    diagnose_korner_i_to_g_transaction,
)


@dataclass(frozen=True)
class RuntimeKornerEvidence:
    """Observed fields from one topology result, with no inferred f transfer.

    ``available_keys`` is intentionally a report of what was present, rather
    than a schema promise.  ``transaction`` is populated only when a future
    production publisher supplies every R1 contract field verbatim.
    """

    event_id: str | None
    provenance: str
    available_keys: tuple[str, ...]
    conversion_cells: tuple[CellConversion, ...]
    transaction: TransactionInput | None
    actual_f_population_transfer: bool
    source_cells: tuple[tuple[int, int, int], ...]
    destination_cells: tuple[tuple[int, int, int], ...]
    replay_reference: str | None


@dataclass(frozen=True)
class RuntimeKornerObserverReport:
    """R1 outcome plus the observed-field audit trail."""

    status: str
    reason: str
    diagnostic_accepted: bool
    physical_accepted: bool
    event_id: str | None
    provenance: str
    available_keys: tuple[str, ...]
    contract_report: TransactionContractReport | None


def _key_paths(value: object, prefix: str = "") -> tuple[str, ...]:
    """List mapping keys, including keys published by list-valued step ledgers."""
    if isinstance(value, list):
        return tuple(path for item in value for path in _key_paths(item, prefix))
    if not isinstance(value, Mapping):
        return ()
    paths: list[str] = []
    for key, item in value.items():
        name = str(key) if not prefix else f"{prefix}.{key}"
        paths.append(name)
        paths.extend(_key_paths(item, name))
    return tuple(paths)


def _cell(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        return None
    return value


def _conversion_cells(conversion_evidence: object) -> tuple[CellConversion, ...]:
    if not isinstance(conversion_evidence, Mapping):
        return ()
    raw_cells = conversion_evidence.get("conversion_cells")
    if not isinstance(raw_cells, tuple):
        return ()
    result: list[CellConversion] = []
    for item in raw_cells:
        if not isinstance(item, Mapping):
            continue
        cell = _cell(item.get("cell"))
        # Production flags are the canonical integer constants GAS=0/I=2.
        if cell is not None and item.get("flag_before") == 2 and item.get("flag_after") == 0:
            result.append(CellConversion(cell, CellState.I, CellState.G))
    return tuple(result)


def extract_runtime_korner_evidence(
    result: Mapping[str, object], *, event_id: str | None = None,
    provenance: str = "shaped_result_mapping_not_claimed_production",
) -> RuntimeKornerEvidence:
    """Extract only published facts from a result mapping.

    A real wrapper should pass ``{"runtime_ledger": ledger,
    "replay_capture": capture}`` and set production provenance.  Arbitrary
    mappings remain explicitly non-production.  In particular, f snapshots,
    ownership mass links, and replay capture are *not* treated as an actual
    f-population transfer unless a publisher emits the four explicit fields.
    """
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping of published runtime outputs")
    runtime_ledger = result.get("runtime_ledger")
    latest: object = None
    if isinstance(runtime_ledger, Mapping):
        steps = runtime_ledger.get("steps")
        if isinstance(steps, list) and steps and isinstance(steps[-1], Mapping):
            latest = steps[-1]
    conversion_evidence = latest.get("conversion_evidence") if isinstance(latest, Mapping) else None
    conversions = _conversion_cells(conversion_evidence)
    observed_event_id = event_id
    if observed_event_id is None and isinstance(latest, Mapping):
        candidate = latest.get("event_id")
        observed_event_id = candidate if isinstance(candidate, str) and candidate.strip() else None
    if observed_event_id is None:
        observed_event_id = "runtime-step/latest" if latest is not None else None

    # Deliberately inspect only an explicit production transfer publication.
    # Existing topology evidence says it has none, so f_before/f_after must not
    # be reverse-engineered into source/destination/replay claims.
    explicit = result.get("population_transfer")
    actual = False
    sources: tuple[tuple[int, int, int], ...] = ()
    destinations: tuple[tuple[int, int, int], ...] = ()
    replay_reference: str | None = None
    if isinstance(explicit, Mapping) and explicit.get("actual_f_population_transfer") is True:
        raw_sources = explicit.get("source_cells")
        raw_destinations = explicit.get("destination_cells")
        raw_replay = explicit.get("replay_reference")
        if isinstance(raw_sources, tuple) and isinstance(raw_destinations, tuple):
            parsed_sources = [_cell(value) for value in raw_sources]
            parsed_destinations = [_cell(value) for value in raw_destinations]
            if all(cell is not None for cell in parsed_sources) and all(cell is not None for cell in parsed_destinations):
                sources = tuple(cell for cell in parsed_sources if cell is not None)
                destinations = tuple(cell for cell in parsed_destinations if cell is not None)
        replay_reference = raw_replay if isinstance(raw_replay, str) and raw_replay.strip() else None
        actual = bool(sources and destinations and replay_reference)

    return RuntimeKornerEvidence(
        event_id=observed_event_id,
        provenance=provenance,
        available_keys=_key_paths(result),
        conversion_cells=conversions,
        transaction=None,
        actual_f_population_transfer=actual,
        source_cells=sources,
        destination_cells=destinations,
        replay_reference=replay_reference,
    )


def observe_korner_runtime_evidence(evidence: RuntimeKornerEvidence) -> RuntimeKornerObserverReport:
    """Produce an R1 report, failing closed before any missing f evidence.

    This ordering is intentional: absent actual f source, destination, or
    replay must be named ``WITHHELD_NO_POPULATION_TRANSFER`` even if other R1
    fields are also absent.  No placeholder owners or transfer records are
    fabricated merely to drive the detached contract validator.
    """
    if not isinstance(evidence, RuntimeKornerEvidence):
        raise TypeError("RuntimeKornerEvidence is required")
    if not evidence.actual_f_population_transfer:
        return RuntimeKornerObserverReport(
            WITHHELD_NO_POPULATION_TRANSFER,
            "actual f population-transfer source, destination, and replay evidence were not published",
            False, False, evidence.event_id, evidence.provenance,
            evidence.available_keys, None,
        )
    if evidence.transaction is None:
        return RuntimeKornerObserverReport(
            WITHHELD_NO_POPULATION_TRANSFER,
            "actual f transfer was published without a complete RuntimeKornerEvidence R1 transaction",
            False, False, evidence.event_id, evidence.provenance,
            evidence.available_keys, None,
        )
    contract = diagnose_korner_i_to_g_transaction(evidence.transaction)
    return RuntimeKornerObserverReport(
        contract.status, contract.reason, contract.diagnostic_accepted,
        contract.physical_accepted, contract.event_id, evidence.provenance,
        evidence.available_keys, contract,
    )


def run_free_surface_step_with_observer(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], RuntimeKornerObserverReport]:
    """Run the real production step and observe its published transaction result.

    The wrapper supplies otherwise optional capture dictionaries; it never
    changes solver tensors or topology arithmetic.
    """
    from .free_surface_lbm import free_surface_step

    runtime_ledger = kwargs.setdefault("runtime_ledger", {})
    replay_capture = kwargs.setdefault("replay_capture", {})
    kwargs.setdefault("capture_replay_stages", True)
    outcome = free_surface_step(*args, **kwargs)
    evidence = extract_runtime_korner_evidence(
        {"runtime_ledger": runtime_ledger, "replay_capture": replay_capture},
        provenance="production_free_surface_step_runtime_ledger",
    )
    return outcome, observe_korner_runtime_evidence(evidence)
