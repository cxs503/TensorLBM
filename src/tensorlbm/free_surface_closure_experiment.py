"""Repeatable, observer-only R1 diagnosis of dynamic free-surface topology.

The experiment invokes the production :func:`free_surface_step` unchanged.  It
never corrects mass, fill, or populations, and its immutable report is always a
diagnostic observation rather than a physical/PV closure claim.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .core.d3q19_stencil import D3Q19_MOVING_Q, roll_from_pull_source
from .d3q19 import C, equilibrium3d
from .free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    free_surface_step,
    total_liquid_inventory,
)
from .free_surface_topology_transaction import ReplayEvidence

DIAGNOSTIC_NOT_PHYSICAL_CLOSURE = "DIAGNOSTIC_NOT_PHYSICAL_CLOSURE"
WITHHELD = "WITHHELD"
FAILED_DIAGNOSTIC = "FAILED_DIAGNOSTIC"


class ClosureExperimentError(ValueError):
    """An experiment input cannot support a fail-closed diagnostic observation."""


@dataclass(frozen=True)
class ClosureSnapshot:
    independent_mass: float
    total_liquid_inventory: float
    population_mass_sum: float
    liquid_cells: int
    interface_cells: int
    gas_cells: int
    direct_liquid_gas_links: int
    finite: bool


@dataclass(frozen=True)
class TopologyEvent:
    operator: str
    net_delta: float
    gross_magnitude: float
    evidence_available: bool
    population_only: bool
    event_count: int


@dataclass(frozen=True)
class ClosureStepEvidence:
    step: int
    independent_mass: float
    total_liquid_inventory: float
    population_mass_sum: float
    tracked_independent_mass_drift: float
    inventory_drift: float
    runtime_ledger: tuple[tuple[str, object], ...] | None
    ownership_ledger: object | None
    ledger_reconciliation_residual: float | None
    ownership_unresolved_categories: tuple[str, ...]
    topology_events: tuple[TopologyEvent, ...]
    topology_event_evidence_available: bool
    abb_population_only: bool
    direct_liquid_gas_links: int
    finite: bool
    failure_reason: str | None
    # Appended with defaults so existing positional construction remains ABI-compatible.
    inventory_reconciliation: tuple[tuple[str, object], ...] | None = None
    replay_evidence: ReplayEvidence | None = None


@dataclass(frozen=True)
class ClosureCaseReport:
    case_id: str
    requested_steps: int
    freeze_topology: bool
    paired_liquid_interface_debit: bool
    status: str
    physical_closure_claim: bool
    initial: ClosureSnapshot
    final: ClosureSnapshot
    mass_drift_curve: tuple[float, ...]
    inventory_drift_curve: tuple[float, ...]
    steps: tuple[ClosureStepEvidence, ...]
    failure_reason: str | None


@dataclass(frozen=True)
class ClosureExperimentReport:
    status: str
    physical_closure_claim: bool
    global_mass_correction_applied: bool
    cases: tuple[ClosureCaseReport, ...]


def _direct_liquid_gas_links(flags: torch.Tensor) -> int:
    liquid = flags == LIQUID
    gas_sources = torch.stack([
        roll_from_pull_source(flags, q) == GAS for q in D3Q19_MOVING_Q
    ])
    return int((liquid.unsqueeze(0) & gas_sources).sum())


def _finite(f: torch.Tensor, fill: torch.Tensor, mass: torch.Tensor) -> bool:
    return bool(torch.isfinite(f).all() and torch.isfinite(fill).all() and torch.isfinite(mass).all())


def _snapshot(f: torch.Tensor, fill: torch.Tensor, flags: torch.Tensor, mass: torch.Tensor) -> ClosureSnapshot:
    return ClosureSnapshot(
        independent_mass=float(mass.sum()),
        total_liquid_inventory=float(total_liquid_inventory(f, fill, flags)),
        population_mass_sum=float(f.sum()),
        liquid_cells=int((flags == LIQUID).sum()),
        interface_cells=int((flags == INTERFACE).sum()),
        gas_cells=int((flags == GAS).sum()),
        direct_liquid_gas_links=_direct_liquid_gas_links(flags),
        finite=_finite(f, fill, mass),
    )


def _frozen_runtime_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (3, 3, 5)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    flags[:, :, 0] = INTERFACE
    flags[:, :, 2] = INTERFACE
    flags[:, :, 3:] = LIQUID
    fill = torch.zeros(shape)
    fill[flags == INTERFACE] = 0.5
    fill[flags == LIQUID] = 1.0
    rho = torch.where(flags == GAS, torch.full_like(fill, 0.001), torch.ones_like(fill))
    x = torch.arange(shape[2], dtype=fill.dtype).view(1, 1, shape[2])
    ux = 0.025 * torch.sin(2.0 * torch.pi * x / shape[2]).expand_as(fill)
    zero = torch.zeros_like(fill)
    return equilibrium3d(rho, ux, zero, zero), fill, flags, torch.zeros(shape, dtype=torch.bool)


def _conversion_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (5, 6, 7)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    fill = torch.zeros(shape)
    centre = (2, 3, 3)
    flags[centre] = INTERFACE
    fill[centre] = 1.0
    for q in D3Q19_MOVING_Q:
        dz, dy, dx = int(C[q, 2]), int(C[q, 1]), int(C[q, 0])
        cell = tuple((index - delta) % extent for index, delta, extent in zip(centre, (dz, dy, dx), shape))
        flags[cell] = INTERFACE
        fill[cell] = 0.5
    zero = torch.zeros(shape)
    return equilibrium3d(torch.ones(shape), zero, zero, zero), fill, flags, torch.zeros(shape, dtype=torch.bool)


def _event_summary(runtime_record: dict[str, object] | None) -> tuple[tuple[TopologyEvent, ...], bool, bool]:
    if runtime_record is None:
        return (), False, False
    attribution = runtime_record.get("operator_attribution")
    raw_events = attribution.get("events", ()) if isinstance(attribution, dict) else ()
    evidence = runtime_record.get("conversion_evidence")
    evidence_available = isinstance(evidence, dict)
    conversion_cells = evidence.get("conversion_cells", ()) if evidence_available else ()
    redistribution_links = evidence.get("redistribution_links", ()) if evidence_available else ()
    events: list[TopologyEvent] = []
    categories = {
        "conversion": ("conversion", len(conversion_cells)),
        "redistribution": ("redistribution", len(redistribution_links)),
        "abb": ("abb", 0),
        "liquid_interface": ("interface_paired_debit", 0),
        # These are tracked-state operators, not anonymous "other" activity.
        "clamp": ("clamp", 0),
        "isolation": ("isolation", 0),
        "boundary": ("boundary", 0),
    }
    for operator, (source_operator, count) in categories.items():
        raw = next((item for item in raw_events if item.get("operator") == source_operator), None)
        if raw is None:
            continue
        events.append(TopologyEvent(
            operator=operator,
            net_delta=float(raw.get("net_delta", 0.0)),
            gross_magnitude=float(raw.get("gross_magnitude", 0.0)),
            evidence_available=evidence_available and operator in {"conversion", "redistribution"},
            population_only=operator == "abb",
            event_count=count,
        ))
    known = {
        "conversion", "redistribution", "abb", "interface_paired_debit",
        "clamp", "isolation", "boundary",
    }
    for raw in raw_events:
        if raw.get("operator") not in known:
            events.append(TopologyEvent(
                operator="other", net_delta=float(raw.get("net_delta", 0.0)),
                gross_magnitude=float(raw.get("gross_magnitude", 0.0)), evidence_available=False,
                population_only=not bool(raw.get("tracked_mass", False)), event_count=0,
            ))
    return tuple(events), evidence_available, any(event.population_only and event.operator == "abb" for event in events)


def _freeze_value(value: object) -> object:
    """Detach nested solver evidence so the returned report is immutable."""
    if isinstance(value, Mapping):
        # Mapping order is evidence here: inventory stages are a canonical
        # chronological ledger, not an unordered set of diagnostics.
        return tuple((str(key), _freeze_value(item)) for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_value(item) for item in value)
    return deepcopy(value)


def _freeze_mapping(value: dict[str, object] | None) -> tuple[tuple[str, object], ...] | None:
    if value is None:
        return None
    frozen = _freeze_value(value)
    assert isinstance(frozen, tuple)
    return frozen


def _validate_case_definition(definition: object) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, bool, bool]:
    if not isinstance(definition, tuple) or len(definition) != 8:
        raise ClosureExperimentError("each extra case must be an 8-item tuple")
    case_id, f, fill, flags, solid, requested_steps, freeze_topology, paired = definition
    if not isinstance(case_id, str) or not case_id:
        raise ClosureExperimentError("case_id must be a non-empty string")
    for name, field in (("f", f), ("fill", fill), ("flags", flags), ("solid", solid)):
        if not isinstance(field, torch.Tensor):
            raise ClosureExperimentError(f"{name} must be a torch.Tensor")
    assert isinstance(f, torch.Tensor) and isinstance(fill, torch.Tensor)
    assert isinstance(flags, torch.Tensor) and isinstance(solid, torch.Tensor)
    if any(field.layout != torch.strided for field in (f, fill, flags, solid)):
        raise ClosureExperimentError("all case tensors must be dense strided tensors")
    if any(dimension <= 0 for dimension in f.shape[1:]):
        raise ClosureExperimentError("spatial dimensions must be positive")
    if f.dtype != torch.float32 or fill.dtype != torch.float32:
        raise ClosureExperimentError("f and fill must be float32")
    if f.device != fill.device or flags.device != fill.device or solid.device != fill.device:
        raise ClosureExperimentError("all case tensors must share one device")
    if f.ndim != 4 or f.shape[0] != 19:
        raise ClosureExperimentError("f must have shape (19, nz, ny, nx)")
    if fill.ndim != 3 or fill.shape != f.shape[1:]:
        raise ClosureExperimentError("fill must have shape (nz, ny, nx) matching f")
    if flags.shape != fill.shape or solid.shape != fill.shape:
        raise ClosureExperimentError("flags and solid must match fill shape")
    if flags.dtype != torch.int8 or solid.dtype != torch.bool:
        raise ClosureExperimentError("flags must be int8 and solid must be bool")
    if isinstance(requested_steps, bool) or not isinstance(requested_steps, int) or requested_steps <= 0:
        raise ClosureExperimentError("requested steps must be a positive integer")
    if not isinstance(freeze_topology, bool) or not isinstance(paired, bool):
        raise ClosureExperimentError("freeze_topology and paired must be bool")
    return case_id, f, fill, flags, solid, requested_steps, freeze_topology, paired


def _run_case(
    case_id: str, f: torch.Tensor, fill: torch.Tensor, flags: torch.Tensor, solid: torch.Tensor,
    requested_steps: int, freeze_topology: bool, paired: bool, enable_i_to_g_ownership_closure: bool,
    capture_replay_stages: bool = False,
) -> ClosureCaseReport:
    f, fill, flags, solid = (field.clone() for field in (f, fill, flags, solid))
    mass = fill.clone()
    initial = _snapshot(f, fill, flags, mass)
    mass_curve = [initial.independent_mass]
    inventory_curve = [initial.total_liquid_inventory]
    observations: list[ClosureStepEvidence] = []
    failure_reason: str | None = None

    for number in range(1, requested_steps + 1):
        pre = _snapshot(f, fill, flags, mass)
        if not pre.finite:
            failure_reason = "non-finite input state"
        elif pre.direct_liquid_gas_links:
            failure_reason = f"direct LIQUID-GAS links before step: {pre.direct_liquid_gas_links}"
        runtime: dict[str, object] = {}
        ownership: dict[str, object] = {}
        inventory: dict[str, object] = {}
        replay_capture: dict[str, object] = {}
        if failure_reason is None:
            try:
                f, fill, flags, mass, _ = free_surface_step(
                    f, fill, flags, solid, mass=mass, tau=1.0, rho_gas=1.0e-3,
                    freeze_topology=freeze_topology, runtime_ledger=runtime,
                    ownership_ledger=ownership, inventory_reconciliation_ledger=inventory,
                    paired_liquid_interface_debit=paired,
                    enable_i_to_g_ownership_closure=enable_i_to_g_ownership_closure,
                    capture_replay_stages=capture_replay_stages,
                    replay_capture=replay_capture if capture_replay_stages else None,
                )
            except (RuntimeError, ValueError) as error:
                failure_reason = str(error)
        current = _snapshot(f, fill, flags, mass)
        raw_steps = runtime.get("steps")
        record = raw_steps[-1] if isinstance(raw_steps, list) and raw_steps else None
        ownership_state = ownership.get("latest")
        events, evidence_available, abb_population_only = _event_summary(record if isinstance(record, dict) else None)
        unresolved = tuple(getattr(ownership_state, "unresolved_categories", ()))
        reconciliation = None if not isinstance(record, dict) else float(record["residual_reconciliation"]["residual"])
        if failure_reason is None and not current.finite:
            failure_reason = "non-finite state after free_surface_step"
        if failure_reason is None and current.direct_liquid_gas_links:
            failure_reason = f"direct LIQUID-GAS links after step: {current.direct_liquid_gas_links}"
        captured_replay_evidence = replay_capture.get("evidence")
        replay_evidence = (
            captured_replay_evidence if failure_reason is None
            and isinstance(captured_replay_evidence, ReplayEvidence) else None
        )
        observations.append(ClosureStepEvidence(
            step=number, independent_mass=current.independent_mass,
            total_liquid_inventory=current.total_liquid_inventory, population_mass_sum=current.population_mass_sum,
            tracked_independent_mass_drift=current.independent_mass - initial.independent_mass,
            inventory_drift=current.total_liquid_inventory - initial.total_liquid_inventory,
            inventory_reconciliation=_freeze_mapping(inventory if inventory else None),
            runtime_ledger=_freeze_mapping(record if isinstance(record, dict) else None), ownership_ledger=_freeze_value(ownership_state),
            ledger_reconciliation_residual=reconciliation, ownership_unresolved_categories=unresolved,
            topology_events=events, topology_event_evidence_available=evidence_available,
            abb_population_only=abb_population_only, direct_liquid_gas_links=current.direct_liquid_gas_links,
            finite=current.finite, failure_reason=failure_reason,
            replay_evidence=replay_evidence,
        ))
        mass_curve.append(current.independent_mass)
        inventory_curve.append(current.total_liquid_inventory)
        if failure_reason is not None:
            break
    final = _snapshot(f, fill, flags, mass)
    return ClosureCaseReport(
        case_id=case_id, requested_steps=requested_steps, freeze_topology=freeze_topology,
        paired_liquid_interface_debit=paired,
        status=FAILED_DIAGNOSTIC if failure_reason is not None else WITHHELD,
        physical_closure_claim=False, initial=initial, final=final,
        mass_drift_curve=tuple(mass_curve), inventory_drift_curve=tuple(inventory_curve),
        steps=tuple(observations), failure_reason=failure_reason,
    )


def run_free_surface_closure_experiment(
    *, extra_cases: tuple[tuple[str, Any, Any, Any, Any, int, bool, bool], ...] = (),
    enable_i_to_g_ownership_closure: bool = False,
    capture_replay_stages: bool = False,
) -> ClosureExperimentReport:
    """Run the fixed R1 diagnostic matrix without emitting files or corrections.

    Case A compares frozen topology with paired L/I accounting off/on. Case B
    uses a legal compact state whose actual solver update deterministically
    converts its centre. Case C repeats that real dynamic topology path across
    several steps as a tiny dam-break-style topology stressor.
    """
    if not isinstance(extra_cases, tuple):
        raise ClosureExperimentError("extra_cases must be a tuple of case definitions")
    if not isinstance(enable_i_to_g_ownership_closure, bool):
        raise ClosureExperimentError("enable_i_to_g_ownership_closure must be bool")
    if not isinstance(capture_replay_stages, bool):
        raise ClosureExperimentError("capture_replay_stages must be bool")
    frozen = _frozen_runtime_state()
    conversion = _conversion_state()
    definitions = (
        ("A_frozen_topology_paired_off", *frozen, 3, True, False),
        ("A_frozen_topology_paired_on", *frozen, 3, True, True),
        ("B_forced_conversion_deterministic", *conversion, 3, False, True),
        ("C_dam_break_style_tiny_dynamic_topology", *conversion, 10, False, True),
        *extra_cases,
    )
    cases = tuple(
        _run_case(
            *_validate_case_definition(definition),
            enable_i_to_g_ownership_closure=enable_i_to_g_ownership_closure,
            capture_replay_stages=capture_replay_stages,
        )
        for definition in definitions
    )
    return ClosureExperimentReport(
        status=DIAGNOSTIC_NOT_PHYSICAL_CLOSURE,
        physical_closure_claim=False,
        global_mass_correction_applied=False,
        cases=cases,
    )
