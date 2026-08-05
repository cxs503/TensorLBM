"""Application boundary for the existing D3Q19 Körner free-surface dam-break runner.

This module deliberately delegates the numerical run unchanged to
:func:`run_dam_break_3d`; it neither implements nor compiles a timestep path.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from .dam_break_3d import DamBreak3DConfig, run_dam_break_3d

RunState = Literal["COMPLETED", "FAILED"]
GateStatus = Literal["PASS", "FAIL", "WITHHELD"]
ValidationStatus = Literal["PASS", "FAIL", "WITHHELD", "NOT_APPLICABLE"]

_FORMULATION = "D3Q19 Körner"


def _freeze(value: object) -> object:
    """Return a recursively immutable snapshot suitable for evidence records."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _mapping_snapshot(value: Mapping[str, object]) -> Mapping[str, object]:
    snapshot = _freeze(value)
    assert isinstance(snapshot, Mapping)
    return snapshot


@dataclass(frozen=True, slots=True)
class FreeSurfaceScenario:
    """A named application scenario around the existing free-surface config only."""

    scenario_id: str
    config: DamBreak3DConfig

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")
        if self.config.model != "fs":
            raise ValueError("FreeSurfaceScenario requires DamBreak3DConfig.model == 'fs'")

    @property
    def metadata(self) -> Mapping[str, object]:
        """Declared formulation scope, not a physical-accuracy validation result."""
        return _mapping_snapshot({
            "lattice": "D3Q19",
            "physics": "single-phase free-surface fill-level tracking",
            "formulation": "Körner",
            "runner": "run_dam_break_3d",
            "validation_scope": "application integration and existing runner accounting only",
        })


@dataclass(frozen=True, slots=True)
class FreeSurfaceApplicationResult:
    """Immutable application-level report of a single delegated cold-path run."""

    state: RunState
    mass_gate_status: GateStatus
    validation_status: ValidationStatus
    run_metadata: Mapping[str, object]
    evidence: Mapping[str, object]


def _base_evidence() -> dict[str, object]:
    return {
        "formulation": _FORMULATION,
        "dynamic_topology_physical_accuracy": "WITHHELD",
        "physical_reference": "WITHHELD",
        "D3Q27_phase_field_equivalence": False,
        "color_gradient_equivalence": False,
        "physical_accuracy_claim": False,
        "unsupported_claims": (
            "D3Q27 phase-field equivalence",
            "color-gradient equivalence",
            "physical accuracy",
            "GPU performance",
            "SDAA support",
            "long-duration dynamic-topology closure",
        ),
    }


def _read_run_metadata(run_dir: Path) -> Mapping[str, object]:
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"runner metadata is not an object: {metadata_path}")
    return _mapping_snapshot(metadata)


def _mass_gate_status(metadata: Mapping[str, object]) -> GateStatus:
    gate = metadata.get("free_surface_quality_gate")
    if not isinstance(gate, Mapping) or not isinstance(gate.get("passed"), bool):
        return "WITHHELD"
    return "PASS" if gate["passed"] else "FAIL"


def run_free_surface_scenario(scenario: FreeSurfaceScenario) -> FreeSurfaceApplicationResult:
    """Run the existing dam-break cold path exactly once and wrap its evidence."""
    try:
        run_dir = run_dam_break_3d(scenario.config)
        metadata = _read_run_metadata(Path(run_dir))
    except RuntimeError as error:
        evidence = _base_evidence()
        evidence["failure_reason"] = str(error)
        evidence["validation_scope"] = "existing runner failure mapping; not physical accuracy"
        return FreeSurfaceApplicationResult(
            state="FAILED",
            mass_gate_status="FAIL",
            validation_status="FAIL",
            run_metadata=_mapping_snapshot({}),
            evidence=_mapping_snapshot(evidence),
        )

    evidence = _base_evidence()
    evidence["mass_gate_source"] = "free_surface_quality_gate.passed"
    evidence["validation_scope"] = "existing runner accounting evidence; physical accuracy withheld"
    return FreeSurfaceApplicationResult(
        state="COMPLETED",
        mass_gate_status=_mass_gate_status(metadata),
        validation_status="WITHHELD",
        run_metadata=metadata,
        evidence=_mapping_snapshot(evidence),
    )


def write_metadata_wrapping_observation(path: Path = Path("/tmp/tensorlbm-free-surface-app-overhead-r1.json")) -> Path:
    """Write an honesty-scoped CPU observation; it is not a solver benchmark."""
    observation = {
        "scope": "CPU metadata wrapping observation",
        "does_not_represent": [
            "solver performance",
            "GPU performance",
            "SDAA performance",
            "physical performance",
        ],
        "operation": "immutable metadata/evidence wrapping only",
    }
    path.write_text(json.dumps(observation, indent=2) + "\n", encoding="utf-8")
    return path
