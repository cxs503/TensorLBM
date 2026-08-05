"""Null-safe R1 resistance/force contract.

This module only classifies supplied observations.  It neither calls a solver
nor converts legacy cell-reset force estimates into link-owned traction.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Mapping


Vector3 = tuple[float, float, float]


def _positive(value: object, name: str, diagnostics: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or value <= 0.0:
        diagnostics.append(f"{name}: missing or not finite positive")
        return None
    return float(value)


def _vector(value: object, name: str, diagnostics: list[str]) -> Vector3 | None:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        diagnostics.append(f"{name}: missing or not a 3-vector")
        return None
    if any(isinstance(component, bool) or not isinstance(component, (int, float)) or not isfinite(float(component)) for component in value):
        diagnostics.append(f"{name}: components must be finite numbers")
        return None
    vector = tuple(float(component) for component in value)
    if sqrt(sum(component * component for component in vector)) == 0.0:
        diagnostics.append(f"{name}: must be non-zero")
        return None
    return vector  # type: ignore[return-value]


def _complete_ownership(value: object, diagnostics: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        diagnostics.append("link_ownership: missing")
        return None
    if value.get("status") != "complete":
        diagnostics.append("link_ownership: status is not complete")
        return None
    if not isinstance(value.get("owner"), str) or not value["owner"]:
        diagnostics.append("link_ownership: missing owner")
        return None
    links = value.get("owned_links")
    if isinstance(links, bool) or not isinstance(links, int) or links <= 0:
        diagnostics.append("link_ownership: owned_links must be a positive integer")
        return None
    return dict(value)


@dataclass(frozen=True)
class ResistanceForceContract:
    reference_area: float | None
    length: float | None
    rho: float | None
    U: float | None
    direction: Vector3 | None
    method: str | None
    sample_phase: str | None
    link_ownership: Mapping[str, Any] | None
    force: Vector3 | None
    status: str
    Ct: float | None
    validated: bool
    diagnostics: tuple[str, ...]


def build_resistance_force_contract(
    *, reference_area: float | None, length: float | None, rho: float | None,
    U: float | None, direction: Vector3 | None, method: str | None,
    sample_phase: str | None, link_ownership: Mapping[str, Any] | None,
    force: Vector3 | None,
) -> ResistanceForceContract:
    """Classify one force observation; incomplete ownership always fails closed.

    A ``Ct`` is constructed only when every normalizing quantity, observation
    descriptor, and a complete link-ownership ledger are present.
    """
    diagnostics: list[str] = []
    area = _positive(reference_area, "reference_area", diagnostics)
    resolved_length = _positive(length, "length", diagnostics)
    density = _positive(rho, "rho", diagnostics)
    speed = _positive(U, "U", diagnostics)
    resolved_direction = _vector(direction, "direction", diagnostics)
    resolved_force = _vector(force, "force", diagnostics)
    if not isinstance(method, str) or not method:
        diagnostics.append("method: missing")
        method = None
    if not isinstance(sample_phase, str) or not sample_phase:
        diagnostics.append("sample_phase: missing")
        sample_phase = None
    ownership = _complete_ownership(link_ownership, diagnostics)

    Ct: float | None = None
    if not diagnostics and area is not None and density is not None and speed is not None and resolved_force is not None and resolved_direction is not None:
        norm = sqrt(sum(component * component for component in resolved_direction))
        unit_direction = tuple(component / norm for component in resolved_direction)
        axial_force = sum(component * axis for component, axis in zip(resolved_force, unit_direction))
        Ct = axial_force / (0.5 * density * speed * speed * area)
        # Link ownership makes this a traceable measured coefficient candidate,
        # not a reference/convergence/uncertainty-backed physical validation.
        status = "measured_candidate"
        validated = False
        diagnostics.append("physical_validation: withheld")
    else:
        status = "diagnostic_only"
        validated = False

    return ResistanceForceContract(
        reference_area=area, length=resolved_length, rho=density, U=speed,
        direction=resolved_direction, method=method, sample_phase=sample_phase,
        link_ownership=ownership, force=resolved_force, status=status, Ct=Ct,
        validated=validated, diagnostics=tuple(diagnostics),
    )


__all__ = ["ResistanceForceContract", "build_resistance_force_contract"]
