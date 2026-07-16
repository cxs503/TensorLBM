"""Solver-independent, immutable case/reference definitions for marine applications.

R1 intentionally records no external SUBOFF, CH, or Körner dimensions.  A
consumer must explicitly provide sourced dimensions before using a reference
for validation; the default is therefore ``withheld``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Mapping


_CONFIGURATIONS = frozenset(("bare_hull", "with_sail", "full"))
_APPLICATIONS = frozenset(("suboff", "ch_hull", "korner_hull"))


def _finite_positive_or_none(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0 when supplied")
    return float(value)


@dataclass(frozen=True)
class MarineReferenceDefinition:
    """Reference quantities with explicit provenance status.

    ``L``, ``D`` and ``Sref`` are deliberately nullable: an absent source is
    represented as absent data, rather than an invented benchmark value.
    """

    L: float | None = None
    D: float | None = None
    Sref: float | None = None
    coordinates: tuple[str, str, str] = ("x", "y", "z")
    units: Mapping[str, str] = field(default_factory=lambda: {
        "length": "m", "diameter": "m", "reference_area": "m^2",
        "coordinates": "m", "density": "kg/m^3", "speed": "m/s", "force": "N",
    })
    source_status: str = "withheld"
    source: str | None = None
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "L", _finite_positive_or_none(self.L, "L"))
        object.__setattr__(self, "D", _finite_positive_or_none(self.D, "D"))
        object.__setattr__(self, "Sref", _finite_positive_or_none(self.Sref, "Sref"))
        if self.coordinates != ("x", "y", "z"):
            raise ValueError("coordinates must declare the canonical ('x', 'y', 'z') axes")
        normalized_units = dict(self.units)
        required_units = {"length", "diameter", "reference_area", "coordinates", "density", "speed", "force"}
        if not required_units.issubset(normalized_units) or not all(
            isinstance(normalized_units[key], str) and normalized_units[key] for key in required_units
        ):
            raise ValueError("units must define non-empty canonical marine units")
        frozen_units = tuple(sorted(normalized_units.items()))
        object.__setattr__(self, "units", MappingProxyType(dict(frozen_units)))
        if self.source_status not in {"withheld", "provided"}:
            raise ValueError("source_status must be 'withheld' or 'provided'")
        if self.source_status == "provided" and not self.source:
            raise ValueError("provided reference data requires a source")
        if self.source_status == "withheld" and self.source is not None:
            raise ValueError("withheld reference data must not name an external source")
        unsigned = {
            "L": self.L, "D": self.D, "Sref": self.Sref,
            "coordinates": self.coordinates, "units": dict(self.units),
            "source_status": self.source_status, "source": self.source,
        }
        encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "sha256", sha256(encoded.encode("utf-8")).hexdigest())

    def __hash__(self) -> int:
        return hash((
            self.L, self.D, self.Sref, self.coordinates, tuple(sorted(self.units.items())),
            self.source_status, self.source, self.sha256,
        ))


@dataclass(frozen=True)
class SuboffCaseDefinition:
    """Shared full-wet case descriptor for SUBOFF and generic hull applications."""

    configuration: str = "bare_hull"
    application: str = "suboff"
    medium: str = "full_wet"
    reference: MarineReferenceDefinition = field(default_factory=MarineReferenceDefinition)
    schema: str = "marine-case-definition-r1"

    def __post_init__(self) -> None:
        if self.configuration not in _CONFIGURATIONS:
            raise ValueError("configuration must be bare_hull, with_sail, or full")
        if self.application not in _APPLICATIONS:
            raise ValueError("application must be suboff, ch_hull, or korner_hull")
        if self.medium != "full_wet":
            raise ValueError("R1 only defines full_wet cases")
        if not isinstance(self.reference, MarineReferenceDefinition):
            raise TypeError("reference must be a MarineReferenceDefinition")


__all__ = ["MarineReferenceDefinition", "SuboffCaseDefinition"]
