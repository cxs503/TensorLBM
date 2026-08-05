"""Explicit force-observation provenance contract.

This module records observation semantics but does not calculate hydrodynamic
loads and makes no physical-validation claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

ForceStatus = Literal["diagnostic_only", "withheld", "measured"]


def _triple(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be an (x, y, z) tuple of length 3")
    result: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)) or not isfinite(component):
            raise ValueError(f"{name} must contain finite numeric coordinates")
        result.append(float(component))
    return tuple(result)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ForceObservation:
    """A force sample with phase, sign, and link-ownership provenance.

    ``measured`` is intentionally fail-closed: it is only permitted when the
    producer can attest that its result owns the contributing wall links.
    """

    method: str
    lattice_id: str
    sample_phase: str
    force_on: str
    origin: tuple[float, float, float]
    status: ForceStatus
    force: tuple[float, float, float] | None = None
    link_ownership: bool = False

    def __post_init__(self) -> None:
        for name in ("method", "lattice_id", "sample_phase", "force_on"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.status not in ("diagnostic_only", "withheld", "measured"):
            raise ValueError("status must be diagnostic_only, withheld, or measured")
        if not isinstance(self.link_ownership, bool):
            raise ValueError("link_ownership must be a bool")
        object.__setattr__(self, "origin", _triple(self.origin, "origin"))
        if self.force is not None:
            object.__setattr__(self, "force", _triple(self.force, "force"))
        if self.status == "measured" and not self.link_ownership:
            raise ValueError("measured force observations require explicit link ownership")

    @classmethod
    def withheld(
        cls,
        *,
        method: str,
        lattice_id: str,
        sample_phase: str,
        force_on: str,
        origin: tuple[float, float, float],
    ) -> "ForceObservation":
        """Construct a no-result record without laundering it as a measurement."""
        return cls(method, lattice_id, sample_phase, force_on, origin, "withheld")


__all__ = ["ForceObservation", "ForceStatus"]
