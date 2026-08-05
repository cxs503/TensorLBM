"""Domain-neutral identities for collision-model families.

These values identify a requested operator family only. They neither select nor
reimplement a collision kernel; applications retain ownership of their existing
BGK, MRT, and cumulant implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class CollisionModel(Protocol):
    """Structural identity contract for a collision-model family."""

    name: str


@dataclass(frozen=True)
class CollisionIdentity:
    """Immutable collision-model family identity with no numerical behaviour."""

    name: str


BGK = CollisionIdentity(name="BGK")
"""Single-relaxation-time collision family identity."""

MRT = CollisionIdentity(name="MRT")
"""Multi-relaxation-time collision family identity."""

CUMULANT = CollisionIdentity(name="CUMULANT")
"""Cumulant collision family identity; no implementation is implied here."""

__all__ = ["BGK", "CUMULANT", "MRT", "CollisionIdentity", "CollisionModel"]
