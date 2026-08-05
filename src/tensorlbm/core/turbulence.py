"""Domain-neutral identities for turbulence-model families.

These values are descriptive metadata, not turbulence closures. Existing
application kernels continue to own any model implementation and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class TurbulenceModel(Protocol):
    """Structural identity contract for a turbulence-model family."""

    name: str


@dataclass(frozen=True)
class TurbulenceIdentity:
    """Immutable turbulence-model family identity with no numerical behaviour."""

    name: str


NONE = TurbulenceIdentity(name="NONE")
"""No turbulence closure selected."""

SMAGORINSKY = TurbulenceIdentity(name="SMAGORINSKY")
"""Smagorinsky LES family identity; no implementation is implied here."""

WALE = TurbulenceIdentity(name="WALE")
"""WALE LES family identity; no implementation is implied here."""

__all__ = ["NONE", "SMAGORINSKY", "WALE", "TurbulenceIdentity", "TurbulenceModel"]
