"""Domain-neutral numerical contracts and adapters.

This package exposes descriptions of existing lattice, collision, and turbulence
families. It deliberately does not own solver evolution or physical models.
"""

from .collision import BGK, CUMULANT, MRT, CollisionIdentity, CollisionModel
from .lattice import D3Q19, D3Q27, LatticeDescriptor
from .turbulence import NONE, SMAGORINSKY, WALE, TurbulenceIdentity, TurbulenceModel

__all__ = [
    "BGK",
    "CUMULANT",
    "D3Q19",
    "D3Q27",
    "MRT",
    "NONE",
    "SMAGORINSKY",
    "WALE",
    "CollisionIdentity",
    "CollisionModel",
    "LatticeDescriptor",
    "TurbulenceIdentity",
    "TurbulenceModel",
]
