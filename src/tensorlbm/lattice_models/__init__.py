"""Lattice model namespace for new grouped imports."""
from __future__ import annotations

from ..d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from ..d3q19 import OPPOSITE as OPPOSITE3D
from ..d3q19 import C as C3D
from ..d3q19 import W as W3D
from ..d3q19 import equilibrium3d, macroscopic3d
from ..d3q27 import OPPOSITE as OPPOSITE27
from ..d3q27 import C as C27
from ..d3q27 import W as W27
from ..d3q27 import equilibrium27, macroscopic27

__all__ = [
    "C",
    "W",
    "OPPOSITE",
    "equilibrium",
    "macroscopic",
    "C3D",
    "W3D",
    "OPPOSITE3D",
    "equilibrium3d",
    "macroscopic3d",
    "C27",
    "W27",
    "OPPOSITE27",
    "equilibrium27",
    "macroscopic27",
]
