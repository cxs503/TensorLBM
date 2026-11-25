"""Compatibility layer: re-export advanced collision kernels.

This module exists solely to satisfy the import path used by
:mod:`free_surface_lbm` (``from .advanced_collision_d3q19 import ...``).
All implementations live in their respective modules:
  - BGK/MRT/regularized/central-stress/cumulant → :mod:`collision_d3q19_advanced`
  - Cascaded → :mod:`cascaded_collision`
  - KBC → :mod:`entropic_kbc`
"""

from __future__ import annotations

from .collision_d3q19_advanced import (
    collide_bgk_d3q19,
    collide_central_stress_d3q19,
    collide_cumulant_d3q19,
    collide_mrt_d3q19,
    collide_regularized_stress_d3q19,
    reconstruct_second_order_stress_d3q19,
    second_order_stress_d3q19,
)
from .cascaded_collision import collide_cascaded_d3q19
from .entropic_kbc import collide_kbc_d3q19

__all__ = [
    "collide_bgk_d3q19",
    "collide_central_stress_d3q19",
    "collide_cascaded_d3q19",
    "collide_cumulant_d3q19",
    "collide_kbc_d3q19",
    "collide_mrt_d3q19",
    "collide_regularized_stress_d3q19",
    "reconstruct_second_order_stress_d3q19",
    "second_order_stress_d3q19",
]
