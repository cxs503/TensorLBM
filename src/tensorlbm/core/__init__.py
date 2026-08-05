"""Domain-neutral numerical contracts and adapters.

This package exposes descriptions of existing lattice, collision, and turbulence
families. It deliberately does not own solver evolution or physical models.
"""

from .collision import BGK, CUMULANT, MRT, CollisionIdentity, CollisionModel
from .d3q19_stencil import (
    D3Q19_MOVING_Q,
    all_moving_neighbor_masks,
    assert_no_direct_phase_links,
    moving_tensor_shifts,
    roll_from_pull_source,
    roll_to_neighbor,
    tensor_shift_for_q,
)
from .lattice import D3Q19, D3Q27, LatticeDescriptor
from .turbulence import NONE, SMAGORINSKY, WALE, TurbulenceIdentity, TurbulenceModel

__all__ = [
    "BGK",
    "CUMULANT",
    "D3Q19_MOVING_Q",
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
    "all_moving_neighbor_masks",
    "assert_no_direct_phase_links",
    "moving_tensor_shifts",
    "roll_from_pull_source",
    "roll_to_neighbor",
    "tensor_shift_for_q",
]
