"""Stable public API surface for TensorLBM.

This module provides a curated subset intended for long-term compatibility.
For experimental features and rapid-evolution interfaces, use
``tensorlbm.experimental``.
"""
from __future__ import annotations

from ._version import __version__
from .boundaries import (
    apply_simple_channel_boundaries,
    apply_zou_he_channel_boundaries,
    bounce_back_cells,
    cylinder_mask,
    make_channel_wall_mask,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from .d2q9 import C, OPPOSITE, W, equilibrium, macroscopic
from .d3q19 import C as C3D
from .d3q19 import OPPOSITE as OPPOSITE3D
from .d3q19 import W as W3D
from .d3q19 import equilibrium3d, macroscopic3d
from .solver import collide_bgk, collide_mrt, collide_rlbm, collide_trt, correct_mass, stream
from .solver3d import (
    collide_bgk3d,
    collide_mrt3d,
    collide_rlbm3d,
    collide_trt3d,
    correct_mass3d,
    stream3d,
)
from .utils import (
    DiagnosticPoint,
    flow_step_image_path,
    get_reproducibility_metadata,
    legacy_snapshot_image_path,
    prepare_run_dir,
    resolve_device,
    write_legacy_snapshot_alias,
)

__all__ = [
    "__version__",
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
    "cylinder_mask",
    "make_channel_wall_mask",
    "bounce_back_cells",
    "zou_he_inlet_velocity",
    "zou_he_outlet_pressure",
    "apply_simple_channel_boundaries",
    "apply_zou_he_channel_boundaries",
    "collide_bgk",
    "collide_mrt",
    "collide_rlbm",
    "collide_trt",
    "stream",
    "correct_mass",
    "collide_bgk3d",
    "collide_mrt3d",
    "collide_rlbm3d",
    "collide_trt3d",
    "stream3d",
    "correct_mass3d",
    "DiagnosticPoint",
    "resolve_device",
    "prepare_run_dir",
    "get_reproducibility_metadata",
    "flow_step_image_path",
    "legacy_snapshot_image_path",
    "write_legacy_snapshot_alias",
]

