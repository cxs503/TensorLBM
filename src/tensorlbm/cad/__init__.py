"""CAD and geometry namespace for grouped imports."""
from __future__ import annotations

from ..offshore_cad import (
    OffshoreStructureType,
    build_offshore_mask,
    export_offshore_stl,
    generate_offshore_previews,
    jacket_mask,
    monopile_mask,
    offshore_statistics,
    semi_sub_mask,
    spar_mask,
)
from ..propeller_cad import (
    optimal_advance_ratio,
    propeller_design,
    propeller_disk_mask,
    wageningen_b_series,
)
from ..ship_cad import (
    ShipHullType,
    export_hull_stl,
    generate_hull_body_plan,
    generate_hull_previews,
    generate_hull_sideprofile,
    generate_hull_waterplane,
    hull_block_coefficient,
    hull_statistics,
    kcs_hull_mask,
    series60_hull_mask,
    ship_lbm_parameters,
    theoretical_block_coefficient,
)
from ..ship_cad import (
    build_hull_mask as build_ship_hull_mask,
)
from ..suboff_cad import (
    SuboffConfig,
    SuboffHullType,
    build_suboff_mask,
    export_suboff_stl,
    generate_suboff_previews,
    suboff_hull_mask,
    suboff_radius_profile,
    suboff_statistics,
)

__all__ = [
    "ShipHullType",
    "series60_hull_mask",
    "kcs_hull_mask",
    "hull_block_coefficient",
    "hull_statistics",
    "theoretical_block_coefficient",
    "generate_hull_body_plan",
    "generate_hull_waterplane",
    "generate_hull_sideprofile",
    "generate_hull_previews",
    "export_hull_stl",
    "build_ship_hull_mask",
    "ship_lbm_parameters",
    "SuboffConfig",
    "SuboffHullType",
    "build_suboff_mask",
    "export_suboff_stl",
    "generate_suboff_previews",
    "suboff_hull_mask",
    "suboff_radius_profile",
    "suboff_statistics",
    "OffshoreStructureType",
    "monopile_mask",
    "jacket_mask",
    "spar_mask",
    "semi_sub_mask",
    "build_offshore_mask",
    "offshore_statistics",
    "generate_offshore_previews",
    "export_offshore_stl",
    "wageningen_b_series",
    "optimal_advance_ratio",
    "propeller_design",
    "propeller_disk_mask",
]
