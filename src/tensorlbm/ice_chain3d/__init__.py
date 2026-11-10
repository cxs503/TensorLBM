"""AG27W2-N08: full-ship broken-ice resistance chain (DEM + D3Q19 LBM + two-way FSI).

The package is a self-contained, physics-based chain that predicts the ice
resistance of a full three-dimensional ship hull in brash ice without empirical
ice-resistance formulas:

======================  ====================================================
module                  role
======================  ====================================================
:mod:`hull`             KVLCC2 analytic double-body surface, signed distance,
                        block/waterplane/section-coefficient gates, ice-level
                        outline for the DEM
:mod:`floes`            brash-ice floe statistics (independent audit), jittered
                        lattice packing with position projection, soft-sphere
                        DEM with hull / floe / wall contacts and water drag
:mod:`flow`             D3Q19 two-relaxation-time LBM (float64), Bouzidi
                        interpolated bounce-back on the analytic hull surface,
                        Darcy-Forchheimer porous ice band, control-volume
                        momentum balance
:mod:`reference`        coefficient-free estimates, the in-repo Riska baseline,
                        ITTC-57 friction reference and GCI algebra
:mod:`chain`            campaign driver: six criteria, figures, ``criteria.json``
======================  ====================================================

Quick start::

    python -m tensorlbm.ice_chain3d.chain
"""

from __future__ import annotations

from .hull import (ICE_LEVEL_Z, KVLCC2Hull, ice_drift_outline, make_hull,
                   outline_entrance_angle, rasterise_mask)
from .floes import (DEMConfig, FloeDEM, FloeField, FloeFieldConfig,
                    audit_floe_field, calibrated_floe_field, generate_floe_field)
from .flow import FlowConfig, HullFlow

__all__ = [
    "ICE_LEVEL_Z", "KVLCC2Hull", "make_hull", "ice_drift_outline",
    "outline_entrance_angle", "rasterise_mask",
    "FloeFieldConfig", "FloeField", "generate_floe_field", "audit_floe_field",
    "calibrated_floe_field", "FloeDEM", "DEMConfig",
    "FlowConfig", "HullFlow",
]