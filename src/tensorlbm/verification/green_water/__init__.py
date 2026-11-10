"""Green water on deck verification suite for TensorLBM.

Covers three verification cases:
1. 2D FPSO green water (Greco 2012 benchmark)
2. 3D S175 bow green water (Lugni 2018 benchmark)
3. Deck drainage hole flow coefficient (ITTC recommended 0.6-0.7)

Plus mass conservation validation (<2% residual per time step).
"""

from .fpso_2d import FPSOGreenWaterConfig, run_fpso_green_water
from .drainage import DrainageConfig, run_drainage_benchmark
from .mass_conservation import check_mass_conservation
from .reference_data import (
    GRECO_2012_FREE_SURFACE,
    LUGNI_2018_WET_AREA,
    ITTC_DRAINAGE_CD,
)

__all__ = [
    "FPSOGreenWaterConfig",
    "run_fpso_green_water",
    "DrainageConfig",
    "run_drainage_benchmark",
    "check_mass_conservation",
    "GRECO_2012_FREE_SURFACE",
    "LUGNI_2018_WET_AREA",
    "ITTC_DRAINAGE_CD",
]
