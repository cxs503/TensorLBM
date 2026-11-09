"""Superhydrophobic surface effective slip length module.

Three-path verification framework:
1. Philip(1972) analytical anchor
2. LBM explicit microstructure simulation
3. Navier slip BC equivalent path

Plus drag reduction quantification.
"""

from .philip1972 import (
    philip_slip_length,
    philip_slip_length_dimensionless,
    gas_area_fraction,
)
from .explicit_microstructure import (
    run_explicit_microstructure,
    extract_slip_length_from_profile,
)
from .navier_slip_equivalent import (
    run_navier_slip_equivalent,
)
from .drag_reduction import (
    compute_drag_reduction,
    friction_coefficient_channel,
)

__all__ = [
    "philip_slip_length",
    "philip_slip_length_dimensionless",
    "gas_area_fraction",
    "run_explicit_microstructure",
    "extract_slip_length_from_profile",
    "run_navier_slip_equivalent",
    "compute_drag_reduction",
    "friction_coefficient_channel",
]
