"""Ship-bottom bubble-curtain / air-layer drag-reduction closure (IN57-N02).

Modules
-------
``bubble3d``   single Shan-Chen bubble rising under net-zero buoyancy, three-box
               confinement sweep and Hadamard-Rybczynski comparison;
``curtain2d``  two-phase plane channel with the hull plate as the ceiling, hole
               gas injection, phase indicator, coverage and drag reduction at
               matched throughput;
``power``      compressor ledger (isothermal / quadrature / multi-stage) and the
               net-power closure;
``campaign``   the driver that produces ``artifacts/bubble_dr_in57_n02``.
"""

from .bubble3d import BubbleConfig, RisingBubble3D, confinement_extrapolation
from .curtain2d import CurtainChannel2D, CurtainConfig, drag_reduction, run_pair
from .power import (
    OperatingPoint,
    compressor_power_density,
    isothermal_work_closed_form,
    isothermal_work_quadrature,
    multistage_work,
    net_closure,
    positive_domain,
)

__all__ = [
    "BubbleConfig",
    "RisingBubble3D",
    "confinement_extrapolation",
    "CurtainConfig",
    "CurtainChannel2D",
    "drag_reduction",
    "run_pair",
    "OperatingPoint",
    "compressor_power_density",
    "isothermal_work_closed_form",
    "isothermal_work_quadrature",
    "multistage_work",
    "net_closure",
    "positive_domain",
]
