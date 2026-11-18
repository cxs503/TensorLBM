"""Air-layer / microbubble drag reduction (D2Q9 phase-field LBM).

Sub-package created for campaign ``IN30-N03`` (船底气层/微气泡减阻 LBM 两相仿真).

Modules
-------
``core``       : two-distribution D2Q9 solver — pressure-based variable-density
                 hydrodynamics + conservative Allen-Cahn phase transport with a
                 Cahn-Hilliard (Korteweg) surface-tension force.
``ledger``     : mass / gas-inventory / power accounting (compressor ledger).
``economics``  : net-power model ``dP_net = P_saved - P_comp``.
``validations``: machine-checkable gates (single-phase Cf, Laplace law, mass).
``campaign``   : DR-theta-Qg sweep driver producing the closure surface.
"""

from .core import (  # noqa: F401
    PhaseFieldParams,
    PhaseFieldChannel,
    SinglePhaseChannel,
)
from .economics import compressor_power, net_power_gain  # noqa: F401
from .ledger import GasLedger  # noqa: F401

__all__ = [
    "PhaseFieldParams",
    "PhaseFieldChannel",
    "SinglePhaseChannel",
    "GasLedger",
    "compressor_power",
    "net_power_gain",
]