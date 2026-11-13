"""Parallel execution helpers (domain decomposition, halo exchange)."""

from .slab_unpacked import (  # noqa: F401
    C,
    CX,
    CY,
    CROSS_DIRS,
    OPP,
    W,
    PackedHaloExchange,
    SerialComm,
    SlabSolver,
    SocketPeers,
    UnpackedHaloExchange,
    tg_analytic,
    uniform_state,
)

__all__ = [
    "C",
    "CX",
    "CY",
    "CROSS_DIRS",
    "OPP",
    "W",
    "PackedHaloExchange",
    "SerialComm",
    "SlabSolver",
    "SocketPeers",
    "UnpackedHaloExchange",
    "tg_analytic",
    "uniform_state",
]