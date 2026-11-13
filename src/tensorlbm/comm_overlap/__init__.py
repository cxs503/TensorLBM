"""Halo-exchange overlap scheduling ported from the TCLB async-copy/wait-any design.

Reference implementation (zero-transcription source of record):
``CFD-GO/TCLB`` ``src/Lattice.cu.Rt:327-386`` (``MPIStream_A`` / ``MPIStream_B``),
``src/Lattice.cu.Rt:496-530`` (``RunBorder`` -> ``MPIStream_A`` -> ``RunInterior`` ->
``MPIStream_B``) and ``src/Lattice.h.Rt:70-72`` (``inStream`` / ``outStream``).
"""

from .backend import BackendStats, TorchDistBackend, VirtualClockBackend
from .virtual_case import run_virtual_case
from .waitany_pipeline import (
    MODES,
    D3Q19OverlapTransport,
    stream_full,
    stream_interior,
    stream_one_column,
)

__all__ = [
    "MODES",
    "BackendStats",
    "D3Q19OverlapTransport",
    "TorchDistBackend",
    "VirtualClockBackend",
    "run_virtual_case",
    "stream_full",
    "stream_interior",
    "stream_one_column",
]