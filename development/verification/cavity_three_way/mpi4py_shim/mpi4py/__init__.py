"""Minimal single-rank stand-in for mpi4py (only what pylbm imports)."""
from . import MPI  # noqa: F401
__version__ = "0.0.0-single-rank-shim"
