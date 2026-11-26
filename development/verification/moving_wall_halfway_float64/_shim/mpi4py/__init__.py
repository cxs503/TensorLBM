"""Minimal serial stand-in for mpi4py (size=1). Only the API surface that
pylbm touches in a serial run is provided."""
from . import MPI  # noqa: F401
