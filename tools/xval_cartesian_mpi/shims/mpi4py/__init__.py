"""Execution-only stand-in for :mod:`mpi4py` used by the XVAL harness.

See :mod:`mpi4py.MPI` for scope and the hard limits of this shim.
"""

from . import MPI  # noqa: F401

__version__ = "0.0.0+xval-shim"