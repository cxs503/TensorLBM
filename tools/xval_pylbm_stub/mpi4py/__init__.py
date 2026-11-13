"""Minimal offline stand-in for ``mpi4py`` (single rank, no-op collectives).

The XVAL reference implementation (``pylbm``, ``/root/oss-cache/pylbm``) imports
``mpi4py.MPI`` at package import time even when the pure-Python scheme algebra is
all that is used.  This shim makes :mod:`pylbm` importable in an offline,
single-rank session so that the reference moment matrices / equilibria can be
*computed* instead of hand-copied.  It implements nothing more than the handful
of attributes ``pylbm`` touches during import and scheme construction; it is
**not** an MPI implementation and must never be used for a parallel run.
"""

from . import MPI  # noqa: F401

__all__ = ["MPI"]