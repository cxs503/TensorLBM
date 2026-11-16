"""Minimal offline stand-in for ``mpi4py`` (single rank, no-op collectives).

Why this exists
---------------
The OSS reference implementation for anchor ``XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME``
is ``pylbm`` (cached read-only at ``/root/oss-cache/pylbm``).  ``pylbm`` imports
``mpi4py.MPI`` at package-import time and calls ``mpi.Compute_dims`` /
``COMM_WORLD.allreduce`` while building a ``Simulation`` even in a single-rank
serial run, and ``mpi4py`` has no wheel on this offline host (the source build
fails without an MPI toolchain).  This shim makes the *serial* code paths of
``pylbm`` usable so the reference numbers can be **computed instead of quoted**.

Scope / honesty
---------------
It implements only the handful of attributes ``pylbm`` touches on the serial
path.  Every collective is a no-op on a single rank and asserts that it is
being called with ``size == 1`` semantics.  It is **not** an MPI implementation
and must never be used for a parallel run; nothing here is used by TensorLBM
itself (it lives under ``tools/`` and is only on ``PYTHONPATH`` for xval
drivers).
"""

from . import MPI  # noqa: F401

__all__ = ["MPI"]