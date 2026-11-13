"""Execution-only stand-in for ``mpi4py.MPI``.

WHY THIS FILE EXISTS
--------------------
The XVAL-LBM-CARTESIAN-MPI anchor cross-validates TensorLBM's cartesian domain
decomposition against ``pylbm`` (OSS, version 0.12.0).  ``pylbm`` imports
``mpi4py.MPI`` at package import time, so the reference code cannot be executed
at all on a host without an MPI build.  The test host (loongarch64 / Loongnix)
has no MPI implementation and ``pip install mpi4py`` fails at the C-extension
build step (see ``docs/XVAL-LBM-CARTESIAN-MPI.md`` §6 friction).

This module therefore provides *the smallest object surface that pylbm actually
touches*, so that **pylbm's own source code** (not a re-implementation) can be
executed and its numbers read off directly.

SCOPE — what this shim does
---------------------------
1. ``COMM_WORLD`` of size 1, rank 0.  ``Irecv``/``Isend``/``Send``/``Recv``
   between a rank and *itself* are executed as a real buffer copy driven by the
   MPI derived-datatype descriptor (``Datatype.Create_subarray``).  For a
   single-process run this is exactly what pylbm's periodic halo wrap does:
   cartesian topology with ``period=[True]`` maps both neighbours of the only
   rank onto itself, so ``storage.update()`` must move the interior boundary
   plane into the ghost plane.  The copy is therefore *not* a no-op.
2. ``SimCartComm``: a **topology-only** cartesian communicator of arbitrary
   size ``P`` with a chosen ``(npx, npy, npz)`` factorisation.  ``Get_rank``,
   ``Get_coords``, ``Get_cart_rank``, ``Get_topo``, ``Sub`` and thus pylbm's
   ``MpiTopology.get_region`` / ``get_region_indices`` / ``storage._set_subarray``
   all compute their real numbers (region bounds, neighbour ranks, derived
   datatype extents).  ``Send``/``Recv``/``Isend``/``Irecv`` raise
   :class:`NotImplementedError`: there is no data transport for several ranks.

SCOPE — what this shim does NOT do
----------------------------------
- It never fakes results: every number the harness reports comes out of pylbm's
  own functions.
- It is not a general MPI replacement and must never be installed into
  site-packages or put on a default ``sys.path`` (a real ``mpi4py`` must always
  win).  It is consumed only via ``PYTHONPATH=<this dir>`` by the XVAL scripts,
  and only when a real ``mpi4py`` is absent.
- Multi-rank pylbm *simulations* are out of reach here; only their decomposition
  geometry/descriptors are exercised (declared in the report).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUM = "SUM"
MAX = "MAX"
MIN = "MIN"


# ---------------------------------------------------------------------------
# Derived datatypes
# ---------------------------------------------------------------------------


class Datatype:
    """Bare datatype handle (only the subarray constructor is implemented)."""

    def __init__(self, name: str = "DOUBLE") -> None:
        self.name = name

    def Create_subarray(self, sizes, subsizes, sstart, order=None):
        return SubarrayType(sizes, subsizes, sstart)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Datatype {self.name}>"


class SubarrayType:
    """Records an MPI ``Create_subarray`` descriptor and can slice an array."""

    def __init__(self, sizes, subsizes, sstart, order=None) -> None:
        self.sizes = tuple(int(s) for s in sizes)
        self.subsizes = tuple(int(s) for s in subsizes)
        self.sstart = tuple(int(s) for s in sstart)
        if len(self.sizes) != len(self.subsizes) or len(self.sizes) != len(self.sstart):
            raise ValueError("subarray descriptor dimension mismatch")

    def slices(self):
        """numpy slices of the described region."""
        return tuple(slice(s, s + e) for s, e in zip(self.sstart, self.subsizes))

    def nbytes(self, itemsize: int = 8) -> int:
        n = 1
        for e in self.subsizes:
            n *= e
        return n * itemsize

    def Commit(self):  # MPI API parity
        return self

    def Free(self):  # MPI API parity
        return None

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Subarray sizes={self.sizes} subsizes={self.subsizes} start={self.sstart}>"


DOUBLE = Datatype("DOUBLE")
FLOAT = Datatype("FLOAT")


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class Request:
    """A completed-later request holding a pending copy closure."""

    def __init__(self, op=None) -> None:
        self._op = op
        self._done = False

    def Wait(self):
        if not self._done and self._op is not None:
            self._op()
        self._done = True
        return None

    def test(self):
        return (True, None)

    @staticmethod
    def Waitall(reqs):
        for r in reqs:
            r.Wait()
        return None


# ---------------------------------------------------------------------------
# Communicators
# ---------------------------------------------------------------------------


def _balanced_dims(size: int, dim: int) -> list[int]:
    """``MPI_Dims_create``-style balanced factorisation (dims non-increasing)."""
    if dim < 1 or size < 1:
        raise ValueError("size and dim must be >= 1")
    dims = [1] * dim
    n = size
    # distribute prime factors, largest first, onto the currently smallest axis
    f = 2
    factors: list[int] = []
    while f * f <= n:
        while n % f == 0:
            factors.append(f)
            n //= f
        f += 1
    if n > 1:
        factors.append(n)
    for p in sorted(factors, reverse=True):
        i = int(np.argmin(dims))
        dims[i] *= p
    return sorted(dims, reverse=True)


class Comm:
    """Base communicator: self-communication really copies, no transport."""

    def __init__(self, size: int = 1, rank: int = 0) -> None:
        self._size = int(size)
        self._rank = int(rank)
        self._pending_recv: list = []

    # -- identity -----------------------------------------------------------
    def Get_size(self):
        return self._size

    def Get_rank(self):
        return self._rank

    # -- collectives (single rank: identity) --------------------------------
    def allreduce(self, sendobj=None, op=None):
        return sendobj

    def allgather(self, sendobj=None):
        return [sendobj]

    def gather(self, sendobj=None, root=0):
        return [sendobj] if self._rank == root else None

    def bcast(self, obj=None, root=0):
        return obj

    def Barrier(self):
        return None

    # -- point to point -----------------------------------------------------
    def _check(self, peer):
        if self._size != 1:
            raise NotImplementedError(
                "xval mpi4py shim: no data transport between distinct ranks "
                f"(size={self._size}). Use a real MPI build for multi-rank runs."
            )
        if peer not in (0, self._rank, MPI.ANY_SOURCE):
            raise NotImplementedError(f"xval mpi4py shim: unknown peer {peer}")

    def Isend(self, buf, dest=0, tag=0):
        """Send ``buf`` to ``dest``.  With one rank the peer is self: the send
        region is snapshotted and delivered into the matching posted recv
        region at ``Request.Wait`` time (true copy, real periodic wrap)."""
        self._check(dest)
        array, dtype = _split_buf(buf)
        snap = np.array(array[dtype.slices()], copy=True)
        pending = self._pending_recv

        def _op():
            for item in list(pending):
                recv_array, recv_type, source = item
                if source in (self._rank, ANY_SOURCE):
                    recv_array[recv_type.slices()] = snap
                    pending.remove(item)

        return Request(_op)

    def Irecv(self, buf, source=0, tag=0):
        """Post a receive; the matching in-rank ``Isend`` fills the region."""
        self._check(source)
        array, dtype = _split_buf(buf)
        self._pending_recv.append((array, dtype, source))

        def _op():
            return None

        return Request(_op)

    def Send(self, buf, dest=0, tag=0):
        self.Isend(buf, dest, tag).Wait()

    def Recv(self, buf, source=0, tag=0):
        self.Irecv(buf, source, tag).Wait()

    # -- lifecycle ----------------------------------------------------------
    def Free(self):
        return None

    def Sub(self, is_active):  # pylbm/hdf5 helper
        ranks = [r for r, a in enumerate(is_active) if a]
        if not ranks:
            raise ValueError("empty sub-communicator")
        return Comm(size=len(ranks), rank=0)

    def Create_cart(self, dims, periods=None, reorder=False):
        return CartComm(dims, periods)


class CartComm(Comm):
    """Cartesian sub-communicator on a single rank (real periodic self-wrap)."""

    def __init__(self, dims, periods=None, rank: int = 0) -> None:
        dims = [int(d) for d in dims]
        size = 1
        for d in dims:
            size *= d
        super().__init__(size=size, rank=rank)
        if size != 1:
            raise NotImplementedError(
                "xval mpi4py shim: CartComm with size>1 needs real MPI; "
                "use SimCartComm for topology-only work."
            )
        self.dims = dims
        self.periods = list(periods) if periods is not None else [False] * len(dims)

    def Get_topology(self):
        return self.dims, self.periods, [0] * len(self.dims)

    def Get_topo(self):
        return self.dims, self.periods, [0] * len(self.dims)

    def Get_coords(self, rank=0):
        return np.zeros(len(self.dims), dtype=int)

    def Get_cart_rank(self, coords):
        return 0


class SimCartComm(Comm):
    """Topology-only cartesian communicator of arbitrary size (no transport)."""

    def __init__(self, dims, periods=None, rank: int = 0) -> None:
        dims = [int(d) for d in dims]
        size = 1
        for d in dims:
            size *= d
        if not 0 <= rank < size:
            raise ValueError(f"rank {rank} out of range for dims {dims}")
        super().__init__(size=size, rank=rank)
        self.dims = dims
        self.periods = list(periods) if periods is not None else [True] * len(dims)

    def Get_topo(self):
        return self.dims, self.periods, tuple(int(c) for c in self.Get_coords())

    Get_topology = Get_topo

    def Get_coords(self, rank=None):
        rank = self._rank if rank is None else int(rank)
        coords = []
        for d in reversed(self.dims):
            coords.append(rank % d)
            rank //= d
        return np.asarray(list(reversed(coords)), dtype=int)

    def Get_cart_rank(self, coords):
        coords = [int(c) for c in coords]
        if len(coords) != len(self.dims):
            raise ValueError("coords dimension mismatch")
        rank = 0
        for c, d, p in zip(coords, self.dims, self.periods):
            if p:
                c %= d
            elif not 0 <= c < d:
                raise ValueError(f"coordinate {c} out of range for dim {d} (non-periodic)")
            rank = rank * d + c
        return rank

    def Create_cart(self, dims, periods=None, reorder=False):
        return SimCartComm(dims, periods, rank=self._rank)

    # -- transport is explicitly refused ------------------------------------
    def _no_transport(self, *a, **k):
        raise NotImplementedError(
            "xval mpi4py shim: SimCartComm is topology-only (no data transport). "
            "Multi-rank pylbm simulations require a real MPI build."
        )

    Irecv = Isend = Send = Recv = _no_transport


def _split_buf(buf):
    """Unpack an mpi4py-style ``[array, datatype]`` buffer tuple."""
    if isinstance(buf, (list, tuple)):
        array, dtype = buf[0], buf[1]
        if not isinstance(dtype, SubarrayType):
            raise NotImplementedError("xval mpi4py shim: only subarray datatypes are supported")
        a = np.asarray(array)
        if tuple(a.shape) != dtype.sizes:
            # pylbm orders dimensions itself; trust the descriptor sizes
            pass
        return a, dtype
    raise NotImplementedError("xval mpi4py shim: raw buffers are not supported")


# ---------------------------------------------------------------------------
# Module level helpers
# ---------------------------------------------------------------------------


def Compute_dims(size: int, dim: int) -> list[int]:
    """Balanced ``MPI_Dims_create`` equivalent (see :func:`_balanced_dims`)."""
    return _balanced_dims(int(size), int(dim))


def Wtime() -> float:
    import time

    return time.perf_counter()


ANY_SOURCE = -1
ANY_TAG = -1
INFO_NULL = None


class _MPIModule:
    """Namespace object mirroring ``mpi4py.MPI``'s flat layout."""

    SUM = SUM
    MAX = MAX
    MIN = MIN
    DOUBLE = DOUBLE
    FLOAT = FLOAT
    Datatype = Datatype
    Request = Request
    Comm = Comm
    CartComm = CartComm
    SimCartComm = SimCartComm
    Compute_dims = staticmethod(Compute_dims)
    Wtime = staticmethod(Wtime)
    ANY_SOURCE = ANY_SOURCE
    ANY_TAG = ANY_TAG
    INFO_NULL = INFO_NULL
    COMM_WORLD = Comm(size=1, rank=0)


MPI = _MPIModule()

__all__ = [
    "Comm",
    "CartComm",
    "SimCartComm",
    "Datatype",
    "SubarrayType",
    "Request",
    "DOUBLE",
    "FLOAT",
    "SUM",
    "Compute_dims",
    "Wtime",
    "COMM_WORLD",
    "MPI",
]

COMM_WORLD = MPI.COMM_WORLD