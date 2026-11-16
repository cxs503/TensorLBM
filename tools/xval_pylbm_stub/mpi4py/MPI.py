"""Single-rank no-op ``mpi4py.MPI`` used only to import and run ``pylbm`` offline."""

import time as _time

import numpy as np


class _Op:
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def __repr__(self):  # pragma: no cover - debugging convenience
        return f"<Op {self.name}>"


def _sum(values):
    total = values[0]
    for v in values[1:]:
        total = total + v
    return total


def _min(values):
    out = values[0]
    for v in values[1:]:
        out = min(out, v)
    return out


def _max(values):
    out = values[0]
    for v in values[1:]:
        out = max(out, v)
    return out


SUM = _Op("SUM", _sum)
MIN = _Op("MIN", _min)
MAX = _Op("MAX", _max)
PROD = _Op("PROD", lambda v: v[0])
LOR = _Op("LOR", lambda v: v[0])
LAND = _Op("LAND", lambda v: v[0])


class _Comm:
    rank = 0
    size = 1

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Barrier(self):
        return None

    def bcast(self, value, root=0):
        return value

    def gather(self, value, root=0):
        return [value]

    def allgather(self, value):
        return [value]

    def allreduce(self, sendobj=None, op=SUM, recvbuf=None):
        return sendobj

    def reduce(self, value, op=SUM, root=0):
        return value

    def scatter(self, value, root=0):
        return value

    def send(self, *args, **kwargs):
        return None

    def recv(self, *args, **kwargs):
        return None

    def Send(self, *args, **kwargs):
        return None

    def Irecv(self, *args, **kwargs):
        return REQUEST

    def Isend(self, *args, **kwargs):
        return REQUEST

    def Wait(self, *args, **kwargs):
        return None

    def Waitall(self, *args, **kwargs):
        return None

    def Recv(self, *args, **kwargs):
        return None

    def Create_cart(self, dims, periods=None, reorder=False):
        """Serial cartesian topology: all dims are 1, the single rank is at the origin."""
        dims = [int(d) for d in dims]
        if any(d != 1 for d in dims):
            raise NotImplementedError("the offline single-rank stub only supports dims == 1")
        periods = [False] * len(dims) if periods is None else [bool(p) for p in periods]
        return _CartComm(dims, periods)


class _CartComm(_Comm):
    def __init__(self, dims, periods):
        self._dims = list(dims)
        self._periods = list(periods)

    def Get_topo(self):
        return (list(self._dims), list(self._periods), [0] * len(self._dims))

    def Get_coords(self, rank=0):
        if rank != 0:
            raise NotImplementedError("only rank 0 exists in the stub")
        return [0] * len(self._dims)

    def Get_dim(self):
        return len(self._dims)

    def Shift(self, direction, disp):
        return (0, 0)

    def Get_cart_rank(self, coords):
        """Single rank: every cartesian coordinate maps onto rank 0 (periodic wrap)."""
        return 0

    def Sendrecv(self, *args, **kwargs):
        return None


COMM_WORLD = _Comm()


class _SubarrayType:
    """Placeholder MPI derived datatype; never used for a real transfer (size 1)."""

    def __init__(self, sizes, subsizes, starts):
        self.sizes, self.subsizes, self.starts = tuple(sizes), tuple(subsizes), tuple(starts)

    def Free(self):
        return None

    def Commit(self):
        return None

    def Get_size(self):  # pragma: no cover - never exercised serially
        return 1


class _Datatype:
    def __init__(self, name="DATATYPE"):
        self.name = name

    def __getattr__(self, name):  # pragma: no cover - import-time convenience
        return _Datatype(name)

    def Create_subarray(self, sizes, subsizes, starts):
        return _SubarrayType(sizes, subsizes, starts)

    def __repr__(self):  # pragma: no cover - debugging convenience
        return f"<mpi datatype {self.name}>"


datatype = _Datatype()
DOUBLE = _Datatype("DOUBLE")
INT = _Datatype("INT")


def Compute_dims(dims, ndims):
    """Serial topology: every dimension is 1 (``mpi4py`` returns ``[1]*ndims``)."""
    if int(dims) != 1:
        raise NotImplementedError("the offline single-rank stub only supports comm.size == 1")
    return [1] * int(ndims)


class _Request:
    @staticmethod
    def Waitall(reqs):
        return None

    def Wait(self):
        return None


Request = _Request
REQUEST = _Request()


def Wtime():
    return _time.time()


def Init():
    return None


def Finalize():
    return None