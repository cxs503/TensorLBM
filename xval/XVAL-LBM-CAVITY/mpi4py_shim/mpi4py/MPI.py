"""Single-rank (size=1, rank=0) subset of mpi4py.MPI.

Used only to run pylbm in serial when no MPI toolchain is installed.
Anything that would need real communication is a no-op where that is exact
for one rank, and raises NotImplementedError otherwise.
"""
import time as _time

SUM = "SUM"
MAX = "MAX"
MIN = "MIN"
class _DataType:
    """stand-in datatype; subarray types are only built for multi-rank runs"""

    def __init__(self, name):
        self.name = name

    def Create_subarray(self, sizes, subsizes, starts, order=None):
        return _SubarrayType(tuple(sizes), tuple(subsizes), tuple(starts), order)

    def __repr__(self):
        return f"<shim datatype {self.name}>"


class _SubarrayType:
    """committed-looking subarray type; never actually transferred in serial"""

    def __init__(self, sizes, subsizes, starts, order=None):
        self.sizes, self.subsizes, self.starts, self.order = sizes, subsizes, starts, order
        self.committed = False

    def Commit(self):
        self.committed = True

    def Free(self):
        self.committed = False


DOUBLE = _DataType("DOUBLE")


class _Comm:
    rank = 0
    size = 1

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def allreduce(self, sendobj, op=None):
        return sendobj

    def Allreduce(self, sendbuf, recvbuf, op=None):
        if isinstance(sendbuf, list):
            recvbuf[...] = sendbuf[0]
        else:
            recvbuf[...] = sendbuf
        return recvbuf

    def bcast(self, obj, root=0):
        return obj

    def Barrier(self):
        return None

    def gather(self, obj, root=0):
        return [obj]

    def Get_coords(self):
        return [0]

    def Get_dim(self):
        return 1

    # -- self-communication: with one rank every peer is rank 0, so an
    # -- Irecv/Isend pair is exactly a copy from the send window into the
    # -- receive window of the same array (the windows are disjoint halo
    # -- strips).  Request.Waitall below performs those copies in order.
    def Irecv(self, buf, source=0, tag=0):
        return _Request(kind="recv", buf=buf)

    def Isend(self, buf, dest=0, tag=0):
        return _Request(kind="send", buf=buf)

    def __getattr__(self, name):
        raise NotImplementedError(f"single-rank mpi4py shim: {name} not implemented")



class _CartComm(_Comm):
    """cartesian communicator of a 1-rank world"""

    def __init__(self, dims, periods):
        self._dims = list(dims)
        self._periods = list(periods)

    def Get_dim(self):
        return len(self._dims)

    def Get_topo(self):
        return (self._dims, self._periods, [0] * len(self._dims))

    def Get_coords(self, rank=0):
        return [0] * len(self._dims)

    def Shift(self, direction, disp):
        return (0, 0)

    def Get_cart_rank(self, coords):
        # one rank: every (periodic) coordinate maps to rank 0
        return 0


def _create_cart(self, dims, periods=None, reorder=False):
    return _CartComm(dims, periods if periods is not None else [False] * len(dims))


_Comm.Create_cart = _create_cart


class _Request:
    """deferred self-transfer for one rank"""

    def __init__(self, kind, buf):
        self.kind = kind
        self.buf = buf

    @staticmethod
    def Waitall(reqs):
        recvs = [r for r in reqs if r.kind == "recv"]
        sends = [r for r in reqs if r.kind == "send"]
        for r, s_ in zip(recvs, sends):
            array_r, sub_r = r.buf
            array_s, sub_s = s_.buf
            array_r[_window(sub_r)] = array_s[_window(sub_s)]

    def Wait(self):
        return None


def _window(sub):
    """numpy slice for a (_SubarrayType) window of the storage array"""
    return tuple(
        slice(int(st), int(st + sz))
        for st, sz in zip(sub.starts, sub.subsizes)
    )


class Request:
    """namespace mirroring mpi4py's Request class"""

    Waitall = staticmethod(lambda reqs: _Request.Waitall(reqs))
    Wait = staticmethod(lambda req=None: None)


COMM_WORLD = _Comm()


def Wtime():
    return _time.perf_counter()


def Compute_dims(nnodes, ndims):
    dims = [1] * ndims
    dims[0] = nnodes
    return dims
