"""Serial (1-rank) subset of mpi4py.MPI, sufficient to run pylbm's serial
NumPy backend.  Only the surface pylbm touches is provided; every collective
is the identity on a single rank.  Used so that a reference-comparison run of
the upstream OSS code does not need an MPI installation.
"""
import time
import numpy as np

class _Token:
    """Opaque stand-in for an MPI op/datatype handle."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<mpi-shim {self.name}>"

    def Create_subarray(self, sizes, subsizes, sstart, order=None):
        tok = _Token(f"{self.name}<sub>")
        tok.sizes = list(sizes)
        tok.subsizes = list(subsizes)
        tok.sstart = list(sstart)
        return tok

    def Create_vector(self, count, blocklength, stride):
        return self

    def Commit(self):
        return self

    def Free(self):
        return None


SUM = _Token("SUM")
DOUBLE = _Token("DOUBLE")
MAX = _Token("MAX")
MIN = _Token("MIN")


def Wtime():
    return time.time()


def Compute_dims(nprocs, ndims):
    return [1] * ndims


class _Comm:
    rank = 0
    size = 1

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Barrier(self):
        return None

    def allreduce(self, sendobj=None, op=None):
        return sendobj

    def bcast(self, obj, root=0):
        return obj

    def Create_cart(self, dims, periods=None, reorder=False):
        self.dims = list(dims)
        self.periods = list(periods) if periods is not None else [False] * len(dims)
        return self

    def Get_topo(self):
        return list(self.dims), list(self.periods), [0] * len(self.dims)

    def Get_coords(self, rank):
        return [0] * len(getattr(self, "dims", [0]))

    def Get_cart_rank(self, coords):
        return 0

    def Shift(self, direction, disp):
        return (0, 0)

    def Send(self, buf, dest=0, tag=0):
        return None

    def Recv(self, buf, source=0, tag=0):
        return None

    def Irecv(self, buf, source=0, tag=0):
        return Request("recv", buf[0], buf[1], tag)

    def Isend(self, buf, dest=0, tag=0):
        return Request("send", buf[0], buf[1], tag)

    def Sub(self, sub):
        return self

    def Split(self, color=0, key=0):
        return self


COMM_WORLD = _Comm()
COMM_SELF = _Comm()


def _slices(type_token):
    """Interior slice of a subarray datatype, in the array's memory order."""
    return tuple(
        slice(int(type_token.sstart[i]), int(type_token.sstart[i]) + int(type_token.subsizes[i]))
        for i in range(len(type_token.subsizes))
    )


class Request:
    """Non-blocking request of the 1-rank self-communication emulation.

    A single-rank MPI job talks to itself: sends are matched to receives by
    tag and completed on ``Waitall`` by copying the described subarrays
    within the (shared) buffer.  Without this, periodic halo exchange is
    silently skipped and the periodic direction is wrong.
    """

    _pending = []

    def __init__(self, kind="done", array=None, type_token=None, tag=0):
        self.kind = kind
        self.array = array
        self.type_token = type_token
        self.tag = tag
        if kind != "done":
            Request._pending.append(self)

    def Wait(self):
        Request.Waitall([self])

    @staticmethod
    def Waitall(requests):
        sends = {r.tag: r for r in Request._pending if r.kind == "send"}
        recvs = [r for r in Request._pending if r.kind == "recv"]
        # snapshot the payloads first: a later copy must not read data that
        # an earlier copy of the same exchange already overwrote
        payload = {tag: (r.array[_slices(r.type_token)].copy(), r) for tag, r in sends.items()}
        for r in recvs:
            if r.tag in payload:
                data, sender = payload[r.tag]
                dest = r.array[_slices(r.type_token)]
                if dest.shape == data.shape:
                    dest[...] = data
        Request._pending = [r for r in Request._pending if r not in requests]
