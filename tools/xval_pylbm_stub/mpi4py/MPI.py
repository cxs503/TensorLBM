"""Single-rank no-op ``mpi4py.MPI`` used only to import ``pylbm`` offline.

See :mod:`mpi4py` (this stub package) for the rationale and the caveat that it
is not an MPI implementation.
"""


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

    def allreduce(self, value, op=None):
        return value

    def reduce(self, value, op=None, root=0):
        return value

    def scatter(self, value, root=0):
        return value

    def send(self, *args, **kwargs):
        return None

    def recv(self, *args, **kwargs):
        return None


COMM_WORLD = _Comm()


class _Datatype:
    def __getattr__(self, name):  # pragma: no cover - import-time convenience
        return None


datatype = _Datatype()


def Init():
    return None


def Finalize():
    return None