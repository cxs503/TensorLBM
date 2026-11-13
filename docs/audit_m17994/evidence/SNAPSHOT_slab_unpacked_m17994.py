"""1-D y-slab D2Q9 LBM with *unpacked* halo exchange and Testall overlap.

This module transplants the distributed pattern of the MIT-licensed
``lbm-principles-practice/code`` MPI variant
(``mpi-1d-slab-unpacked-halo-testall-overlap``) into TensorLBM:

* **1-D slab along y**, ranks wrap around (``rankp1 = (rank+1) % nprocs``).
* **direction-major layout** ``f[q, y, x]`` with ``f0`` kept as ``q = 0``, so
  the halo block of one face is ``(ndir-1) * NX`` contiguous float64 values
  (``f[1:, y, :]``).  It is therefore sent straight out of the population
  array: **no packing into a staging buffer** (that is what "unpacked" means
  here), in contrast with the staging-buffer + selective-direction scheme of
  :mod:`tensorlbm.domain_decomposition`.
* **MPI_Testall progress polling**: the halo request set is polled after every
  streamed interior row, so the communication overlaps the remaining
  collision/streaming work; only the two boundary rows have to wait.
* **one Allreduce per diagnostic window** over the global scalars
  (7 of them in the OSS code: mass, three momenta, kinetic energy, ...).

The transport is pluggable:

``SocketPeers``
    one UNIX ``socketpair`` per rank pair, ``sendall`` / ``recv_into`` issued
    from a background thread so the request handle can be polled
    (``Testall``) -- real multi-process execution, an *mpi4py shim*, not MPI.
``SerialComm``
    single rank; halo blocks are copied from the local array, so a NP=1 run is
    the bit-exact reference arm.

Everything is float64 numpy; the code is a verification instrument, not a
performance port.
"""

from __future__ import annotations

import queue
import socket
import threading
import time

import numpy as np

# --------------------------------------------------------------- D2Q9 lattice
CX = np.array([0, 1, -1, 0, 0, 1, -1, 1, -1], dtype=np.int64)
CY = np.array([0, 0, 0, 1, -1, 1, -1, -1, 1], dtype=np.int64)
C = tuple((int(CX[q]), int(CY[q])) for q in range(9))
W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4, dtype=np.float64)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7], dtype=np.int64)
#: directions whose y-component is non-zero -> the only ones that can need a
#: ghost row.  ``q = 1, 2`` never cross a y face, the OSS unpacked block sends
#: them anyway (8 instead of 6 rows per face).
CROSS_DIRS = tuple(q for q in range(9) if CY[q] != 0)
NONREST = tuple(range(1, 9))


def equilibrium2d(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium, shape ``(9, ...)``."""
    cu = CX.reshape((9,) + (1,) * rho.ndim) * ux + CY.reshape((9,) + (1,) * rho.ndim) * uy
    usq = ux * ux + uy * uy
    return W.reshape((9,) + (1,) * rho.ndim) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def uniform_state(ny_local: int, nx: int, rho0: float = 1.0) -> np.ndarray:
    """Neutral populations on ``ny_local + 2`` rows (2 ghost rows)."""
    rho = np.full((ny_local + 2, nx), rho0, dtype=np.float64)
    return equilibrium2d(rho, np.zeros_like(rho), np.zeros_like(rho))


def shear_wave_analytic(ny: int, nx: int, t: float, nu: float, u0: float) -> np.ndarray:
    """Transverse (shear) wave ``ux = u0 cos(k y) exp(-nu k^2 t)``, ``uy = 0``.

    Divergence-free (``div u = 0``), so the linear acoustic mode is not
    excited: this is the cleanest analytic anchor for a y-slab decomposition,
    because the field varies exactly along the decomposed axis while staying
    periodic in both directions.
    """
    k = 2.0 * np.pi / ny
    y = np.arange(ny, dtype=np.float64)[:, None]
    return u0 * np.cos(k * y) * np.exp(-nu * k * k * t) * np.ones((1, nx))


def tg_analytic(nx: int, ny: int, t: float, nu: float, u0: float) -> tuple[np.ndarray, np.ndarray]:
    """Taylor-Green vortex velocity field (periodic in x and y).

    ``ux = u0 cos(kx) cos(ky) exp(-2 nu k^2 t)``,
    ``uy = -u0 sin(kx) sin(ky) exp(-2 nu k^2 t)`` with ``k = 2 pi / L`` and one
    wavelength per box (``L = nx = ny`` is expected; the argument uses ``nx``).
    """
    k = 2.0 * np.pi / nx
    x = np.arange(nx, dtype=np.float64)[None, :]
    y = np.arange(ny, dtype=np.float64)[:, None]
    decay = np.exp(-2.0 * nu * k * k * t)
    return u0 * np.cos(k * x) * np.cos(k * y) * decay, -u0 * np.sin(k * x) * np.sin(k * y) * decay


# ------------------------------------------------------------------- transport
class _Req:
    """Request handle; ``test()`` is the shim's ``MPI_Test``."""

    __slots__ = ("event", "buf")

    def __init__(self, event: threading.Event, buf) -> None:
        self.event = event
        self.buf = buf

    def test(self) -> bool:
        return self.event.is_set()

    def wait(self) -> None:
        self.event.wait()


class _Channel:
    """One *directed* socket with a FIFO worker thread.

    Sends and receives posted on the same channel are executed in posting
    order, which is what keeps the two byte streams aligned without tags (an
    ordered stream socket is the shim's substitute for MPI tags).  Each
    direction owns its own socketpair, so a blocked send can never stall a
    receive on the same rank.
    """

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.q: "queue.Queue" = queue.Queue()
        self.nbytes = 0
        self.nmsg = 0
        th = threading.Thread(target=self._run, daemon=True)
        th.start()

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            kind, buf, ev = item
            try:
                if kind == "s":
                    self.sock.sendall(memoryview(np.ascontiguousarray(buf)).cast("B"))
                else:
                    self.sock.recv_into(memoryview(buf).cast("B"), memoryview(buf).nbytes)
            finally:
                ev.set()

    def post(self, kind: str, buf) -> _Req:
        ev = threading.Event()
        n = int(memoryview(np.ascontiguousarray(buf)).nbytes) if kind == "s" else int(buf.nbytes)
        self.nbytes += n
        self.nmsg += 1
        self.q.put((kind, buf, ev))
        return _Req(ev, buf)

    def close(self) -> None:
        self.q.put(None)
        try:
            self.sock.close()
        except OSError:
            pass


class QueueChannel:
    """In-process directed channel with the same non-blocking request API.

    No MPI/mpi4py stack is installed in the verification image, so the ranks
    run as threads and each direction owns a ``queue.Queue``.  The channel
    thread pairs payloads and receive registrations in FIFO order (the shim's
    substitute for MPI tags) and completes the request, so ``Testall`` polling
    and the overlap are measured on a real interleaving.  The payload is
    copied, i.e. the shim pays exactly the staging memcpy that the unpacked
    scheme avoids on a real wire (reported as a comparison limit).
    """

    def __init__(self, q: "queue.Queue", readable: bool = True) -> None:
        self.q = q
        self.nbytes = 0
        self.nmsg = 0
        self.readable = readable
        # only the receiving end of a direction runs the pairing thread,
        # otherwise the sender would consume its own payloads
        self._th = threading.Thread(target=self._run, daemon=True) if readable else None
        if self._th is not None:
            self._th.start()

    def _run(self) -> None:
        pend_pay = []
        pend_recv = []
        while True:
            try:
                item = self.q.get(timeout=0.05)
            except queue.Empty:
                item = None
            if item is not None:
                if item[0] == "pay":
                    pend_pay.append(item)
                elif item[0] == "rec":
                    pend_recv.append(item)
                else:  # stop
                    return
            while pend_pay and pend_recv:
                _, payload, ev_s = pend_pay.pop(0)
                _, buf, ev_r = pend_recv.pop(0)
                buf[...] = payload
                ev_s.set()
                ev_r.set()

    def post_send(self, buf) -> "_Req":
        ev = threading.Event()
        payload = np.array(buf, copy=True)
        self.nbytes += payload.nbytes
        self.nmsg += 1
        self.q.put(("pay", payload, ev))
        return _Req(ev, buf)

    def post_recv(self, buf) -> "_Req":
        ev = threading.Event()
        self.nbytes += buf.nbytes
        self.nmsg += 1
        self.q.put(("rec", buf, ev))
        return _Req(ev, buf)

    def post(self, kind: str, buf) -> "_Req":
        return self.post_send(buf) if kind == "s" else self.post_recv(buf)

    def close(self) -> None:
        self.q.put(("stop", None, None))


class ThreadPeers:
    """``SocketPeers`` twin over :class:`QueueChannel` (no OS sockets)."""

    def __init__(self, rank: int, size: int, out_ch: dict, in_ch: dict) -> None:
        self.rank = rank
        self.size = size
        self.out = out_ch
        self.inn = in_ch
        self.n_messages = 0
        self.n_testall = 0

    def isend(self, buf, dest: int, tag: int = 0) -> "_Req":
        self.n_messages += 1
        return self.out[dest].post_send(buf)

    def irecv(self, buf, src: int, tag: int = 0) -> "_Req":
        self.n_messages += 1
        return self.inn[src].post_recv(buf)

    def testall(self, reqs) -> bool:
        self.n_testall += 1
        return all(r.test() for r in reqs)

    @staticmethod
    def waitall(reqs) -> None:
        for r in reqs:
            r.wait()

    def barrier(self) -> None:
        if self.size == 1:
            return
        nxt = (self.rank + 1) % self.size
        prv = (self.rank - 1) % self.size
        tok = np.zeros(1, dtype=np.float64)
        if self.rank == 0:
            self.out[nxt].post_send(tok).wait()
            self.inn[prv].post_recv(np.zeros(1)).wait()
        else:
            self.inn[prv].post_recv(np.zeros(1)).wait()
            self.out[nxt].post_send(tok).wait()

    def allreduce_sum(self, values) -> list[float]:
        arr = np.asarray(values, dtype=np.float64)
        n = arr.size
        if self.size == 1:
            return arr.tolist()
        if self.rank == 0:
            total = arr.copy()
            for src in range(1, self.size):
                got = np.empty(n, dtype=np.float64)
                self.inn[src].post_recv(got).wait()
                total += got
            for dst in range(1, self.size):
                self.out[dst].post_send(total).wait()
            return total.tolist()
        self.out[0].post_send(arr).wait()
        got = np.empty(n, dtype=np.float64)
        self.inn[0].post_recv(got).wait()
        return got.tolist()

    def close(self) -> None:
        for ch in list(self.out.values()) + list(self.inn.values()):
            ch.close()


class SerialComm:
    """Single-rank stand-in: no wire at all (see ``UnpackedHaloExchange``)."""

    rank = 0
    size = 1
    n_messages = 0
    n_testall = 0

    def isend(self, buf, dest: int, tag: int = 0) -> _Req:
        raise RuntimeError("SerialComm has no peers")

    def irecv(self, buf, src: int, tag: int = 0) -> _Req:
        raise RuntimeError("SerialComm has no peers")

    @staticmethod
    def testall(reqs) -> bool:
        return all(r.test() for r in reqs)

    @staticmethod
    def waitall(reqs) -> None:
        for r in reqs:
            r.wait()


class SocketPeers:
    """mpi4py-flavoured shim over UNIX socketpairs (one per direction).

    ``isend``/``irecv`` are non-blocking from the caller's point of view: the
    payload is handed to the peer's channel thread and a pollable request is
    returned, which is what makes ``Testall`` progress polling and the
    overlap measurable at all.  This is a *shim*, not MPI: no derived
    datatypes, no real network, no rendezvous protocol.
    """

    def __init__(self, rank: int, size: int, out_ch: dict, in_ch: dict) -> None:
        self.rank = rank
        self.size = size
        self.out = out_ch   # peer -> _Channel on which we send
        self.inn = in_ch    # peer -> _Channel on which we receive
        self.n_messages = 0
        self.n_testall = 0

    def isend(self, buf, dest: int, tag: int = 0) -> _Req:
        self.n_messages += 1
        return self.out[dest].post("s", buf)

    def irecv(self, buf, src: int, tag: int = 0) -> _Req:
        self.n_messages += 1
        return self.inn[src].post("r", buf)

    def testall(self, reqs) -> bool:
        self.n_testall += 1
        return all(r.test() for r in reqs)

    def barrier(self) -> None:
        if self.size == 1:
            return
        tok = bytearray(8)
        nxt = (self.rank + 1) % self.size
        prv = (self.rank - 1) % self.size
        if self.rank == 0:
            self.out[nxt].sock.sendall(b"\x01" * 8)
            self.inn[prv].sock.recv_into(tok, 8)
        else:
            self.inn[prv].sock.recv_into(tok, 8)
            self.out[nxt].sock.sendall(tok)

    def allreduce_sum(self, values) -> list[float]:
        """Allreduce of a short float64 vector: rank-0 gather then broadcast."""
        arr = np.asarray(values, dtype=np.float64)
        n = arr.size
        if self.size == 1:
            return arr.tolist()
        if self.rank == 0:
            total = arr.copy()
            for src in range(1, self.size):
                got = np.empty(n, dtype=np.float64)
                self.inn[src].sock.recv_into(memoryview(got).cast("B"), n * 8)
                total += got
            for dst in range(1, self.size):
                self.out[dst].sock.sendall(memoryview(total).cast("B"))
            return total.tolist()
        self.out[0].sock.sendall(memoryview(arr).cast("B"))
        got = np.empty(n, dtype=np.float64)
        self.inn[0].sock.recv_into(memoryview(got).cast("B"), n * 8)
        return got.tolist()

    def close(self) -> None:
        for ch in list(self.out.values()) + list(self.inn.values()):
            ch.close()


# --------------------------------------------------------------- halo exchange
class UnpackedHaloExchange:
    """Halo blocks sent straight out of ``f[1:, row, :]`` (no staging copy).

    Both faces of the slab use the *full* non-rest block (``ndir - 1`` rows),
    which is exactly the OSS convention: contiguous, zero packing cost, but
    ``2 / 6`` more wire bytes than exchanging only the crossing directions.
    """

    kind = "unpacked"

    def __init__(self, comm, ny_local: int, nx: int, selective: bool = False) -> None:
        self.comm = comm
        self.ny_local = ny_local
        self.nx = nx
        self.selective = selective
        self.dirs = CROSS_DIRS if selective else NONREST
        self.block = np.empty((len(self.dirs), nx), dtype=np.float64)
        self.ghost_lo = np.zeros((len(self.dirs), nx), dtype=np.float64)
        self.ghost_hi = np.zeros((len(self.dirs), nx), dtype=np.float64)
        self.copies_bytes = 0  # bytes moved by an explicit staging memcpy

    # ---- send side ------------------------------------------------------
    def post(self, f_post: np.ndarray, rank: int):
        comm = self.comm
        if comm.size == 1:  # periodic self-wrap: no wire at all
            sel = np.asarray(CROSS_DIRS)
            f_post[sel, 0, :] = f_post[sel, self.ny_local, :]
            f_post[sel, self.ny_local + 1, :] = f_post[sel, 1, :]
            return []
        np1 = (rank + 1) % comm.size
        nm1 = (rank - 1) % comm.size
        if self.selective:
            # staging copy required: take only the crossing directions
            sel = np.asarray(self.dirs)
            send_hi = f_post[sel, self.ny_local, :]
            send_lo = f_post[sel, 1, :]
            self.copies_bytes += send_hi.nbytes + send_lo.nbytes
        else:
            # zero copy: f[1:, row, :] is already contiguous
            send_hi = f_post[1:, self.ny_local, :]
            send_lo = f_post[1:, 1, :]
        # For size == 2 both faces share one directed channel, so the receive
        # order must mirror the peer's send order: the peer sends its top row
        # first, and that row is exactly our *bottom* ghost row.
        return [
            comm.isend(send_hi, np1, 1),   # our top row -> neighbour above
            comm.isend(send_lo, nm1, 2),   # our bottom row -> neighbour below
            comm.irecv(self.ghost_lo, nm1, 1),   # peer's top row (its send #1)
            comm.irecv(self.ghost_hi, np1, 2),   # peer's bottom row (its send #2)
        ]

    def complete(self, f_post: np.ndarray) -> None:
        sel = np.asarray(self.dirs)
        f_post[sel, 0, :] = self.ghost_lo
        f_post[sel, self.ny_local + 1, :] = self.ghost_hi

    def bytes_per_step(self) -> int:
        return 4 * len(self.dirs) * self.nx * 8

    def messages_per_step(self) -> int:
        return 4


class PackedHaloExchange(UnpackedHaloExchange):
    """In-repo style control: selective directions *and* staging buffers.

    Mirrors :class:`tensorlbm.domain_decomposition.HaloExchange` semantics
    (only the populations that cross the face are exchanged, but they are
    packed into contiguous staging buffers, ``3.8x`` fewer wire bytes for
    D3Q19 / ``8/6`` here, at the price of two extra memcpys per face).
    """

    kind = "packed-selective"

    def __init__(self, comm, ny_local: int, nx: int) -> None:
        super().__init__(comm, ny_local, nx, selective=True)

    def post(self, f_post: np.ndarray, rank: int):
        comm = self.comm
        if comm.size == 1:  # periodic self-wrap: no wire at all
            sel = np.asarray(CROSS_DIRS)
            f_post[sel, 0, :] = f_post[sel, self.ny_local, :]
            f_post[sel, self.ny_local + 1, :] = f_post[sel, 1, :]
            return []
        np1 = (rank + 1) % comm.size
        nm1 = (rank - 1) % comm.size
        sel = np.asarray(self.dirs)
        self.send_hi = np.ascontiguousarray(f_post[sel, self.ny_local, :])
        self.send_lo = np.ascontiguousarray(f_post[sel, 1, :])
        self.copies_bytes += self.send_hi.nbytes + self.send_lo.nbytes
        return [
            comm.isend(self.send_hi, np1, 1),
            comm.isend(self.send_lo, nm1, 2),
            comm.irecv(self.ghost_lo, nm1, 1),
            comm.irecv(self.ghost_hi, np1, 2),
        ]


# ---------------------------------------------------------------------- solver
class SlabSolver:
    """Periodic D2Q9 BGK solver on one y-slab of a ``NY x NX`` periodic box.

    ``ny_local`` owned rows; ``f`` has two extra ghost rows (index 0 and
    ``ny_local + 1``) that hold the *post-collision* populations of the
    neighbouring ranks, as required by a pull-streaming scheme.
    """

    def __init__(self, ny_local: int, nx: int, tau: float = 0.8, comm=None,
                 exchange=None, overlap: bool = True, poll_every: int = 1) -> None:
        self.ny_local = ny_local
        self.nx = nx
        self.tau = tau
        self.nu = (tau - 0.5) / 3.0
        self.comm = comm if comm is not None else SerialComm()
        self.exchange = exchange if exchange is not None else UnpackedHaloExchange(self.comm, ny_local, nx)
        self.overlap = overlap
        self.poll_every = poll_every
        self.f = uniform_state(ny_local, nx)
        self.post = np.zeros_like(self.f)
        self.steps = 0
        self.poll_total = 0
        self.poll_skipped = 0  # polls avoided because comm finished early
        self.n_wait = 0

    # ---- init -----------------------------------------------------------
    def init_taylor_green(self, u0: float) -> None:
        k = 2.0 * np.pi / self.nx
        y0 = self.comm.rank * self.ny_local
        x = np.arange(self.nx, dtype=np.float64)[None, :]
        y = (y0 + np.arange(self.ny_local, dtype=np.float64))[:, None]  # physical y of owned row
        rho = np.ones((self.ny_local + 2, self.nx), dtype=np.float64)
        ux = np.zeros_like(rho)
        uy = np.zeros_like(rho)
        ux[1:-1] = u0 * np.cos(k * x) * np.cos(k * y)
        uy[1:-1] = -u0 * np.sin(k * x) * np.sin(k * y)
        self.f = equilibrium2d(rho, ux, uy)

    def init_shear_wave(self, u0: float) -> None:
        k = 2.0 * np.pi / (self.ny_local * self.comm.size)
        y0 = self.comm.rank * self.ny_local
        y = (y0 + np.arange(self.ny_local, dtype=np.float64))[:, None]
        rho = np.ones((self.ny_local + 2, self.nx), dtype=np.float64)
        ux = np.zeros_like(rho)
        uy = np.zeros_like(rho)
        ux[1:-1] = u0 * np.cos(k * y)
        self.f = equilibrium2d(rho, ux, uy)

    # ---- diagnostics ----------------------------------------------------
    def macroscopic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = self.f[:, 1 : self.ny_local + 1, :]
        rho = f.sum(axis=0)
        ux = np.tensordot(CX.astype(np.float64), f, axes=(0, 0)) / rho
        uy = np.tensordot(CY.astype(np.float64), f, axes=(0, 0)) / rho
        return rho, ux, uy

    def local_scalars(self) -> list[float]:
        """Seven scalars, the OSS Allreduce payload."""
        rho, ux, uy = self.macroscopic()
        ke = float(np.sum(rho * (ux * ux + uy * uy)))
        return [
            float(np.sum(rho)),
            float(np.sum(rho * ux)),
            float(np.sum(rho * uy)),
            float(np.sum(np.abs(ux))),
            float(np.sum(np.abs(uy))),
            ke,
            float(np.max(np.sqrt(ux * ux + uy * uy))),
        ]

    def global_scalars(self) -> list[float]:
        return self.comm.allreduce_sum(self.local_scalars())

    # ---- one time step --------------------------------------------------
    def step(self) -> None:
        ny = self.ny_local
        f = self.f
        post = self.post
        # 1) collide every owned row (BGK)
        r = f[:, 1 : ny + 1, :]
        rho = r.sum(axis=0)
        ux = np.tensordot(CX.astype(np.float64), r, axes=(0, 0)) / rho
        uy = np.tensordot(CY.astype(np.float64), r, axes=(0, 0)) / rho
        feq = equilibrium2d(rho, ux, uy)
        post[:, 1 : ny + 1, :] = r - (r - feq) / self.tau

        # 2) post the halo messages right after collision (unpacked: zero copy)
        # (size == 1: the exchange performs the periodic self-wrap in memory)
        reqs = self.exchange.post(post, self.comm.rank)

        # 3) overlap: stream interior rows in chunks of ``poll_every`` rows and
        #    poll the request set (MPI_Testall) after every chunk
        chunk = max(1, self.poll_every)
        row = 2
        if reqs and self.overlap:
            while row < ny:
                hi = min(row + chunk, ny)
                self._stream_chunk(row, hi)
                self.poll_total += 1
                row = hi
                if self.comm.testall(reqs):
                    self.poll_skipped += max(0, (ny - hi) // chunk)
                    break
        # 4) blocking wait for the halo, then unpack it into the ghost rows
        if reqs:
            self.n_wait += 1
            self.comm.waitall(reqs)
            self.exchange.complete(post)
        if row < ny:  # communication finished early (or overlap disabled)
            self._stream_chunk(row, ny)
        # 5) the two rows that need ghost data
        self._stream_chunk(1, 2)
        if ny > 1:
            self._stream_chunk(ny, ny + 1)

        # no buffer swap: ``f`` now holds the streamed state and ``post`` stays
        # the post-collision scratch (its ghost rows are refilled next step)
        self.steps += 1

    def _stream_chunk(self, lo: int, hi: int) -> None:
        """Pull-stream owned rows ``[lo, hi)`` of the ghost-padded array."""
        if hi <= lo:
            return
        dst = self.f
        src = self.post
        for q in range(9):
            cy = int(CY[q])
            cx = int(CX[q])
            block = src[q, lo - cy : hi - cy, :]
            if cx:
                block = np.roll(block, cx, axis=1)
            dst[q, lo:hi, :] = block

