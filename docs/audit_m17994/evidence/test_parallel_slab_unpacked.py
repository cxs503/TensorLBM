"""Regression tests for the unpacked 1-D y-slab halo (OSS MPI pattern transplant).

What is pinned here:
  * decomposition neutrality: NP = 2 and NP = 4 reproduce the NP = 1 field
    bit for bit, for all three halo conventions and for overlap on/off;
  * the unpacked block carries ``8/6`` the wire bytes of the selective block
    and pays zero staging memcpy;
  * the solver reproduces the analytic transverse (shear) wave to < 1 %.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tensorlbm.parallel import slab_unpacked as su  # noqa: E402

N = 32
STEPS = 40


def _run_serial() -> np.ndarray:
    s = su.SlabSolver(N, N, tau=0.8)
    s.init_shear_wave(0.04)
    for _ in range(STEPS):
        s.step()
    return s.macroscopic()[1]


def _run_ranks(size: int, halo: str, overlap: bool = True) -> np.ndarray:
    """Thread ranks over directed queue channels (same shim as the xval driver)."""
    import queue
    import threading

    pairs = {(i, j): queue.Queue() for i in range(size) for j in range(size) if i != j}
    out = {}

    def _rank(r: int) -> None:
        ny_local = N // size
        comm = su.ThreadPeers(
            r, size,
            {j: su.QueueChannel(pairs[(r, j)], readable=False) for j in range(size) if j != r},
            {j: su.QueueChannel(pairs[(j, r)], readable=True) for j in range(size) if j != r},
        )
        if halo == "unpacked":
            ex = su.UnpackedHaloExchange(comm, ny_local, N, selective=False)
        elif halo == "unpacked-selective":
            ex = su.UnpackedHaloExchange(comm, ny_local, N, selective=True)
        else:
            ex = su.PackedHaloExchange(comm, ny_local, N)
        solver = su.SlabSolver(ny_local, N, tau=0.8, comm=comm, exchange=ex, overlap=overlap)
        solver.init_shear_wave(0.04)
        for _ in range(STEPS):
            solver.step()
        out[r] = solver.macroscopic()[1]

    threads = [threading.Thread(target=_rank, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    return np.vstack([out[r] for r in range(size)])


@pytest.mark.parametrize("halo", ["unpacked", "unpacked-selective", "packed-selective"])
@pytest.mark.parametrize("size", [2, 4])
def test_decomposition_is_exact(halo: str, size: int) -> None:
    ref = _run_serial()
    got = _run_ranks(size, halo)
    assert got.shape == ref.shape
    np.testing.assert_array_equal(got, ref)  # bit-exact, not just close


@pytest.mark.parametrize("size", [2, 4])
def test_overlap_matches_blocking(size: int) -> None:
    a = _run_ranks(size, "unpacked", overlap=True)
    b = _run_ranks(size, "unpacked", overlap=False)
    np.testing.assert_array_equal(a, b)


class _NoopReq:
    def test(self):
        return True

    def wait(self):
        pass


class _NoopComm:
    rank, size = 0, 4

    def isend(self, buf, dest, tag=0):
        return _NoopReq()

    def irecv(self, buf, src, tag=0):
        return _NoopReq()


def _post(ex):
    f = np.zeros((9, 18, N))
    f[:, 1:-1, :] = 1.0
    ex.post(f, 0)
    return ex


def test_unpacked_block_pays_no_staging_and_sends_8_of_6() -> None:
    un = _post(su.UnpackedHaloExchange(_NoopComm(), 16, N, selective=False))
    sel = _post(su.UnpackedHaloExchange(_NoopComm(), 16, N, selective=True))
    pk = _post(su.PackedHaloExchange(_NoopComm(), 16, N))
    assert un.copies_bytes == 0          # zero copy: f[1:, row, :] is contiguous
    assert sel.copies_bytes > 0          # selecting directions forces a copy
    assert pk.copies_bytes > 0
    assert un.bytes_per_step() == 4 * 8 * N * 8
    assert sel.bytes_per_step() == 4 * 6 * N * 8
    assert un.bytes_per_step() / sel.bytes_per_step() == pytest.approx(8 / 6)
    assert sel.copies_bytes == pk.copies_bytes  # same payload, only the pack differs


def test_shear_wave_matches_analytic() -> None:
    s = su.SlabSolver(64, 64, tau=0.8)
    s.init_shear_wave(0.04)
    for _ in range(200):
        s.step()
    _, ux, uy = s.macroscopic()
    ana = su.shear_wave_analytic(64, 64, 200, s.nu, 0.04)
    err = np.sqrt(np.sum((ux - ana) ** 2)) / np.sqrt(np.sum(ana ** 2))
    assert err < 0.01
    assert np.abs(uy).max() < 1e-12