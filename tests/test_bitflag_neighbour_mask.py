"""XVAL-LBM-BITFLAG-NEIGHBOUR-MASK: operator equivalence and physics anchors.

The tests pin three things:

1. **Operator equivalence.** The per-direction neighbour mask reproduces the
   OSS-faithful scan sweep (``lbm/streamcollide.py:44-92``) bit-for-bit, in both
   the scalar and the vectorised implementation, with and without a moving wall.
2. **Mask semantics.** The mask bit set == the OSS bounce-back branch condition
   (closed neighbour or out-of-domain link), verified cell by cell.
3. **Physics unchanged.** The solver reproduces the analytic plane-Poiseuille
   parabola and the analytic Couette line within the predeclared 3 % band, in
   both boundary modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# the tests must exercise the working tree, not an installed copy of tensorlbm
# (the editable install may point at another worktree; purge and re-resolve)
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
for _name in [k for k in list(sys.modules) if k == "tensorlbm" or k.startswith("tensorlbm.")]:
    del sys.modules[_name]

from tensorlbm.bitflag_d2q9 import (
    BitflagD2Q9,
    couette_profile,
    make_channel,
    make_cylinder_channel,
)
from tensorlbm.bitflag_neighbour_mask import (
    D2Q9_C,
    StreamStats,
    mask_stats,
    neighbour_mask,
    open_cells,
    oss_border_guard_wraps,
    shifted,
    stream_mask_scalar,
    stream_mask_vector,
    stream_scan_scalar,
    stream_scan_vector,
)


def test_shifted_semantics():
    a = np.arange(12.0).reshape(3, 4)
    out = shifted(a, 1, 0, fill=-1.0)
    assert np.array_equal(out[:2], a[1:])
    assert np.all(out[2] == -1.0)
    out = shifted(a, 0, -1, fill=-1.0)
    assert np.array_equal(out[:, 1:], a[:, :3])
    assert np.all(out[:, 0] == -1.0)


def test_mask_equals_oss_branch_condition():
    """Bit (i-1) set <=> the OSS pull for direction i would bounce."""
    ct = make_cylinder_channel(nx=32, ny=20, centre=(10.5, 10.0), radius=3.0)
    mask = neighbour_mask(ct)
    op = open_cells(ct)
    for i in range(1, 9):
        cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
        nb_open = shifted(op, -cx, -cy, fill=False)
        bit = (mask & np.uint8(1 << (i - 1))) != 0
        # only open nodes are ever streamed, so only they need to agree
        assert np.array_equal(bit[op], (~nb_open)[op])


def test_mask_kernels_match_scan_kernels():
    rng = np.random.default_rng(7)
    ct = make_cylinder_channel(nx=28, ny=18, centre=(9.0, 9.0), radius=2.5)
    mask = neighbour_mask(ct)
    op = open_cells(ct)[None]
    f = rng.random((9, 28, 18))

    def cmp(a, b):
        return np.array_equal(np.where(op, a, 0.0), np.where(op, b, 0.0))

    assert cmp(stream_mask_vector(f, ct, mask), stream_scan_vector(f, ct))
    assert cmp(stream_mask_scalar(f, ct, mask), stream_scan_scalar(f, ct))
    assert cmp(stream_mask_vector(f, ct, mask), stream_mask_scalar(f, ct))

    uw = np.zeros((28, 18))
    uw[:, 9:] = 0.05
    wall_dot = np.stack(
        [np.array([0, 0, 0, -1, -1, -1, 1, 1, 1])[i] * uw for i in range(9)]
    )
    assert cmp(
        stream_mask_vector(f, ct, mask, wall_velocity=wall_dot),
        stream_scan_vector(f, ct, wall_velocity=wall_dot),
    )


def test_interior_nodes_need_no_direction_decision():
    ct = make_cylinder_channel(nx=40, ny=24, centre=(12.5, 12.0), radius=4.0)
    mask = neighbour_mask(ct)
    rng = np.random.default_rng(11)
    f = rng.random((9, 40, 24))
    st_scan, st_mask = StreamStats(), StreamStats()
    stream_scan_scalar(f, ct, stats=st_scan)
    stream_mask_scalar(f, ct, mask, stats=st_mask)
    stats = mask_stats(ct)
    assert st_scan.neighbour_type_tests == 8 * stats.open_nodes
    assert st_mask.neighbour_type_tests == 0
    assert st_mask.mask_word_tests == stats.open_nodes
    assert st_mask.bit_tests == 8 * stats.open_nodes
    assert st_scan.decision_count > 2.0 * st_mask.decision_count
    # identical physical work underneath
    assert st_scan.bounces == st_mask.bounces
    assert st_scan.streams == st_mask.streams


def test_oss_border_guard_does_not_catch_negative_index():
    """``0 < ix_next >= nx`` (lbm/streamcollide.py:76-78) misses ``ix_next < 0``."""
    assert oss_border_guard_wraps(40, 3, (40, 20)) is True  # upper bound caught
    assert oss_border_guard_wraps(-1, 3, (40, 20)) is False  # negative -> wraps
    assert oss_border_guard_wraps(0, 3, (40, 20)) is False


@pytest.mark.parametrize("mode", ["mask", "scan"])
def test_poiseuille_reproduces_analytic_parabola(mode):
    nx, ny, tau, u_max = 48, 16, 0.8, 0.02
    h = ny - 1.0
    prof = np.zeros(ny)
    prof[1:-1] = u_max * 4.0 * ((np.arange(1, ny - 1) - 0.5) * (h - (np.arange(1, ny - 1) - 0.5))) / h**2
    s = BitflagD2Q9(
        make_channel(nx, ny, inlet=True, outlet=True),
        tau=tau,
        mode=mode,
        inlet_profile=prof,
        outlet_mode="zero_gradient_all",
    )
    for _ in range(20000):
        s.step()
    rho, ux, uy = s.macros()
    got = ux[nx // 2, 1:-1]
    err = 100.0 * np.abs(got - prof[1:-1]).max() / prof.max()
    assert err < 3.0, f"{mode} Poiseuille error {err:.2f}%"
    assert np.abs(uy[1:-1, 1:-1]).max() < 0.05 * u_max


@pytest.mark.parametrize("mode", ["mask", "scan"])
def test_couette_reproduces_analytic_line(mode):
    nx, ny, u_wall = 24, 14, 0.02
    h = ny - 1.0
    prof = np.zeros(ny)
    prof[1:-1] = u_wall * (np.arange(1, ny - 1) - 0.5) / h
    uw = np.zeros((nx, ny))
    uw[:, ny - 1] = u_wall
    s = BitflagD2Q9(
        make_channel(nx, ny, inlet=True, outlet=True),
        tau=0.8,
        mode=mode,
        inlet_profile=prof,
        u_wall_x=uw,
        outlet_mode="zero_gradient_all",
    )
    for _ in range(12000):
        s.step()
    _rho, ux, _uy = s.macros()
    y, ana = couette_profile(ny, u_wall)
    got = ux[:-1, nx // 2][1 : ny - 1]
    err = 100.0 * np.abs(got - ana).max() / ana.max()
    assert err < 3.0, f"{mode} Couette error {err:.2f}%"


def test_mask_and_scan_solvers_agree_bit_for_bit():
    nx, ny = 28, 14
    h = ny - 1.0
    prof = np.zeros(ny)
    prof[1:-1] = 0.02 * 4.0 * ((np.arange(1, ny - 1) - 0.5) * (h - (np.arange(1, ny - 1) - 0.5))) / h**2
    sol = {}
    for mode in ("mask", "scan"):
        s = BitflagD2Q9(
            make_channel(nx, ny, inlet=True, outlet=True),
            tau=0.8,
            mode=mode,
            inlet_profile=prof,
            outlet_mode="zero_gradient_all",
        )
        for _ in range(500):
            s.step()
        sol[mode] = s.f
    assert np.array_equal(sol["mask"], sol["scan"])