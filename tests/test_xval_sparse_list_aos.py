"""Tests for the sparse list-based AoS LBM port (XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM).

The reference implementation is the OSS ``lbmpy.sparse`` module shipped with
lbmpy 2.0 (``mapping.py``: cell list + per-cell/per-direction pull offsets,
no-slip by index aliasing, periodic wrap across ghost layers).  These tests pin

1. the index array to *integer equality* with the OSS implementation,
2. the structural meaning of the entries (aliasing / periodicity),
3. bit-level equivalence of the list-based traversal with an independent dense
   traversal of the same discrete scheme,
4. both arms against the analytic plane-Poiseuille profile and the
   Taylor-Green decay rate,
5. mass conservation, plus a negative control showing that the second-order
   Guo force term is *required* (without it the forcing leaks mass),
6. that mixing the AoS stride convention with the OSS (SoA) offsets is detected.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.sparse_list_aos import (
    C_D2Q9,
    C_D2Q9_LBMPY,
    FLUID,
    GHOST,
    WALL,
    SparseListLBM,
    build_index_array,
    dense_collide_stream,
    equilibrium,
    poiseuille_analytic,
    velocity_from,
    weights_for,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def channel_flags(length: int, height: int) -> np.ndarray:
    flags = np.full((length + 2, height + 2), GHOST, dtype=np.uint32)
    flags[1:-1, 1:-1] = FLUID
    flags[:, 0] = WALL
    flags[:, -1] = WALL
    return flags


def box_flags(length: int, height: int) -> np.ndarray:
    flags = np.full((length + 2, height + 2), GHOST, dtype=np.uint32)
    flags[1:-1, 1:-1] = FLUID
    return flags


def run_poiseuille(height, tau=0.8, umax=0.03, length=4, steps=3000):
    nu = (tau - 0.5) / 3.0
    force_x = 8.0 * nu * umax / height**2
    flags = channel_flags(length, height)
    sp = SparseListLBM(flags, tau=tau, force=(force_x, 0.0))
    for _ in range(steps):
        sp.step()
    _, ux, _ = sp.macroscopic()
    ys = sp.coords[:, 1].astype(int)
    profile = np.array([ux[ys == y].mean() for y in range(1, height + 1)])
    return profile, poiseuille_analytic(np.arange(height), height, force_x, nu), sp


# --------------------------------------------------------------------------- #
# 1. index array vs the OSS implementation
# --------------------------------------------------------------------------- #
def test_index_array_equals_oss_lbmpy():
    """The ported index array is bit-identical to ``lbmpy.sparse`` output."""
    lbmpy = pytest.importorskip("lbmpy")
    from lbmpy.sparse import SparseLbMapper
    from lbmpy.stencils import LBStencil

    flags = channel_flags(4, 8)
    _, mine = build_index_array(flags, stencil=C_D2Q9_LBMPY)
    mapper = SparseLbMapper(
        LBStencil(lbmpy.enums.Stencil.D2Q9),
        flags.copy(),
        fluid_flag=FLUID,
        no_slip_flag=WALL,
        other_boundary_mask=0,
    )
    oss = mapper.create_index_array(ghost_layers=1)
    assert mine.shape == oss.shape
    assert mine.dtype == np.uint32
    assert np.array_equal(mine, oss)


# --------------------------------------------------------------------------- #
# 2. structure of the index array
# --------------------------------------------------------------------------- #
def test_index_array_aliasing_and_periodic_wrap():
    """Structural meaning of every entry, checked against the flag grid.

    ``soa`` offsets decode as ``dir = off // N``, ``cell = off % N``; direction
    ``d`` occupies column ``d-1`` (the rest direction has no entry).
    """
    from tensorlbm.sparse_list_aos import OPP_D2Q9

    length, height = 4, 6
    flags = channel_flags(length, height)
    coords, idx = build_index_array(flags)
    n_cells = len(coords)
    lookup = {tuple(int(v) for v in c): i for i, c in enumerate(coords)}
    shape = np.array(flags.shape)

    assert idx.shape == (n_cells, len(C_D2Q9) - 1)

    n_aliased = n_periodic = n_interior = 0
    for cell, x in enumerate(coords.astype(int)):
        for d in range(1, len(C_D2Q9)):
            nb = x - C_D2Q9[d]
            inside = bool(np.all(nb >= 0) and np.all(nb < shape))
            off = int(idx[cell, d - 1])
            target_dir, target_cell = divmod(off, n_cells)
            if not (inside and flags[tuple(nb)] & WALL):
                # a fluid/periodic pull keeps its own direction label
                assert target_dir == d
            if inside and flags[tuple(nb)] & FLUID:
                assert target_cell == lookup[tuple(int(v) for v in nb)]
                n_interior += 1
            elif inside and flags[tuple(nb)] & WALL:
                # halfway bounce-back by index aliasing: own cell, opposite dir
                assert target_cell == cell and target_dir == OPP_D2Q9[d]
                n_aliased += 1
            else:
                wrapped = nb.copy()
                for k in range(2):
                    if wrapped[k] == 0:
                        wrapped[k] += shape[k] - 2
                    elif wrapped[k] == shape[k] - 1:
                        wrapped[k] -= shape[k] - 2
                assert target_cell == lookup[tuple(int(v) for v in wrapped)]
                n_periodic += 1
    assert n_aliased > 0 and n_periodic > 0 and n_interior > 0


# --------------------------------------------------------------------------- #
# 3. list-based vs dense traversal of the same scheme
# --------------------------------------------------------------------------- #
def test_one_step_sparse_equals_dense():
    length, height = 4, 6
    flags = channel_flags(length, height)
    sp = SparseListLBM(flags, tau=0.8, force=(1e-4, 0.0))
    rng = np.random.default_rng(1234)
    perturbed = sp.populations() * (1.0 + 1e-3 * rng.standard_normal((sp.n_cells, 9)))
    sp.pdf[sp._write_offsets] = perturbed
    dense = np.ascontiguousarray(sp.dense_view((length + 2, height + 2))[:, 1:-1, 1:-1])

    sp.step()
    sparse_state = sp.dense_view((length + 2, height + 2))[:, 1:-1, 1:-1]
    dense_state = dense_collide_stream(dense, tau=0.8, force=(1e-4, 0.0), periodic=(True, False))
    assert np.nanmax(np.abs(sparse_state - dense_state)) < 1e-15


def test_sparse_equals_dense_after_steady_state():
    height = 16
    nu = 0.1
    force_x = 8.0 * nu * 0.03 / height**2
    flags = channel_flags(4, height)
    sp = SparseListLBM(flags, tau=0.8, force=(force_x, 0.0))
    w = weights_for(C_D2Q9)
    f = np.repeat(w[:, None, None], 4, axis=1).repeat(height, axis=2).astype(np.float64)
    for _ in range(6000):
        sp.step()
        f = dense_collide_stream(f, tau=0.8, force=(force_x, 0.0), periodic=(True, False))
    _, ux_sp, _ = sp.macroscopic()
    _, ux_d, _ = velocity_from(f.transpose(1, 2, 0), (force_x, 0.0))
    ys = sp.coords[:, 1].astype(int)
    prof_sp = np.array([ux_sp[ys == y].mean() for y in range(1, height + 1)])
    assert np.max(np.abs(prof_sp - ux_d.mean(axis=0))) < 1e-13 * prof_sp.max()


# --------------------------------------------------------------------------- #
# 4. analytic references
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("height", [16, 32])
def test_poiseuille_profile_within_one_percent(height):
    profile, ref, _ = run_poiseuille(height, steps=8000)
    assert np.abs(profile - ref).max() / ref.max() < 0.01


def test_taylor_green_decay_rate():
    size, tau, u0 = 32, 0.8, 0.02
    nu = (tau - 0.5) / 3.0
    k = 2.0 * np.pi / size
    omega = 2.0 * nu * k * k
    steps, skip = int(2.0 / omega), 50
    flags = box_flags(size, size)
    x = np.arange(size) + 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    sp = SparseListLBM(flags, tau=tau)
    ix = sp.coords[:, 0].astype(int) - 1
    iy = sp.coords[:, 1].astype(int) - 1
    sp.reset(rho=1.0, ux=u0 * np.sin(k * X)[ix, iy] * np.cos(k * Y)[ix, iy],
             uy=-u0 * np.cos(k * X)[ix, iy] * np.sin(k * Y)[ix, iy])
    proj = np.outer(np.sin(k * x), np.cos(k * x))
    norm = float((proj * proj).sum())
    ts, amps = [], []
    for it in range(1, steps + 1):
        sp.step()
        if it % 10 == 0:
            _, vx, _ = sp.macroscopic()
            field = np.zeros((size, size))
            field[ix, iy] = vx
            ts.append(it)
            amps.append(float((field * proj).sum() / norm))
    ts, amps = np.array(ts, float), np.array(amps)
    sel = ts >= skip
    measured = -np.polyfit(ts[sel], np.log(amps[sel]), 1)[0]
    assert abs(measured - omega) / omega < 0.02


# --------------------------------------------------------------------------- #
# 5. mass conservation + negative control on the Guo second-order term
# --------------------------------------------------------------------------- #
def test_mass_is_conserved_with_full_guo_term():
    flags = channel_flags(4, 16)
    sp = SparseListLBM(flags, tau=0.8, force=(1e-4, 0.0))
    mass0 = float(sp.populations().sum())
    drift = 0.0
    for _ in range(2000):
        sp.step()
        drift = max(drift, abs(float(sp.populations().sum()) - mass0) / mass0)
    # round-off accumulation only: ~1e-16 relative per step, so 1e-12 bounds
    # 2e3 steps; a genuine leak (missing O(Ma F) Guo term) gives > 1e-3.
    assert drift < 1e-12


def test_truncated_guo_term_leaks_mass_negative_control():
    """Dropping the O(Ma F) term must break mass conservation -- otherwise the
    test above would not be discriminating."""
    flags = channel_flags(4, 16)
    sp = SparseListLBM(flags, tau=0.8, force=(1e-4, 0.0))
    st, w = sp.stencil, weights_for(sp.stencil)
    f = sp.populations().copy()
    mass0 = float(f.sum())
    fx = 1e-4
    for _ in range(2000):
        # pull (identical index-array traversal, but a truncated force term)
        fpre = np.empty_like(f)
        fpre[:, 0] = f[:, 0]
        for d in range(1, sp.q):
            fpre[:, d] = sp.pdf[sp.index_array[:, d - 1]].reshape(-1)
        rho = fpre.sum(axis=1)
        ux = (fpre @ st[:, 0] + 0.5 * fx) / rho
        uy = fpre @ st[:, 1] / rho
        feq = equilibrium(rho, ux, uy, st)
        omega = 1.0 / sp.tau
        trunc = 3.0 * w * ((st[None, :, 0] - ux[:, None]) * fx) / rho[:, None]
        fpost = fpre + omega * (feq - fpre) + (1.0 - 0.5 * omega) * trunc
        new = np.empty_like(sp.pdf)
        new[sp._write_offsets] = fpost
        sp.pdf = new
        f = sp.populations()
    drift = abs(float(f.sum()) - mass0) / mass0
    assert drift > 1e-3


# --------------------------------------------------------------------------- #
# 6. stride-convention mismatch (AoS label vs SoA offsets)
# --------------------------------------------------------------------------- #
def test_aos_layout_with_soa_offsets_is_detected():
    flags = channel_flags(4, 8)
    good = SparseListLBM(flags, tau=0.8, force=(1e-4, 0.0))
    bad = SparseListLBM(flags, tau=0.8, force=(1e-4, 0.0), layout="aos",
                        index_array=good.index_array.copy())
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(500):
            good.step()
            bad.step()
    _, ux_good, _ = good.macroscopic()
    ref = poiseuille_analytic(np.arange(8), 8, 1e-4, 0.1)
    # the consistent arm stays physical ...
    assert np.isfinite(ux_good).all()
    # ... while the strided mis-addressing either blows up or is grossly wrong
    bad_pop = bad.populations()
    if np.all(np.isfinite(bad_pop)):
        _, ux_bad, _ = bad.macroscopic()
        assert np.abs(ux_bad).max() > 10.0 * ref.max()
    else:
        assert not np.all(np.isfinite(bad_pop))


def test_aos_offsets_use_its_own_index_array():
    """Sanity: the AoS solver builds its own consistent offsets (cell*Q+d)."""
    flags = channel_flags(3, 4)
    sp = SparseListLBM(flags, tau=0.8, layout="aos")
    assert sp.index_array.shape == (sp.n_cells, 8)
    assert sp._write_offsets.max() == sp.n_cells * sp.q - 1
    # and running it stays finite and mass conserving (self-consistent pair)
    mass0 = float(sp.populations().sum())
    for _ in range(200):
        sp.step()
    assert abs(float(sp.populations().sum()) - mass0) / mass0 < 1e-14