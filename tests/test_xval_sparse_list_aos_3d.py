"""Round-2 tests: D3Q19 generalisation of the list-based AoS port + interior solids.

Round 1 (``tests/test_xval_sparse_list_aos.py``) pins the 2-D D2Q9 port.  These
tests pin the round-2 additions:

1. the D3Q19 stencil tables / weights,
2. the 3-D index array against the OSS ``lbmpy.sparse.SparseLbMapper`` (D3Q19),
3. dense-vs-list equivalence in 3-D, and stencil-order invariance,
4. the analytic references: 3-D plane Poiseuille, rectangular duct (incl. its
   plane-channel limit), shear-wave decay, and a periodic array of *interior*
   solid slabs (solid cells inside the list, no border walls),
5. the layout-mismatch control (SoA offsets must not be silently accepted by the
   AoS layout).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.sparse_list_aos import (
    C_D2Q9,
    C_D3Q19,
    C_D3Q19_LBMPY,
    FLUID,
    GHOST,
    WALL,
    SparseListLBM,
    build_index_array,
    dense_collide_stream,
    duct_poiseuille_analytic,
    equilibrium,
    equilibrium_vec,
    poiseuille_analytic,
    velocity_from,
    velocity_from_vec,
    weights_for,
)

TAU = 0.8
NU = (TAU - 0.5) / 3.0
U_MAX = 0.03


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def channel_flags(nx, height, nz):
    flags = np.full((nx + 2, height + 2, nz + 2), GHOST, dtype=np.uint32)
    flags[1:-1, 1:-1, 1:-1] = FLUID
    flags[:, 0, :] = WALL
    flags[:, -1, :] = WALL
    return flags


def duct_flags(nx, hy, hz):
    flags = channel_flags(nx, hy, hz)
    flags[:, :, 0] = WALL
    flags[:, :, -1] = WALL
    return flags


def slab_flags(nx, hc, channels, nz):
    """Periodic box with ``channels`` one-cell solid slabs inside the fluid list.

    The first slab sits on row 2 so that row 1's periodic wrap (which lands on the
    last interior row) sees fluid, not solid -- the wrap target must be fluid for
    the index array to exist at all (same constraint as lbmpy's mapper).
    """
    ny = channels * (hc + 1)
    flags = np.full((nx + 2, ny + 2, nz + 2), GHOST, dtype=np.uint32)
    flags[1:-1, 1:-1, 1:-1] = FLUID
    slab_rows = [2 + k * (hc + 1) for k in range(channels)]
    for r in slab_rows:
        flags[:, r, :] = WALL
    return flags, slab_rows


def march(sp, u_ref, tol=1e-10, cap=200_000, check=200):
    prev, steps, drift = None, 0, float("inf")
    while steps < cap:
        for _ in range(check):
            sp.step()
        steps += check
        ux = sp.macroscopic()[1]
        if prev is not None:
            drift = float(np.max(np.abs(ux - prev))) / u_ref
            if drift < tol:
                break
        prev = ux.copy()
    return steps, drift


def profile(sp, shape, height):
    ux = sp.macroscopic()[1]
    coords = sp.coords.astype(int) - 1
    return np.array([ux[coords[:, 1] == y].mean() for y in range(height)])


# --------------------------------------------------------------------------- #
# 1. stencils / weights
# --------------------------------------------------------------------------- #
def test_d3q19_stencil_tables():
    assert C_D3Q19.shape == (19, 3) and C_D3Q19_LBMPY.shape == (19, 3)
    # same direction set, different order
    assert sorted(map(tuple, C_D3Q19.tolist())) == sorted(map(tuple, C_D3Q19_LBMPY.tolist()))
    assert pytest.approx(1.0) == float(weights_for(C_D3Q19).sum())
    assert pytest.approx(1.0) == float(weights_for(C_D3Q19_LBMPY).sum())
    w = weights_for(C_D3Q19)
    norms = (C_D3Q19**2).sum(axis=1)
    assert pytest.approx(1.0 / 3.0) == float(w[norms == 0][0])
    assert pytest.approx(1.0 / 18.0) == float(w[norms == 1][0])
    assert pytest.approx(1.0 / 36.0) == float(w[norms == 2][0])
    with pytest.raises(ValueError):
        weights_for(np.zeros((5, 4), dtype=np.int64))


def test_equilibrium_wrappers_agree():
    rng = np.random.default_rng(0)
    ux, uy = rng.normal(size=7) * 0.05, rng.normal(size=7) * 0.05
    feq2 = equilibrium(np.ones(7), ux, uy, C_D2Q9)
    feq2v = equilibrium_vec(np.ones(7), np.stack([ux, uy], axis=-1), C_D2Q9)
    assert np.array_equal(feq2, feq2v)
    _, a, b = velocity_from(feq2, (1e-4, -2e-4), C_D2Q9)
    _, u = velocity_from_vec(feq2, (1e-4, -2e-4), C_D2Q9)
    assert np.allclose(a, u[:, 0]) and np.allclose(b, u[:, 1])
    with pytest.raises(ValueError):
        equilibrium(np.ones(7), ux, uy, C_D3Q19)


# --------------------------------------------------------------------------- #
# 2. index array vs the OSS implementation in 3-D
# --------------------------------------------------------------------------- #
def test_index_array_matches_oss_d3q19():
    lbmpy = pytest.importorskip("lbmpy")
    from lbmpy.sparse import SparseLbMapper
    from lbmpy.stencils import LBStencil

    flags = duct_flags(2, 6, 8)
    coords, arr = build_index_array(flags, layout="soa", stencil=C_D3Q19_LBMPY)
    mapper = SparseLbMapper(LBStencil("D3Q19"), np.asarray(flags, dtype=np.uint32), FLUID, WALL, 0)
    arr_oss = mapper.create_index_array()
    assert arr.shape == arr_oss.shape == (int(np.count_nonzero(flags & FLUID)), 18)
    assert arr.dtype == arr_oss.dtype == np.uint32
    assert np.array_equal(arr, arr_oss)
    oss_coords = np.ascontiguousarray(mapper.fluid_coordinates).view(np.uint32).reshape(-1, 3)
    assert np.array_equal(oss_coords, coords)


def test_index_array_structure_3d():
    """No-slip aliases to the own cell; the interior of a slab wall is unreachable."""
    flags, slab_rows = slab_flags(2, 4, 2, 2)
    coords, arr = build_index_array(flags, layout="soa", stencil=C_D3Q19_LBMPY)
    n = coords.shape[0]
    # every stored offset must point at a stored population
    assert arr.max() < n * 19
    assert list(slab_rows) == [2, 7]
    # the fluid cell just below the second slab must alias its own opposite row
    below = np.where((coords[:, 1] == 6) & (coords[:, 0] == 1) & (coords[:, 2] == 1))[0][0]
    aliased = np.count_nonzero(arr[below].astype(int) % n == below)
    assert aliased >= 1  # the population streaming out of the slab comes from own cell
    # and no stored offset may address a missing (solid) cell: offsets must be in range
    assert arr.max() < n * 19


# --------------------------------------------------------------------------- #
# 3. equivalence: dense vs list, both layouts, both stencil orders
# --------------------------------------------------------------------------- #
def test_dense_vs_sparse_3d_equivalence():
    h, nx, nz = 8, 2, 4
    g = 8.0 * NU * U_MAX / h**2
    flags = channel_flags(nx, h, nz)
    states = {}
    for layout in ("soa", "aos"):
        sp = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout=layout, stencil=C_D3Q19)
        for _ in range(120):
            sp.step()
        states[layout] = np.stack(sp.macroscopic()[1:], axis=-1)
    coords = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout="soa",
                           stencil=C_D3Q19).coords.astype(int) - 1
    # dense arm
    f = np.moveaxis(
        equilibrium_vec(np.ones(nx * h * nz), np.zeros((nx * h * nz, 3)), C_D3Q19)
        .reshape((nx, h, nz, 19)), -1, 0).copy()
    for _ in range(120):
        f = dense_collide_stream(f, tau=TAU, force=(g, 0.0, 0.0), periodic=(True, False, True),
                                 stencil=C_D3Q19)
    _, u_d = velocity_from_vec(np.moveaxis(f, 0, -1).reshape(-1, 19), (g, 0.0, 0.0), C_D3Q19)
    u_d = u_d.reshape(nx, h, nz, 3)
    assert np.max(np.abs(states["soa"] - states["aos"])) / U_MAX < 1e-12
    dense_at_cells = u_d[coords[:, 0], coords[:, 1], coords[:, 2]]
    assert np.max(np.abs(states["soa"] - dense_at_cells)) / U_MAX < 1e-12


def test_stencil_order_invariance():
    h, g = 6, 8.0 * NU * U_MAX / 6**2
    flags = channel_flags(2, h, 3)
    profs = []
    for st in (C_D3Q19, C_D3Q19_LBMPY):
        sp = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout="soa", stencil=st)
        march(sp, U_MAX, tol=1e-9, cap=20_000)
        profs.append(profile(sp, flags.shape, h))
    assert np.max(np.abs(profs[0] - profs[1])) / U_MAX < 1e-12


# --------------------------------------------------------------------------- #
# 4. analytic physics
# --------------------------------------------------------------------------- #
def test_plane_poiseuille_3d_within_15pct():
    h, g = 8, 8.0 * NU * U_MAX / 8**2
    flags = channel_flags(2, h, 4)
    ref = poiseuille_analytic(np.arange(h, dtype=float), h, g, NU)
    for layout in ("soa", "aos"):
        sp = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout=layout, stencil=C_D3Q19)
        march(sp, ref.max())
        got = profile(sp, flags.shape, h)
        assert np.max(np.abs(got - ref)) / ref.max() < 0.15


def test_duct_analytic_plane_limit():
    """b -> infinity must reproduce the plane-channel parabola exactly."""
    h = 8
    g = 3.75e-4
    y = np.arange(h, dtype=float)
    wide = duct_poiseuille_analytic(y[:, None], np.array([[9999.5]]), h, 20000, g, NU)[:, 0]
    plane = poiseuille_analytic(y, h, g, NU)
    assert np.allclose(wide, plane, rtol=1e-6, atol=1e-12)
    # and the square duct must be slower than the plane channel at every point
    square = duct_poiseuille_analytic(y[:, None], np.arange(h)[None, :], h, h, g, NU)
    assert np.all(square < plane[:, None] + 1e-12)


def test_duct_3d_within_15pct():
    hy, hz = 8, 12
    unit = duct_poiseuille_analytic(np.arange(hy)[:, None], np.arange(hz)[None, :], hy, hz, 1.0, 1.0)
    g = U_MAX / float(unit.max())
    ref = duct_poiseuille_analytic(np.arange(hy)[:, None], np.arange(hz)[None, :], hy, hz, g, NU)
    flags = duct_flags(2, hy, hz)
    sp = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout="aos", stencil=C_D3Q19)
    march(sp, ref.max())
    u = np.stack(sp.macroscopic()[1:], axis=-1)
    coords = sp.coords.astype(int) - 1
    got = np.empty((hy, hz))
    for iy in range(hy):
        for iz in range(hz):
            got[iy, iz] = u[(coords[:, 1] == iy) & (coords[:, 2] == iz), 0].mean()
    assert np.max(np.abs(got - ref)) / ref.max() < 0.15


def test_shear_wave_decay_3d():
    """u = A sin(k z) exp(-nu k^2 t) is an exact NS solution (no nonlinearity)."""
    n, amp = 8, 0.02
    flags = np.full((n + 2, n + 2, n + 2), GHOST, dtype=np.uint32)
    flags[1:-1, 1:-1, 1:-1] = FLUID
    k = 2.0 * np.pi / n
    omega_ref = NU * k**2
    times = [0, 8, 16, 24, 32]
    grid = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), axis=-1)
    phase = k * grid[..., 2]
    u0 = np.zeros((n**3, 3))
    u0[:, 0] = amp * np.sin(phase).ravel()
    sp = SparseListLBM(flags, tau=TAU, force=(0.0, 0.0, 0.0), layout="soa", stencil=C_D3Q19)
    sp.reset(rho=1.0, u=u0)
    amps, done = [], 0
    for t in times:
        while done < t:
            sp.step()
            done += 1
        u = np.stack(sp.macroscopic()[1:], axis=-1)
        amps.append(float(2.0 * np.mean(u[:, 0] * np.sin(phase).ravel())))
    slope = np.polyfit(times, np.log(amps), 1)[0]
    assert abs(-slope - omega_ref) / omega_ref < 0.15
    assert abs(sp.populations().sum() / sp.n_cells - 1.0) < 1e-12


def test_interior_solid_slab_channels_within_15pct():
    """Solid cells *inside* the list: 2 channels carved out of one periodic box."""
    hc, nx, nz = 6, 2, 2
    g = 8.0 * NU * U_MAX / hc**2
    ref = poiseuille_analytic(np.arange(hc, dtype=float), hc, g, NU)
    flags, slab_rows = slab_flags(nx, hc, 2, nz)
    ny = 2 * (hc + 1)
    # channel A: rows 3..slab2-1 ; channel B: rows slab2+1..ny then 1
    chan = [[y - 1 for y in range(3, slab_rows[1])],
            [y - 1 for y in list(range(slab_rows[1] + 1, ny + 1)) + [1]]]
    for layout in ("soa", "aos"):
        sp = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout=layout, stencil=C_D3Q19)
        _, drift = march(sp, ref.max(), tol=1e-9, cap=100_000)
        u = np.stack(sp.macroscopic()[1:], axis=-1)
        coords = sp.coords.astype(int) - 1
        for rows in chan:
            got = np.array([u[coords[:, 1] == y, 0].mean() for y in rows])
            assert np.max(np.abs(got - ref)) / ref.max() < 0.15
        assert abs(sp.populations().sum() / sp.n_cells - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# 5. layout mismatch must not be silent
# --------------------------------------------------------------------------- #
def test_layout_mismatch_detected():
    flags = duct_flags(2, 6, 8)
    _, arr_soa = build_index_array(flags, layout="soa", stencil=C_D3Q19_LBMPY)
    _, arr_aos = build_index_array(flags, layout="aos", stencil=C_D3Q19_LBMPY)
    assert np.count_nonzero(arr_soa != arr_aos) > 0
    g = 1e-4
    good = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout="soa",
                         index_array=arr_soa, stencil=C_D3Q19_LBMPY)
    bad = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout="aos",
                        index_array=arr_soa, stencil=C_D3Q19_LBMPY)
    try:
        for _ in range(50):
            good.step()
            bad.step()
        _, u_g, _ = good.macroscopic()
        _, u_b, _ = bad.macroscopic()
        dev = np.max(np.abs(u_g - u_b)) / max(np.max(np.abs(u_g)), 1e-30)
        assert not np.isfinite(dev) or dev > 1e-6
    except (IndexError, ValueError):
        pass  # blowing up is an acceptable (and detectable) failure mode