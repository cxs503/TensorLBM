"""Tests for the D3Q19 mixed bounce-back / specular (partial-slip) wall.

The module's claims, in the order the tests check them:

1. **algebra** -- the mirror table is a specular reflection about the ``y``
   wall; the mixed operator is bounce-back at ``sigma=0``, specular at
   ``sigma=1`` and the convex combination in between;
2. **C0 reduction exactness** -- the full D3Q19 field reproduces the 1D-column
   solution to round-off for an x/z-uniform channel, so the reduction used by
   ``partial_slip_wall`` is exact rather than approximate;
3. **C1 inversion** -- all three slip-length estimators (quadratic fit,
   fixed-curvature model, bulk mean) recover a *known* slip length from an
   exact analytic slip-Poiseuille profile, plane and pipe;
4. **C4 negative control** -- ``sigma=0`` recovers the no-slip analytic profile
   within 0.5% in 3D;
5. **C5 free-slip limit** -- ``sigma=1`` produces the documented b -> inf
   signature (plug profile + secular momentum growth) instead of a crash.

Wall-clock is kept small: the 3D cases use ``ny=17`` and a few thousand steps,
which is enough for the algebraic and control claims (the report's headline
numbers come from the converged runs in ``artifacts/P106W1-N04``).
"""

import numpy as np
import pytest

from tensorlbm.boundary import partial_slip_wall as ps
from tensorlbm.boundary import partial_slip_wall3d as p3

TAU = 0.8
NU = ps.nu_from_tau(TAU)
G = 1.0e-5


# ---------------------------------------------------------------------------
# 1. algebra
# ---------------------------------------------------------------------------
def test_mirror_table_is_specular_about_y():
    for q in range(len(p3.C19)):
        m = int(p3.MIRROR_Y[q])
        assert p3.MIRROR_Y[m] == q                       # involution
        assert p3.C19[m][0] == p3.C19[q][0]              # c_x kept
        assert p3.C19[m][2] == p3.C19[q][2]              # c_z kept
        assert p3.C19[m][1] == -p3.C19[q][1]             # c_y flipped


def test_weight_and_opposite_tables_are_consistent():
    assert np.isclose(p3.W19.sum(), 1.0)
    for q in range(len(p3.C19)):
        assert np.all(p3.C19[p3.OPP19[q]] == -p3.C19[q])


def _random_post_collision(rng):
    f = rng.random((19, 4, 5)) + 0.1
    return f / f.sum(0)


def test_mixed_wall_3d_sigma0_is_bounce_back():
    rng = np.random.default_rng(0)
    f = _random_post_collision(rng)
    row = p3.mixed_wall_row_3d(f, 0.0, top=False)
    for q in range(19):
        if p3.C19[q][1] > 0:                             # streams into the fluid
            assert np.allclose(row[q], f[p3.OPP19[q]])


def test_mixed_wall_3d_sigma1_is_specular():
    rng = np.random.default_rng(1)
    f = _random_post_collision(rng)
    row = p3.mixed_wall_row_3d(f, 1.0, top=True)
    for q in range(19):
        if p3.C19[q][1] < 0:
            assert np.allclose(row[q], f[p3.MIRROR_Y[q]])


def test_mixed_wall_3d_is_convex_combination():
    rng = np.random.default_rng(2)
    f = _random_post_collision(rng)
    s = 0.31
    mix = p3.mixed_wall_row_3d(f, s, top=False)
    bb = p3.mixed_wall_row_3d(f, 0.0, top=False)
    sp = p3.mixed_wall_row_3d(f, 1.0, top=False)
    assert np.allclose(mix, (1.0 - s) * bb + s * sp)


def test_sigma_out_of_range_is_rejected():
    f = np.ones((19, 2, 2))
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            p3.mixed_wall_row_3d(f, bad, top=False)


# ---------------------------------------------------------------------------
# 2. C0 -- the 1D reduction is exact
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def collapse_pair():
    ny, steps = 17, 3000
    r1 = ps.channel_poiseuille(0.5, ny=ny, tau=TAU, g=G, steps=steps, tol=-1.0)
    r3 = p3.channel_poiseuille_3d(0.5, ny=ny, nx=3, nz=3, tau=TAU, g=G,
                                  steps=steps, tol=-1.0)
    return r1, r3


def test_reduction_matches_1d_column(collapse_pair):
    r1, r3 = collapse_pair
    err = p3.collapse_error(r3, r1)
    assert err["rel_profile_deviation"] < 1e-12          # round-off, not modelling
    assert err["plane_uniformity"] < 1e-14               # x/z invariance of the field


def test_3d_field_is_uniform_across_the_periodic_planes(collapse_pair):
    _r1, r3 = collapse_pair
    assert r3.plane_spread < 1e-14
    assert r3.u_slice.shape == (r3.ny - 2, r3.nx)


# ---------------------------------------------------------------------------
# 3. C1 -- estimators recover a known slip length (no solver involved)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("b_true", [0.5, 1.0, 2.0])
def test_estimators_recover_analytic_plane_slip_length(b_true):
    ny = 33
    H = (ny - 2) / 2.0
    y = np.arange(1, ny - 1, dtype=float)
    yc = float(np.mean(y))
    u = ps.analytic_plane_profile(y, H, NU, G, b_true, center=yc)
    quad = ps.slip_length_from_quadratic_fit(y, u, length=H, nu=NU, g=G, geometry="plane")
    model = p3.model_slip_length(y, u, H, NU, G, geometry="plane")
    bulk = p3.bulk_mean_slip_length(u, H, NU, G, geometry="plane", coord=y)
    assert quad["b"] == pytest.approx(b_true, rel=1e-10)
    assert model["b"] == pytest.approx(b_true, rel=1e-10)
    assert bulk["b"] == pytest.approx(b_true, rel=1e-10)


@pytest.mark.parametrize("b_true", [0.5, 1.0, 2.0])
def test_estimators_recover_analytic_pipe_slip_length(b_true):
    nr = 33
    R = nr - 1.5
    r = np.arange(0, nr - 1, dtype=float)
    u = ps.analytic_pipe_profile(r, R, NU, G, b_true)
    quad = ps.slip_length_from_quadratic_fit(r, u, length=R, nu=NU, g=G,
                                             center=0.0, geometry="pipe",
                                             core_fraction=0.6, inner_fraction=0.2)
    model = p3.model_slip_length(r, u, R, NU, G, geometry="pipe", band=(0.55, 1.0))
    um = float(np.sum(r * u) / np.sum(r))
    bulk = p3.bulk_mean_slip_length(u, R, NU, G, geometry="pipe", coord=r, u_mean=um)
    # the continuum second moment is biased by O(h^2): keep that as a control
    bulk_cont = p3.bulk_mean_slip_length(u, R, NU, G, geometry="pipe", u_mean=um)
    assert abs(bulk_cont["b"] - b_true) > 1e-5
    assert quad["b"] == pytest.approx(b_true, rel=1e-10)
    assert model["b"] == pytest.approx(b_true, rel=1e-10)
    assert bulk["b"] == pytest.approx(b_true, rel=1e-10)


# ---------------------------------------------------------------------------
# 4. C4 -- negative control in 3D
# ---------------------------------------------------------------------------
def test_sigma0_recovers_no_slip_profile_in_3d():
    ny = 17
    res = p3.channel_poiseuille_3d(0.0, ny=ny, nx=2, nz=2, tau=TAU, g=G,
                                   steps=20000, tol=1e-9)
    assert res.converged
    H = (ny - 2) / 2.0
    y = np.arange(1, ny - 1, dtype=float)
    ana = ps.analytic_plane_profile(y, H, NU, G, 0.0, center=float(np.mean(y)))
    dev = float(np.max(np.abs(res.u_profile - ana)) / ana.max())
    assert dev < 5e-3
    inv = p3.model_slip_length(y, res.u_profile, H, NU, G, geometry="plane")
    assert abs(inv["b"]) / H < 5e-3


# ---------------------------------------------------------------------------
# 5. monotonicity and the free-slip limit
# ---------------------------------------------------------------------------
def test_slip_length_monotone_in_sigma_3d():
    bs = []
    for s in (0.0, 0.4, 0.7):
        res = p3.channel_poiseuille_3d(s, ny=17, nx=1, nz=1, tau=TAU, g=G,
                                       steps=20000, tol=1e-9)
        bs.append(p3.model_slip_length(np.arange(1, 16.0), res.u_profile,
                                       res.length, NU, G, geometry="plane")["b"])
    assert all(b2 > b1 for b1, b2 in zip(bs, bs[1:]))


def test_sigma1_free_slip_signature_is_recorded_not_crashing():
    res = p3.channel_poiseuille_3d(1.0, ny=17, nx=1, nz=1, tau=TAU, g=G,
                                   steps=2000, tol=1e-12, track_momentum=True)
    sig = p3.freeslip_signature(res)
    assert sig["converged"] is False                     # no steady state exists
    assert sig["profile_flatness"] < 1e-10               # plug flow
    assert abs(sig["curvature_ratio"]) < 1e-6            # zero curvature
    assert sig["momentum_growth_per_1000_steps"] > 0.0   # secular energy input
    assert np.isfinite(res.u_profile).all()


def test_law_fit_reproduces_the_measured_slope():
    """The fitted law must expose ``a`` and reproduce the data it came from."""
    sig = np.array([0.0, 0.25, 0.5, 0.75])
    b = 0.3 * sig / (1.0 - sig) - 0.005
    law = ps.fit_slip_law(sig, b)
    assert law.a == pytest.approx(0.3, rel=1e-10)
    assert law.c == pytest.approx(-0.005, rel=1e-10)
    assert law.r2 > 0.999
    assert law.monotone
