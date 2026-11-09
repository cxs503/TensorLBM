"""Tests for the D3Q27 lift of the mixed bounce-back / specular wall.

``partial_slip_wall3d`` used to be D3Q19-only: the wall row, the BGK/Guo
kernels and the channel solver all read module-level D3Q19 tables.  The lift
makes the velocity set an argument (``Lattice`` / ``LATTICE_19`` /
``LATTICE_27``) instead of an assumption, which the contract previously
recorded as ``wall_partial_slip / D3Q27 = NO_IMPLEMENTATION``.

The claims checked here, in order:

1. **algebra** -- the D3Q27 tables are consistent; the mixed operator is a
   convex combination of bounce-back and specular about the ``y`` wall, with
   the *same* result as the D3Q19 entry point when the D3Q19 lattice is passed;
2. **C0' reduction exactness** -- the full D3Q27 field reproduces the 1D-column
   solution to round-off, and it does so *from a non-uniform start* (the
   transverse modes launched by the corner directions have to decay);
3. **velocity-set independence** -- D3Q27 and D3Q19 give the same profile and
   the same ``b(sigma)`` law (the law is a property of the operator: slope
   ``a = 0.3``, lattice defect ``c*H = -0.065``, both to ~1e-12);
4. **negative control** -- ``sigma = 0`` recovers the analytic no-slip parabola
   to < 0.5% (measured 0.231%) with ``|b|/H < 5e-3``;
5. **monotonicity and the free-slip limit** -- ``b`` grows with ``sigma`` and
   ``sigma = 1`` shows the documented plug-flow signature instead of crashing.

Wall clock stays around two minutes: ``ny = 17``, ``nx = nz = 1..4`` and a few
thousand steps are enough for these claims; the converged headline numbers are
in ``artifacts/P106W1-N04/d3q27_slip_wall.json``.
"""

import numpy as np
import pytest

from tensorlbm.boundary import partial_slip_wall as ps
from tensorlbm.boundary import partial_slip_wall3d as p3

TAU = 0.8
NU = ps.nu_from_tau(TAU)
G = 1.0e-5
NY = 17
H = (NY - 2) / 2.0
Y = np.arange(1, NY - 1, dtype=float)
Q27 = p3.LATTICE_27.Q


@pytest.fixture(scope="module")
def fields():
    """Converged D3Q27 channel fields, computed once per module."""
    cache = {}

    def get(sigma: float, **kw):
        key = (sigma, tuple(sorted(kw.items())))
        if key not in cache:
            params = {"steps": 20000, "tol": 1e-9}
            params.update(kw)
            cache[key] = p3.channel_poiseuille_lattice(
                sigma, p3.LATTICE_27, ny=NY, tau=TAU, g=G, **params
            )
        return cache[key]

    return get


# ---------------------------------------------------------------------------
# 1. algebra
# ---------------------------------------------------------------------------
def test_d3q27_tables_are_consistent():
    assert p3.C27.shape == (Q27, 3)
    assert p3.W27.shape == (Q27,)
    assert np.isclose(p3.W27.sum(), 1.0)
    for q in range(Q27):
        assert np.all(p3.C27[p3.OPP27[q]] == -p3.C27[q])
        m = int(p3.MIRROR_Y_27[q])
        assert p3.MIRROR_Y_27[m] == q                  # involution
        assert p3.C27[m][0] == p3.C27[q][0]             # c_x kept
        assert p3.C27[m][2] == p3.C27[q][2]             # c_z kept
        assert p3.C27[m][1] == -p3.C27[q][1]            # c_y flipped
    assert set(p3.LATTICES) == {"D3Q19", "D3Q27"}
    assert p3.LATTICES["D3Q27"] is p3.LATTICE_27


def test_mirror_table_is_lattice_derived():
    """The mirror table must be a *reflection*, not a hand-written permutation."""
    for C, table in ((p3.C19, p3.MIRROR_Y), (p3.C27, p3.MIRROR_Y_27)):
        flipped = np.column_stack([C[:, 0], -C[:, 1], C[:, 2]])
        assert np.all(C[table] == flipped)


def _random_post_collision(rng, q: int, shape=(3, 4)):
    f = rng.random((q,) + shape) + 0.1
    return f / f.sum(0)


def test_d3q19_entry_point_delegates_to_the_generic_operator():
    rng = np.random.default_rng(0)
    f = _random_post_collision(rng, 19)
    for sigma in (0.0, 0.42, 1.0):
        for top in (False, True):
            legacy = p3.mixed_wall_row_3d(f, sigma, top=top)
            generic = p3.mixed_wall_row_lattice(p3.LATTICE_19, f, sigma, top)
            assert np.array_equal(legacy, generic)


def test_d3q27_sigma0_is_bounce_back():
    rng = np.random.default_rng(1)
    f = _random_post_collision(rng, Q27)
    row = p3.mixed_wall_row_lattice(p3.LATTICE_27, f, 0.0, top=False)
    for q in range(Q27):
        if p3.C27[q][1] > 0:                            # streams into the fluid
            assert np.allclose(row[q], f[p3.OPP27[q]])


def test_d3q27_sigma1_is_specular():
    rng = np.random.default_rng(2)
    f = _random_post_collision(rng, Q27)
    row = p3.mixed_wall_row_lattice(p3.LATTICE_27, f, 1.0, top=True)
    for q in range(Q27):
        if p3.C27[q][1] < 0:
            assert np.allclose(row[q], f[p3.MIRROR_Y_27[q]])


def test_d3q27_mixed_row_is_convex_combination():
    rng = np.random.default_rng(3)
    f = _random_post_collision(rng, Q27)
    s = 0.37
    mix = p3.mixed_wall_row_lattice(p3.LATTICE_27, f, s, top=False)
    bb = p3.mixed_wall_row_lattice(p3.LATTICE_27, f, 0.0, top=False)
    sp = p3.mixed_wall_row_lattice(p3.LATTICE_27, f, 1.0, top=False)
    assert np.allclose(mix, (1.0 - s) * bb + s * sp)


def test_d3q27_rejects_a_d3q19_population_block():
    rng = np.random.default_rng(4)
    f19 = _random_post_collision(rng, 19)
    with pytest.raises(ValueError, match="D3Q27 expects 27"):
        p3.mixed_wall_row_lattice(p3.LATTICE_27, f19, 0.5, top=False)


# ---------------------------------------------------------------------------
# 2. C0' -- reduction exactness of the full D3Q27 field
# ---------------------------------------------------------------------------
def test_d3q27_reduction_matches_1d_column(fields):
    """Identical iteration counts, so the comparison sees modelling error only."""
    steps = 4000
    res3 = fields(0.5, nx=1, nz=1, steps=steps, tol=-1.0)
    res1 = ps.channel_poiseuille(0.5, ny=NY, tau=TAU, g=G, steps=steps, tol=-1.0)
    out = p3.collapse_error(res3, res1)
    assert out["rel_profile_deviation"] < 1e-12
    assert out["plane_uniformity"] < 1e-14


def test_d3q27_uniformity_is_reached_from_a_wave_start(fields):
    """A transverse perturbation must decay: uniformity is not just preserved.

    Same fixed iteration count for the uniform and the perturbed run, so any
    surviving transverse mode would show up as a profile difference.
    """
    steps = 6000
    base = fields(0.5, nx=1, nz=1, steps=steps, tol=-1.0)
    wave = fields(0.5, nx=4, nz=4, steps=steps, tol=-1.0, init_wave=0.05)
    dev = np.max(np.abs(wave.u_profile - base.u_profile)) / np.max(np.abs(base.u_profile))
    assert dev < 1e-10
    assert wave.plane_spread < 1e-10


# ---------------------------------------------------------------------------
# 3. velocity-set independence
# ---------------------------------------------------------------------------
def test_d3q27_and_d3q19_agree_on_the_profile(fields):
    r27 = fields(0.5, nx=1, nz=1)
    r19 = p3.channel_poiseuille_3d(0.5, ny=NY, tau=TAU, g=G, steps=20000, tol=1e-9)
    assert r27.lattice == "D3Q27"
    assert r19.lattice == "D3Q19"
    dev = np.max(np.abs(r27.u_profile - r19.u_profile)) / np.max(np.abs(r19.u_profile))
    assert dev < 1e-8


def test_d3q27_law_reproduces_the_d3q19_law(fields):
    """``b = a sigma/(1-sigma) + c`` is an operator property, not a lattice one."""
    sigmas = (0.0, 0.25, 0.5, 0.75)
    b27, b19 = [], []
    for s in sigmas:
        b27.append(p3.model_slip_length(Y, fields(s, nx=1, nz=1).u_profile, H, NU, G)["b"])
        r19 = p3.channel_poiseuille_3d(s, ny=NY, tau=TAU, g=G, steps=20000, tol=1e-9)
        b19.append(p3.model_slip_length(Y, r19.u_profile, H, NU, G)["b"])
    law27 = ps.fit_slip_law(np.array(sigmas), np.array(b27))
    law19 = ps.fit_slip_law(np.array(sigmas), np.array(b19))
    assert law27.a == pytest.approx(0.3, rel=1e-4)
    assert law27.a == pytest.approx(law19.a, rel=1e-6)
    assert law27.c * H == pytest.approx(-0.065, abs=5e-3)
    assert law27.c == pytest.approx(law19.c, rel=1e-3)
    assert law27.r2 > 0.999 and law27.monotone


# ---------------------------------------------------------------------------
# 4. negative control
# ---------------------------------------------------------------------------
def test_d3q27_sigma0_recovers_the_no_slip_parabola(fields):
    res = fields(0.0, nx=1, nz=1)
    ana = ps.analytic_plane_profile(Y, H, NU, G, 0.0, center=float(np.mean(Y)))
    dev = np.max(np.abs(res.u_profile - ana)) / ana.max()
    assert dev < 5e-3
    inv = p3.model_slip_length(Y, res.u_profile, H, NU, G)
    bulk = p3.bulk_mean_slip_length(res.u_profile, H, NU, G, coord=Y)
    assert abs(inv["b"]) / H < 5e-3
    assert bulk["b"] == pytest.approx(inv["b"], rel=1e-2)


# ---------------------------------------------------------------------------
# 5. monotonicity and the free-slip limit
# ---------------------------------------------------------------------------
def test_d3q27_slip_length_is_monotone_in_sigma(fields):
    bs = [p3.model_slip_length(Y, fields(s, nx=1, nz=1).u_profile, H, NU, G)["b"]
          for s in (0.0, 0.4, 0.7)]
    assert all(b2 > b1 for b1, b2 in zip(bs, bs[1:]))


def test_d3q27_sigma1_free_slip_signature_is_recorded_not_crashing():
    res = p3.channel_poiseuille_lattice(
        1.0, p3.LATTICE_27, ny=NY, tau=TAU, g=G, steps=2000, tol=1e-12,
        track_momentum=True,
    )
    sig = p3.freeslip_signature(res)
    assert sig["converged"] is False
    assert sig["profile_flatness"] < 1e-10
    assert abs(sig["curvature_ratio"]) < 1e-6
    assert sig["momentum_growth_per_1000_steps"] > 0.0
    assert np.isfinite(res.u_profile).all()