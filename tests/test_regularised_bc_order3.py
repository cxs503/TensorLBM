"""Order-3 (third-moment) wall reconstruction + force-aware boundary row (xval R5).

Anchor: ``XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC``, arm R5.

What these tests pin -- and what they deliberately do *not* pin:

* :func:`tensorlbm.regularised_bc.project_third_order` keeps ``rho``, the
  momentum and the non-equilibrium stress exactly, and keeps the third moment
  of the non-equilibrium part to machine precision for lattice data on both
  D2Q9 and D3Q19 (the map is the least-squares/Gram form, so a lattice
  distribution -- whose third moment already lies in the representable
  subspace -- is reproduced exactly).  Idempotent.
* the third-order Hermite basis has no 0th/1st/2nd moments, and on D2Q9
  ``H3_xxx = c_x^3 - 3 cs^2 c_x`` vanishes identically on the velocity set:
  that is *why* the D2Q9 third moment is fully representable with only two
  surviving directions, while D3Q19 keeps 6 of 10.
* ``order=3`` is a strict superset of ``order=2``: with a vanishing third
  moment the two closures agree to machine precision, so the shipped
  behaviour is unchanged unless the caller asks for the third moment.
* ``stress_source="neighbour"`` really does use the interior plane (it differs
  from the completion whenever the field is not uniform) and leaves the
  closure's density/momentum prescription untouched.
* the R5 physics result, as a *regression*: in the body-force channel the
  boundary row must receive the Guo half-step force (``wall_force=1``);
  omitting it (the shipped default) costs a wall slip proportional to the
  force and roughly doubles the profile error.  Measured on the analytic
  parabola at two grid sizes.

The heavy sweeps (grid ladder, tau sweep, force sweep, Stokes layer) live in
``development/verification/regularised_bc_noneq_stress/run_xval_r5.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm import regularised_bc as rb

D2_NORMALS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _random_field(spec, shape=(4, 5), seed=11, amp=0.02):
    rng = np.random.default_rng(seed)
    f = np.ones((spec.q,) + shape) * spec.w.reshape((-1,) + (1,) * len(shape))
    return f + amp * rng.standard_normal((spec.q,) + shape)


# ---------------------------------------------------------------------------
# third-order projection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [rb.D2Q9, rb.D3Q19], ids=["d2q9", "d3q19"])
def test_third_order_projection_keeps_rho_momentum_stress_and_third_moment(spec):
    f = _random_field(spec)
    rho, u = rb.macroscopic(spec, f)
    out = rb.project_third_order(spec, f, rho=rho, u=u)

    rho2, u2 = rb.macroscopic(spec, out)
    assert np.allclose(rho2, rho, atol=1e-14)
    assert np.allclose(u2, u, atol=1e-14)

    s1 = rb.nonequilibrium_stress(spec, f, rho=rho, u=u)
    s2 = rb.nonequilibrium_stress(spec, out, rho=rho2, u=u2)
    assert np.allclose(s2, s1, atol=1e-14)

    t1 = rb.third_moment(spec, f - rb.equilibrium(spec, rho, u))
    t2 = rb.third_moment(spec, out - rb.equilibrium(spec, rho2, u2))
    rel = np.linalg.norm(t2 - t1) / np.linalg.norm(t1)
    assert rel < 1e-12, f"third moment not reproduced: rel={rel:.3e}"

    again = rb.project_third_order(spec, out, rho=rho2, u=u2)
    assert np.allclose(again, out, atol=1e-14)


@pytest.mark.parametrize("spec", [rb.D2Q9, rb.D3Q19], ids=["d2q9", "d3q19"])
def test_third_order_reduces_to_second_order_for_vanishing_third_moment(spec):
    """order=3 with T=0 must be bit-for-bit the shipped order-2 projection."""
    f = _random_field(spec, seed=5)
    rho, u = rb.macroscopic(spec, f)
    d = spec.dims
    two = rb.project_second_order(spec, f, rho=rho, u=u)
    three = rb.project_third_order(spec, f, rho=rho, u=u, t_ne=np.zeros((d,) * 3 + f.shape[1:]))
    assert np.allclose(three, two, atol=1e-15)


def test_hermite_third_basis_has_no_low_moments_and_d2q9_xxx_vanishes():
    for spec in (rb.D2Q9, rb.D3Q19):
        h3 = rb.hermite_third_basis(spec)
        d = spec.dims
        w = spec.w
        c = spec.c.astype(float)
        # zero-th, first and second (momentum) moments: the order-3 term can
        # never move rho, u or Pi -- only the third and higher moments.
        assert np.allclose(np.tensordot(w, h3, axes=(0, 0)), 0.0, atol=1e-15)
        for a in range(d):
            assert np.allclose(np.tensordot(w * c[:, a], h3, axes=(0, 0)), 0.0, atol=1e-15)
            for b in range(d):
                m2 = np.tensordot(w * c[:, a] * c[:, b], h3, axes=(0, 0))
                assert np.allclose(m2, 0.0, atol=1e-15)
    # D2Q9: c^3 == c for c_x in {0, +-1}, so the pure-direction basis function
    # is identically zero and the third moment has only the two mixed
    # directions -> fully representable.
    h3 = rb.hermite_third_basis(rb.D2Q9)
    assert np.allclose(h3[:, 0, 0, 0], 0.0, atol=1e-15)
    assert np.allclose(h3[:, 1, 1, 1], 0.0, atol=1e-15)


def test_third_moment_representability_reports_rank_deficiency():
    d2 = rb.third_moment_representability(rb.D2Q9)
    d3 = rb.third_moment_representability(rb.D3Q19, shape=(3, 3, 3))
    assert d2["symmetric_dim"] == 4 and d2["rank_gram3"] == 2
    assert d3["symmetric_dim"] == 10 and d3["rank_gram3"] == 6
    # ...yet the *lattice* third moment is reproduced exactly on both sets.
    assert d2["max_rel_residual"] < 1e-12
    assert d3["max_rel_residual"] < 1e-12


# ---------------------------------------------------------------------------
# closure plumbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("normal", D2_NORMALS)
def test_regularised_face_order3_keeps_wall_density_and_stress(normal):
    spec = rb.D2Q9
    f = _random_field(spec, shape=(6,), seed=3)
    face2 = rb.regularised_face(spec, f, normal, kind="velocity", u_wall=0.0, order=2)
    face3 = rb.regularised_face(spec, f, normal, kind="velocity", u_wall=0.0, order=3)
    rho2, u2 = rb.macroscopic(spec, face2)
    rho3, u3 = rb.macroscopic(spec, face3)
    assert np.allclose(rho3, rho2, atol=1e-13)
    assert np.allclose(u3, u2, atol=1e-13)
    s2 = rb.nonequilibrium_stress(spec, face2, rho=rho2, u=u2)
    s3 = rb.nonequilibrium_stress(spec, face3, rho=rho3, u=u3)
    assert np.allclose(s3, s2, atol=1e-13)
    with pytest.raises(ValueError):
        rb.regularised_face(spec, f, normal, order=4)


def test_apply_regularised_wall_neighbour_stress_is_wired():
    spec = rb.D2Q9
    rng = np.random.default_rng(2)
    rho = np.ones((7, 3))
    u = np.stack([0.02 * rng.standard_normal((7, 3)), np.zeros((7, 3))])
    f = rb.equilibrium(spec, rho, u) + 0.01 * rng.standard_normal((9, 7, 3))
    completed = rb.apply_regularised_wall(spec, f, axis=0, side="high", order=2)
    neighbour = rb.apply_regularised_wall(
        spec, f, axis=0, side="high", order=2, stress_source="neighbour"
    )
    # same reconstruction source -> identical; different source -> different node
    assert np.allclose(completed, rb.apply_regularised_wall(spec, f, axis=0, side="high"))
    assert not np.allclose(completed[:, -1, :], neighbour[:, -1, :])
    # the neighbour variant still lands on the prescribed wall state
    rho_n, u_n = rb.macroscopic(spec, neighbour)
    assert np.allclose(u_n[1, -1, :], 0.0, atol=1e-13)
    # control: both variants must be attainable (finite, positive density)
    assert np.all(np.isfinite(neighbour)) and np.all(rho_n > 0)
    with pytest.raises(ValueError):
        rb.apply_regularised_wall(spec, f, axis=0, side="high", stress_source="bogus")


# ---------------------------------------------------------------------------
# physics regression: the boundary row must carry the Guo half-step force
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ny", [16, 24])
def test_boundary_row_force_removes_part_of_the_wall_slip(ny):
    """R5: the shipped default leaves a wall slip ~proportional to the force.

    Measured on the analytic body-force parabola: with ``wall_force=1`` the
    profile error and the uniform offset both drop, and the slip is a constant
    offset (``residual_after_offset_l2`` is orders of magnitude below
    ``l2_rel``) -- i.e. the defect is a wall condition, not a viscosity error.
    """
    kw = dict(case="poiseuille", ny=ny, nx=2, tau=0.8, body_force=2e-5, steps=400000, tol=1e-11)
    shipped = rb.run_channel(rb.ChannelConfig(wall_bc="regularised", wall_force=0.0, **kw))
    forced = rb.run_channel(rb.ChannelConfig(wall_bc="regularised", wall_force=1.0, **kw))
    for rec in (shipped, forced):
        assert rec["converged"]
        assert rec["mass_drift"] < 1e-12
    assert forced["l2_rel"] < shipped["l2_rel"]
    ratio = forced["l2_rel"] / shipped["l2_rel"]
    # measured 0.44 (ny=24) and 0.60 (ny=16) in R5; the band only asserts the
    # direction and the size of the win, not a fitted constant
    assert 0.2 < ratio < 0.75, f"l2 ratio {ratio:.3f} outside the measured band"
    y = np.asarray(shipped["y"])
    off = lambda rec: (np.asarray(rec["ux"]) - np.asarray(rec["ux_exact"])).mean()
    assert abs(off(forced)) < abs(off(shipped))
    # the error is a uniform slip, not a shape error
    exact = np.asarray(shipped["ux_exact"])
    for rec in (shipped, forced):
        err = np.asarray(rec["ux"]) - exact
        assert np.ptp(err) / np.max(np.abs(exact)) < 1e-6
        # measured offset / u_max: 0.86 % (ny=16) and 0.36 % (ny=24) shipped,
        # 0.24 % / 0.16 % with the boundary force -- it is *grid independent*
        # in absolute terms (offset ~ F), which is the signature of a wall
        # condition artefact, not of a discretisation error
        assert abs(err.mean() / np.max(np.abs(exact))) < 2e-2
    assert y.size == shipped["height"] - 1


def test_couette_is_unchanged_by_the_boundary_force_arm():
    """No body force -> ``wall_force`` must be inert (and Couette stays exact)."""
    kw = dict(case="couette", ny=16, nx=2, tau=0.8, u_wall=0.02, steps=400000, tol=1e-11)
    a = rb.run_channel(rb.ChannelConfig(wall_bc="regularised", wall_force=0.0, **kw))
    b = rb.run_channel(rb.ChannelConfig(wall_bc="regularised", wall_force=1.0, **kw))
    assert a["l2_rel"] < 1e-9
    assert np.allclose(a["ux"], b["ux"], atol=0.0)


def test_wall_bc_arm_table_is_consistent():
    assert rb.WALL_BCS[0] == "regularised" and rb.WALL_BCS[-1] == "bounce_back"
    expected = {
        "regularised": (2, "completed"),
        "regularised_neighbour_stress": (2, "neighbour"),
        "regularised3_completed": (3, "completed"),
        "regularised3_neighbour": (3, "neighbour"),
    }
    for name, (order, source) in expected.items():
        cfg = rb.ChannelConfig(wall_bc=name)
        assert (cfg.projection_order, cfg.stress_source) == (order, source)
        assert cfg.is_regularised and cfg.wall_origin == 0.0
    bb = rb.ChannelConfig(wall_bc="bounce_back")
    assert not bb.is_regularised and bb.wall_origin == 0.5
    with pytest.raises(ValueError):
        rb.run_channel(rb.ChannelConfig(wall_bc="nope"))