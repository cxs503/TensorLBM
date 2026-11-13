"""Tests for the regularised (non-equilibrium stress) boundary condition.

Anchor: XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC.
The closure is a semantics transplant of joconnor22/LIFE
``src/Grid.cpp:386-466``; the assertions below pin the OSS invariants
(second-moment preservation, ghost-mode removal, wall velocity closure), the
agreement with a literal scalar transcription of the OSS routine, and the two
analytic channel solutions used by the cross-validation runner.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm import regularised_bc as rb


def _random_face(seed: int = 0, shape=(4, 3)):
    rng = np.random.default_rng(seed)
    f = np.ones((9,) + shape) * rb.D2Q9.w[:, None, None]
    return f + 0.01 * rng.standard_normal((9,) + shape)


def test_lattice_tables_match_repo():
    import torch

    from tensorlbm import d2q9, d3q19

    assert np.array_equal(rb.D2Q9.c, d2q9.C.numpy())
    assert np.allclose(rb.D2Q9.w, d2q9.W.numpy(), atol=1e-15)
    assert np.array_equal(rb.D2Q9.opposite, d2q9.OPPOSITE.numpy())
    assert np.array_equal(rb.D3Q19.c, d3q19.C.numpy())
    assert np.array_equal(rb.D3Q19.opposite, d3q19.OPPOSITE.numpy())


@pytest.mark.parametrize("normal", [(0, 1), (0, -1), (1, 0), (-1, 0)])
def test_stress_projection_identity(normal):
    f = _random_face(seed=1)
    normal = np.asarray(normal)
    rho_in, u_in = rb.macroscopic(rb.D2Q9, f)
    u_wall = (0.03, -0.01)
    u = np.zeros((2,) + f.shape[1:])
    u[0], u[1] = u_wall
    cn = rb.D2Q9.c @ normal
    rho = (2.0 * f[cn == -1].sum(axis=0) + f[cn == 0].sum(axis=0)) / (1.0 - float(normal @ u_wall))

    out = rb.regularised_face(rb.D2Q9, f, normal, kind="velocity", u_wall=u_wall)
    completed = rb.complete_unknowns(rb.D2Q9, f, normal, rho=rho, u=u)
    s_in = rb.nonequilibrium_stress(rb.D2Q9, completed, rho=rho, u=u)
    s_out = rb.nonequilibrium_stress(rb.D2Q9, out, rho=rho, u=u)
    assert np.max(np.abs(s_out - s_in)) / np.max(np.abs(s_in)) < 1e-14

    feq = rb.equilibrium(rb.D2Q9, rho, u)
    ghost = rb.third_moment(rb.D2Q9, out - feq)
    assert np.max(np.abs(ghost)) < 1e-14
    rho_out, u_out = rb.macroscopic(rb.D2Q9, out)
    assert np.max(np.abs(rho_out - rho)) < 1e-14
    assert np.max(np.abs(u_out - u)) < 1e-14


def test_matches_oss_scalar_transcription():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / (
        "xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval.py"
    )
    spec = importlib.util.spec_from_file_location("xval_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = _random_face(seed=7)
    u_wall = (0.02, 0.01)
    for normal in (np.array([0, 1]), np.array([1, 0])):
        out = rb.regularised_face(rb.D2Q9, f, normal, kind="velocity", u_wall=u_wall)
        for j in range(f.shape[1]):
            for i in range(f.shape[2]):
                ref = mod._oss_with_wall(rb.D2Q9, f[:, j, i], normal, u_wall)
                assert np.max(np.abs(ref - out[:, j, i])) < 1e-13


@pytest.mark.parametrize("wall_bc", ["regularised", "bounce_back"])
def test_couette_analytic(wall_bc):
    cfg = rb.ChannelConfig(case="couette", wall_bc=wall_bc, ny=16, u_wall=0.02, steps=60000, tol=1e-9)
    rec = rb.run_channel(cfg)
    assert rec["converged"]
    assert rec["l2_rel"] < 1e-6
    # wall location convention: the closure is the wall, bounce back sits half a link out
    expected = 0.0 if wall_bc == "regularised" else 0.5
    assert abs(rec["fitted_wall_low"] - expected) < 1e-3
    assert abs(rec["fitted_wall_high"] - (cfg.height + expected)) < 1e-3


def test_poiseuille_analytic():
    nu = 0.1
    ny = 16
    height = ny - 1
    cfg = rb.ChannelConfig(
        case="poiseuille",
        wall_bc="regularised",
        ny=ny,
        body_force=8.0 * nu * 0.02 / height**2,
        steps=60000,
        tol=1e-9,
    )
    rec = rb.run_channel(cfg)
    assert rec["l2_rel"] < 0.03
    assert np.isfinite(rec["ux"]).all()


def test_oscillating_wall_beats_bounce_back():
    common = dict(case="oscillating_wall", ny=32, u_wall=0.02, omega=2 * np.pi / 200, steps=800)
    reg = rb.run_channel(rb.ChannelConfig(wall_bc="regularised", **common))
    bb = rb.run_channel(rb.ChannelConfig(wall_bc="bounce_back", **common))
    assert reg["l2_rel_complex"] < 0.15
    assert reg["l2_rel_complex"] < bb["l2_rel_complex"]


def test_channel_kernel_numpy_integration():
    from tensorlbm.channel_kernel_numpy import ChannelKernelConfig, run_channel

    for wall_bc in ("bounce_back", "regularised"):
        cfg = ChannelKernelConfig(
            nx=2, ny=8, nz=2, steps=40, warmup_steps=10, sample_interval=10,
            initialization="laminar", wall_bc=wall_bc,
        )
        rec = run_channel(cfg)
        assert rec["configuration"]["wall_bc"] == wall_bc
        profile = np.asarray(rec["statistics"]["mean_velocity_profile"])
        assert np.isfinite(profile).all()
