"""Parked regression gate: TensorLBM collision kernels vs lbmpy pointwise reference.

Transplant of the lssfau/lbmpy practice: construct the collision operator
symbolically in the OSS project, evaluate it pointwise on the same PDF field
the in-repo kernel sees, and require agreement at the predeclared tolerance
(see ``xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/predeclared.json``).

Skips when lbmpy is not installed so the suite stays runnable without it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lbmpy")
pytest.importorskip("pystencils")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "xval" / "XVAL-LBM-POINTWISE-REFERENCE-REGRESSION"))

import lbmpy_pointwise_ref as ref  # noqa: E402
from tensorlbm import solver  # noqa: E402

GATE = 1e-10
TAUS = (0.55, 0.7, 1.0)
MRT = np.array(solver._M_D2Q9_DATA, dtype=float)


def _run(kernel, f, **kw):
    return kernel(torch.from_numpy(np.ascontiguousarray(f)), **kw).numpy().astype(np.float64)


def _eps(got, f_ref):
    return ref.compare(got, f_ref)["eps_scaled_inf"]


def test_direction_permutation_is_a_bijection():
    for lat, q in (("D2Q9", 9), ("D3Q19", 19)):
        perm = ref.direction_permutation(lat)
        assert perm.shape == (q,)
        assert sorted(perm.tolist()) == list(range(q))


def test_mrt_moment_basis_matches_in_repo_matrix():
    assert ref.moment_basis_check(MRT, "D2Q9") < 1e-12


@pytest.mark.parametrize("tau", TAUS)
def test_collide_bgk_matches_lbmpy_srt(tau):
    f = ref.make_test_field("D2Q9", tau=tau, seed=0, amplitude=0.05)
    f_ref = ref.PointwiseReference(lattice="D2Q9", model="srt", tau=tau).apply(f)
    assert _eps(_run(solver.collide_bgk, f, tau=tau), f_ref) < GATE


@pytest.mark.parametrize("tau", TAUS)
def test_collide_bgk_matmul_matches_lbmpy_srt(tau):
    f = ref.make_test_field("D2Q9", tau=tau, seed=1, amplitude=0.05)
    f_ref = ref.PointwiseReference(lattice="D2Q9", model="srt", tau=tau).apply(f)
    assert _eps(_run(solver.collide_bgk_matmul, f, tau=tau), f_ref) < GATE


@pytest.mark.parametrize("tau", TAUS)
def test_collide_trt_matches_lbmpy_trt(tau):
    f = ref.make_test_field("D2Q9", tau=tau, seed=2, amplitude=0.05)
    f_ref = ref.PointwiseReference(lattice="D2Q9", model="trt", tau_plus=tau).apply(f)
    got = _run(solver.collide_trt, f, tau_plus=tau, lambda_trt=3.0 / 16.0)
    assert _eps(got, f_ref) < GATE


@pytest.mark.parametrize("tau", TAUS)
def test_collide_mrt_matches_lbmpy_moment_method(tau):
    f = ref.make_test_field("D2Q9", tau=tau, seed=3, amplitude=0.05)
    spec = dict(lattice="D2Q9", model="mrt_from_matrix", moment_matrix=MRT, tau=tau,
                relaxation_rates=[1.0, 1.64, 1.54, 1.0, 1.7, 1.0, 1.7, 1.0 / tau, 1.0 / tau])
    f_ref = ref.PointwiseReference(**spec).apply(f)
    got = _run(solver.collide_mrt, f, tau=tau, s_e=1.64, s_eps=1.54, s_q=1.7)
    assert _eps(got, f_ref) < GATE


def test_d3q19_bgk_matches_lbmpy_discrete_equilibrium():
    from tensorlbm.collision_d3q19_advanced import collide_bgk_d3q19

    f = ref.make_test_field("D3Q19", tau=0.7, seed=4, amplitude=0.05)
    f_ref = ref.PointwiseReference(lattice="D3Q19", model="srt", tau=0.7,
                                  continuous_equilibrium=False).apply(f)
    assert _eps(_run(collide_bgk_d3q19, f, tau=0.7), f_ref) < 1e-10


def test_rlbm_post_collision_third_order_moments_vanish():
    """No OSS counterpart: check the defining invariant instead."""
    f = ref.make_test_field("D2Q9", tau=0.6, seed=5, amplitude=0.05, noise=1e-2)
    post = _run(solver.collide_rlbm, f, tau=0.6)
    vel = ref.tensorlbm_velocities("D2Q9").astype(np.float64)
    rho = f.sum(0)
    u = np.tensordot(vel, f, axes=(0, 0)) / rho[None]
    w = ref._weights("D2Q9").reshape((9, 1, 1))
    cu = np.tensordot(vel, u, axes=(1, 0))
    feq = rho[None] * w * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * (u ** 2).sum(0)[None])
    third = np.einsum("qa,qb,qc,q...->...abc", vel, vel, vel, post - feq)
    assert np.abs(third).max() < 1e-13


def test_gate_is_sensitive_positive_control():
    """A 1e-3 relative perturbation of tau must be detected by the same metric."""
    f = ref.make_test_field("D2Q9", tau=0.7, seed=0, amplitude=0.05)
    ref_ok = ref.PointwiseReference(lattice="D2Q9", model="srt", tau=0.7).apply(f)
    ref_off = ref.PointwiseReference(lattice="D2Q9", model="srt", tau=0.7 * (1 + 1e-3)).apply(f)
    assert _eps(_run(solver.collide_bgk, f, tau=0.7), ref_ok) < GATE
    assert _eps(_run(solver.collide_bgk, f, tau=0.7), ref_off) > 1e-6
