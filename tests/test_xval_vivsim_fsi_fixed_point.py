"""Tests for the XVAL harness ``xval_vivsim_fsi_fixed_point``.

Anchor: ``XVAL-LBM-PURE-FUNCTION-FSI-FIXED-POINT`` (OSS ``haimingz/vivsim``).

The point of these tests is that the *cross-implementation identities* of the
report are machine-checkable, not prose: TensorLBM's partitioned solver must
reproduce the OSS coupling loop when configured identically, and must
reproduce the "dry" dynamics when the OSS added-mass line is switched on.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "benchmarks" / "xval_vivsim_fsi_fixed_point" / "run_xval.py"
ART = ROOT / "benchmarks" / "xval_vivsim_fsi_fixed_point" / "artifacts"


def _load():
    spec = importlib.util.spec_from_file_location("xval_fsi_fp", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


X = _load()


# ------------------------------------------------------------------ provenance
def test_vendored_oss_module_is_byte_identical():
    """The executed OSS source must be the vendored, MIT-licensed original."""
    dyn = ROOT / "benchmarks" / "xval_vivsim_fsi_fixed_point" / "vendor" / "vivsim" / "dyn.py"
    import hashlib

    assert hashlib.sha256(dyn.read_bytes()).hexdigest() == X.OSS_DYN_SHA256
    prov = json.loads(
        (ROOT / "benchmarks" / "xval_vivsim_fsi_fixed_point" / "vendor" /
         "PROVENANCE.json").read_text()
    )
    assert prov["vendored_files"]["vendor/vivsim/dyn.py"] == X.OSS_DYN_SHA256
    assert prov["upstream"].endswith("haimingz/vivsim")
    assert "MIT" in prov["license"]


def test_oss_constants_are_the_oss_ones():
    assert X.OSS_COMMIT == "fb84b12f02e18f3b9dfdf738d6a1a318d81539f9"
    assert X.OSS_SRC == "examples/2d/vortex_induced_vibration.py"
    assert X.TOL_PCT == 15.0            # owner precision standard
    assert X.FSI_ITER == 1              # OSS :70
    assert X.D == 64.0 and X.U0 == 0.05 and X.UR == 5.0


# ------------------------------------------------------------- model / physics
def test_stiffness_is_calibrated_on_the_wet_mass():
    """OSS :58 calibrates K on ``m_s + m_a``; the wet frequency must be FN."""
    for mr in X.MASS_RATIOS:
        m = X.build_model(mr)
        wn = math.sqrt(m["k"] / (m["m_s"] + m["m_a"]))
        assert wn == pytest.approx(m["wn"], rel=1e-12)
        assert m["m_a"] / m["m_s"] == pytest.approx(1.0 / mr, rel=1e-12)


def test_analytic_response_solves_the_ode_and_the_initial_conditions():
    m = X.build_model(10.0)
    t = np.linspace(0.0, 3.0 * X.TN, 4001)
    x = X.analytic(t, m)
    mtot = m["m_s"] + m["m_a"]
    f = m["f0"] * np.sin(m["w_f"] * t)
    # second derivative by central differences on a fine grid
    d2 = np.gradient(np.gradient(x, t), t)
    d1 = np.gradient(x, t)
    interior = slice(50, -50)
    resid = mtot * d2[interior] + m["c"] * d1[interior] + m["k"] * x[interior] - f[interior]
    assert np.max(np.abs(resid)) < 1e-3 * m["f0"]
    assert abs(x[0]) < 1e-12
    assert abs(d1[0]) < 1e-9


def test_vendored_newmark_is_the_newmark_that_tensorlbm_uses():
    a0 = np.array([1.0, -0.5])
    v0 = np.array([0.3, -0.2])
    d0 = np.array([0.1, 0.05])
    h = np.array([2.0, -1.0])
    m_s, k, c, dt = 10.0, 3.0, 0.4, 0.7
    a1, v1, d1 = X.dyn.newmark_2dof(a0, v0, d0, h, m_s, k, c, dt)
    c1, c2 = 0.5 * dt, 0.25 * dt * dt
    v_p = v0 + dt * 0.5 * a0
    d_p = d0 + dt * v0 + dt * dt * 0.25 * a0
    a_tl = (h - c * v_p - k * d_p) / (m_s + c1 * c + c2 * k)
    assert np.allclose(a1, a_tl, rtol=0, atol=1e-14)
    assert np.allclose(v1, v_p + c1 * a_tl, rtol=0, atol=1e-14)
    assert np.allclose(d1, d_p + c2 * a_tl, rtol=0, atol=1e-14)


# ------------------------------------------------- cross-implementation identities
DT = 16.0
N_STEPS = 1600  # 4 wet periods


def _series(leg: str, mr: float, n_steps: int = N_STEPS) -> np.ndarray:
    return X.run_leg(leg, X.build_model(mr), DT, n_steps)["x"]


def test_q1_tensorlbm_reproduces_the_oss_coupling_loop():
    """TL ``explicit``/no-accelerator/1 iteration == OSS loop, ``FSI_ITER=1``."""
    for mr in (2.0, 10.0):
        a = _series("oss_plain_1", mr)
        b = _series("tl_exp_1", mr)
        assert np.max(np.abs(a - b)) / np.max(np.abs(b)) < 1e-12


def test_q2_oss_fixed_point_is_tensorlbm_strong_coupling():
    """OSS ``FSI_ITER=40`` fixed point == TL Aitken / IQN-ILS converged solution."""
    a = _series("oss_plain_40", mr=10.0)
    for leg in ("tl_aitken", "tl_iqn"):
        b = _series(leg, 10.0)
        assert np.max(np.abs(a - b)) / np.max(np.abs(b)) < 1e-8


def test_q3_oss_added_mass_line_turns_the_scheme_into_the_dry_oscillator():
    """Line 136 (``h += a*CYL_AREA``) cancels the interface added mass.

    Under the documented linear interface operator this is an algebraic
    identity, not an approximation: the compensated loop equals TensorLBM run
    on a model with ``m_added = 0``.
    """
    for mr in (2.0, 10.0):
        a = _series("oss_compen_1", mr)
        b = _series("tl_dry", mr)
        assert np.max(np.abs(a - b)) / np.max(np.abs(b)) < 1e-12


def test_q3b_compensated_loop_is_iteration_invariant():
    """With the added-mass line on, ``FSI_ITER`` no longer changes the answer."""
    a = _series("oss_compen_1", 10.0)
    b = _series("oss_compen_10", 10.0)
    assert np.max(np.abs(a - b)) / np.max(np.abs(b)) < 1e-12


def test_plain_fixed_point_does_not_converge_at_mass_ratio_one():
    """``beta = m_a/m_s = 1``: more sub-iterations without relaxation diverge."""
    out = X.run_leg("oss_plain_40", X.build_model(1.0), DT, N_STEPS)
    assert out["unstable"] or np.max(np.abs(out["x"])) > 5.0 * X.X_TARGET
    # the accelerated TensorLBM solver stays bounded on the same problem
    out_tl = X.run_leg("tl_aitken", X.build_model(1.0), DT, N_STEPS)
    assert np.all(np.isfinite(out_tl["x"]))
    assert np.max(np.abs(out_tl["x"])) < 5.0 * X.X_TARGET


def test_metrics_are_zero_for_the_analytic_record_itself():
    m = X.build_model(10.0)
    t = (np.arange(N_STEPS) + 1.0) * DT
    out = dict(x=X.analytic(t, m), t=t, unstable=False,
               iters=np.ones(N_STEPS), wall=0.0)
    met = X.metrics(out, m)
    assert met["l2_rel_pct"] < 1e-10
    assert abs(met["amp_rel_pct"]) < 1e-6
    assert abs(met["phase_err_deg"]) < 1e-6


# --------------------------------------------------------------- artifact gate
ARTIFACT = ART / "xval_fsi_fixed_point.json"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="run_xval.py has not been run")
def test_artifact_verdicts_against_the_15pct_standard():
    rows = json.loads(ARTIFACT.read_text())["sweep"]["rows"]

    def get(leg, mr, dt):
        for r in rows:
            if r["leg"] == leg and r["mass_ratio"] == mr and r["dt"] == dt:
                return r
        raise KeyError((leg, mr, dt))

    # converged strong coupling (TL) is inside the owner tolerance everywhere
    for mr in (2.0, 10.0):
        for dt in (16.0, 64.0):
            r = get("tl_aitken", mr, dt)
            assert r["l2_rel_pct"] < X.TOL_PCT, (mr, dt, r["l2_rel_pct"])

    # the OSS added-mass line (= dry dynamics) misses the 15 % bar at m* <= 2
    for dt in (16.0, 64.0):
        assert get("oss_compen_1", 2.0, dt)["l2_rel_pct"] > X.TOL_PCT
        assert get("oss_compen_1", 10.0, dt)["l2_rel_pct"] > X.TOL_PCT

# ------------------------------------------- capability: theta-split coupling
def test_theta_split_is_exact_for_the_fixed_point():
    """``variant="theta_split", theta=1`` == converged explicit+Aitken in 1 iteration."""
    import math

    from tensorlbm.fsi_partitioned import AddedMassModel1D
    from tensorlbm.fsi_partitioned import make_accelerator, solve_partitioned_1d

    model = AddedMassModel1D(m_struct=1.0, m_added=2.0, stiffness=100.0, damping=0.2)
    force = lambda t: 0.5 * math.sin(2.0 * math.pi * t)  # noqa: E731
    kw = dict(n_steps=400, dt=0.01, forcing=force, tol=1e-14, ref_scale=1.0)
    ref = solve_partitioned_1d(model, variant="explicit", max_iter=200,
                               accelerator=make_accelerator("aitken"), **kw)
    got = solve_partitioned_1d(model, variant="theta_split", theta=1.0,
                               max_iter=1, **kw)
    scale = np.max(np.abs(ref.displacement))
    assert np.max(np.abs(ref.displacement - got.displacement)) / scale < 1e-12
    assert int(got.iterations.max()) == 1
    assert ref.iterations.mean() > 1.5          # the iterate really was needed

    zero = solve_partitioned_1d(model, variant="theta_split", theta=0.0,
                                max_iter=1, **kw)
    plain = solve_partitioned_1d(model, variant="explicit", max_iter=1, **kw)
    assert np.array_equal(zero.displacement, plain.displacement)


@pytest.mark.skipif(not ARTIFACT.exists(), reason="run_xval.py has not been run")
def test_theta_split_arm_is_inside_the_tolerance_at_every_mass_ratio():
    rows = json.loads(ARTIFACT.read_text())["sweep"]["rows"]
    got = [r for r in rows if r["leg"] == "tl_theta1"]
    assert got
    for r in got:
        assert r["l2_rel_pct"] is not None, r
        assert r["l2_rel_pct"] < X.TOL_PCT, (r["mass_ratio"], r["dt"], r["l2_rel_pct"])
        assert r["iters_max"] == 1.0
