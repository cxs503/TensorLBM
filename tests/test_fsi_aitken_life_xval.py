"""Tests for the displacement-Aitken under-relaxation XVAL
(anchor XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXE).

Acceptance lines pinned here:

T1 recurrence identity: ``AitkenRelaxation(clip=False)`` reproduces the raw
   joconnor22/LIFE secant ``relax = -relax (r_{k-1}.dr)/|dr|^2`` on the same
   residual sequence (relative deviation < 1e-12, three residual magnitudes),
T2 the order-2 interface predictor (LIFE's ``2.5 U_n - 2 U_{n-1} + 0.5 U_{n-2}``)
   never costs iterations on the added-mass problem and saves them when the
   fixed point is slow,
T3 the pre-fix clip floor ``omega_min = 0.05`` is *divergent* for
   ``beta_eff > 39`` while the shipped default (``1e-4``) and ``clip=False``
   converge, and all converged arms land on the monolithic Newmark solution,
T4 the LIFE transcription and the repo driver driven with LIFE conventions
   agree on the recovered relaxation factor ``1/(1+beta_eff)`` (< 15 %) and on
   the converged interface displacement (< 15 %).
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "xval", "aitken_underrelaxation_life"))

import reference as R  # noqa: E402

from tensorlbm.fsi_partitioned import (  # noqa: E402
    AddedMassModel1D,
    AitkenRelaxation,
    solve_partitioned_1d,
)

TOL_REL = 1e-8
REF_LEN = 1.0


def _case(beta: float, steps_per_period: int = 200):
    p0 = R.Piston(m_struct=1.0, beta_eff=beta, stiffness=100.0, damping=0.3, dt=0.0)
    p = R.Piston(m_struct=1.0, beta_eff=beta, stiffness=100.0, damping=0.3,
                 dt=p0.period / steps_per_period)
    omega_f = 2.0 * math.pi / p.period
    f_ext = lambda t: math.sin(omega_f * t)  # noqa: E731
    model = AddedMassModel1D(m_struct=p.m_struct, m_added=p.m_added,
                             stiffness=p.stiffness, damping=p.damping)
    scale = REF_LEN / (R.BETA_N * p.dt * p.dt)
    return p, f_ext, model, scale


def test_t1_recurrence_identity_with_life_secant():
    rng = np.random.default_rng(20261113)
    for scale in (1.0, 1e-3, 1e3):
        seq = rng.normal(size=(12, 3)) * scale
        acc = AitkenRelaxation(omega0=1.0, clip=False)
        ours, life, relax, r_prev = [], [], 1.0, None
        for r in seq:
            _, w = acc.step(np.zeros(3), r)
            if r_prev is not None:
                dr = r - r_prev
                den = float(dr @ dr)
                if den > 1e-300:
                    cand = -relax * float(r_prev @ dr) / den
                    if math.isfinite(cand):
                        relax = cand
                life.append(relax)
            r_prev = r
            ours.append(w)
        a = np.asarray(ours[1:])
        b = np.asarray(life)
        dev = np.max(np.abs(a - b) / np.abs(b))
        assert dev < 1e-12, f"recurrence deviation {dev} at residual scale {scale}"


def test_t1b_clipping_is_the_only_difference_at_high_beta():
    """With clip on, the factor is floored at omega_min; with clip off it is raw."""
    seq = [np.array([1.0]), np.array([0.6]), np.array([0.55]), np.array([0.552])]
    acc_clip = AitkenRelaxation(omega0=1.0, clip=True, omega_min=0.05)
    acc_raw = AitkenRelaxation(omega0=1.0, clip=False)
    w_clip, w_raw = [], []
    for r in seq:
        w_clip.append(acc_clip.step(np.zeros(1), r)[1])
        w_raw.append(acc_raw.step(np.zeros(1), r)[1])
    assert w_clip[-1] >= 0.05 - 1e-15
    assert abs(w_raw[-1] - w_clip[-1]) > 0.0 or abs(w_raw[-1]) < 0.05


@pytest.mark.parametrize("beta", [2.0, 25.0])
def test_t2_predictor_never_costs_iterations(beta):
    p, f_ext, model, scale = _case(beta)
    it = {}
    for pred in ("none", "linear", "quadratic"):
        acc = AitkenRelaxation(omega0=0.5)
        res = solve_partitioned_1d(model, n_steps=200, dt=p.dt, variant="explicit",
                                   forcing=f_ext, accelerator=acc, tol=TOL_REL,
                                   max_iter=200, ref_scale=scale, predictor=pred)
        assert res.converged
        it[pred] = float(np.mean(res.iterations))
    assert it["quadratic"] <= it["none"] + 1e-9
    assert it["linear"] <= it["none"] + 1e-9


def test_t3_clip_floor_005_diverges_above_beta_39():
    beta = 50.0
    p, f_ext, model, scale = _case(beta, steps_per_period=100)
    n = 60

    def run(**kw):
        acc = AitkenRelaxation(omega0=0.5, **kw)
        return solve_partitioned_1d(model, n_steps=n, dt=p.dt, variant="explicit",
                                    forcing=f_ext, accelerator=acc, tol=TOL_REL,
                                    max_iter=200, ref_scale=scale), acc

    old, _ = run(omega_min=0.05)
    new, acc_new = run()                    # shipped default: omega_min = 1e-4
    raw, _ = run(clip=False)
    assert old.converged is False, "clip floor 0.05 must not converge at beta=50"
    assert new.converged and raw.converged
    # both converged arms reproduce the monolithic Newmark solution
    mono = R.monolithic_newmark(p, n_steps=n, f_ext=f_ext)
    scale_d = float(np.max(np.abs(mono["d"])))
    for res in (new, raw):
        err = float(np.max(np.abs(res.displacement - mono["d"]))) / scale_d
        assert err < 1e-4, err
    assert abs(acc_new.omega - p.omega_star) / p.omega_star < 0.15


@pytest.mark.parametrize("beta", [2.0, 10.0, 50.0])
def test_t4_life_transcription_matches_repo_driver(beta):
    p, f_ext, model, scale = _case(beta)
    n = 200

    acc = AitkenRelaxation(omega0=1.0, clip=False)
    ours = solve_partitioned_1d(model, n_steps=n, dt=p.dt, variant="explicit",
                               forcing=f_ext, accelerator=acc, tol=TOL_REL,
                               max_iter=200, ref_scale=scale, predictor="quadratic")
    life = R.life_run(p, n_steps=n, f_ext=f_ext, tol=TOL_REL, max_iter=200, ref_len=REF_LEN)
    assert ours.converged and life["converged"]
    # recovered relaxation factor, measured on the bare fixed-point map (the
    # driver's last step may already converge on the predictor, in which case
    # no factor is applied and acc.omega keeps its initial value)
    m_eff = p.m_eff_solid
    a, acc_probe, w_probe = 1.0, AitkenRelaxation(omega0=1.0, clip=False), 1.0
    for _ in range(20):
        a_s = (1.0 - p.m_added * a) / m_eff
        r = a_s - a
        if abs(r) < 1e-14:
            break
        x_new, w_probe = acc_probe.step(np.array([a]), np.array([r]))
        a = float(x_new[0])
    dev = abs(w_probe - p.omega_star) / p.omega_star
    assert dev < 0.15, (w_probe, p.omega_star)
    w_life = float(life["omega"][-1])
    assert abs(w_life - p.omega_star) / p.omega_star < 1e-12
    # converged interface displacement (normalised to the monolithic amplitude)
    mono = R.monolithic_newmark(p, n_steps=n, f_ext=f_ext)
    scale_d = float(np.max(np.abs(mono["d"])))
    e_ours = float(np.max(np.abs(ours.displacement - mono["d"]))) / scale_d
    e_life = float(np.max(np.abs(life["d"] - mono["d"]))) / scale_d
    assert e_ours < 0.15 and e_life < 0.15
    # sub-iteration cost: the order-2 predictor can only save iterations.  The
    # residual difference observed at small beta (2.18 vs 3.00 at beta=2) is the
    # predictor + stop-convention effect, predeclared as "attribute, do not gate";
    # at beta >= 10 the two loops must agree within the owner 15 % criterion.
    it_ours = float(np.mean(ours.iterations))
    it_life = float(np.mean(life["iters"]))
    assert it_ours <= 1.05 * it_life
    if beta >= 10.0:
        assert abs(it_ours - it_life) / it_life < 0.15, (it_ours, it_life)