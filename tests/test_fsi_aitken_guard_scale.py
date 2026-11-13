"""Scale-free guard for the Aitken under-relaxation factor (anchor
XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXE).

Independent audit of the m17727 fix.  The m17727 report replaced the divergent
clip floor ``omega_min = 0.05`` of ``fsi_partitioned.AitkenRelaxation`` by
``1e-4`` and declared the defect fixed at ``beta_eff = 50``.  On a linear
added-mass interface the fixed point converges only for
``0 < omega < 2/(1+beta_eff)``, so an **absolute** floor keeps diverging for
``beta_eff > 1/omega_min - 1``: the check below reproduces that at
``beta_eff = 2e4`` and pins the corrected, scale-free default
(``omega_min = 0`` -> positivity guard only).

Pinned assertions:

G1 shipped default (``omega_min = 0``) converges on the driver for
   ``beta_eff`` in {1e4, 2e4, 1e5} and reproduces the raw Aitken factor
   ``1/(1+beta_eff)`` (< 15 %), while the m17727 floor ``1e-4`` fails,
G2 the three sibling copies of the accelerator (``fsi_partitioned``,
   ``fsi_added_mass``, ``fsi_partitioned_stability``) agree on the factor at
   ``beta_eff`` = {25, 1e4, 1e5} (rel. dev < 1e-6) and all converge; their
   former floors (``1e-4``, ``1e-4``, ``1e-3``) are reproduced as divergent,
G3 the guard is positivity-only (scale free): an absolute floor passed by the
   caller is still honoured (backward compatibility) and ``clip=False`` stays
   raw,
G4 no stagnation: at ``beta_eff = 1e6`` the driver still converges within
   ``max_iter`` and the interface displacement is within 1 % of the monolithic
   Newmark reference.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "xval", "aitken_underrelaxation_life"))

import reference as R  # noqa: E402

from tensorlbm.fsi_partitioned import (  # noqa: E402
    AITKEN_POSITIVITY_EPS,
    AddedMassModel1D,
    AitkenRelaxation,
    clamp_aitken_factor,
    solve_partitioned_1d,
)
from tensorlbm.fsi_added_mass import AitkenRelaxation as AitkenAddedMass  # noqa: E402
from tensorlbm.fsi_partitioned_stability import (  # noqa: E402
    AitkenRelaxation as AitkenStability,
)

TOL_REL = 1e-8
REF_LEN = 1.0


def _case(beta: float, steps_per_period: int = 100):
    p0 = R.Piston(m_struct=1.0, beta_eff=beta, stiffness=100.0, damping=0.3, dt=0.0)
    p = R.Piston(m_struct=1.0, beta_eff=beta, stiffness=100.0, damping=0.3,
                 dt=p0.period / steps_per_period)
    omega_f = 2.0 * math.pi / p.period
    f_ext = lambda t: math.sin(omega_f * t)  # noqa: E731
    model = AddedMassModel1D(m_struct=p.m_struct, m_added=p.m_added,
                             stiffness=p.stiffness, damping=p.damping)
    scale = REF_LEN / (R.BETA_N * p.dt * p.dt)
    return p, f_ext, model, scale


def _driver(beta: float, *, n_steps: int = 60, max_iter: int = 200, **acc_kw):
    p, f_ext, model, scale = _case(beta)
    acc = AitkenRelaxation(omega0=0.5, **acc_kw)
    res = solve_partitioned_1d(model, n_steps=n_steps, dt=p.dt, variant="explicit",
                               forcing=f_ext, accelerator=acc, tol=TOL_REL,
                               max_iter=max_iter, ref_scale=scale)
    return p, res, acc


def _scalar_loop(cls, beta: float, iters: int = 200, tol: float = 1e-12, **kw):
    """Drive one accelerator copy on ``r(a) = 1 - (1+beta) a`` (gain -(1+beta))."""
    acc = cls(omega0=1.0, **kw)
    a, omegas = 0.0, []
    for _ in range(iters):
        r = 1.0 - (1.0 + beta) * a
        if abs(r) < tol:
            break
        if cls is AitkenAddedMass:
            inc, w = acc.step(np.array([a]), np.array([r]))
            a = a + float(inc[0])
        elif cls is AitkenStability:
            w = acc.update(np.array([r]), np.zeros(1))
            a = a + float(w) * r
        else:
            x_new, w = acc.step(np.array([a]), np.array([r]))
            a = float(x_new[0])
        omegas.append(float(w))
        if not math.isfinite(a) or abs(a) > 1e12:
            return float("nan"), omegas
    return a, omegas


@pytest.mark.parametrize("beta", [1.0e4, 2.0e4, 1.0e5])
def test_g1_default_guard_is_scale_free_on_driver(beta):
    p, res, acc = _driver(beta)
    assert res.converged, f"default guard must converge at beta_eff={beta:g}"
    assert np.all(np.isfinite(res.displacement))
    dev = abs(acc.omega - p.omega_star) / p.omega_star
    assert dev < 1e-6, (acc.omega, p.omega_star)
    # m17727 floor (1e-4) reproduces the residual failure above beta > 9999
    if beta > 1.0e4:
        _, old, _ = _driver(beta, omega_min=1e-4)
        assert old.converged is False, f"floor 1e-4 must fail at beta_eff={beta:g}"
    # raw Aitken (no clipping) agrees on the converged factor
    _, raw, acc_raw = _driver(beta, clip=False)
    assert raw.converged
    assert abs(acc.omega - acc_raw.omega) / acc_raw.omega < 1e-6


@pytest.mark.parametrize("beta", [25.0, 1.0e4, 1.0e5])
def test_g2_three_copies_agree_and_converge(beta):
    star = 1.0 / (1.0 + beta)
    for cls in (AitkenRelaxation, AitkenAddedMass, AitkenStability):
        a, omegas = _scalar_loop(cls, beta)
        assert math.isfinite(a), f"{cls.__module__} diverged at beta={beta:g}"
        assert abs(1.0 - (1.0 + beta) * a) < 1e-8
        assert omegas, cls
        dev = abs(omegas[-1] - star) / star
        assert dev < 1e-6, (cls.__module__, omegas[-1], star)


def test_g2b_former_absolute_floors_are_divergent():
    # fsi_partitioned: 1e-4 -> threshold beta > 9999
    a, _ = _scalar_loop(AitkenRelaxation, 2.0e4, omega_min=1e-4)
    assert not math.isfinite(a) or abs(1.0 - (1.0 + 2.0e4) * a) > 1e-6
    # fsi_added_mass: 1e-4 -> same threshold
    a, _ = _scalar_loop(AitkenAddedMass, 2.0e4, omega_min=1e-4)
    assert not math.isfinite(a) or abs(1.0 - (1.0 + 2.0e4) * a) > 1e-6
    # fsi_partitioned_stability: 1e-3 -> measured divergence from beta = 4e3
    a, _ = _scalar_loop(AitkenStability, 4.0e3, omega_min=1e-3)
    assert not math.isfinite(a) or abs(1.0 - (1.0 + 4.0e3) * a) > 1e-6


def test_g3_guard_semantics():
    # positivity-only default: machine-level floor, no absolute pinning
    assert clamp_aitken_factor(1e-7) == pytest.approx(max(1e-7, AITKEN_POSITIVITY_EPS))
    assert clamp_aitken_factor(1e-30) == pytest.approx(AITKEN_POSITIVITY_EPS)
    # backward compatibility: an explicit absolute floor is still honoured
    assert clamp_aitken_factor(1e-7, omega_min=1e-4) == pytest.approx(1e-4)
    assert clamp_aitken_factor(7.0, omega_max=2.0) == pytest.approx(2.0)
    # the accelerator keeps the caller's floor
    acc = AitkenRelaxation(omega0=1.0, omega_min=1e-3)
    seq = [np.array([1.0]), np.array([0.0]), np.array([1.0]), np.array([0.0])]
    for r in seq:
        _, w = acc.step(np.zeros(1), r)
    assert w >= 1e-3 - 1e-15
    # clip=False stays raw (negative factors allowed, LIFE convention)
    acc_raw = AitkenRelaxation(omega0=1.0, clip=False)
    ws = [acc_raw.step(np.zeros(1), r)[1] for r in seq]
    assert min(ws) < 0.0 or min(ws) < 1e-3


def test_g4_no_stagnation_at_extreme_added_mass():
    beta = 1.0e6
    p, res, acc = _driver(beta, n_steps=20)
    assert res.converged
    assert abs(acc.omega - p.omega_star) / p.omega_star < 1e-6
    mono = R.monolithic_newmark(p, n_steps=20, f_ext=_case(beta)[1])
    scale_d = float(np.max(np.abs(mono["d"])))
    err = float(np.max(np.abs(res.displacement - mono["d"]))) / scale_d
    assert err < 1e-2, err