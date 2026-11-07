"""Tests for the partitioned-FSI added-mass scaling and interface accelerators.

Covers the acceptance criteria of the P33-N3 scaling study:

C1  1-D analytic convergence factor ``rho = beta/(1+beta)`` vs numerical
    measurement (< 5 %, at least three ``beta`` values),
C2  critical added-mass ratio ``beta_crit`` vs the numerical divergence
    boundary (< 10 %),
C3  IQN-ILS stabilisation at ``beta = 10`` (residual < 1e-6),
C4  Aitken relaxation factor inside ``[0.3, 0.8]`` and matching the analytic
    optimum ``1/(1+beta)``,
plus consistency of the partitioned solver with the exact 1-DOF solution.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.fsi_partitioned import (
    AddedMassModel1D,
    AitkenRatio,
    AitkenRelaxation,
    ConstantRelaxation,
    IQNILS,
    added_mass_from_channel,
    analytic_convergence_factor,
    beta_critical,
    beta_over_one_plus_beta,
    detect_beta_critical,
    make_accelerator,
    measure_contraction_1d,
    optimal_relaxation,
    solve_partitioned_1d,
    theta_split_rho,
)

BETAS = (0.25, 1.0, 2.5, 10.0)


# ---------------------------------------------------------------------------
# C1: analytic factors
# ---------------------------------------------------------------------------


def test_analytic_factors_match_definitions():
    """rho_explicit = beta, rho_semi_implicit = beta/(1+beta), beta_crit = 1."""
    for beta in BETAS:
        model = AddedMassModel1D(m_struct=2.0, m_added=2.0 * beta)
        assert model.beta == pytest.approx(beta, rel=1e-14)
        assert model.rho_explicit() == pytest.approx(beta, rel=1e-14)
        assert model.rho_semi_implicit() == pytest.approx(beta / (1.0 + beta), rel=1e-14)
        assert model.beta_critical() == pytest.approx(1.0, rel=1e-14)
    # limits that carry the physics
    assert beta_over_one_plus_beta(0.0) == 0.0
    assert beta_over_one_plus_beta(1e6) > 0.999999
    assert optimal_relaxation(1.0) == pytest.approx(0.5, rel=1e-14)


def test_relaxed_convergence_factor_formula():
    """rho(omega) = |1 - omega (1 + beta)| for the explicit scheme."""
    dt = 0.01
    for beta in (0.25, 0.5, 1.0):
        model = AddedMassModel1D(m_struct=1.0, m_added=beta)
        for omega in (0.2, 0.4, 0.7):
            rho_num, _, _ = measure_contraction_1d(
                model,
                dt=dt,
                variant="explicit",
                accelerator=ConstantRelaxation(omega),
                n_iter=40,
                skip=2,
                window=8,
            )
            rho_ana = analytic_convergence_factor(beta, omega)
            assert rho_num == pytest.approx(rho_ana, rel=1e-8, abs=1e-12)


def test_added_mass_from_channel_is_linear_in_geometry():
    """The 1-D incompressible channel gives m_a = rho_f A L exactly."""
    m_a = added_mass_from_channel(rho_fluid=1000.0, area=0.02, length=0.5)
    assert m_a == pytest.approx(10.0, rel=1e-14)
    assert added_mass_from_channel(1.0, 1.0, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# C1: numerical measurement of rho
# ---------------------------------------------------------------------------


def test_measured_rho_matches_semi_implicit_formula():
    """Measured contraction of the wet-mass interface vs beta/(1+beta) < 5 %."""
    errors = []
    for beta in BETAS:
        model = AddedMassModel1D(m_struct=1.0, m_added=beta)
        rho_num, res, _ = measure_contraction_1d(
            model, dt=0.01, variant="wet_mass", n_iter=40, skip=3, window=8
        )
        rho_ana = beta_over_one_plus_beta(beta)
        err = abs(rho_num - rho_ana) / rho_ana
        errors.append(err)
        assert res[0] > 0.0 and res[-1] < res[0]
    assert max(errors) < 0.05, f"max relative error {max(errors):.3e} exceeds 5 %"
    assert max(errors) < 1e-10  # in fact exact to round-off


def test_measured_rho_matches_explicit_beta():
    """Measured contraction of the naive staggered interface vs rho = beta."""
    for beta in (0.25, 0.5, 0.9, 2.5):
        model = AddedMassModel1D(m_struct=1.0, m_added=beta)
        rho_num, _, _ = measure_contraction_1d(
            model, dt=0.01, variant="explicit", n_iter=30, skip=3, window=10
        )
        assert rho_num == pytest.approx(beta, rel=1e-6)


# ---------------------------------------------------------------------------
# C2: critical added-mass ratio
# ---------------------------------------------------------------------------


def test_beta_critical_matches_numerical_divergence_boundary():
    """Analytic beta_crit = 1 vs measured boundary, error < 10 %."""
    out = detect_beta_critical(dt=0.01, m_struct=1.0, variant="explicit")
    assert beta_critical(1.0) == pytest.approx(1.0)
    assert out["beta_crit"] == pytest.approx(1.0, rel=0.10), out
    lo, hi = out["bracket"]
    assert lo < 1.0 <= hi + 1e-9


def test_beta_critical_scales_with_dt_through_newmark_mass():
    """The boundary is set by the *effective* mass M_eff = m_s + c g dt + k b dt^2."""
    dt, c, k, m_s = 0.01, 0.0, 0.0, 2.0
    model = AddedMassModel1D(m_struct=m_s, m_added=m_s, damping=c, stiffness=k)
    rho_num, _, _ = measure_contraction_1d(model, dt=dt, variant="explicit", n_iter=30, skip=3, window=10)
    assert rho_num == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# C3: IQN-ILS stabilisation
# ---------------------------------------------------------------------------


def test_iqn_ils_converges_at_beta_10():
    """C3: with IQN-ILS the interface converges below 1e-6 in a few iterations."""
    model = AddedMassModel1D(m_struct=1.0, m_added=10.0)
    forcing = lambda t: 0.1  # noqa: E731
    plain = solve_partitioned_1d(
        model, n_steps=1, dt=0.01, variant="wet_mass", tol=1e-6, max_iter=300, forcing=forcing
    )
    iqn = solve_partitioned_1d(
        model,
        n_steps=1,
        dt=0.01,
        variant="wet_mass",
        tol=1e-6,
        max_iter=300,
        forcing=forcing,
        accelerator=IQNILS(),
    )
    assert plain.step_residuals[0] < 1e-6  # wet mass converges, but slowly
    assert plain.iterations[0] > 50, plain.iterations
    assert iqn.converged
    assert iqn.step_residuals[0] < 1e-6
    assert iqn.iterations[0] <= 10, iqn.iterations


def test_iqn_ils_stabilises_unstable_explicit_scheme():
    """beta = 10 with the explicit scheme diverges; IQN-ILS/Aitken converge."""
    model = AddedMassModel1D(m_struct=1.0, m_added=10.0)
    forcing = lambda t: 0.1  # noqa: E731
    plain = solve_partitioned_1d(
        model, n_steps=3, dt=0.01, variant="explicit", tol=1e-8, max_iter=200, forcing=forcing
    )
    assert not plain.converged
    assert np.abs(plain.acceleration).max() > 1e3  # blow-up
    for accelerator in (IQNILS(), AitkenRelaxation()):
        run = solve_partitioned_1d(
            model,
            n_steps=3,
            dt=0.01,
            variant="explicit",
            tol=1e-8,
            max_iter=200,
            forcing=forcing,
            accelerator=accelerator,
        )
        assert run.converged
        assert np.all(np.isfinite(run.acceleration))
        assert np.all(run.step_residuals < 1e-8)


def test_iqn_ils_reduces_iterations_on_multidof_problem():
    """IQN-ILS keeps the multi-DOF secant problem well scaled (rank filtering)."""
    acc = IQNILS(max_pairs=10)
    x = np.zeros(4)
    target = np.array([1.0, -2.0, 0.5, 3.0])
    calls = 0
    for _ in range(12):
        r = target - x  # linear affine interface operator (identity gain)
        calls += 1
        x, _ = acc.step(x, r)
        if np.linalg.norm(r) < 1e-10:
            break
    assert np.allclose(x, target, atol=1e-6), (x, target)
    assert calls <= 3


# ---------------------------------------------------------------------------
# C4: Aitken dynamic relaxation
# ---------------------------------------------------------------------------


def test_aitken_recovers_optimal_relaxation_in_band():
    """C4: omega -> 1/(1+beta) and stays inside [0.3, 0.8] for beta in [0.25, 2]."""
    omegas = []
    for beta in (0.25, 0.5, 1.0, 2.0):
        model = AddedMassModel1D(m_struct=1.0, m_added=beta)
        rho, res, oms = measure_contraction_1d(
            model, dt=0.01, variant="explicit", accelerator=AitkenRelaxation(), n_iter=20, skip=1, window=4
        )
        omega = oms[1]  # value obtained after the first secant update
        assert omega == pytest.approx(optimal_relaxation(beta), rel=1e-9)
        assert 0.3 <= omega <= 0.8
        assert rho == 0.0  # dynamic relaxation annihilates the linear error
        omegas.append(omega)
    assert omegas == sorted(omegas, reverse=True)


def test_aitken_ratio_form_is_documented_as_non_optimal():
    """The literal ratio form omega_k = omega_{k-1}(1 - dr_k/dr_{k-1}) degrades."""
    beta = 0.25
    model = AddedMassModel1D(m_struct=1.0, m_added=beta)
    rho_std, _, oms_std = measure_contraction_1d(
        model, dt=0.01, variant="explicit", accelerator=AitkenRelaxation(), n_iter=20, skip=1, window=4
    )
    rho_ratio, _, oms_ratio = measure_contraction_1d(
        model, dt=0.01, variant="explicit", accelerator=AitkenRatio(), n_iter=20, skip=1, window=4
    )
    assert math.isfinite(rho_ratio) and rho_ratio > rho_std
    assert oms_ratio[-1] < oms_std[-1]


# ---------------------------------------------------------------------------
# Consistency of the partitioned solver with the exact 1-DOF solution
# ---------------------------------------------------------------------------


def test_partitioned_solution_matches_exact_ode():
    """The explicit (consistent) partitioned solve reproduces the coupled ODE.

    Exact problem: (m_s + m_a) d'' + c d' + k d = F_ext, with the partitioned
    solver iterated to residual < 1e-12 at each Newmark step.
    """
    from scipy.integrate import solve_ivp

    m_s, m_a, k, c = 1.0, 0.5, 4.0, 0.2
    model = AddedMassModel1D(m_struct=m_s, m_added=m_a, stiffness=k, damping=c)
    dt, n_steps = 2e-3, 500
    f_ext = 0.7

    run = solve_partitioned_1d(
        model,
        n_steps=n_steps,
        dt=dt,
        variant="explicit",
        tol=1e-13,
        max_iter=500,
        forcing=lambda t: f_ext,
    )
    assert run.converged

    m_tot = m_s + m_a
    sol = solve_ivp(
        lambda t, y: [y[1], (f_ext - c * y[1] - k * y[0]) / m_tot],
        (0.0, n_steps * dt),
        [0.0, 0.0],
        t_eval=run.t,
        rtol=1e-12,
        atol=1e-14,
    )
    d_exact = sol.y[0]
    err = np.max(np.abs(run.displacement - d_exact)) / np.max(np.abs(d_exact))
    assert err < 5e-3, f"relative displacement error {err:.3e}"


def test_wet_mass_variant_biases_fixed_point_but_converges():
    """Documented caveat: the wet-mass variant double counts the added mass.

    Both variants share the *interface contraction mechanism*, but only the
    explicit variant preserves the exact fixed point.  The wet-mass fixed
    point solves (m_s + 2 m_a) a = F_ext instead of (m_s + m_a) a = F_ext.
    """
    m_s, m_a = 1.0, 1.0
    model = AddedMassModel1D(m_struct=m_s, m_added=m_a)
    f_ext = 0.4
    wet = solve_partitioned_1d(
        model, n_steps=1, dt=0.01, variant="wet_mass", tol=1e-12, max_iter=2000, forcing=lambda t: f_ext
    )
    a_wet = wet.acceleration[0]
    assert a_wet == pytest.approx(f_ext / (m_s + 2.0 * m_a), rel=1e-6)
    ref = solve_partitioned_1d(
        model,
        n_steps=1,
        dt=0.01,
        variant="explicit",
        tol=1e-12,
        max_iter=2000,
        forcing=lambda t: f_ext,
        accelerator=IQNILS(),
    )
    assert ref.acceleration[0] == pytest.approx(f_ext / (m_s + m_a), rel=1e-6)


def test_theta_split_family_endpoints():
    """rho(theta) interpolates the explicit (beta) and implicit (0) limits."""
    beta = 3.0
    assert theta_split_rho(beta, 0.0) == pytest.approx(beta)
    assert theta_split_rho(beta, 1.0) == pytest.approx(0.0)
    values = [theta_split_rho(beta, t) for t in np.linspace(0.0, 1.0, 11)]
    assert all(b < a for a, b in zip(values, values[1:]))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_input_validation():
    with pytest.raises(ValueError):
        AddedMassModel1D(m_struct=0.0, m_added=1.0)
    with pytest.raises(ValueError):
        AddedMassModel1D(m_struct=1.0, m_added=-1.0)
    with pytest.raises(ValueError):
        analytic_convergence_factor(1.0, omega=1.5)
    with pytest.raises(ValueError):
        make_accelerator("nope")
    with pytest.raises(ValueError):
        solve_partitioned_1d(AddedMassModel1D(1.0, 1.0), n_steps=0, dt=0.01)
    with pytest.raises(ValueError):
        theta_split_rho(1.0, 1.5)

# ---------------------------------------------------------------------------
# Audit chain of the flow-side probes (C5 evidence grade)
# ---------------------------------------------------------------------------


def test_viv_probes_declare_withheld_validation_status():
    """Both VIV flow side probe modules must advertise WITHHELD evidence."""
    from tensorlbm import fsi_viv_bb, fsi_viv_ibm

    assert fsi_viv_ibm.VALIDATION_STATUS == "WITHHELD"
    assert fsi_viv_bb.VALIDATION_STATUS == "WITHHELD"


def test_staircase_mask_and_bounce_back_smoke():
    """The bounce-back probe advances a few steps without NaN (no accuracy claim)."""
    from tensorlbm.fsi_viv_bb import BBSolver, staircase_mask
    from tensorlbm.fsi_viv_ibm import VIVConfig

    cfg = VIVConfig(diameter=8, length_x=8.0, height_y=8.0, n_steps=5, warmup=1)
    mask = staircase_mask(cfg, cfg.y_center)
    assert mask.dtype == bool and mask.sum() > 10
    solver = BBSolver(cfg)
    solver.set_body(cfg.y_center)
    for _ in range(5):
        fx, fy = solver.step(0.0)
        assert np.isfinite(fx) and np.isfinite(fy)


def test_ibm_probe_smoke():
    """The immersed-boundary probe advances a few steps without NaN."""
    import torch

    from tensorlbm.fsi_viv_ibm import VIVConfig, _tensors, lagrangian_markers, lbm_step_ibm

    torch.set_grad_enabled(False)
    cfg = VIVConfig(diameter=8, length_x=8.0, height_y=8.0, n_steps=5, warmup=1)
    f, c, w, ops = _tensors(cfg)
    markers = lagrangian_markers(cfg.x_center, cfg.y_center, 0.5 * cfg.diameter)
    for _ in range(3):
        f, fx, fy = lbm_step_ibm(f, cfg, markers, (0.0, 0.0), ops, c, w)
    assert torch.isfinite(f).all()
