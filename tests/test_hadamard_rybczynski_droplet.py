"""Tests for the revitalised Hadamard-Rybczynski droplet benchmark (node IN26-L2).

Covers three layers:

* analytic HR law + its two limits (``tensorlbm.hadamard_rybczynski``),
* viscosity-ratio / density-ratio decoupling, i.e. library gap G2
  (``tensorlbm.two_phase_viscosity``),
* the axisymmetric Stokes drop-drag solver and its post-processing.

The heavy production grid lives in
``benchmarks/hadamard_rybczynski_droplet_revival/run_revival.py``; the values
asserted here use deliberately coarse settings so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

import tensorlbm.hadamard_rybczynski as HR
import tensorlbm.two_phase_viscosity as TPV


# ---------------------------------------------------------------------------
# analytic Hadamard-Rybczynski law
# ---------------------------------------------------------------------------
def test_drag_factor_endpoints_and_midpoint():
    """(2+3L)/(1+L) -> 2 as L->0, -> 3 as L->inf, = 2.5 at L = 1."""
    assert HR.drag_coefficient_factor(0.0) == pytest.approx(2.0)
    assert HR.drag_coefficient_factor(1.0) == pytest.approx(2.5)
    assert HR.drag_coefficient_factor(1e12) == pytest.approx(3.0, rel=1e-9)
    assert HR.drag_coefficient_factor(1e-4) == pytest.approx(2.0, abs=1e-3)
    # strictly increasing in lambda (a more viscous drop is harder to deform
    # internally -> larger drag)
    lam = np.geomspace(1e-3, 1e3, 50)
    vals = np.array([HR.drag_coefficient_factor(v) for v in lam])
    assert np.all(np.diff(vals) > 0.0)


def test_drag_limit_functions_match_law_limits():
    mu, R, U = 0.7, 1.3, 0.02
    assert HR.hr_drag(mu, R, U, 1e-10) == pytest.approx(HR.bubble_limit_drag(mu, R, U), rel=1e-8)
    assert HR.hr_drag(mu, R, U, 1e10) == pytest.approx(HR.stokes_limit_drag(mu, R, U), rel=1e-8)
    assert HR.hr_drag(mu, R, U, 1.0) == pytest.approx(HR.equal_viscosity_drag(mu, R, U))


def test_hr_velocity_limits_are_classical():
    """lambda->0 gives the Rybczynski bubble speed, lambda->inf the Stokes one."""
    drho, g, R, mu = 10.0, 0.5, 0.2, 1.5
    u_bubble = HR.hr_velocity(drho, g, R, mu, 0.0)
    u_stokes = HR.hr_velocity(drho, g, R, mu, 1e12)
    assert u_bubble == pytest.approx(drho * g * R**2 / (3.0 * mu))
    assert u_stokes == pytest.approx(2.0 * drho * g * R**2 / (9.0 * mu))
    # a drop always settles between the bubble and the rigid-sphere speed
    for lam in (0.01, 0.3, 1.0, 10.0, 100.0):
        assert u_stokes < HR.hr_velocity(drho, g, R, mu, lam) < u_bubble


def test_force_balance_between_drag_and_velocity():
    """Buoyancy (4/3) pi R^3 drho g must equal the HR drag at the HR velocity."""
    drho, g, R, mu = 3.0, 0.7, 0.35, 0.9
    for lam in (0.0, 0.05, 1.0, 7.0, 1e6):
        U = HR.hr_velocity(drho, g, R, mu, lam)
        F = HR.hr_drag(mu, R, U, lam)
        assert F == pytest.approx((4.0 / 3.0) * np.pi * R**3 * drho * g, rel=1e-12)


def test_infer_lambda_roundtrip():
    mu, R, U = 1.1, 0.8, 0.05
    for lam in (1e-3, 0.02, 0.5, 1.0, 3.0, 50.0, 1e4):
        F = HR.hr_drag(mu, R, U, lam)
        assert HR.infer_lambda_from_drag(mu, R, U, F) == pytest.approx(lam, rel=1e-8)


def test_terminal_velocity_from_drag_matches_hr():
    drho, g, R, mu = 2.0, 0.4, 0.5, 1.3
    for lam in (0.01, 1.0, 100.0):
        F = HR.hr_drag(mu, R, 1.0, lam)
        assert HR.terminal_velocity_from_drag(F, drho, g, R) == pytest.approx(
            HR.hr_velocity(drho, g, R, mu, lam)
        )


@pytest.mark.parametrize("lam", [-1.0, np.nan, np.inf])
def test_invalid_lambda_rejected(lam):
    with pytest.raises(ValueError):
        HR.hr_drag(1.0, 1.0, 1.0, lam)


@pytest.mark.parametrize("kwarg", [{"R": 0.0}, {"mu_c": -1.0}, {"U": 0.0}])
def test_non_physical_inputs_rejected(kwarg):
    args = {"mu_c": 1.0, "R": 1.0, "U": 1.0, "lam": 1.0, **kwarg}
    with pytest.raises(ValueError):
        HR.hr_drag(**args)


def test_lambda_ladder_has_eight_log_spaced_points_with_both_endpoints():
    lams = HR.hr_lambda_ladder(8, 0.01, 100.0)
    assert lams.size == 8
    assert lams[0] == pytest.approx(0.01)
    assert lams[-1] == pytest.approx(100.0)
    ratios = lams[1:] / lams[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-10)  # log-spaced


# ---------------------------------------------------------------------------
# viscosity-ratio / density-ratio decoupling (library gap G2)
# ---------------------------------------------------------------------------
def test_decoupled_mapping_is_exact_for_every_density_ratio():
    """mu_d/mu_c == lambda regardless of rho_d/rho_c (the G2 fix)."""
    rng = np.random.default_rng(7)
    phi = np.round(rng.random((5, 6)))
    for rho_ratio in (1.0, 1.5, 2.0, 7.0, 25.0):
        rho = np.where(phi > 0.5, rho_ratio, 1.0)
        for lam in (0.01, 0.25, 1.0, 4.0, 100.0):
            tau, mu = TPV.pseudopotential_relaxation(phi, rho, 1.0, lam)
            assert TPV.viscosity_ratio_from_tau(tau, rho, phi) == pytest.approx(lam, rel=1e-12)
            assert np.all(tau > 0.5)


def test_legacy_mapping_reproduces_the_g2_lock():
    """The pre-fix constant-tau path ties lambda_eff to the density ratio."""
    phi = np.array([[0.0, 1.0], [0.0, 1.0]])
    for rho_ratio in (1.0, 2.0, 6.0):
        rho = np.where(phi > 0.5, rho_ratio, 1.0)
        tau = TPV.legacy_uniform_relaxation(rho, 1.0, 4.0)  # lambda=4 ignored
        assert np.allclose(tau, tau[0])                     # one tau for both phases
        assert TPV.viscosity_ratio_from_tau(tau, rho, phi) == pytest.approx(rho_ratio)


def test_phase_viscosity_schemes_hit_pure_phase_limits():
    phi = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    for scheme in ("linear", "harmonic", "geometric"):
        mu = TPV.phase_viscosity(phi, mu_c=2.0, lam=0.25, scheme=scheme)
        assert mu[0] == pytest.approx(2.0)
        assert mu[-1] == pytest.approx(0.5)
        assert np.all(np.diff(mu) < 0.0)  # monotone between the two phases
    # harmonic mean is the smallest of the three at the interface
    mid = {s: TPV.phase_viscosity(np.array([0.5]), 2.0, 0.25, scheme=s)[0]
           for s in ("linear", "harmonic", "geometric")}
    assert mid["harmonic"] < mid["geometric"] < mid["linear"]


def test_relaxation_time_inverts_the_viscosity_definition():
    rho = np.array([1.0, 2.0, 8.0])
    mu = np.array([0.1, 0.4, 3.2])
    tau = TPV.relaxation_time(rho, mu)
    assert np.allclose(rho * TPV.CS2 * (tau - 0.5), mu, rtol=1e-14)


def test_lattice_shear_wave_recovers_bgk_viscosity():
    """Independent check: the tau -> mu map on the real BGK kernel."""
    for tau in (0.6, 0.8, 1.2):
        nu = TPV.measure_uniform_viscosity(tau, ny=64, steps=600, sample_every=20)
        assert nu == pytest.approx(TPV.CS2 * (tau - 0.5), rel=5e-3)


def test_two_phase_contrast_realises_lambda_at_density_ratio_two():
    """Density ratio 2 with lambda = 4: mu ratio from measured nu must be 4."""
    phi = np.array([0.0, 1.0])
    rho = np.array([1.0, 2.0])
    tau, _ = TPV.pseudopotential_relaxation(phi, rho, 0.1, 4.0)
    nu_c = TPV.measure_uniform_viscosity(float(tau[0]))
    nu_d = TPV.measure_uniform_viscosity(float(tau[1]))
    mu_ratio = (rho[1] * nu_d) / (rho[0] * nu_c)
    assert mu_ratio == pytest.approx(4.0, rel=1e-2)


# ---------------------------------------------------------------------------
# axisymmetric Stokes drop-drag solver
# ---------------------------------------------------------------------------
def _extrapolated(lam, hrs=(10.0, 20.0), h_wall=0.1):
    drags = [HR.HRDragSolver(R=1.0, H_ratio=h, h_wall=h_wall).solve(lam).drag for h in hrs]
    return HR.wall_extrapolate(hrs, drags, order=1)["drag_inf"]


def test_solver_reproduces_rigid_sphere_stokes_drag():
    """lambda >> 1 must recover 6 pi mu R U in the unbounded limit."""
    drag_inf = _extrapolated(1e6)
    assert drag_inf == pytest.approx(6.0 * np.pi, rel=0.03)


def test_solver_reproduces_equal_viscosity_drop_drag():
    """lambda = 1 must recover 5 pi mu R U (the HR midpoint)."""
    drag_inf = _extrapolated(1.0)
    assert drag_inf == pytest.approx(5.0 * np.pi, rel=0.03)


def test_solver_drag_increases_with_viscosity_ratio():
    drags = [HR.HRDragSolver(R=1.0, H_ratio=10.0, h_wall=0.1).solve(lam).drag
             for lam in (0.01, 0.1, 1.0)]
    assert drags[0] < drags[1] < drags[2]


def test_solver_constraint_is_penalty_insensitive():
    """The element-mean divergence penalty must not bias the answer."""
    a = HR.HRDragSolver(R=1.0, H_ratio=10.0, h_wall=0.1, penalty=1e4).solve(1.0).drag
    b = HR.HRDragSolver(R=1.0, H_ratio=10.0, h_wall=0.1, penalty=1e7).solve(1.0).drag
    assert a == pytest.approx(b, rel=1e-3)


def test_solver_finite_container_drag_exceeds_unbounded_limit():
    """Confinement must increase the drag (sanity of the flow direction)."""
    confined = HR.HRDragSolver(R=1.0, H_ratio=5.0, h_wall=0.1).solve(1.0).drag
    loose = HR.HRDragSolver(R=1.0, H_ratio=20.0, h_wall=0.1).solve(1.0).drag
    assert confined > loose > 5.0 * np.pi


def test_solver_grid_convergence_order_at_least_one():
    H_ratio = 10.0
    ref = HR.HRDragSolver(R=1.0, H_ratio=H_ratio, h_wall=0.05).solve(1.0).drag
    coarse = HR.HRDragSolver(R=1.0, H_ratio=H_ratio, h_wall=0.4).solve(1.0).drag
    medium = HR.HRDragSolver(R=1.0, H_ratio=H_ratio, h_wall=0.2).solve(1.0).drag
    e_coarse = abs(coarse / ref - 1.0)
    e_medium = abs(medium / ref - 1.0)
    assert e_medium < e_coarse
    assert np.log(e_coarse / e_medium) / np.log(2.0) >= 1.0


def test_wall_extrapolate_recovers_a_linear_intercept():
    x = np.array([0.1, 0.05, 0.025])
    truth = 17.0
    drags = truth * (1.0 + 2.0 * x)
    fit = HR.wall_extrapolate(1.0 / x, drags, order=1)
    assert fit["drag_inf"] == pytest.approx(truth, rel=1e-12)
    assert fit["slope"] == pytest.approx(2.0 * truth, rel=1e-12)


def test_richardson_order_of_a_second_order_sequence():
    h = np.array([0.4, 0.2, 0.1])
    val = 3.0 + 5.0 * h**2
    assert HR.richardson_order(val[0], val[1], val[2], ratio=2.0) == pytest.approx(2.0)


def test_solver_reports_grid_and_constraint_diagnostics():
    res = HR.HRDragSolver(R=1.0, H_ratio=8.0, h_wall=0.15).solve(0.5)
    assert res.n_unknowns > 0 and res.n_elements > 0
    assert res.drag > 0.0
    assert np.isfinite(res.div_l2)
    assert res.drag_factor() == pytest.approx(res.drag / (2.0 * np.pi), rel=1e-12)
    assert res.rel_error_vs_hr() == pytest.approx(
        res.drag / HR.hr_drag(1.0, 1.0, 1.0, 0.5) - 1.0, rel=1e-12
    )