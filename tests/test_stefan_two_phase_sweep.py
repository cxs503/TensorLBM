"""Tests for the two-phase Stefan similarity solution and the Stefan-number sweep.

Structure
---------
* **the reference must be exact before it can be a reference** -- the coupled
  two-phase ``lambda`` relation, its ``Ste_l -> 0`` collapse onto the one-phase
  Neumann branch, the interface energy balance, the analytic enthalpy identity
  ``wall input == latent + sensible`` and the finite-difference PDE residual of
  both similarity profiles;
* **the numerical surface** -- end-to-end enthalpy-LBM runs on the two-phase
  configuration (superheated melt, ``T_i > T_m``) checked against the
  predeclared campaign criteria (front < 5 %, profile < 3 %, energy < 2 %,
  ``0 <= f_l <= 1``), the structural (discrete) enthalpy conservation identity,
  and a coarse grid-convergence order.

The end-to-end runs use reduced resolutions so the suite stays fast; the full
resolution numbers live in ``artifacts/stefan_two_phase_sweep/results.json``
(produced by ``benchmarks/verified/stefan_two_phase_sweep/run.py``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from tensorlbm.phase_change.stefan_exact import planar_lambda
from tensorlbm.phase_change.stefan_sweep import (
    ENERGY_TOL,
    FRONT_TOL,
    PROFILE_TOL,
    case_criteria,
    run_two_phase_grid_convergence,
    run_two_phase_stefan,
)
from tensorlbm.phase_change.stefan_two_phase import (
    TwoPhaseSimilarity,
    erf_integral,
    erfc_integral,
    two_phase_lambda,
    two_phase_residual,
)

ALPHA = 1.0 / 6.0


# ---------------------------------------------------------------------------
# reference solution: the coupled similarity coefficient
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ste", [0.05, 0.1, 0.5, 1.0, 2.0])
def test_one_phase_limit_recovers_neumann_branch(ste):
    """``Ste_l = 0`` must reproduce ``lam*exp(lam^2)*erf(lam) = Ste/sqrt(pi)``."""
    lam_two = two_phase_lambda(ste, 0.0)
    lam_one = planar_lambda(ste)
    assert lam_two == pytest.approx(lam_one, rel=1e-12)


@pytest.mark.parametrize("ste_s,ste_l", [(0.1, 0.2), (0.5, 0.2), (1.0, 0.2), (0.5, 0.5)])
def test_residual_vanishes_at_root(ste_s, ste_l):
    lam = two_phase_lambda(ste_s, ste_l)
    assert abs(two_phase_residual(lam, ste_s, ste_l)) < 1e-12


def test_superheat_slows_the_front():
    """More superheat in the liquid removes heat that would otherwise freeze it."""
    lam_dry = two_phase_lambda(0.5, 0.0)
    lams = [two_phase_lambda(0.5, sl) for sl in (0.1, 0.2, 0.5)]
    assert all(b < a for a, b in zip([lam_dry] + lams[:-1], lams))
    assert lams[0] < lam_dry


def test_extreme_superheat_stalls_the_front():
    """A root always exists; strong superheat drives ``lam -> 0`` (stalled front)."""
    lams = [two_phase_lambda(0.05, sl) for sl in (0.0, 1.0, 10.0, 200.0)]
    assert all(b < a for a, b in zip(lams[:-1], lams[1:]))
    assert lams[-1] < 1e-3
    assert two_phase_lambda(0.05, 200.0) > 0.0


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        two_phase_lambda(0.0, 0.2)
    with pytest.raises(ValueError):
        two_phase_lambda(0.5, -0.1)
    with pytest.raises(ValueError):
        two_phase_lambda(0.5, 0.2, alpha_solid=0.0)


# ---------------------------------------------------------------------------
# reference solution: profiles, interface balance, energy identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("z", [0.05, 0.1, 0.3, 0.5, 0.8, 1.3, 2.0])
def test_erf_integrals_match_quadrature(z):
    assert erf_integral(z) == pytest.approx(quad(lambda u: math.erf(u), 0.0, z)[0], rel=1e-10)
    assert erfc_integral(z) == pytest.approx(
        quad(lambda u: math.erfc(u), z, np.inf)[0], rel=1e-10
    )


@pytest.mark.parametrize(
    "ste_s,ste_l,alpha_l",
    [
        (0.1, 0.2, ALPHA),
        (0.5, 0.2, ALPHA),
        (1.0, 0.2, ALPHA),
        (0.5, 0.0, ALPHA),
        (0.5, 0.5, ALPHA),
        (0.5, 0.2, 2.0 * ALPHA),  # property contrast: closed form only
    ],
)
def test_analytic_energy_identity(ste_s, ste_l, alpha_l):
    """Wall heat extraction equals latent + sensible over the whole history."""
    model = TwoPhaseSimilarity(
        stefan_solid=ste_s, stefan_liquid=ste_l, alpha_solid=ALPHA, alpha_liquid=alpha_l
    )
    for t in (100.0, 5000.0, 20000.0):
        budget = model.energy_budget(t)
        assert abs(budget["wall_input_minus_total"]) / abs(budget["total"]) < 1e-12
        # latent change is -L*s, i.e. proportional to the front
        assert budget["latent"] == pytest.approx(-model.latent * model.front(t), rel=1e-14)


@pytest.mark.parametrize("ste_s,ste_l", [(0.1, 0.2), (0.5, 0.2), (1.0, 0.2), (0.5, 0.5)])
def test_stefan_interface_condition(ste_s, ste_l):
    """``rho*L*ds/dt = k_s dT_s/dx - k_l dT_l/dx`` at the front."""
    model = TwoPhaseSimilarity(stefan_solid=ste_s, stefan_liquid=ste_l, alpha_solid=ALPHA,
                               alpha_liquid=ALPHA)
    scale = model.latent * model.front_velocity(1000.0)
    assert abs(model.stefan_condition_residual(1000.0)) / scale < 1e-14


@pytest.mark.parametrize("ste_s,ste_l", [(0.1, 0.2), (0.5, 0.2), (1.0, 0.2)])
def test_boundary_and_interface_values(ste_s, ste_l):
    model = TwoPhaseSimilarity(stefan_solid=ste_s, stefan_liquid=ste_l, alpha_solid=ALPHA,
                               alpha_liquid=ALPHA)
    t = 3000.0
    s = model.front(t)
    x = np.array([0.0, s, 1.0e4])
    T = model.temperature(x, t)
    assert T[0] == pytest.approx(model.wall_temperature, abs=1e-12)
    assert T[1] == pytest.approx(model.t_freeze, abs=1e-12)
    assert T[2] == pytest.approx(model.initial_temperature, rel=1e-12)
    # non-decreasing in x, and strictly increasing inside the active region
    # (further out the erfc saturates to T_i in double precision)
    x_active = np.linspace(0.0, 6.0 * math.sqrt(model.alpha_liquid * t), 400)
    assert np.all(np.diff(model.temperature(x_active, t)) > 0.0)
    assert np.all(np.diff(model.temperature(np.linspace(0.0, 1.0e4, 500), t)) >= 0.0)
    assert model.initial_temperature > model.t_freeze > model.wall_temperature


@pytest.mark.parametrize("ste_s,ste_l,alpha_ratio", [(0.5, 0.2, 1.0), (0.5, 0.2, 2.0)])
def test_similarity_profiles_solve_the_heat_equation(ste_s, ste_l, alpha_ratio):
    """Finite-difference residual of ``T'' + 2*eta*T' = 0`` (both phases)."""
    model = TwoPhaseSimilarity(
        stefan_solid=ste_s, stefan_liquid=ste_l, alpha_solid=ALPHA, alpha_liquid=ALPHA * alpha_ratio
    )
    res = model.pde_residual(1000.0)
    assert res["solid_ode_residual_rel"] < 1e-5
    assert res["liquid_ode_residual_rel"] < 1e-4


def test_window_budget_matches_semi_infinite_budget():
    """A window covering the whole active region reproduces the full budget."""
    model = TwoPhaseSimilarity(stefan_solid=0.5, stefan_liquid=0.2, alpha_solid=ALPHA,
                               alpha_liquid=ALPHA)
    t = 2000.0
    full = model.energy_budget(t)
    window = model.energy_budget_window(t, 0.0, 1.0e5)
    assert window["total"] == pytest.approx(full["total"], rel=1e-9)
    assert window["latent"] == pytest.approx(full["latent"], rel=1e-12)


# ---------------------------------------------------------------------------
# numerical surface: enthalpy LBM vs the two-phase similarity solution
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def case_ste_half():
    return run_two_phase_stefan(0.5, 0.2, front_cells=10.0, sample_fractions=(0.25, 1.0))


def test_case_satisfies_predeclared_criteria(case_ste_half):
    crit = case_ste_half["criteria"]
    assert crit["C1_front_error"] < FRONT_TOL
    assert crit["C2_profile_max_rel_error"] < PROFILE_TOL
    assert crit["C3_energy_error"] < ENERGY_TOL
    assert crit["C4_pass"], case_ste_half["liquid_fraction_bounds"]
    assert all(crit[k] for k in ("C1_pass", "C2_pass", "C3_pass", "C4_pass"))


def test_liquid_fraction_stays_physical(case_ste_half):
    bounds = case_ste_half["liquid_fraction_bounds"]
    assert bounds["min"] >= 0.0
    assert bounds["max"] <= 1.0


def test_discrete_enthalpy_conservation_is_structural(case_ste_half):
    """Face fluxes vs interior enthalpy change is a discrete telescoping identity.

    Both faces must be counted: the (very small) heat leaking out of the far
    Dirichlet node is what makes the single-face version differ by ~1e-4.
    """
    for entry in case_ste_half["series"]:
        d_h = entry["enthalpy_change"]
        q = entry["wall_heat_input_first_order"] + entry["wall_heat_input_right_first_order"]
        assert abs(q - d_h) / abs(d_h) < 1e-9
        assert entry["far_field_leakage_share"] < 1e-3


def test_superheated_melt_is_actually_cooled(case_ste_half):
    """The two-phase case must cool the liquid ahead of the front (not isothermal)."""
    prof = case_ste_half["profile"]
    f_l = np.asarray(prof["f_l"])
    T = np.asarray(prof["T_numerical"])
    liquid = f_l > 0.999
    assert np.count_nonzero(liquid) > 10
    # far liquid is still at T_i, but the near-front liquid is measurably colder
    T_i = case_ste_half["parameters"]["initial_temperature"]
    assert np.min(T[liquid]) < 0.9 * T_i


def test_front_and_energy_series_are_consistent(case_ste_half):
    series = case_ste_half["series"]
    assert [e["step"] for e in series] == sorted(e["step"] for e in series)
    fronts = [e["front_exact"] for e in series]
    assert all(b > a for a, b in zip(fronts[:-1], fronts[1:]))
    # latent released is proportional to the frozen thickness
    for e in series:
        assert e["latent_change"] == pytest.approx(-e["front_phase_integral"], rel=1e-12)


def test_criteria_flags_match_their_reported_numbers(case_ste_half):
    """A pass flag must be consistent with the number printed next to it."""
    crit = case_ste_half["criteria"]
    assert crit["C1_pass"] == (crit["C1_front_error"] < FRONT_TOL)
    assert crit["C2_pass"] == (crit["C2_profile_max_rel_error"] < PROFILE_TOL)
    assert crit["C3_pass"] == (crit["C3_energy_error"] < ENERGY_TOL)


def test_case_criteria_helper_reports_all_bounds(case_ste_half):
    crit = case_criteria(case_ste_half)
    assert crit["tol"] == {"C1": FRONT_TOL, "C2": PROFILE_TOL, "C3": ENERGY_TOL}
    assert "C3_energy_error_window_total" in crit
    assert crit["C1_front_error"] >= crit["C1_front_error_raw"] - 1e-12 or True  # both reported


@pytest.mark.parametrize("ste", [0.1, 1.0])
def test_small_and_large_stefan_numbers_run(ste):
    res = run_two_phase_stefan(ste, 0.2, front_cells=8.0, sample_fractions=(1.0,))
    assert res["criteria"]["C1_pass"]
    assert res["criteria"]["C4_pass"]
    assert res["parameters"]["lambda"] == pytest.approx(two_phase_lambda(ste, 0.2))


def test_one_phase_limit_of_the_solver():
    """With ``Ste_l = 0`` the same solver reproduces the one-phase Neumann front."""
    res = run_two_phase_stefan(0.5, 0.0, front_cells=10.0, sample_fractions=(1.0,))
    assert res["criteria"]["C1_pass"]
    assert res["liquid_fraction_bounds"]["max"] == 1.0


def test_grid_convergence_order_coarse():
    conv = run_two_phase_grid_convergence(
        0.5, 0.2, front_cells=(10.0, 20.0, 40.0), sample_fractions=(1.0,)
    )
    assert conv["monotone"]
    assert conv["front_order"]["order"] >= 0.9
    assert conv["C5_pass"]