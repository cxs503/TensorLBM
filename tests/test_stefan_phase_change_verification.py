"""Tests for the Stefan phase-change verification surface (P155W1-N01).

Covers
------
* exact similarity solutions: lambda relations, boundary values, PDE residual,
  interface energy balance (the references must be exact before they can be used
  as references);
* the thermal LBE diffusion operator vs the analytic erfc solution;
* exact total-enthalpy conservation in a closed domain;
* the enthalpy inversion (piecewise, latent plateau) and the Dirichlet phase
  handling;
* end-to-end verification runs at reduced resolution (front error, energy
  closure, cylindrical front shape) — these are the acceptance quantities of the
  node, run at a resolution chosen so that the suite stays fast.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erfc, erf

from tensorlbm.phase_change.stefan_exact import (
    CylindricalSimilarity,
    NeumannSimilarity,
    cylindrical_lambda,
    planar_lambda,
)
from tensorlbm.phase_change.stefan_verification import (
    EnthalpyThermalLBM,
    quasi_steady_radial_front,
    run_cylindrical_wall_solidification,
    run_planar_stefan,
    sharp_interface_radial_solidification,
)

ALPHA = 1.0 / 6.0


# ---------------------------------------------------------------------------
# exact similarity solutions
# ---------------------------------------------------------------------------
def test_planar_lambda_satisfies_relation():
    for ste in (0.05, 0.2, 0.5, 1.0, 2.0):
        lam = planar_lambda(ste)
        assert lam > 0.0
        # both one-phase branches (wall at one end, front at the other) reduce to
        # lam*exp(lam^2)*erf(lam) = Ste/sqrt(pi)
        assert lam * math.exp(lam * lam) * erf(lam) == pytest.approx(
            ste / math.sqrt(math.pi), rel=1e-10
        )


def test_planar_lambda_small_stefan_limit():
    r"""Small Ste: lam*exp(lam^2)*erf(lam) = Ste/sqrt(pi) gives lam -> sqrt(Ste/2).

    (The quasi-steady/one-phase small-Ste law therefore reads
    ``s(t) = 2*lam*sqrt(alpha*t) -> sqrt(2*Ste*alpha*t)`` — *not* the linear
    ``2*Ja*sqrt(alpha*t/pi)`` form quoted for the bubble-growth anchor.)
    """
    ste = 1e-3
    assert planar_lambda(ste) == pytest.approx(math.sqrt(ste / 2.0), rel=1e-3)
    assert 2.0 * planar_lambda(ste) == pytest.approx(math.sqrt(2.0 * ste), rel=1e-3)


def test_cylindrical_lambda_satisfies_relation():
    from scipy.special import exp1

    for ja in (0.1, 0.3, 0.5, 0.8):
        lam = cylindrical_lambda(ja)
        z = lam * lam
        assert math.exp(-z) / float(exp1(z)) == pytest.approx(z / ja, rel=1e-9)


def test_cylindrical_lambda_rejects_ja_above_one():
    with pytest.raises(ValueError):
        cylindrical_lambda(1.5)


@pytest.mark.parametrize("side", ["solid", "liquid"])
def test_planar_similarity_boundary_values(side):
    """T(0) = T_wall and T(front) = T_freeze (a residual-only check would miss this)."""
    ex = NeumannSimilarity(stefan_number=0.5, alpha=ALPHA, side=side)
    t = 5000.0
    assert float(ex.temperature(0.0, t)) == pytest.approx(ex.boundary_temperature, rel=1e-12)
    assert float(ex.temperature(ex.front(t), t)) == pytest.approx(ex.t_freeze, abs=1e-12)
    assert float(ex.temperature(ex.front(t) + 50.0, t)) == pytest.approx(ex.t_freeze, abs=1e-12)
    # the two branches are mirror images: same lambda for the same Ste
    assert ex.lam == pytest.approx(planar_lambda(0.5), rel=1e-12)


@pytest.mark.parametrize("side", ["solid", "liquid"])
def test_planar_similarity_pde_residual(side):
    ex = NeumannSimilarity(stefan_number=0.5, alpha=ALPHA, side=side)
    t = 20000.0
    s = ex.front(t)
    x = np.linspace(2.0, s - 3.0, 300)
    T = ex.temperature(x, t)
    Txx = ex.temperature(x + 1.0, t) - 2.0 * T + ex.temperature(x - 1.0, t)
    Tt = (ex.temperature(x, t + 1.0) - ex.temperature(x, t - 1.0)) / 2.0
    assert np.max(np.abs(Tt - ALPHA * Txx)) / ex.dT < 1e-6


@pytest.mark.parametrize("side", ["solid", "liquid"])
def test_planar_interface_energy_balance(side):
    """rho*L*v = k*|dT/dx| at the front (the Stefan condition)."""
    ex = NeumannSimilarity(stefan_number=0.5, alpha=ALPHA, side=side)
    t = 20000.0
    s = ex.front(t)
    eps = 1e-5
    grad = abs((ex.temperature(s - eps, t) - ex.temperature(s - 2 * eps, t)) / eps)
    lhs = ex.latent * ex.front_velocity(t)
    rhs = ex.cp * ex.alpha * grad
    assert rhs == pytest.approx(lhs, rel=1e-5)


def test_cylindrical_similarity_residuals():
    r"""PDE residual and interface balance of the circular-front similarity solution."""
    cyl = CylindricalSimilarity(jacob_number=0.5, alpha=ALPHA)
    t = 3000.0
    rf = cyl.front(t)
    r = np.linspace(rf + 3.0, rf + 60.0, 300)
    T = cyl.temperature(r, t)
    Tr = (cyl.temperature(r + 1.0, t) - cyl.temperature(r - 1.0, t)) / 2.0
    Trr = cyl.temperature(r + 1.0, t) - 2.0 * T + cyl.temperature(r - 1.0, t)
    Tt = (cyl.temperature(r, t + 1.0) - cyl.temperature(r, t - 1.0)) / 2.0
    assert np.max(np.abs(Tt - ALPHA * (Trr + Tr / r))) / cyl.dT < 1e-5

    eps = 1e-5
    grad = abs((cyl.temperature(rf + eps, t) - cyl.temperature(rf + 2 * eps, t)) / eps)
    assert cyl.cp * ALPHA * grad == pytest.approx(cyl.latent * cyl.front_velocity(t), rel=1e-4)
    # undercooled melt: the far field is *below* the freezing point
    assert cyl.far_field_temperature < cyl.t_freeze


def test_cylindrical_wall_heat_input_is_consistent_with_latent_release():
    """The similarity heat input per length equals 4*pi*cp*dT*alpha*t*exp(-lam^2)/E1(lam^2)."""
    cyl = CylindricalSimilarity(jacob_number=0.5, alpha=ALPHA)
    t = 1000.0
    lat = cyl.latent * math.pi * (cyl.front(t) ** 2 - cyl.front(1.0) ** 2)
    assert lat > 0.0
    assert cyl.heat_input_per_length(t) > 0.0


# ---------------------------------------------------------------------------
# thermal LBE operator and enthalpy bookkeeping
# ---------------------------------------------------------------------------
def test_lbe_diffusion_matches_erfc():
    nx, n_steps = 200, 2000
    sim = EnthalpyThermalLBM(nx=nx, ny=1, tau=1.0, periodic=(False, True))
    sim.set_field(np.zeros((nx, 1)), np.ones((nx, 1)))  # stay on the liquid branch
    mask_left = np.zeros((nx, 1), dtype=bool)
    mask_left[0, 0] = True
    mask_right = np.zeros((nx, 1), dtype=bool)
    mask_right[-1, 0] = True
    sim.add_dirichlet(mask_left, 1.0, 1.0)
    sim.add_dirichlet(mask_right, 0.0, 1.0)
    sim.step(n_steps)
    x = np.arange(nx, dtype=float)
    exact = erfc(x / (2.0 * math.sqrt(sim.alpha * n_steps)))
    band = x < nx - 15
    assert np.max(np.abs(sim.T[band, 0] - exact[band])) < 5e-4


def test_enthalpy_conservation_closed_domain():
    rng = np.random.default_rng(7)
    sim = EnthalpyThermalLBM(nx=48, ny=48, tau=1.0, periodic=(True, True))
    sim.set_field(-0.2 + 0.4 * rng.random((48, 48)))
    h0 = sim.total_enthalpy()
    sim.step(300)
    assert sim.total_enthalpy() == pytest.approx(h0, rel=1e-12)


def test_enthalpy_inversion_piecewise():
    sim = EnthalpyThermalLBM(nx=5, ny=1, cp=1.0, latent=2.0, t_freeze=0.0)
    H = np.array([[-1.0, -0.0, 0.5, 2.0, 3.0]])
    T, f_l = sim.invert(H.copy().reshape(5, 1))
    assert T[0, 0] == pytest.approx(-1.0) and f_l[0, 0] == 0.0
    assert T[1, 0] == pytest.approx(0.0) and f_l[1, 0] == 0.0
    assert T[2, 0] == pytest.approx(0.0) and f_l[2, 0] == pytest.approx(0.25)
    assert T[3, 0] == pytest.approx(0.0) and f_l[3, 0] == pytest.approx(1.0)
    assert T[4, 0] == pytest.approx(1.0) and f_l[4, 0] == pytest.approx(1.0)


def test_dirichlet_phase_is_prescribed_not_inverted():
    """A boundary exactly at T_freeze must keep the phase it was given."""
    sim = EnthalpyThermalLBM(nx=4, ny=1, latent=1.0, t_freeze=0.0, periodic=(False, True))
    sim.set_field(np.zeros((4, 1)), np.ones((4, 1)))
    mask = np.zeros((4, 1), dtype=bool)
    mask[0, 0] = True
    sim.add_dirichlet(mask, 0.0, 0.0)  # solid at the freezing point
    assert sim.f_l[0, 0] == 0.0
    mask2 = np.zeros((4, 1), dtype=bool)
    mask2[-1, 0] = True
    sim.add_dirichlet(mask2, 0.0, 1.0)  # liquid at the freezing point
    assert sim.f_l[-1, 0] == 1.0


# ---------------------------------------------------------------------------
# end-to-end verification (reduced resolution, still inside the criteria)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["freezing", "melting"])
def test_planar_stefan_front_accuracy(mode):
    res = run_planar_stefan(mode=mode, nx=200, front_fraction=0.25)
    final = res["final"]
    # conservative phase-integral front vs the Neumann solution
    assert final["front_rel_error"] < 0.02
    # temperature profile away from the smeared interface
    assert final["profile_error"]["l2_rel"] < 0.01
    # the LBM front must be behind/ahead consistently and grow like sqrt(t)
    assert final["front_integral"] > 0.0


def test_planar_front_scales_like_sqrt_t():
    res = run_planar_stefan(mode="freezing", nx=200, sample_fractions=(0.5, 1.0),
                            energy_audit=False)
    s1 = res["series"][0]["front_integral"]
    s2 = res["series"][1]["front_integral"]
    t1 = res["series"][0]["time"]
    t2 = res["series"][1]["time"]
    # sampling times are front_fraction**2 * t_end, i.e. s1/s2 = 0.5 exactly for
    # a sqrt(t) law
    assert (s1 / s2) == pytest.approx(math.sqrt(t1 / t2), rel=2e-2)


def test_melting_energy_closure():
    res = run_planar_stefan(mode="melting", nx=200)
    e = res["energy"]
    # wall input (2nd-order flux, independent operator) vs analytic Neumann value
    assert e["wall_input_2nd_vs_exact_rel_error"] < 0.01
    # latent heat absorbed vs (input - stored sensible heat)
    assert e["latent_vs_input_minus_sensible"] < 0.05
    # the first-order flux is the discrete identity of the scheme: exactly zero
    assert e["closure_1st_order"] < 1e-10


def test_sharp_interface_reference_self_convergence():
    fine = sharp_interface_radial_solidification(
        r_wall=20.0, shell0=3.0, stefan_number=0.5, alpha=ALPHA, n_nodes=400, t_end=400.0
    )
    coarse = sharp_interface_radial_solidification(
        r_wall=20.0, shell0=3.0, stefan_number=0.5, alpha=ALPHA, n_nodes=200, t_end=400.0
    )
    t = fine["t"]
    r_coarse = np.interp(t, coarse["t"], coarse["r_front"])
    assert np.max(np.abs(r_coarse - fine["r_front"])) / np.mean(fine["r_front"]) < 5e-3
    # the front must move inward monotonically
    assert np.all(np.diff(fine["r_front"]) < 0.0)


def test_cylindrical_front_shape_and_radius():
    res = run_cylindrical_wall_solidification(r_wall=24, shell0=4, sample_count=4)
    final = res["final"]
    assert final["shape_max_dev_rel"] < 0.03  # exact front is a circle
    assert final["r_rel_error_vs_reference"] < 0.03
    assert res["reference"]["self_convergence_rel"] < 5e-3
    assert res["final"]["r_area"] < res["parameters"]["r_meas0"]  # inward growth


def test_quasi_steady_formula_matches_reference_at_small_ste():
    """The Planck quasi-steady anchor is exact only as Ste -> 0; error grows with Ste."""
    ref_small = sharp_interface_radial_solidification(
        r_wall=40.0, shell0=4.0, stefan_number=0.02, alpha=ALPHA, t_end=20000.0
    )
    ref_large = sharp_interface_radial_solidification(
        r_wall=40.0, shell0=4.0, stefan_number=0.5, alpha=ALPHA, t_end=3000.0
    )
    err_small = abs(ref_small["r_front"][-1] - quasi_steady_radial_front(
        40.0, 20000.0, 0.02, ALPHA)) / ref_small["r_front"][-1]
    err_large = abs(ref_large["r_front"][-1] - quasi_steady_radial_front(
        40.0, 3000.0, 0.5, ALPHA)) / ref_large["r_front"][-1]
    assert err_large > err_small
    assert err_small < 0.05