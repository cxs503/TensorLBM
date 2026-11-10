"""Gauss-Seidel radiosity solver: convergence certificate and energy closure.

Acceptance criteria pinned here (P93W3-N02):

* (b) the iteration residual of a closed enclosure is ``<= 1e-6`` and the
  converged radiosity agrees with the dense LU solution of
  :class:`tensorlbm.gray_enclosure_radiosity.GrayEnclosure`;
* (c) an **isothermal** closed cavity exchanges no net heat: the enclosure
  energy residual is ``<= 0.1 %`` (in fact ~1e-16 here) for every emissivity
  combination, including the black-body and the pure-reradiating limits.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.gray_enclosure_radiosity import (
    STEFAN_BOLTZMANN,
    GrayEnclosure,
    GraySurface,
    analytic_two_surface_flux,
    concentric_spheres_enclosure,
    cube_enclosure,
)
from tensorlbm.radiosity_gauss_seidel import (
    enclosure_vectors,
    iteration_matrix,
    solve_radiosity_gauss_seidel,
    spectral_radius,
)

RESIDUAL_TOL = 1e-6  # machine criterion (b)
ENERGY_TOL = 1e-3  # machine criterion (c): 0.1 %


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def test_gauss_seidel_matches_lu_and_meets_residual_gate() -> None:
    """Cube enclosure (two eps levels) -> same answer as the direct solve."""
    enc = cube_enclosure(1.0, eps_opposite=0.9, eps_adjacent=0.2, T_opposite=700.0, T_adjacent=400.0)
    lu = enc.solve()
    gs_gate = solve_radiosity_gauss_seidel(enc, tol=RESIDUAL_TOL)
    assert gs_gate.converged
    assert gs_gate.final_residual <= RESIDUAL_TOL  # criterion (b)
    assert gs_gate.energy_residual < ENERGY_TOL  # criterion (c)
    gs = solve_radiosity_gauss_seidel(enc, tol=1e-13)
    assert np.max(np.abs(gs.radiosity - lu.radiosity)) / np.max(np.abs(lu.radiosity)) < 1e-11
    assert gs.algebraic_residual < 1e-12


def test_two_surface_parallel_plates_vs_series_resistance() -> None:
    """GS on a 2-surface enclosure reproduces the analytic series resistance."""
    T1, T2, e1, e2 = 800.0, 500.0, 0.9, 0.7
    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    enc = GrayEnclosure(
        [GraySurface(1.0, e1, T1, name="p1"), GraySurface(1.0, e2, T2, name="p2")], F
    )
    gs = solve_radiosity_gauss_seidel(enc, tol=1e-12)
    q_ana = analytic_two_surface_flux(T1, T2, e1, e2, 1.0, 1.0)
    assert rel(gs.net_flux[0], q_ana) < 1e-10
    assert rel(gs.net_flux[1], -q_ana) < 1e-10


def test_isothermal_cavity_energy_closure() -> None:
    """Criterion (c): an isothermal cavity has zero net exchange (<= 0.1 %)."""
    T = 450.0
    for eps in (0.02, 0.1, 0.5, 0.9, 1.0):
        enc = cube_enclosure(2.0, eps_opposite=eps, eps_adjacent=eps, T_opposite=T, T_adjacent=T)
        gs = solve_radiosity_gauss_seidel(enc, tol=RESIDUAL_TOL)
        assert gs.energy_residual < ENERGY_TOL
        # net flux must vanish to the iteration tolerance, not to machine zero
        assert float(np.max(np.abs(gs.net_flux))) / (STEFAN_BOLTZMANN * T**4) < 1e-5
        assert gs.converged


def test_isothermal_spheres_cavity() -> None:
    T = 600.0
    enc = concentric_spheres_enclosure(1.0, 4.0, 0.8, 0.4, T, T)
    gs = solve_radiosity_gauss_seidel(enc, tol=1e-12)
    assert gs.energy_residual < ENERGY_TOL
    assert float(np.max(np.abs(gs.net_flux))) / (STEFAN_BOLTZMANN * T**4) < 1e-9


def test_reradiating_surface_row_is_zero_flux() -> None:
    """eps = 0 / T = None rows give q ~ 0 without inventing a temperature.

    Enclosure: A = (1, 2, 3) m2, F12 = 0.4, F13 = 0.6 (so F21 = 0.2, F23 = 0.8,
    F31 = 0.2, F32 = 0.5333, and the concave surface 3 sees itself with
    F33 = 0.2667).  Summation and reciprocity are exact by construction.
    """
    A = np.array([1.0, 2.0, 3.0])
    F = np.array(
        [
            [0.0, 0.4, 0.6],
            [0.2, 0.0, 0.8],
            [0.2, 1.6 / 3.0, 1.0 - 0.2 - 1.6 / 3.0],
        ]
    )
    surfaces = [
        GraySurface(A[0], 0.9, 800.0, name="hot"),
        GraySurface(A[1], 0.1, 400.0, name="cold"),
        GraySurface(A[2], 0.0, None, name="reradiating"),
    ]
    enc = GrayEnclosure(surfaces, F)
    gs = solve_radiosity_gauss_seidel(enc, tol=1e-12)
    assert abs(gs.net_flux[2]) / (STEFAN_BOLTZMANN * 800.0**4) < 1e-10
    assert np.isfinite(gs.radiosity[2])
    # the reradiating surface floats between the two active temperatures
    T_eff_2 = (gs.radiosity[2] / STEFAN_BOLTZMANN) ** 0.25
    assert 400.0 < T_eff_2 < 800.0


def test_all_reradiating_cavity_is_rejected() -> None:
    F = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    enc = GrayEnclosure([GraySurface(1.0, 0.0, None, name=f"r{i}") for i in range(3)], F)
    with pytest.raises(ValueError, match="reradiating"):
        solve_radiosity_gauss_seidel(enc)


def test_spectral_radius_bounds_the_contraction() -> None:
    """rho(diag(1-eps) F) <= max(1-eps) for a closed enclosure."""
    enc = cube_enclosure(1.0, eps_opposite=0.9, eps_adjacent=0.2, T_opposite=700.0, T_adjacent=400.0)
    rho = spectral_radius(enc)
    _, eps, _, F = enclosure_vectors(enc)
    assert rho <= float(np.max(1.0 - eps)) + 1e-12
    assert 0.0 < rho < 1.0
    assert np.allclose(F.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(iteration_matrix(enc), (1.0 - eps)[:, None] * F)


@pytest.mark.parametrize("omega", [0.5, 1.0, 1.5])
def test_relaxation_sweep_converges_to_same_solution(omega: float) -> None:
    enc = cube_enclosure(1.0, eps_opposite=0.6, eps_adjacent=0.3, T_opposite=650.0, T_adjacent=350.0)
    ref = enc.solve().radiosity
    gs = solve_radiosity_gauss_seidel(enc, tol=1e-12, relaxation=omega)
    assert gs.converged
    assert np.max(np.abs(gs.radiosity - ref)) / np.max(np.abs(ref)) < 1e-10


def test_honest_non_convergence_is_reported() -> None:
    """Too few sweeps -> converged = False with the residual actually reached."""
    enc = cube_enclosure(1.0, eps_opposite=0.9, eps_adjacent=0.9, T_opposite=900.0, T_adjacent=300.0)
    gs = solve_radiosity_gauss_seidel(enc, tol=1e-14, max_iter=3, record_history=True)
    assert gs.converged is False
    assert gs.final_residual > 1e-14
    assert len(gs.residual_history) == 3
    assert gs.as_dict()["converged"] is False


def test_argument_validation() -> None:
    enc = cube_enclosure(1.0, 0.5, 0.5, 600.0, 400.0)
    with pytest.raises(ValueError):
        solve_radiosity_gauss_seidel(enc, relaxation=0.0)
    with pytest.raises(ValueError):
        solve_radiosity_gauss_seidel(enc, relaxation=2.0)
    with pytest.raises(ValueError):
        solve_radiosity_gauss_seidel(enc, tol=0.0)
    with pytest.raises(ValueError):
        solve_radiosity_gauss_seidel(enc, initial=np.zeros(3))
    with pytest.raises(ValueError):
        solve_radiosity_gauss_seidel(enc, initial=np.full(6, np.nan))


def test_open_enclosure_converges() -> None:
    """A deck/sky/sea patch enclosure (rows sum to 1, open to the rest)."""
    A, A_p = 24.0, 1000.0 * 24.0
    F = np.array(
        [
            [0.0, 0.75, 0.25],
            [A * 0.75 / A_p, 1.0 - A * 0.75 / A_p, 0.0],
            [A * 0.25 / A_p, 0.0, 1.0 - A * 0.25 / A_p],
        ]
    )
    enc = GrayEnclosure(
        [
            GraySurface(A, 0.9, 320.0, name="deck"),
            GraySurface(A_p, 1.0, 280.0, name="sky"),
            GraySurface(A_p, 0.96, 293.0, name="sea"),
        ],
        F,
        summation_tol=1e-6,
    )
    gs = solve_radiosity_gauss_seidel(enc, tol=RESIDUAL_TOL)
    lu = enc.solve()
    assert gs.converged
    assert np.max(np.abs(gs.radiosity - lu.radiosity)) / np.max(np.abs(lu.radiosity)) < 1e-9
