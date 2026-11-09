"""Verification of the gray-diffuse enclosure radiosity solver.

Criterion (b) of P92W3-N02: the energy-conservation residual of a closed
enclosure must be <= 1e-6.  Each enclosure is additionally checked against an
analytic anchor that is independent of the linear solve:

* two parallel plates / concentric spheres -> series-resistance solution,
* three-surface enclosure with one reradiating surface -> radiation-network
  solution,
* cube enclosure -> closed-form family summation rule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.gray_enclosure_radiosity import (
    STEFAN_BOLTZMANN,
    EnclosureSolution,
    GrayEnclosure,
    GraySurface,
    analytic_reradiating_network_flux,
    analytic_two_surface_flux,
    concentric_spheres_enclosure,
    cube_enclosure,
)

ENERGY_TOL = 1e-6  # machine criterion (b)


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------


def test_parallel_plates_vs_series_resistance() -> None:
    """A1 = A2, F12 = 1 -> sigma(T1^4-T2^4)/(1/eps1+1/eps2-1)."""
    T1, T2, e1, e2 = 800.0, 500.0, 0.9, 0.7
    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    sol = GrayEnclosure(
        [GraySurface(1.0, e1, T1, name="p1"), GraySurface(1.0, e2, T2, name="p2")], F
    ).solve()
    q_ana = analytic_two_surface_flux(T1, T2, e1, e2, 1.0, 1.0)
    assert rel_err(sol.net_flux[0], q_ana) < 1e-12
    assert rel_err(sol.net_flux[1], -q_ana) < 1e-12
    assert sol.energy_residual < ENERGY_TOL


def test_concentric_spheres_vs_series_resistance() -> None:
    """Inner sphere (F12 = 1) inside outer sphere (F21 = (r1/r2)^2)."""
    r1, r2, e1, e2, T1, T2 = 1.0, 4.0, 0.8, 0.4, 600.0, 300.0
    enc = concentric_spheres_enclosure(r1, r2, e1, e2, T1, T2)
    sol = enc.solve()
    A1, A2 = 4.0 * math.pi * r1**2, 4.0 * math.pi * r2**2
    q_ana = analytic_two_surface_flux(T1, T2, e1, e2, A1, A2)
    assert rel_err(sol.net_flux[0], q_ana) < 1e-12
    assert sol.energy_residual < ENERGY_TOL
    assert sol.summation_residual < 1e-12


def test_cube_enclosure_closed_form_family() -> None:
    """Six-face cube: F_opp from the family complement, F_adj closed form."""
    enc = cube_enclosure(side=2.0, eps_opposite=0.9, eps_adjacent=0.85, T_opposite=420.0, T_adjacent=300.0)
    assert enc.summation_residual() < 1e-12
    assert enc.reciprocity_residual() < 1e-12
    sol = enc.solve()
    assert sol.energy_residual < ENERGY_TOL
    # symmetry: all four adjacent faces see the same net flux
    assert abs(sol.net_flux[2] - sol.net_flux[3]) < 1e-12
    assert abs(sol.net_flux[4] - sol.net_flux[5]) < 1e-12
    # the hot faces must give energy to the cold ones
    assert sol.net_flux[0] > 0.0 and sol.net_flux[2] < 0.0
    assert abs(sol.total_power_leaving.sum()) / np.abs(sol.total_power_leaving).sum() < ENERGY_TOL


def test_reradiating_surface_vs_network_solution() -> None:
    """epsilon = 0, T = None -> q = 0 exactly and the network anchor matches."""
    F = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    enc = GrayEnclosure(
        [
            GraySurface(1.0, 0.9, 800.0, name="hot"),
            GraySurface(1.0, 0.6, 400.0, name="cold"),
            GraySurface(1.0, 0.0, None, name="reradiating"),
        ],
        F,
    )
    sol = enc.solve()
    q_ana = analytic_reradiating_network_flux(800.0, 400.0, 0.9, 0.6, 1.0, 1.0, 0.5, 0.5, 0.5)
    assert rel_err(sol.net_flux[0], q_ana) < 1e-10
    assert abs(sol.net_flux[2]) < 1e-6  # reradiating: zero net flux [W/m2]
    assert sol.energy_residual < ENERGY_TOL


def test_blackbody_surface_limit() -> None:
    """eps = 1 -> J = sigma T^4 and q = sigma T^4 - G."""
    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    sol = GrayEnclosure(
        [GraySurface(1.0, 1.0, 700.0, name="black"), GraySurface(1.0, 1.0, 300.0, name="black2")], F
    ).solve()
    T1, T2 = 700.0, 300.0
    assert abs(sol.radiosity[0] - STEFAN_BOLTZMANN * T1**4) < 1e-9
    assert abs(sol.net_flux[0] - STEFAN_BOLTZMANN * (T1**4 - T2**4)) < 1e-9


def test_energy_conservation_sweep() -> None:
    """Criterion (b) over a temperature/emissivity sweep on the cube enclosure."""
    worst = 0.0
    for T_hot in (320.0, 400.0, 600.0, 900.0):
        for eps in (0.05, 0.3, 0.85, 1.0):
            enc = cube_enclosure(1.0, eps, eps, T_hot, 300.0)
            sol = enc.solve()
            worst = max(worst, sol.energy_residual, sol.closure_residual, sol.linear_residual)
    assert worst < ENERGY_TOL, worst


def test_energy_conservation_multisurface_random_temperatures() -> None:
    """Cube with six different temperatures: internal exchange must cancel."""
    F_adj = 0.20004
    F_opp = 1.0 - 4.0 * F_adj
    F = np.zeros((6, 6))
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        F[i, j] = F[j, i] = F_opp
    for i in range(6):
        for j in range(6):
            if i != j and F[i, j] == 0.0:
                F[i, j] = F_adj
    temps = [320.0, 350.0, 400.0, 450.0, 500.0, 550.0]
    eps = [0.9, 0.7, 0.5, 0.3, 0.95, 0.2]
    enc = GrayEnclosure([GraySurface(1.0, e, T) for e, T in zip(eps, temps)], F)
    sol = enc.solve()
    assert sol.energy_residual < ENERGY_TOL
    assert abs(sum(sol.total_power_leaving)) < 1e-6 * max(1.0, float(np.abs(sol.total_power_leaving).sum()))


def test_open_enclosure_sky_patch_energy_conservation() -> None:
    """Deck + sky patch + sea patch: the radiative exchange inside still cancels."""
    A = 24.0
    A_p = 1000.0 * A
    F_sky, F_sea = 1.0, 0.0
    F = np.array(
        [
            [0.0, F_sky, F_sea],
            [A * F_sky / A_p, 1.0 - A * F_sky / A_p, 0.0],
            [A * F_sea / A_p, 0.0, 1.0 - A * F_sea / A_p],
        ]
    )
    enc = GrayEnclosure(
        [
            GraySurface(A, 0.9, 320.0, name="deck"),
            GraySurface(A_p, 0.85, 280.0, name="sky"),
            GraySurface(A_p, 0.96, 293.0, name="sea"),
        ],
        F,
    )
    sol = enc.solve()
    assert sol.energy_residual < ENERGY_TOL


# ---------------------------------------------------------------------------
# fail-closed behaviour
# ---------------------------------------------------------------------------


def test_rejects_bad_summation() -> None:
    F = np.array([[0.0, 0.4], [0.4, 0.0]])  # rows sum to 0.4
    with pytest.raises(ValueError, match="summation rule"):
        GrayEnclosure([GraySurface(1.0, 0.9, 400.0), GraySurface(1.0, 0.9, 300.0)], F)


def test_rejects_bad_reciprocity() -> None:
    # rows sum to one, but A1 F12 = 1 != A2 F21 = 2
    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="reciprocity"):
        GrayEnclosure([GraySurface(1.0, 0.9, 400.0), GraySurface(2.0, 0.9, 300.0)], F)


def test_rejects_wrong_shape_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="shape"):
        GrayEnclosure([GraySurface(1.0, 0.9, 400.0), GraySurface(1.0, 0.9, 300.0)], np.ones((3, 3)))
    bad = np.array([[0.0, np.nan], [np.nan, 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        GrayEnclosure([GraySurface(1.0, 0.9, 400.0), GraySurface(1.0, 0.9, 300.0)], bad)


def test_rejects_invalid_surface_properties() -> None:
    with pytest.raises(ValueError):
        GraySurface(area=0.0, emissivity=0.9, temperature=300.0)
    with pytest.raises(ValueError):
        GraySurface(area=1.0, emissivity=1.5, temperature=300.0)
    with pytest.raises(ValueError):
        GraySurface(area=1.0, emissivity=0.9, temperature=-5.0)


def test_reradiating_surface_has_no_invented_temperature() -> None:
    s = GraySurface(area=1.0, emissivity=0.9, temperature=None)
    assert s.temperature is None
    assert s.emissivity == 0.0