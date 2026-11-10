"""Tests for the ice/water phase-change interface surface (node p348w2-n06).

Coverage: 1-D similarity anchor + solver (melting and freezing), two-phase
(subcooled) anchor integrity, cylindrical melting vs the quasi-steady
literature solution, latent-heat / energy conservation, interface thickness,
interface temperature continuity, grid convergence and the pure-conduction
negative control.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.ice_water_front import (
    Enthalpy2D,
    FrontTracking1D,
    IceWaterCase,
    RadialFrontTracking,
    TwoPhaseNeumann,
    energy_audit,
    interface_thickness_cells,
    quasi_steady_cylinder_front,
    quasi_steady_cylinder_time,
    temperature_continuity,
    run_verification_suite,
)
from tensorlbm.ice_water_front.audit import vof_circle_field
from tensorlbm.ice_water_front.enthalpy2d import vof_circle_thickness
from tensorlbm.ice_water_front.front1d import solve_pure_conduction
from tensorlbm.ice_water_front.properties import T_MELT


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------
def test_properties_dimensionless_groups():
    case = IceWaterCase(t_wall=T_MELT + 5.0)
    assert case.melting
    assert case.inner.name == "water" and case.outer.name == "ice"
    assert case.stefan_inner == pytest.approx(4217.0 * 5.0 / 333550.0, rel=1e-12)
    # ice diffuses heat ~8x faster than water
    assert case.diffusivity_ratio > 5.0


def test_one_phase_similarity_limits():
    """Small-Ste limit ``lam -> sqrt(Ste/2)`` and the erf relation."""
    for dT in (1.0, 5.0, 10.0):
        case = IceWaterCase(t_wall=T_MELT + dT, t_initial=T_MELT)
        lam = TwoPhaseNeumann(case).lam
        ste = case.stefan_inner
        assert lam * math.exp(lam ** 2) * math.erf(lam) == pytest.approx(
            ste / math.sqrt(math.pi), rel=1e-10
        )
        assert lam == pytest.approx(math.sqrt(ste / 2.0), rel=0.05)


def test_two_phase_similarity_satisfies_pde_and_stefan_condition():
    """The subcooled anchor must satisfy the PDE and the interface balance."""
    case = IceWaterCase(t_wall=T_MELT + 5.0, t_initial=T_MELT - 2.0)
    sim = TwoPhaseNeumann(case)
    xs = np.linspace(5e-5, 6e-2, 4001)
    assert sim.pde_residual(xs, 20.0) < 5e-3
    assert abs(sim.stefan_residual(20.0)) < 0.02   # ~1 %: density-equal idealisation
    # a colder bulk must retard the front
    warm = TwoPhaseNeumann(IceWaterCase(t_wall=T_MELT + 5.0))
    assert sim.lam < warm.lam


def test_quasi_steady_cylinder_inverts_consistently():
    case = IceWaterCase(t_wall=T_MELT + 5.0, length=30e-3)
    t = quasi_steady_cylinder_time(case, 3.5e-3, 5.0e-3, 30e-3)
    assert 0.0 < t < 3600.0
    r_f = quasi_steady_cylinder_front(case, t, 5.0e-3, 30e-3)
    assert r_f == pytest.approx(3.5e-3, rel=1e-6)
    # the ice core shrinks monotonically
    assert quasi_steady_cylinder_front(case, 2 * t, 5e-3, 30e-3) < r_f


# ---------------------------------------------------------------------------
# 1-D front tracking
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dT", [2.0, 5.0, 10.0])
def test_front_position_matches_similarity(dT):
    """C1: front position vs the Neumann similarity, three Stefan numbers."""
    case = IceWaterCase(t_wall=T_MELT + dT, t_initial=T_MELT)
    run = FrontTracking1D(case, n_cells=40, t_end=12.0, domain_factor=2.0).run()
    assert run.front_error < 0.01
    assert np.all(np.diff(run.front) > 0.0)


def test_freezing_branch_runs_and_conserves_energy():
    """The freezing branch (opposite latent-heat sign) must conserve energy."""
    case = IceWaterCase(t_wall=T_MELT - 5.0, t_initial=T_MELT)
    run = FrontTracking1D(case, n_cells=40, t_end=12.0, domain_factor=2.0).run()
    assert np.all(np.diff(run.front) > 0.0)
    assert run.energy_residual < 0.02
    assert run.front_error < 0.10          # see docs: pin-node bias, informational


def test_two_phase_subcooled_numerical_against_similarity():
    case = IceWaterCase(t_wall=T_MELT + 5.0, t_initial=T_MELT - 2.0)
    run = FrontTracking1D(
        case, n_cells=300, t_end=8.0, t_start=8.0 / 16, domain_factor=32.0
    ).run()
    assert run.front_error < 0.02
    assert run.energy_residual < 0.02


def test_energy_audit_closes():
    """C3: latent heat + sensible enthalpy change == boundary heat input."""
    case = IceWaterCase(t_wall=T_MELT + 5.0, t_initial=T_MELT)
    run = FrontTracking1D(case, n_cells=40, t_end=12.0, domain_factor=2.0).run()
    audit = energy_audit(run)
    assert audit["relative_residual"] < 0.02
    assert audit["latent_heat_J"] > 0.0
    assert audit["closure_ratio"] == pytest.approx(1.0, abs=0.03)


def test_interface_temperature_continuity():
    """C5: both one-sided extrapolations of T at the front give T_m."""
    case = IceWaterCase(t_wall=T_MELT + 5.0, t_initial=T_MELT)
    run = FrontTracking1D(case, n_cells=40, t_end=12.0, domain_factor=2.0).run()
    cont = temperature_continuity(run, T_MELT)
    assert cont["max_offset_K"] < 0.1
    assert cont["jump_K"] < 0.1


def test_grid_convergence_of_front():
    """C6: doubling the resolution changes the front by well under 3 %."""
    case = IceWaterCase(t_wall=T_MELT + 5.0, t_initial=T_MELT)
    coarse = FrontTracking1D(case, n_cells=40, t_end=12.0, domain_factor=2.0).run()
    fine = FrontTracking1D(case, n_cells=80, t_end=12.0, domain_factor=2.0).run()
    change = abs(fine.front[-1] - coarse.front[-1]) / coarse.front[-1]
    assert change < 0.03


def test_pure_conduction_negative_control():
    """C7: with phase change absent the solver reproduces the erfc solution."""
    res = solve_pure_conduction(IceWaterCase(t_wall=T_MELT + 5.0), t_end=20.0, n_cells=200)
    assert res["relative_max_error"] < 0.01
    assert res["rms_error_K"] < 5e-3


def test_no_phase_change_when_wall_equals_melting_point():
    """Degenerate case: an isothermal wall must not move the front."""
    case = IceWaterCase(t_wall=T_MELT, t_initial=T_MELT)
    with pytest.raises(ValueError):
        FrontTracking1D(case, n_cells=20, t_end=5.0)


# ---------------------------------------------------------------------------
# 2-D cylinder
# ---------------------------------------------------------------------------
def test_cylinder_melting_matches_quasi_steady_literature_anchor():
    """C2: 2-D circular melting vs the quasi-steady literature solution."""
    case = IceWaterCase(t_wall=T_MELT + 5.0, t_initial=T_MELT, length=30e-3)
    run = RadialFrontTracking(case, r0=5e-3, r_outer=30e-3, n_cells=61, t_end=1500.0).run()
    assert run.front_error < 0.10
    assert run.energy_residual < 0.02
    assert run.continuity_residual < 0.1
    assert run.front[-1] < 0.95 * run.front[0]    # the ice core shrinks


def test_cylinder_grid_convergence():
    case = IceWaterCase(t_wall=T_MELT + 5.0, t_initial=T_MELT, length=30e-3)
    coarse = RadialFrontTracking(case, r0=5e-3, r_outer=30e-3, n_cells=41, t_end=800.0).run()
    fine = RadialFrontTracking(case, r0=5e-3, r_outer=30e-3, n_cells=81, t_end=800.0).run()
    change = abs(fine.front[-1] - coarse.front[-1]) / coarse.front[-1]
    assert change < 0.03


# ---------------------------------------------------------------------------
# interface thickness contrast
# ---------------------------------------------------------------------------
def test_interface_thickness_metrics():
    # a perfect step is one interface cell, a linear ramp is many
    step = np.where(np.arange(20) < 10, 1.0, 0.0)
    ramp = np.clip(1.0 - np.arange(20) / 19.0, 0.0, 1.0)
    assert interface_thickness_cells(step)["max_cells"] <= 1
    assert interface_thickness_cells(ramp)["max_cells"] >= 5


def test_sharp_interface_within_two_cells():
    """C4: the tracked front occupies <= 2 grid cells (sharp representation)."""
    res = vof_circle_thickness(rf=2.3, half_width=6.0, n=72)
    assert res["max_cells"] <= 2
    # the enthalpy method on the same grid smears the interface over >= 2 cells
    enthalpy = Enthalpy2D(ste=0.1, r0=2.0, half_width=6.0, n=48, t_end=2.0).run()
    assert enthalpy.interface_thickness_cells >= 2
    assert enthalpy.front_radius < 2.0      # the ice core shrinks


def test_vof_circle_field_limits():
    h = 0.25
    x = np.array([0.0, 5.0])
    X, Y = np.meshgrid(x, x, indexing="xy")
    frac = vof_circle_field(X, Y, radius=5.0, cell=h, sub=8)
    assert frac[0, 0] == pytest.approx(1.0)      # fully inside
    Xf, Yf = np.meshgrid(np.array([10.0]), np.array([10.0]), indexing="xy")
    far = vof_circle_field(Xf, Yf, radius=1.0, cell=h)
    assert far[0, 0] == pytest.approx(0.0)       # fully outside


# ---------------------------------------------------------------------------
# suite level
# ---------------------------------------------------------------------------
def test_verification_suite_acceptance():
    """Full acceptance driver: every non-informational criterion must pass."""
    res = run_verification_suite(
        t_end_1d=8.0, n_cells_1d=40, radial_n=41, radial_t_end=600.0,
        run_convergence=True, run_enthalpy_contrast=True,
    )
    crit = res["criteria"]
    assert crit["C1"]["pass"], crit["C1"]
    assert crit["C2"]["pass"], crit["C2"]
    assert crit["C3"]["pass"], crit["C3"]
    assert crit["C4"]["pass"], crit["C4"]
    assert crit["C5"]["pass"], crit["C5"]
    assert crit["C6"]["pass"], crit["C6"]
    assert crit["C7"]["pass"], crit["C7"]
    assert res["all_pass"]
    assert crit["anchor_integrity"]["stefan_residual_two_phase"] < 0.02
    assert crit["anchor_integrity"]["pde_residual_two_phase"] < 5e-3