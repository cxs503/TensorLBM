"""Unit tests for the NACA0012 transonic / sonic-boom verification module (IN421-W2-N03)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.compressible import kinetic_flux as kf
from tensorlbm.compressible import transonic_aero as ta


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_naca0012_reference_geometry():
    ref = ta.naca0012_reference_geometry()
    assert ref["max_thickness"] == pytest.approx(0.12, abs=2e-4)
    assert ref["x_max_thickness"] == pytest.approx(0.30, abs=5e-3)
    # 2 * integral of yt over the chord (Abbott & von Doenhoff, sec. 6.1)
    assert ref["area_over_chord2"] == pytest.approx(0.0817, abs=1e-3)


def test_validate_geometry_accepts_true_profile_and_rejects_thickened_one():
    x, y = kf.contour_loop(kf.naca0012_half_thickness, n_points=2000)
    good = ta.validate_geometry(x, y)
    assert good["ok"] is True
    assert abs(good["chord_error_pct"]) < 1e-3
    assert abs(good["max_thickness_error_pct"]) < 0.2

    bad = ta.validate_geometry(x, 1.1 * y)          # 10 % too thick
    assert bad["ok"] is False
    assert bad["max_thickness_error_pct"] > 5.0


def test_diamond_loop_is_closed_and_hits_the_apex():
    theta = 10.0
    x, y = ta.diamond_loop(theta)
    assert x.min() == pytest.approx(0.0)
    assert x.max() == pytest.approx(1.0)
    assert y.max() == pytest.approx(0.5 * math.tan(math.radians(theta)), rel=1e-12)
    assert y.min() == pytest.approx(-y.max(), rel=1e-12)
    # no duplicated node (the O-grid would get a zero-length face)
    ds = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    assert ds.min() > 0.25 * ds.max()


# ---------------------------------------------------------------------------
# mesh: the wall must be the body surface, not an offset curve
# ---------------------------------------------------------------------------
def test_radial_ladder_starts_on_the_surface_and_keeps_the_first_cell():
    for nj in (24, 48, 64, 96, 160):
        lv = kf.radial_levels_from_surface(nj, 12.0, first_cell=0.002)
        assert lv[0] == 0.0
        assert lv[-1] == pytest.approx(12.0, rel=1e-12)
        # regression: the ladder used to be rescaled with a fixed growth ratio,
        # which crushed the wall cell height from 2e-3 c to 3e-5 c at nj=96
        assert (lv[1] - lv[0]) == pytest.approx(0.002, rel=1e-9)
        assert np.all(np.diff(lv) > 0.0)


def test_wall_nodes_are_the_airfoil_surface():
    x_loop, y_loop = kf.contour_loop(kf.naca0012_half_thickness, n_points=80)
    mesh = kf.build_annulus_mesh(
        x_loop, y_loop, kf.radial_levels_from_surface(24, 8.0, first_cell=0.002))
    surf_x = 0.5 * (x_loop + np.roll(x_loop, -1))
    surf_y = 0.5 * (y_loop + np.roll(y_loop, -1))
    assert np.allclose(mesh.wall_x, surf_x)
    assert np.allclose(mesh.wall_y, surf_y)
    # every wall node lies on the NACA0012 thickness distribution (up to the
    # chord sagitta between two contour points, O(ds^2))
    yt = kf.naca0012_half_thickness(np.clip(mesh.wall_x, 0.0, 1.0))
    assert np.max(np.abs(np.abs(mesh.wall_y) - yt)) < 1e-2


def test_old_offset_wall_behaviour_is_documented_regression():
    """``radial_levels`` (r0 first level) leaves the body offset by r0."""
    x_loop, y_loop = kf.contour_loop(kf.naca0012_half_thickness, n_points=80)
    old = kf.radial_levels(24, 0.02, 8.0)
    mesh_old = kf.build_annulus_mesh(x_loop, y_loop, old)
    surf_x = 0.5 * (x_loop + np.roll(x_loop, -1))
    surf_y = 0.5 * (y_loop + np.roll(y_loop, -1))
    dist = np.hypot(mesh_old.wall_x - surf_x, mesh_old.wall_y - surf_y)
    assert np.median(dist) == pytest.approx(0.02, rel=1e-3)   # offset by r0


def test_mesh_quality_flags_no_folded_cells():
    x_loop, y_loop = kf.contour_loop(kf.naca0012_half_thickness, n_points=120)
    mesh = kf.build_annulus_mesh(
        x_loop, y_loop, kf.radial_levels_from_surface(32, 8.0, first_cell=0.004))
    q = kf.mesh_quality(mesh)
    assert q["n_cells"] > 0
    assert q["min_abs_area"] > 0.0
    # the closed-trailing-edge O-grid keeps at most the two TE slivers
    assert q["n_folded"] <= 2


# ---------------------------------------------------------------------------
# reference hierarchies
# ---------------------------------------------------------------------------
def test_thin_airfoil_and_prandtl_glauert_slopes():
    assert ta.thin_airfoil_lift_slope(0.0) == pytest.approx(2.0 * math.pi)
    assert ta.prandtl_glauert_lift_slope(0.3) == pytest.approx(
        2.0 * math.pi / math.sqrt(1.0 - 0.09), rel=1e-12)


def test_thickness_corrected_lift_slope_is_about_nine_percent_above_2pi():
    thick = ta.thickness_corrected_lift_slope(0.12)
    assert thick / (2.0 * math.pi) == pytest.approx(1.0 + 0.77 * 0.12, rel=1e-12)
    assert 8.0 < 100.0 * (thick / (2.0 * math.pi) - 1.0) < 10.0


def test_lift_slope_least_squares_recovers_analytic_slope():
    slope = 2.0 * math.pi / math.sqrt(1.0 - 0.7 ** 2)
    alpha = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    cl = slope * np.radians(alpha)
    out = ta.lift_slope(alpha, cl, alpha_max=4.0)
    assert out["dcl_dalpha_per_rad"] == pytest.approx(slope, rel=1e-10)
    assert out["max_residual_window"] < 1e-12


def test_oblique_shock_and_ackeret_at_mach2_theta10():
    beta = ta.oblique_shock_angle(2.0, 10.0)
    assert math.degrees(beta) == pytest.approx(39.31, abs=0.02)
    p_ratio = ta.oblique_shock_pressure_ratio(2.0, 10.0)
    assert p_ratio - 1.0 == pytest.approx(0.7066, abs=1e-3)
    ref = ta.wedge_reference_table(2.0, 10.0)
    # Ackeret linear theory is itself ~20 % low here: the acceptance criterion
    # "dp vs Ackeret < 10 %" cannot be met by a correct solver
    assert ref["ackeret_error_pct_vs_exact"] == pytest.approx(-20.1, abs=0.5)


def test_prandtl_meyer_expansion_reference_of_the_rear_wedge():
    ref = ta.wedge_expansion_reference(2.0, 10.0)
    assert ref["turning_angle_deg"] == pytest.approx(20.0)
    assert ref["rear_p_ratio"] == pytest.approx(0.5508, abs=2e-3)
    assert ref["mach_rear_wedge"] > ref["mach_behind_front_shock"]


def test_surface_mach_from_cp_is_consistent_with_isentropic_relations():
    mach_inf = 0.85
    mach = np.array([0.3, 0.7, 1.0, 1.4])
    # build Cp from the isentropic relation, then invert it
    stag = 1.0 + 0.5 * 0.4 * mach_inf ** 2
    p_ratio = (1.0 + 0.2 * mach ** 2) ** (-3.5) * stag ** 3.5
    cp = 2.0 * (p_ratio - 1.0) / (1.4 * mach_inf ** 2)
    back = ta.surface_mach_from_cp(cp, mach_inf)
    assert np.allclose(back, mach, rtol=1e-9)


def test_shock_detection_uses_the_supersonic_pocket_not_the_cp_gradient():
    x = np.linspace(0.0, 1.0, 401)
    # supersonic pocket that closes at x/c = 0.6 on the upper surface
    cp = np.where(x < 0.6,
                  -0.9 + 0.5 * x,                 # supersonic pocket
                  -0.3 + 1.2 * (x - 0.6))         # subsonic recovery
    out = ta.shock_from_surface_mach(x, cp, 0.85)
    assert out is not None and out["supersonic"] is True
    assert out["x_shock"] == pytest.approx(0.6, abs=0.02)
    assert out["p_ratio_rankine_hugoniot"] > 1.0
    assert out["strength_error_pct"] < 100.0


def test_shock_detection_returns_subcritical_for_low_mach():
    x = np.linspace(0.0, 1.0, 101)
    cp = -0.5 * (1.0 - x)
    out = ta.shock_from_surface_mach(x, cp, 0.3)
    assert out["supersonic"] is False
    assert out["x_shock"] is None


def test_isentropic_deviation_zero_for_uniform_field():
    rho = np.ones((10, 5))
    p = np.ones((10, 5))
    mach = np.full((10, 5), 0.2)
    out = ta.isentropic_deviation(rho, p, mach)
    assert out["max_rel_dev_subsonic"] == pytest.approx(0.0, abs=1e-14)
    p2 = p.copy()
    p2[0, 0] *= 1.01
    out2 = ta.isentropic_deviation(rho, p2, mach)
    assert out2["max_rel_dev_subsonic"] == pytest.approx(1.0 - 1.0 / 1.01, rel=1e-6)


# ---------------------------------------------------------------------------
# grid convergence index
# ---------------------------------------------------------------------------
def test_gci_recovers_second_order_convergence():
    r = 2.0
    f_ext, c, p = 1.25, 0.4, 2.0
    h = np.array([1.0, 0.5, 0.25])
    f = f_ext - c * h ** p
    out = ta.gci_roache(f[0], f[1], f[2], r)
    assert out["apparent_order"] == pytest.approx(p, rel=1e-6)
    assert out["f_extrapolated"] == pytest.approx(f_ext, rel=1e-8)
    assert out["gci_fine_pct"] < 5.0
    assert out["converged"] is True


def test_gci_flags_non_monotone_ladder():
    out = ta.gci_roache(1.0, 1.05, 1.04, 1.5)
    assert out["converged"] is False
    assert math.isnan(out["apparent_order"])


def test_critical_mach_number_from_suction_peak():
    # deeper suction peak -> earlier critical Mach number
    m_weak = ta.critical_mach_from_cp_min(-0.3)
    m_strong = ta.critical_mach_from_cp_min(-1.0)
    assert 0.5 < m_weak < 1.0
    assert m_strong < m_weak


# ---------------------------------------------------------------------------
# end-to-end smoke test (small grid, few iterations)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mach", [0.3, 0.85])
def test_run_naca_case_is_finite_and_physical(mach):
    res = ta.run_naca_case(mach, 0.0, n_points=64, nj=24, first_cell=0.008,
                           r_out=8.0, n_iter=300)
    cl = res["forces"]["cl"]
    cp = res["surface"]["cp"]
    assert np.isfinite(cl) and abs(cl) < 0.5          # symmetric -> CL ~ 0
    assert np.all(np.isfinite(cp))
    assert cp.min() < 0.0 < cp.max()                  # suction + stagnation
    assert res["field"]["max_local_mach"] < 3.0
    assert res["mesh"]["cells"] == 64 * 23


def test_run_wedge_case_matches_the_oblique_shock_within_a_few_percent():
    res = ta.run_wedge_case(2.0, 10.0, n_points=120, nj=32, first_cell=0.01,
                            r_out=6.0, n_iter=800)
    assert res["reference"]["exact_dp_over_p"] > 0.0
    assert abs(res["error_vs_exact_oblique_pct"]) < 10.0
    assert res["rear_wedge"]["p_ratio"] < 1.0          # expansion on the rear wedge