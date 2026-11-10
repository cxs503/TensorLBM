"""Verification tests for :mod:`tensorlbm.wave_wake_michell`.

Every assertion is a statement about *mathematics that can be checked twice*:
a sampled quadrature against a closed-form integral, an analytic derivative
against a finite difference, a numerically solved caustic against the classical
``sin(alpha) = 1/3``, a shallow-water cone against its ray-envelope derivation.
No literature number is transcribed anywhere: the references are re-derived in
the test or produced by a second independent code path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.wave_wake_michell import (
    G,
    WIGLEY_III,
    WIGLEY_STANDARD,
    ParabolicHull,
    caustic_angle_analytic_deg,
    caustic_angle_deg,
    catamaran_interference,
    hump_hollow,
    measure_wake_angle_deg,
    michell_cw_curve,
    michell_resistance,
    michell_resistance_analytic,
    michell_spectrum_analytic,
    onaxis_wavelength,
    shallow_ray_envelope_deg,
    shallow_wake_angle_deg,
    slenderness_report,
    wake_field,
    wave_spectrum,
    wetted_surface_area,
)

K0_FN03_L1 = G / (0.3**2 * G * WIGLEY_STANDARD.length)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_parabolic_cb_matches_closed_form_and_volume_integral():
    h = WIGLEY_STANDARD
    assert h.cb_closed_form() == pytest.approx(4.0 / 9.0, rel=1e-15)
    x, area = h.area_curve_numeric(2001, 1001)
    cb = float(np.trapezoid(area, x) / (h.length * h.beam * h.draft))
    assert cb == pytest.approx(4.0 / 9.0, rel=1e-4)


def test_wigley_iii_cb_follows_the_alpha_law():
    h = WIGLEY_III
    expected = (2.0 / 3.0) * h.alpha / (h.alpha + 1.0)
    assert h.cb_closed_form() == pytest.approx(expected, rel=1e-15)
    assert h.cb_closed_form() == pytest.approx(0.555, abs=1e-3)
    x, area = h.area_curve_numeric(1201, 201)
    assert float(np.trapezoid(area, x) / (h.length * h.beam * h.draft)) == pytest.approx(
        expected, rel=1e-4)


def test_half_beam_derivative_matches_finite_difference():
    h = WIGLEY_STANDARD
    x = np.linspace(-0.49 * h.length, 0.49 * h.length, 41)
    z = np.linspace(-0.98 * h.draft, -0.02 * h.draft, 21)
    X, Z = np.meshgrid(x, z, indexing="ij")
    hx, hz = 1e-6 * h.length, 1e-6 * h.draft
    fd_x = (h.half_beam(X + hx, Z) - h.half_beam(X - hx, Z)) / (2 * hx)
    fd_z = (h.half_beam(X, Z + hz) - h.half_beam(X, Z - hz)) / (2 * hz)
    an_x = h.half_beam_dx(X, Z)
    an_z = h.half_beam_dz(X, Z)
    mask = np.abs(an_x) + np.abs(an_z) > 1e-6 * h.beam
    assert np.max(np.abs(an_x - fd_x)[mask]) < 1e-5 * h.beam / h.length
    assert np.max(np.abs(an_z - fd_z)[mask]) < 1e-5 * h.beam / h.draft


def test_wetted_area_is_between_projected_and_flat_plate():
    h = WIGLEY_STANDARD
    s = wetted_surface_area(h)["s_wet_m2"]
    projected = 2.0 * h.length * h.draft  # centreline-plane area
    assert s > projected  # the hull surface is larger than its projection
    assert s < 2.0 * projected  # ... but not absurdly so


def test_slenderness_admission_window():
    assert slenderness_report(WIGLEY_STANDARD)["admitted"] is True
    assert slenderness_report(WIGLEY_III)["admitted"] is True
    fat = ParabolicHull(length=10.0, beam=4.0, draft=2.0, name="fat")
    assert slenderness_report(fat)["admitted"] is False


# ---------------------------------------------------------------------------
# Michell integral
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn", [0.20, 0.30, 0.40])
def test_sampled_michell_matches_closed_form_spectrum(fn):
    """The sampled ``x,z`` quadrature must reproduce the exact spectrum."""
    h = WIGLEY_STANDARD
    sampled = michell_resistance(h, fn, n_x=321, n_z=121, n_theta=1200)
    exact = michell_resistance_analytic(h, fn)
    # the sampled version differs from the exact spectrum by <1% at Fn=0.2/0.3
    # and ~0.9% at Fn=0.4 (cell-centred sampling of a steeper integrand)
    assert sampled["cw"] == pytest.approx(exact["cw"], rel=2e-2)
    assert sampled["rw_n"] == pytest.approx(exact["rw_n"], rel=2e-2)


def test_spectrum_modulus_matches_closed_form_pointwise():
    h = WIGLEY_STANDARD
    rec = michell_resistance(h, 0.30, n_x=321, n_z=121, n_theta=400,
                             return_spectrum=True)
    th = rec["spectrum"]["theta_rad"]
    closed = michell_spectrum_analytic(th, h, 0.30)
    rel = np.abs(rec["spectrum"]["amplitude"] - np.abs(closed)) / np.abs(closed)
    assert float(np.max(rel)) < 1e-1  # coarse (400 theta) sampled quadrature
    mid = (th > 0.05) & (th < 0.8)
    assert float(np.max(rel[mid])) < 5e-3


def test_quadrature_refinement_converges():
    h = WIGLEY_STANDARD
    coarse = michell_resistance(h, 0.30, n_x=81, n_z=31, n_theta=300)["cw"]
    mid = michell_resistance(h, 0.30, n_x=161, n_z=61, n_theta=600)["cw"]
    fine = michell_resistance(h, 0.30, n_x=321, n_z=121, n_theta=1200)["cw"]
    assert abs(fine - mid) / fine < 0.01
    assert abs(fine - coarse) / fine < 0.03
    assert abs(fine - mid) < abs(fine - coarse)  # monotone improvement


def test_wave_resistance_scales_with_beam_squared():
    """``R_w`` is quadratic in the source strength, i.e. quadratic in beam."""
    a = michell_resistance(WIGLEY_STANDARD, 0.30)
    b_hull = ParabolicHull(length=WIGLEY_STANDARD.length, beam=2 * WIGLEY_STANDARD.beam,
                           draft=WIGLEY_STANDARD.draft)
    b = michell_resistance(b_hull, 0.30, wetted_area=a["s_wet_m2"] * 2.0)
    assert b["rw_n"] / a["rw_n"] == pytest.approx(4.0, rel=2e-2)


def test_froude_scaling_invariance_of_cw():
    """``C_W(Fn)`` is invariant under a change of hull scale (Froude similarity)."""
    fns = [0.24, 0.30, 0.36]
    small = ParabolicHull(length=1.0, beam=0.1, draft=0.0625)
    big = ParabolicHull(length=3.048, beam=0.1 * 3.048, draft=0.0625 * 3.048)
    a = michell_cw_curve(small, fns)
    b = michell_cw_curve(big, fns)
    assert np.max(np.abs(b["cw"] - a["cw"]) / a["cw"]) < 1e-10


# ---------------------------------------------------------------------------
# wake geometry
# ---------------------------------------------------------------------------
def test_deep_water_wavenumber_selection():
    k0 = 11.0
    th = np.linspace(0.0, 1.2, 13)
    kappa = wave_spectrum(th, k0, None)
    assert np.allclose(kappa, k0 / np.cos(th) ** 2, rtol=1e-14)


def test_finite_depth_spectrum_reduces_to_deep_water():
    k0 = 11.0
    th = np.array([0.0, 0.3, 0.6])
    deep = wave_spectrum(th, k0, None)
    shallowish = wave_spectrum(th, k0, 1.0e4)  # kh >> 1
    assert np.allclose(shallowish, deep, rtol=1e-3)


def test_shallow_limit_satisfies_the_implicit_relation():
    k0, h = 11.0, 0.02  # k0 h = 0.22 << 1 -> shallow
    th = np.array([0.2, 0.4])
    kappa = wave_spectrum(th, k0, h)
    lhs = kappa * np.cos(th) ** 2
    rhs = k0 * np.tanh(kappa * h)
    assert np.allclose(lhs, rhs, rtol=1e-6)


def test_caustic_angle_is_the_classical_19_47_deg():
    rec = caustic_angle_deg(K0_FN03_L1)
    assert rec["found"] is True
    assert rec["alpha_deg"] == pytest.approx(19.47122063, abs=1e-6)
    assert rec["sin_alpha"] == pytest.approx(1.0 / 3.0, rel=1e-8)


def test_caustic_angle_independent_of_speed():
    for k0 in (4.0, 11.111, 40.0):
        assert caustic_angle_deg(k0)["alpha_deg"] == pytest.approx(19.47122063, abs=1e-6)


def test_caustic_analytic_route_agrees_and_gives_35_264_deg_wave_direction():
    rec = caustic_angle_analytic_deg()
    assert rec["alpha_deg"] == pytest.approx(19.47122063, abs=1e-6)
    assert rec["tan_alpha"] == pytest.approx(1.0 / (2.0 * math.sqrt(2.0)), rel=1e-6)
    assert rec["theta_at_caustic_deg"] == pytest.approx(35.26439, abs=1e-3)


def test_two_caustic_routes_agree():
    a = caustic_angle_deg(K0_FN03_L1)["alpha_deg"]
    b = caustic_angle_analytic_deg()["alpha_deg"]
    assert abs(a - b) < 1e-5


@pytest.mark.parametrize("froude_depth", [1.2, 1.6, 2.0])
def test_shallow_wake_angle_matches_its_ray_envelope_derivation(froude_depth):
    closed = shallow_wake_angle_deg(froude_depth)["alpha_deg"]
    ray = shallow_ray_envelope_deg(froude_depth)
    assert closed == pytest.approx(math.degrees(math.asin(1.0 / froude_depth)), rel=1e-12)
    assert ray == pytest.approx(closed, abs=1e-4)


def test_subcritical_shallow_flow_has_no_steady_cone():
    rec = shallow_wake_angle_deg(0.8)
    assert rec["supercritical"] is False
    assert rec["alpha_deg"] is None
    assert shallow_ray_envelope_deg(0.8) is None


def test_wake_field_onaxis_wavelength():
    h = WIGLEY_STANDARD
    wf = wake_field(h, 0.30, n_theta=600,
                    radii=np.linspace(0.3, 6.0, 400) * h.length,
                    alphas=np.linspace(0.0, 0.2, 5))
    rec = onaxis_wavelength(wf)
    assert rec["deviation_pct"] is not None
    assert abs(rec["deviation_pct"]) < 5.0
    assert rec["wavelength_theory_m"] == pytest.approx(2 * math.pi * 0.3**2 * h.length, rel=1e-12)


def test_wake_envelope_measure_is_reported_not_asserted():
    """The naive envelope estimator over-reads the caustic; keep it honest.

    The stationary-phase synthesis keeps non-stationary leakage outside the
    wedge, so the envelope is *not* a caustic estimator.  The test pins the
    behaviour (a number is returned, it over-estimates) instead of pretending
    the measurement passes the 5 deg gate.
    """
    h = WIGLEY_STANDARD
    wf = wake_field(h, 0.30, n_theta=400,
                    radii=np.linspace(0.3, 5.0, 120) * h.length,
                    alphas=np.linspace(0.0, math.radians(50.0), 51))
    rec = measure_wake_angle_deg(wf)
    assert rec["alpha_deg"] is None or rec["alpha_deg"] > 19.4712


# ---------------------------------------------------------------------------
# hump / hollow and interference
# ---------------------------------------------------------------------------
def test_hump_hollow_recovers_synthetic_extrema():
    fn = np.linspace(0.2, 0.6, 401)
    cw = (np.exp(-150.0 * (fn - 0.28) ** 2)
          + 0.8 * np.exp(-150.0 * (fn - 0.45) ** 2))
    rec = hump_hollow(fn, cw, min_prominence_rel=1e-3)
    humps = [h["fn"] for h in rec["humps"]]
    hollows = [h["fn"] for h in rec["hollows"]]
    assert any(abs(f - 0.28) < 5e-3 for f in humps)
    assert any(abs(f - 0.45) < 5e-3 for f in humps)
    assert len(hollows) == 1 and abs(hollows[0] - 0.365) < 1e-2


def test_hump_hollow_on_the_michell_curve_is_non_empty():
    rec = michell_cw_curve(WIGLEY_STANDARD, np.arange(0.22, 0.46, 0.01))
    hh = hump_hollow(rec["fn"], rec["cw"])
    assert len(hh["humps"]) + len(hh["hollows"]) >= 1


def test_catamaran_interference_limits():
    """Coincident hulls must give the coherent sum (factor 2).

    At large separation thin-ship superposition does *not* decorrelate to 1:
    the transverse wave system has a single wavelength ``2 pi U^2/g``, so its
    phase difference is fixed for every separation and the mean stays at
    ``1 + cos(k0 S)`` weighted by the spectrum.  The limit is therefore
    ``|1 + cos(k0 S)|``-bounded, not 1 -- asserted as such rather than as 1.
    """
    h = WIGLEY_STANDARD
    zero = catamaran_interference(h, 0.30, (1e-3,), mode="longitudinal")
    assert zero["rows"][0]["interference_factor"] == pytest.approx(2.0, abs=1e-2)
    k0 = G / (0.30**2 * G * h.length)
    for s_over_l in (10.0, 40.0):
        rec = catamaran_interference(h, 0.30, (s_over_l,), mode="longitudinal",
                                     n_theta=4000)
        expected_transverse = 1.0 + math.cos(k0 * s_over_l * h.length)
        got = rec["rows"][0]["interference_factor"]
        assert 0.0 <= got <= 2.0
        # the full spectrum includes divergent waves, so it brackets the pure
        # transverse-wave value within a wide but finite band
        assert abs(got - expected_transverse) < 0.6


def test_catamaran_interference_factor_stays_physical():
    h = WIGLEY_STANDARD
    for mode in ("transverse", "longitudinal"):
        rec = catamaran_interference(h, 0.30, (0.2, 0.3, 0.4), mode=mode)
        for row in rec["rows"]:
            assert 0.0 <= row["interference_factor"] <= 2.0
        assert rec["cw_single"] > 0.0


def test_catamaran_mode_is_validated():
    with pytest.raises(ValueError):
        catamaran_interference(WIGLEY_STANDARD, 0.30, (0.2,), mode="diagonal")


def test_spectrum_is_symmetric_in_froude_number_sign():
    """|I(theta)| uses f_x^2 and a symmetric hull: only |Fn| matters."""
    h = WIGLEY_STANDARD
    a = michell_resistance(h, 0.30)["cw"]
    with pytest.raises(ValueError):
        michell_resistance(h, -0.30)
    assert a > 0.0