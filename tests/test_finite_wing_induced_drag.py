"""Acceptance tests for the finite-wing induced-drag verification surface.

Covers the Trefftz-plane ledger node (P218-N08): elliptic loading law, Prandtl
induced-drag formula, Trefftz energy ledger, tip-vortex roll-up position,
aspect-ratio scaling law and Biot-Savart/VLM code verification.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.finite_wing_induced_drag import (
    NACA0012_SECTION,
    VLMWake,
    VortexLattice,
    _segment_velocity,
    aspect_ratio_scaling,
    elliptic_loading_deviation,
    elliptic_spanload,
    elliptic_twist,
    prandtl_induced_drag,
    solve_lifting_line,
    tip_vortex_position_prediction,
    tip_vortex_rollup,
    trefftz_energy_kirchhoff,
    wake_from_loading,
    wing_geometry,
)

A0 = NACA0012_SECTION["a0"]


# ---------------------------------------------------------------------------
# geometry / analytic anchors
# ---------------------------------------------------------------------------
def test_geometry_area_and_aspect_ratio():
    for kind in ("elliptic", "rectangular"):
        for ar in (6.0, 8.0, 10.0):
            g = wing_geometry(ar, kind=kind)
            assert g.span == pytest.approx(math.sqrt(ar), rel=1e-12)
            y = np.linspace(-0.5 * g.span, 0.5 * g.span, 2001)
            area = float(np.trapezoid(g.chord(y), y))
            assert area == pytest.approx(1.0, rel=5e-3)


def test_elliptic_spanload_anchor():
    b, cl0 = 4.0, 0.5
    y = np.linspace(-b / 2, b / 2, 401)
    cl = elliptic_spanload(y, b, cl0)
    assert cl.max() == pytest.approx(cl0, rel=1e-6)
    assert abs(cl[0]) < 1e-6 and abs(cl[-1]) < 1e-6
    assert np.allclose(cl, cl[::-1], atol=1e-12)


def test_prandtl_formula():
    assert prandtl_induced_drag(1.0, 8.0) == pytest.approx(1.0 / (math.pi * 8.0), rel=1e-12)
    assert prandtl_induced_drag(1.0, 8.0, e=0.9) > prandtl_induced_drag(1.0, 8.0)


# ---------------------------------------------------------------------------
# lifting line
# ---------------------------------------------------------------------------
def test_elliptic_planform_lifting_line_is_exactly_elliptic():
    ll = solve_lifting_line(8.0, kind="elliptic", alpha_deg=4.0, n_terms=40)
    assert ll.span_efficiency == pytest.approx(1.0, abs=1e-6)
    assert abs(ll.delta) < 1e-8
    # only the first Fourier mode survives on an elliptic planform
    assert abs(ll.a_n[2] / ll.a_n[0]) < 1e-6


def test_lifting_line_matches_elliptic_wing_closed_form():
    """Prandtl's elliptic-wing lift slope a = a0/(1 + a0/(pi AR))."""
    for ar in (6.0, 8.0, 10.0):
        ll = solve_lifting_line(ar, kind="elliptic", alpha_deg=4.0)
        alpha = math.radians(4.0)
        cl_exact = A0 * alpha / (1.0 + A0 / (math.pi * ar))
        assert ll.cl_total == pytest.approx(cl_exact, rel=2e-3)


def test_lifting_line_2d_limit():
    """At AR = 200 the lifting line must recover the 2-D section slope."""
    ll = solve_lifting_line(200.0, kind="rectangular", alpha_deg=4.0)
    cl_2d = 2.0 * math.pi * math.sin(math.radians(4.0))
    assert ll.cl_total == pytest.approx(cl_2d, rel=0.02)


def test_rectangular_planform_span_efficiency_literature_range():
    """Untwisted rectangular wings: Oswald e ~ 0.92-0.96 at AR 6-10."""
    for ar in (6.0, 8.0, 10.0):
        e = solve_lifting_line(ar, kind="rectangular", alpha_deg=4.0).span_efficiency
        assert 0.90 < e < 0.97


def test_zero_alpha_gives_no_lift_and_no_drag():
    ll = solve_lifting_line(8.0, kind="elliptic", alpha_deg=0.0)
    assert abs(ll.cl_total) < 1e-12
    assert abs(ll.cd_i) < 1e-12
    vlm = VortexLattice(8.0, kind="elliptic", alpha_deg=0.0, n_span=12, n_chord=3)
    assert abs(vlm.cl_total) < 1e-10
    assert abs(vlm.cd_i_near) < 1e-12


# ---------------------------------------------------------------------------
# Biot-Savart / VLM code verification
# ---------------------------------------------------------------------------
def test_biot_savart_reduces_to_infinite_line():
    gamma, h = 0.3, 0.7
    p = np.array([[0.0, 0.0, h]])
    a = np.array([-1.0e4, 0.0, 0.0])
    b = np.array([1.0e4, 0.0, 0.0])
    v = _segment_velocity(p, a, b, gamma)
    assert abs(v[0, 1]) == pytest.approx(gamma / (2.0 * math.pi * h), rel=2e-4)
    assert abs(v[0, 0]) < 1e-6 and abs(v[0, 2]) < 1e-6


def test_vlm_matches_lifting_line_in_2d_limit():
    ll = solve_lifting_line(200.0, kind="rectangular", alpha_deg=4.0)
    vlm = VortexLattice(200.0, kind="rectangular", alpha_deg=4.0,
                        n_span=24, n_chord=4)
    assert vlm.cl_total == pytest.approx(ll.cl_total, rel=5e-3)
    # in the 2-D limit the circulation is uniform over the inner span
    inner = np.abs(vlm.y_mid) < 0.8 * 0.5 * vlm.geometry.span
    gam_inner = vlm.gamma_strip[inner] / vlm.gamma_strip[inner].max()
    assert np.max(np.abs(gam_inner - 1.0)) < 1e-2


def test_vlm_spanwise_convergence():
    cl = [VortexLattice(8.0, kind="elliptic", alpha_deg=4.0, n_span=ns,
                        n_chord=4).cl_total for ns in (20, 40, 80)]
    assert abs(cl[-1] - cl[-2]) / cl[-1] < 0.01


# ---------------------------------------------------------------------------
# C1: elliptic loading law
# ---------------------------------------------------------------------------
def test_c1_elliptic_planform_loading_law():
    """Spanwise lift distribution of the elliptic planform is the elliptic law."""
    ll = solve_lifting_line(8.0, kind="elliptic", alpha_deg=4.0)
    vlm = VortexLattice(8.0, kind="elliptic", alpha_deg=4.0, n_span=40, n_chord=6)
    gam = ll.circulation(vlm.y_mid) / ll.circulation_root
    dev = elliptic_loading_deviation(vlm.y_mid, gam, vlm.geometry.span)
    assert dev["l2_rel"] < 0.02


def test_c1_elliptic_twist_gives_elliptic_section_lift():
    """Elliptic washout on a constant-chord wing reproduces cl(y) elliptic law."""
    ar, alpha = 8.0, 4.0
    twist = elliptic_twist(ar, alpha)
    ll = solve_lifting_line(ar, kind="rectangular", alpha_deg=alpha, twist_deg=twist)
    assert ll.span_efficiency == pytest.approx(1.0, abs=2e-2)
    assert twist(1.4) < 0.0                     # washout at the tip
    vlm = VortexLattice(ar, kind="rectangular", alpha_deg=alpha, n_span=40,
                        n_chord=6, twist_fn=twist)
    dev = elliptic_loading_deviation(vlm.y_mid, vlm.cl_strip, vlm.geometry.span)
    assert dev["l2_rel"] < 0.05


# ---------------------------------------------------------------------------
# C2: Prandtl induced-drag formula
# ---------------------------------------------------------------------------
def test_c2_prandtl_induced_drag():
    ll = solve_lifting_line(8.0, kind="elliptic", alpha_deg=4.0)
    vlm = VortexLattice(8.0, kind="elliptic", alpha_deg=4.0, n_span=40, n_chord=6)
    pred = prandtl_induced_drag(vlm.cl_total, 8.0, e=ll.span_efficiency)
    assert abs(vlm.cd_i_near - pred) / pred < 0.15


# ---------------------------------------------------------------------------
# C3: Trefftz-plane energy ledger
# ---------------------------------------------------------------------------
def test_c3_trefftz_ledger_closes():
    ll = solve_lifting_line(8.0, kind="elliptic", alpha_deg=4.0)
    vlm = VortexLattice(8.0, kind="elliptic", alpha_deg=4.0, n_span=40, n_chord=6)
    wake = wake_from_loading(ll.circulation, ll.geometry.span, n_wake=200)
    far = trefftz_energy_kirchhoff(wake)
    residual = abs(far["d_i_far"] - vlm.cd_i_near) / vlm.cd_i_near
    assert residual < 0.10
    assert far["energy_cross"] > 0.0


def test_c3_ledger_converges_with_sheet_refinement():
    ll = solve_lifting_line(8.0, kind="elliptic", alpha_deg=4.0)
    vlm = VortexLattice(8.0, kind="elliptic", alpha_deg=4.0, n_span=40, n_chord=6)
    vals = [trefftz_energy_kirchhoff(
        wake_from_loading(ll.circulation, ll.geometry.span, n_wake=n))["d_i_far"]
        for n in (50, 200)]
    r50 = abs(vals[0] - vlm.cd_i_near) / vlm.cd_i_near
    r200 = abs(vals[1] - vlm.cd_i_near) / vlm.cd_i_near
    assert r200 < r50 + 0.05          # refinement does not degrade the ledger
    assert r200 < 0.10


# ---------------------------------------------------------------------------
# C4: tip-vortex roll-up position
# ---------------------------------------------------------------------------
def test_c4_tip_vortex_position_vs_betz():
    ll = solve_lifting_line(8.0, kind="elliptic", alpha_deg=4.0)
    span = ll.geometry.span
    res = tip_vortex_rollup(ll.circulation, span, n_sheet=160)
    chord = 1.0 / span
    err_chord = abs(abs(res.tip_y) - tip_vortex_position_prediction(span)) / chord
    assert err_chord < 0.10
    # the rolled-up tip vortices are symmetric about the centre line
    assert abs(abs(res.tip_y) - abs(res.tip_y_left)) / span < 0.02
    # roll-up moves vorticity outboard only (no spurious inboard drift)
    assert abs(res.tip_y) > 0.25 * span


# ---------------------------------------------------------------------------
# C5: aspect-ratio scaling law
# ---------------------------------------------------------------------------
def test_c5_aspect_ratio_scaling():
    pts = []
    for ar in (6.0, 8.0, 10.0):
        vlm = VortexLattice(ar, kind="elliptic", alpha_deg=4.0, n_span=20, n_chord=4)
        pts.append({"ar": ar, "cl": vlm.cl_total, "cd_i": vlm.cd_i_near})
    fit = aspect_ratio_scaling(pts)
    assert fit["r2"] > 0.98
    assert abs(fit["slope"] * math.pi - 1.0) < 0.15      # slope -> 1/(pi e), e~1
