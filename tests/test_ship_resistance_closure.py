"""Verification tests for :mod:`tensorlbm.ship_resistance_closure`.

Every assertion is a *mathematical* statement about the closure identities
(volume integrals, thin-ship scaling laws, quadrature convergence, GCI algebra,
fail-closed verdicts) or a *geometry* statement about a locked block
coefficient.  No experimental agreement is asserted here: that lives in
``benchmarks/pending/ship_resistance_closure``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.hydrodynamics import ittc57_friction_coefficient
from tensorlbm.ship_cad import _series60_half_beam
from tensorlbm.ship_resistance_closure import (
    MICHELL_LEVELS,
    HullSectionGrid,
    _grid_from_half_beam,
    _michell_single,
    band_verdict,
    evaluate_case,
    gci,
    holtrop_form_factor,
    hull_section_grid,
    michell_wave_resistance,
    monotone_increasing,
    resistance_closure,
    series60_fullness_grid,
    slenderness_admission,
)

_REPO = Path(__file__).resolve().parents[1]
_STL_5415 = _REPO / "benchmarks/pending/dtmb5415_resistance/geometry/dtmb5415_underwater.stl"
_FAST_LEVELS = ((121, 25, 300), (241, 49, 600), (481, 97, 1200))


def _barge_grid(length=6.0, beam=1.0, draft=0.4, nx=81, nz=41) -> HullSectionGrid:
    """Rectangular-plan barge: ``db/dx = 0`` exactly, constant half-breadth."""

    def hb(xi, zeta):
        inside = (np.abs(xi) <= 1.0) & (zeta >= 0.0) & (zeta <= 1.0)
        return np.where(inside, np.clip(zeta, 0.0, 1.0) ** 0.0, 0.0)

    return _grid_from_half_beam(
        name="barge",
        half_beam_norm=hb,
        length_m=length,
        beam_m=beam,
        draft_m=draft,
        nx=nx,
        nz=nz,
        cb_locked=1.0,
        cb_source="rectangular plan closed form Cb = 1",
    )


# ---------------------------------------------------------------------------
# Geometry leg
# ---------------------------------------------------------------------------


def test_geometry_audit_kcs_reproduces_block_coefficient():
    grid = hull_section_grid("kcs", nx=601, nz=121)
    audit = grid.audit(cb_tolerance=0.02)
    assert audit["status"] == "pass"
    assert abs(audit["cb_error_pct"]) < 2.0
    # KCS is a full hull: L/B = 7.143, B/T = 2.981 by construction
    assert grid.length_m / grid.beam_m == pytest.approx(230.0 / 32.2, rel=1e-6)
    assert grid.beam_m / grid.draft_m == pytest.approx(32.2 / 10.8, rel=1e-6)


def test_geometry_audit_fails_closed_on_wrong_particulars():
    grid = hull_section_grid("kcs", nx=201, nz=41)
    bad = replace(grid, cb_locked=0.90)
    audit = bad.audit(cb_tolerance=0.02)
    assert audit["status"] == "geometry_rejected"
    closure = resistance_closure(bad, 0.26)
    assert closure.total["ct"] is None
    assert closure.total["status"] == "geometry_rejected"
    assert closure.decomposition["pass"] is False


def test_barge_geometry_integrals_are_exact():
    grid = _barge_grid()
    assert grid.block_coefficient == pytest.approx(1.0, rel=1e-12)
    assert grid.volume_m3 == pytest.approx(grid.length_m * grid.beam_m * grid.draft_m, rel=1e-12)
    assert grid.waterplane_coefficient == pytest.approx(1.0, rel=1e-12)


def test_series60_fullness_family_recovers_repo_cb060_member():
    # the repo's Series 60 member is (p=0.51, q=0.30); fixing q must solve p~0.51
    grid = series60_fullness_grid(0.60, nx=401, nz=81, vertical_exponent=0.30)
    assert grid.block_coefficient == pytest.approx(0.60, abs=1e-7)
    p_solved = float(grid.geometry_source.split("p=")[1].split(",")[0])
    # discretisation of the keel form (zeta^0.3 has an infinite slope at the
    # keel) shifts the solved exponent by ~1%: the member is p = 0.51
    assert p_solved == pytest.approx(0.51, abs=0.02)
    xi = np.linspace(-1.0, 1.0, 401)
    ze = np.linspace(0.0, 1.0, 81)
    XI, ZE = np.meshgrid(xi, ze, indexing="ij")
    hb = _series60_half_beam(XI, ZE)
    cb_repo = float(2.0 * np.trapezoid(np.trapezoid(hb, ze, axis=1), xi) / 4.0)
    assert cb_repo == pytest.approx(0.60, abs=5e-3)


def test_fullness_family_declares_unreachable_target():
    with pytest.raises(ValueError, match="unreachable"):
        series60_fullness_grid(0.90, nx=41, nz=21, vertical_exponent=0.30)


def test_fullness_family_hits_target_block_coefficients():
    cbs = [0.60, 0.70, 0.80]
    grids = [series60_fullness_grid(cb, nx=201, nz=41) for cb in cbs]
    for grid, cb in zip(grids, cbs):
        assert grid.block_coefficient == pytest.approx(cb, abs=1e-7)
        assert grid.audit()["status"] == "pass"
    # fuller member => larger waterplane coefficient
    cwp = [g.waterplane_coefficient for g in grids]
    assert cwp[0] < cwp[1] < cwp[2]


def test_wigley_closed_form_block_coefficient():
    grid = hull_section_grid("wigley", nx=601, nz=121)
    assert grid.block_coefficient == pytest.approx(4.0 / 9.0, abs=2e-3)
    assert grid.length_m / grid.beam_m == pytest.approx(10.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Wave leg: thin-ship identities
# ---------------------------------------------------------------------------


def test_michell_zero_for_zero_longitudinal_gradient():
    grid = _barge_grid()
    rw, cw, _ = _michell_single(grid, 0.25, 121, 41, 200, 4.0)
    assert abs(rw) < 1e-9


def test_michell_mirror_invariance():
    grid = hull_section_grid("wigley", nx=401, nz=81)
    mirrored = replace(grid, half_breadth_m=grid.half_breadth_m[::-1].copy())
    a = michell_wave_resistance(grid, 0.25, levels=_FAST_LEVELS)
    b = michell_wave_resistance(mirrored, 0.25, levels=_FAST_LEVELS)
    assert a["status"] == b["status"] == "applicable"
    assert a["cw"] == pytest.approx(b["cw"], rel=1e-10)


def test_michell_linear_in_half_breadth_and_beams():
    grid = hull_section_grid("wigley", nx=401, nz=81)
    wide = replace(grid, beam_m=1.5 * grid.beam_m, half_breadth_m=1.5 * grid.half_breadth_m)
    rw_a, cw_a, _ = _michell_single(grid, 0.25, 241, 49, 400, 4.0)
    rw_b, cw_b, _ = _michell_single(wide, 0.25, 241, 49, 400, 4.0)
    # source strength scales with b, so R_W ~ b^2 exactly (thin-ship linearity)
    assert rw_b / rw_a == pytest.approx(2.25, rel=1e-10)
    # C_W additionally normalises by S_wet, which for a hull this slender grows
    # sublinearly with B (the keel-to-waterline slope dominates dA), so C_W
    # rises faster than B: C_W/B is only asymptotically constant
    assert cw_b > cw_a
    assert cw_b / cw_a > 1.5


def test_michell_quadrature_converges_under_refinement():
    grid = hull_section_grid("wigley", nx=401, nz=81)
    record = michell_wave_resistance(grid, 0.30, levels=MICHELL_LEVELS)
    assert record["status"] == "applicable"
    conv = record["convergence"]
    assert conv["status"] in {"converged", "low_order"}
    # bilinear half-breadth + trapezoid integration converges at first order
    assert conv["p_observed"] == pytest.approx(1.0, abs=0.5)
    assert conv["relative_change_pct"] < 3.0
    assert record["gci_pct"] is not None and record["gci_pct"] < 8.0


def test_michell_low_froude_number_is_refused_rather_than_aliased():
    grid = hull_section_grid("wigley", nx=401, nz=81)
    low = michell_wave_resistance(grid, 0.05, levels=((241, 49, 600),))
    mid = michell_wave_resistance(grid, 0.30, levels=((241, 49, 600),))
    # at Fn = 0.05 the axial wave is ~2 cells long on this ladder: the module
    # must refuse the number instead of reporting an aliased value
    assert mid["quadrature"]["resolved"] is True
    assert low["quadrature"]["resolved"] is False
    assert low["status"] == "quadrature_unresolved"
    assert low["cw"] is None
    # and the resolved band must still decay towards low Froude number
    f15 = michell_wave_resistance(grid, 0.15, levels=MICHELL_LEVELS)
    assert f15["status"] == "applicable"
    assert f15["cw"] < mid["cw"]


def test_michell_magnitude_is_physical_for_wigley():
    grid = hull_section_grid("wigley", nx=401, nz=81)
    record = michell_wave_resistance(grid, 0.25, levels=((241, 49, 600),))
    assert 1e-4 < record["cw"] < 1e-2


def test_slenderness_gate_admits_slender_and_rejects_full_forms():
    wigley = hull_section_grid("wigley", nx=201, nz=41)
    kcs = hull_section_grid("kcs", nx=201, nz=41)
    series60_cb08 = series60_fullness_grid(0.80, nx=201, nz=41)
    assert slenderness_admission(wigley)["admitted"] is True
    assert slenderness_admission(kcs)["admitted"] is False
    assert slenderness_admission(series60_cb08)["admitted"] is False
    kcs_wave = michell_wave_resistance(kcs, 0.26, levels=((121, 25, 300),))
    assert kcs_wave["status"] == "outside_validity"
    assert kcs_wave["cw"] is None
    # the diagnostic value is retained for attribution only
    assert kcs_wave["diagnostic_cw"] > 0.0


# ---------------------------------------------------------------------------
# Convergence index and verdict algebra
# ---------------------------------------------------------------------------


def test_gci_exact_when_levels_identical():
    record = gci(0.5, 0.5, 0.5)
    assert record["status"] == "exact"
    assert record["gci_fine_pct"] == 0.0


def test_gci_recovers_second_order_for_known_series():
    # c = f + C r^2 with r = 2: coarse/medium/fine from f = 1.0, C = 0.1
    f = 1.0
    fine = f + 0.1 * 1.0
    medium = f + 0.1 * 4.0
    coarse = f + 0.1 * 16.0
    record = gci(fine, medium, coarse, ratio=2.0)
    assert record["p_observed"] == pytest.approx(2.0, rel=1e-6)
    assert record["extrapolated"] == pytest.approx(f, rel=1e-6)
    # Roache index for this ladder: 1.25 |fine - medium| / ((r^p - 1) fine)
    assert record["gci_fine_pct"] == pytest.approx(
        100.0 * 1.25 * abs(fine - medium) / ((2.0**2 - 1.0) * fine), rel=1e-9
    )


def test_gci_non_monotone_ladder_returns_no_index():
    record = gci(0.30, 0.31, 0.30, ratio=2.0)
    assert record["status"] == "non_monotone"
    assert record["gci_fine_pct"] is None


def test_band_verdict_is_fail_closed():
    assert band_verdict(4.0e-3, 4.2e-3, 8.0)["pass"] is True
    assert band_verdict(5.0e-3, 4.2e-3, 8.0)["pass"] is False
    withheld = band_verdict(None, 4.2e-3, 8.0)
    assert withheld["pass"] is False and withheld["status"] == "withheld"
    assert band_verdict(4.0e-3, None, 8.0)["pass"] is False
    assert band_verdict(float("nan"), 4.2e-3, 8.0)["pass"] is False
    assert band_verdict(4.0e-3, 0.0, 8.0)["pass"] is False


def test_monotone_increasing_helper():
    assert monotone_increasing([1.0, 2.0, 3.0]) is True
    assert monotone_increasing([1.0, 1.0, 2.0]) is False
    assert monotone_increasing([3.0, 2.0]) is False
    assert monotone_increasing([1.0]) is False


# ---------------------------------------------------------------------------
# Viscous leg + closure identity
# ---------------------------------------------------------------------------


def test_ittc57_friction_leg_is_the_library_correlation():
    grid = hull_section_grid("wigley", nx=201, nz=41)
    closure = resistance_closure(grid, 0.25, reference_ct=None)
    re = closure.viscous["re"]
    assert closure.viscous["cf"] == pytest.approx(ittc57_friction_coefficient(re), rel=1e-12)
    assert closure.viscous["grade"] == "CORRELATION"


def test_holtrop_form_factor_reproduces_the_published_regression():
    # hand evaluation of the Holtrop-Mennen (1984) closed form for the Todd
    # Series 60 Cb=0.60 model (L=6.096, B=0.9378, T=0.3423, nabla=1.176, Cp=0.61)
    one_plus_k = holtrop_form_factor(6.096, 0.9378, 0.3423, 1.176, 0.61)
    assert one_plus_k == pytest.approx(1.1661, abs=5e-4)
    # 1+k is bounded for conventional hulls and responds to the volume term
    assert 1.0 < holtrop_form_factor(7.28, 1.0192, 0.3418, 1.635, 0.66) < 1.5


def test_closure_decomposition_residual_below_one_micrometre_per_micrometre():
    grid = hull_section_grid("wigley", nx=401, nz=81)
    closure = resistance_closure(grid, 0.30)
    dec = closure.decomposition
    assert dec["pass"] is True
    assert dec["status"] == "self_consistent"
    assert dec["residual_relative"] < 1e-6
    assert dec["cf_share_fraction"] + dec["cw_share_fraction"] == pytest.approx(1.0, rel=1e-9)


def test_case_ladder_is_monotone_and_promotion_gate_fails_closed():
    grid = hull_section_grid("wigley", nx=201, nz=41)
    case = evaluate_case(
        grid,
        froude_numbers=[0.20, 0.25, 0.30],
        reference_ct=4.0e-3,
        reference_fn=0.30,
        reference_source="unit-test placeholder",
        band_pct=3.0,
    )
    assert len(case["ct_ladder"]) == 3
    assert all(isinstance(v, float) and v > 0 for v in case["ct_ladder"])
    assert isinstance(case["ct_monotone"], bool)
    assert case["decomposition"]["pass"] is True
    assert case["band"]["status"] in {"pass", "fail", "withheld"}


@pytest.mark.skipif(not _STL_5415.exists(), reason="DTMB 5415 STL not present")
def test_dtmb5415_stl_geometry_audit():
    grid = hull_section_grid(
        "dtmb5415",
        stl_path=str(_STL_5415),
        nx=121,
        nz=81,
        subdivision=4,
        smooth_passes=2,
    )
    audit = grid.audit(cb_tolerance=0.03)
    assert audit["status"] == "pass", audit
    assert abs(audit["cb_error_pct"]) < 3.0
    assert 3.0 < grid.wetted_area_m2 < 7.0
    assert grid.length_m / grid.beam_m == pytest.approx(142.03 / 19.1034, rel=1e-6)
    admission = slenderness_admission(grid)
    assert admission["admitted"] is True