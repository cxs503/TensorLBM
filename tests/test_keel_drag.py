"""Unit and integration tests for :mod:`tensorlbm.keel_drag` (mission P46W3-N01).

Split into three groups:

* **geometry** -- ``(h_k, b_k, s_k) -> mask`` conventions, periodicity, the two
  frontal-area conventions (section area vs silhouette) and the validation guards;
* **analysis** -- the amplitude renormalisation algebra, the correlation/fit
  helpers against analytic answers, the ITTC-78 closed form, the criterion
  functions on synthetic records (including the failing direction, so a criterion
  that cannot fail is caught);
* **kernel integration** -- one level-ice run and one shallow-keel run on a small
  lattice, checking the structural properties the campaign relies on
  (``C_d^form == 0`` exactly on level ice, positive measured stress, the split
  closing on the imposed momentum balance to a loose tolerance).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm import keel_drag as kd


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_level_ice_mask_is_empty_and_has_no_frontal_area():
    geom = kd.KeelGeometry(h_k=0.0, s_k=16.0, nx=16, ny=32, nz=24)
    mask = kd.keel_mask(geom)
    assert mask.shape == (24, 32, 16)
    assert not mask.any()
    audit = kd.keel_geometry_audit(geom, h_eff=20.0)
    assert audit["S_fr_x_over_plan"] == 0.0
    assert audit["coverage_nominal"] == 0.0


def test_triangular_profile_depth_and_volume():
    geom = kd.KeelGeometry(h_k=4.0, b_k=8.0, s_k=16.0, nx=32, ny=32, nz=66)
    mask = kd.keel_mask(geom)
    n_layers = int(np.count_nonzero(mask.any(axis=(1, 2))))  # occupied k-layers
    assert n_layers == 4
    # plan-mean depth of a triangle of base b and height h repeated at pitch s
    assert float(mask.mean()) * 66 == pytest.approx(0.5 * geom.coverage * geom.h_k, rel=0.12)


def test_mask_is_periodic_in_the_transverse_direction():
    geom = kd.KeelGeometry(h_k=3.0, b_k=6.0, s_k=16.0, nx=32, ny=32, nz=24)
    mask = kd.keel_mask(geom)
    # theta = 90: the profile runs along x with period s_k, and is uniform in y
    assert np.array_equal(mask[:, :, :16], mask[:, :, 16:])
    assert np.all(mask == mask[:, :1, :])
    # a pitch that does not divide the box is refused (it would break periodicity)
    with pytest.raises(ValueError):
        kd.KeelGeometry(h_k=3.0, b_k=6.0, s_k=12.0, nx=32, ny=32, nz=24)


def test_frontal_area_conventions_disagree_by_two_over_bk():
    geom = kd.KeelGeometry(h_k=4.0, b_k=8.0, s_k=16.0, nx=32, ny=32, nz=66)
    audit = kd.keel_geometry_audit(geom, h_eff=32.0)
    # realised x-projected voxel faces = silhouette = h_k/s_k
    assert audit["S_fr_x_over_plan"] == pytest.approx(geom.h_k / geom.s_k, rel=1e-12)
    assert audit["silhouette_over_plan"] == pytest.approx(geom.h_k / geom.s_k)
    # the section-area entry is divided by h_eff, the silhouette is not, so the
    # ratio between the two entries is 0.5 b_k/h_eff (per plan area they differ
    # by 2/b_k -- see the docstring of frontal_area_ratio)
    assert audit["section_area_over_plan"] == pytest.approx(
        audit["silhouette_over_plan"] * 0.5 * geom.b_k / 32.0)


def test_streamwise_ridges_have_no_frontal_area():
    geom = kd.KeelGeometry(h_k=4.0, b_k=8.0, s_k=16.0, nx=32, ny=32, nz=66, theta_deg=0.0)
    audit = kd.keel_geometry_audit(geom, h_eff=32.0)
    assert audit["S_fr_x_over_plan"] == 0.0        # structural negative control
    assert audit["keel_volume_fraction"] > 0.0     # the roughness is still there


def test_geometry_validation_guards():
    with pytest.raises(ValueError):
        kd.KeelGeometry(h_k=4.0, b_k=16.0, s_k=16.0)       # ridges touch
    with pytest.raises(ValueError):
        kd.KeelGeometry(h_k=40.0, b_k=8.0, s_k=16.0, nz=66)  # deeper than 45% clearance
    with pytest.raises(ValueError):
        kd.KeelGeometry(h_k=4.0, b_k=0.0, s_k=16.0)
    with pytest.raises(NotImplementedError):
        kd.KeelGeometry(h_k=4.0, b_k=8.0, s_k=16.0, theta_deg=45.0)


def test_coverage_and_aspect_properties():
    geom = kd.KeelGeometry(h_k=2.0, b_k=8.0, s_k=16.0, ny=32)
    assert geom.coverage == pytest.approx(0.5)
    assert geom.aspect == pytest.approx(4.0)
    assert geom.n_ridges == 2
    level = kd.KeelGeometry(h_k=0.0, s_k=16.0)
    assert level.coverage == 0.0 and math.isinf(level.aspect) and level.n_ridges == 0


def test_wetted_area_ratio_grows_with_depth():
    flat = kd.KeelGeometry(h_k=0.0, s_k=16.0, nx=16, ny=32, nz=24)
    shallow = kd.KeelGeometry(h_k=1.0, b_k=8.0, s_k=16.0, nx=16, ny=32, nz=24)
    deep = kd.KeelGeometry(h_k=4.0, b_k=8.0, s_k=16.0, nx=16, ny=32, nz=24)
    assert kd.wetted_area_ratio(flat) == pytest.approx(1.0)
    assert kd.wetted_area_ratio(shallow) > 1.0
    assert kd.wetted_area_ratio(deep) > kd.wetted_area_ratio(shallow)


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------


def _fake_record(**kw):
    """A synthetic kernel record with a self-consistent momentum balance."""
    tau_balance = 1.0e-5
    tau_me = kw.pop("tau_me", tau_balance)
    skin_frac = kw.pop("skin_frac", 0.4)
    u_bulk = kw.pop("u_bulk", 0.01)
    rho = 1.0
    q = rho * u_bulk ** 2
    rec = {
        "tau_me": tau_me, "tau_total": tau_balance,
        "tau_skin": skin_frac * tau_me, "tau_form": (1.0 - skin_frac) * tau_me,
        "cd_skin": skin_frac * tau_me / q, "cd_form": (1.0 - skin_frac) * tau_me / q,
        "cd_total": tau_balance / q, "u_bulk": u_bulk, "u_tau_target": 0.002,
        "u_tau_meas": 0.002, "nu": 0.0032, "tau": 0.5096, "h_eff": 32.0,
        "re_tau": 20.0, "re_h": 100.0, "closure_residual": 0.0,
        "momentum_balance_residual": 0.0, "stationarity_drift": 0.0,
        "cd_form_face_legacy": 0.0, "cd_skin_gradient_legacy": 0.0,
        "cd_skin_yface": 0.0, "max_mach": 0.02, "wall_seconds": 1.0,
        "backend": "test", "config": {"steps": 10}, "u_plus_bulk": 5.0,
        "profile": {"k": [], "y_plus": [], "u_plus": []}, "log_fit": None,
    }
    rec.update(kw)
    return rec


def test_amplitude_correction_is_the_momentum_ratio():
    balanced = _fake_record()
    c = kd.amplitude_correction(balanced)
    assert c["factor"] == pytest.approx(1.0)
    assert c["cd_skin"] == pytest.approx(balanced["cd_skin"])

    over = _fake_record(tau_me=2.0 * 1.0e-5)
    c2 = kd.amplitude_correction(over)
    assert c2["factor"] == pytest.approx(2.0)
    # C_d ~ 1/A and A ~ U_bulk, so the corrected C_d and the corrected U_bulk move
    # in opposite directions by exactly the factor
    assert c2["cd_skin"] == pytest.approx(2.0 * over["cd_skin"])
    assert c2["u_bulk"] == pytest.approx(over["u_bulk"] / 2.0)


def test_amplitude_correction_rejects_diverged_runs():
    with pytest.raises(ValueError):
        kd.amplitude_correction(_fake_record(tau_me=float("nan")))


def test_cd_and_cf_conventions_are_factor_two():
    rec = _fake_record()
    assert rec["cd_total"] == pytest.approx(rec["tau_total"] / rec["u_bulk"] ** 2)


def test_pearson_and_spearman_recover_known_relations():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert kd.pearson(x, 2 * x + 1)["r"] == pytest.approx(1.0)
    anti = kd.pearson(x, -3 * x + 7)
    assert anti["r"] == pytest.approx(-1.0)
    assert anti["p"] == pytest.approx(0.0, abs=1e-12)
    sp = kd.spearman(x, x ** 3)          # monotone but not linear
    assert sp["r"] == pytest.approx(1.0)
    assert kd.pearson(x, x ** 3)["r"] < 1.0


def test_pearson_ignores_non_finite_and_needs_three_points():
    assert math.isnan(kd.pearson([1.0, 2.0], [1.0, 2.0])["r"])
    r = kd.pearson([1.0, 2.0, 3.0, 4.0, np.nan], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert r["n"] == 4 and r["r"] == pytest.approx(1.0)


def test_linear_fit_recovers_a_line_with_standard_errors():
    x = np.linspace(0.0, 5.0, 9)
    fit = kd.linear_fit(x, 3.0 * x - 2.0)
    assert fit["slope"] == pytest.approx(3.0)
    assert fit["intercept"] == pytest.approx(-2.0)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["se_slope"] == pytest.approx(0.0, abs=1e-12)


def test_ittc78_formula_matches_the_published_closed_form():
    # dCf = [105 (ks/L)^(1/3) - 0.64] x 1e-3, clipped at zero
    assert float(kd.ittc78_allowance(1.0e-5)) == pytest.approx(
        (105.0 * (1e-5) ** (1.0 / 3.0) - 0.64) * 1e-3, rel=1e-12)
    assert float(kd.ittc78_allowance(1.0e-9)) == 0.0        # clipped
    assert float(kd.ittc78_allowance(1.0e-4)) > float(kd.ittc78_allowance(1.0e-5))


def test_ittc78_comparison_reports_every_convention_and_flags_validity():
    rec = {"cf_balance": 0.02, "cf_smooth": 0.01, "h_eff": 32.0,
           "geometry": {"h_k": 4.0, "nx": 32}, "re_h": 100.0}
    smooth = {"cf_balance": 0.01}
    comp = kd.ittc78_comparison(rec, smooth)
    assert set(comp["conventions"]) == set(kd.ITTC78_CONVENTIONS)
    assert comp["dCf_measured"] == pytest.approx(0.01)
    for row in comp["conventions"].values():
        assert row["ks_over_L"] > 0.0
        assert math.isfinite(row["dCf_ittc78"])
        # a lattice keel channel sits far above the calibrated window
        assert row["in_validity_window"] is False


def test_drag_decomposition_closure_and_form_share():
    rec = {"cd_skin": 0.004, "cd_form": 0.006, "cd_total_balance": 0.01}
    dec = kd.drag_decomposition(rec)
    assert dec["closure_residual"] == pytest.approx(0.0)
    assert dec["share_form"] == pytest.approx(0.6)


def test_frontal_drag_coefficient_matches_its_definition():
    rec = {"cd_form": 0.01, "geometry": {"S_fr_x_over_plan": 0.125}}
    assert kd.frontal_drag_coefficient(rec) == pytest.approx(2 * 0.01 / 0.125)
    with pytest.raises(ValueError):
        kd.frontal_drag_coefficient({"cd_form": 0.0,
                                     "geometry": {"S_fr_x_over_plan": 0.0}})


def _record(case_id, family, h_k, cd_skin, cd_form, cd_total, re_h=100.0, **kw):
    rec = {"case_id": case_id, "family": family,
           "geometry": {"h_k": h_k, "nx": 32,
                        "S_fr_x_over_plan": (h_k / 32.0) if h_k > 0 else 0.0},
           "cd_skin": cd_skin,
           "cd_form": cd_form, "cd_total_balance": cd_total,
           "cf_balance": 2 * cd_total, "re_h": re_h, "re_h_converged": re_h,
           "re_tau": kw.pop("re_tau", 20.0), "closure_identity_residual": 0.0,
           "momentum_balance_residual": kw.pop("momentum_balance_residual", 0.0),
           "h_eff": 32.0, "u_bulk": 0.01, "nu": 0.0032}
    rec.update(kw)
    return rec


def test_criterion_a_closure_passes_and_fails():
    good = [_record("a", "depth", 2.0, 0.02, 0.005, 0.025),
            _record("b", "depth", 4.0, 0.02, 0.010, 0.030)]
    assert kd.criterion_a_closure(good)["pass"] is True
    bad = [_record("a", "depth", 2.0, 0.02, 0.005, 0.023)]
    assert kd.criterion_a_closure(bad)["pass"] is False


def test_criterion_b_form_vs_depth_detects_monotonicity():
    ws = kd.criterion_b_form_vs_depth([
        _record("d1", "depth", 1.0, 0.02, 0.002, 0.022),
        _record("d2", "depth", 2.0, 0.02, 0.004, 0.024),
        _record("d3", "depth", 4.0, 0.02, 0.008, 0.028),
        _record("d4", "depth", 6.0, 0.02, 0.012, 0.032)])
    assert ws["pass"] is True
    assert ws["pearson"]["r"] == pytest.approx(1.0)
    wired = kd.criterion_b_form_vs_depth([
        _record("d1", "depth", 1.0, 0.02, 0.008, 0.028),
        _record("d2", "depth", 2.0, 0.02, 0.004, 0.024),
        _record("d3", "depth", 4.0, 0.02, 0.002, 0.022)])
    assert wired["pass"] is False


def test_criterion_c_skin_vs_re_detects_decay():
    rows = [_record(f"r{i}", "reynolds", 4.0, 0.02 / (10 ** i), 0.01, 0.03,
                    re_h=100.0 * 10 ** i) for i in range(4)]
    res = kd.criterion_c_skin_vs_re(rows)
    assert res["pass"] is True
    # a 1/Re law is monotone but not linear: the rank correlation is -1 while the
    # raw Pearson is far from it -- which is exactly why both are reported
    assert res["spearman"]["r"] == pytest.approx(-1.0)
    assert res["loglog"]["r"] == pytest.approx(-1.0)
    assert res["loglog"]["slope"] == pytest.approx(-1.0, rel=1e-6)
    assert res["pearson"]["r"] > -0.6
    flat = [_record(f"r{i}", "reynolds", 4.0, 0.02, 0.01, 0.03,
                    re_h=100.0 * 10 ** i) for i in range(4)]
    assert kd.criterion_c_skin_vs_re(flat)["pass"] is False


def test_criterion_e_negative_control_three_subchecks():
    records = [_record("smooth", "smooth", 0.0, 0.0225, 0.0, 0.0225),
               _record("d1", "depth", 1.0, 0.02, 0.003, 0.023),
               _record("d2", "depth", 2.0, 0.02, 0.006, 0.026),
               _record("d3", "depth", 4.0, 0.02, 0.012, 0.032)]
    res = kd.criterion_e_negative_control(records)
    assert res["level_ice_form_exactly_zero"] is True
    assert res["exponent_ok"] is True and res["C_D_bounded"] is True
    assert res["linear_intercept_diagnostic"]["gating"] is False
    assert res["pass"] is True
    # an offset in the depth law must be caught
    offset = [_record("smooth", "smooth", 0.0, 0.0225, 0.0, 0.0225),
              _record("d1", "depth", 1.0, 0.02, 0.010, 0.030),
              _record("d2", "depth", 4.0, 0.02, 0.012, 0.032)]
    assert kd.criterion_e_negative_control(offset)["pass"] is False


def test_evaluate_aggregates_verdicts():
    records = [_record("smooth", "smooth", 0.0, 0.02, 0.0, 0.02)]
    records += [_record(f"d{i}", "depth", float(h), 0.02, 0.003 * h, 0.02 + 0.003 * h)
                for i, h in enumerate((1.0, 2.0, 4.0), start=1)]
    records += [_record(f"r{i}", "reynolds", 4.0, 0.02 / (10 ** i), 0.01, 0.03,
                        re_tau=10.0 * (i + 1)) for i in range(3)]
    out = kd.evaluate(records, [records[0]])
    assert set(out["criteria"]) == {"a_closure", "b_form_vs_depth", "c_skin_vs_re",
                                    "d_ittc78", "e_negative_control"}
    assert out["n_criteria"] == 5
    assert isinstance(out["all_pass"], bool)


def test_keel_depth_distribution_is_normalised():
    dist = kd.keel_depth_distribution(n_bins=60)
    assert dist["pdf"].sum() == pytest.approx(1.0)
    mean = float((dist["h"] * dist["pdf"]).sum())
    # truncated exponential: mean slightly below the 2.5 m scale
    assert 2.0 < mean < 2.6


def test_predict_field_drag_scales_with_the_ridge_drag_coefficient():
    base = kd.predict_field_drag(c_d_keel=1.5, h_eff_m=30.0, re_full=5.6e6)
    double = kd.predict_field_drag(c_d_keel=3.0, h_eff_m=30.0, re_full=5.6e6)
    assert double["cd_form"] == pytest.approx(2.0 * base["cd_form"])
    assert double["cd_skin"] == pytest.approx(base["cd_skin"])   # form/skin independent
    denser = kd.predict_field_drag(c_d_keel=1.5, h_eff_m=30.0, re_full=5.6e6, density=0.3)
    assert denser["cd_form"] > base["cd_form"]
    assert base["observed_band"] == kd.OBSERVED_CD_BAND_KEEL


# ---------------------------------------------------------------------------
# kernel integration (small lattice, few steps)
# ---------------------------------------------------------------------------


def test_level_ice_run_has_no_form_drag_and_balances():
    case = kd.smooth_case(grid=(8, 8, 14), re_tau=5.0, u_tau=0.005, steps=800,
                          warmup=300, threads=2, case_id="tiny_smooth")
    rec = kd.run_keel_case(case, renorm=True, probe_steps=300, max_passes=3)
    assert rec["cd_form"] == 0.0                    # no x-normal faces at all
    assert rec["tau_me"] > 0.0
    assert rec["cd_skin"] == pytest.approx(rec["cd_total_measured"], rel=1e-12)
    assert abs(rec["cd_closure_residual"]) < 0.05   # amplitude iteration converged
    assert rec["amplitude_factor"] > 0.0


def test_shallow_keel_run_produces_a_positive_form_share():
    geom = kd.KeelGeometry(h_k=1.0, b_k=4.0, s_k=8.0, nx=8, ny=8, nz=14)
    case = kd.keel_case("tiny_keel", "depth", geom, re_tau=5.0, u_tau=0.005,
                        steps=800, warmup=300, threads=2)
    rec = kd.run_keel_case(case, renorm=True, probe_steps=200, max_passes=2)
    assert rec["geometry"]["S_fr_x_over_plan"] > 0.0
    assert rec["cd_total_balance"] > 0.0
    assert math.isfinite(rec["cd_skin"]) and math.isfinite(rec["cd_form"])
    assert len(rec["amplitude_passes"]) >= 2
    assert rec["amplitude_passes"][-1]["pass"] == "production"