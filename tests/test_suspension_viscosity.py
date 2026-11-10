"""Tests for the LBM-calibrated suspension effective-viscosity closure.

Run with::

    PYTHONPATH=src pytest tests/test_suspension_viscosity.py -q

The suite is analytic and therefore fast (<5 s): every function is exercised
against its closed-form limit or against synthetic data with known parameters.
The LBM calibration itself is reproduced by
``benchmarks/suspension/run_lbm_calibration.py`` +
``benchmarks/suspension/fit_calibration.py`` (its acceptance JSON is asserted
here when the artefact is present, and skipped otherwise).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensorlbm import suspension_viscosity as sv

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYSIS = os.path.join(REPO, "benchmarks", "suspension", "artifacts", "analysis.json")


# ---------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------
def test_einstein_dilute_limit_is_exact():
    phi = np.array([0.0, 1e-6, 0.001, 0.01])
    eta = sv.relative_viscosity(phi, "einstein")
    assert eta[0] == 1.0
    assert np.allclose(eta, 1.0 + sv.INTRINSIC_VISCOSITY_SPHERE * phi, rtol=0, atol=0)
    # the slope at phi -> 0 IS the intrinsic viscosity
    eps = 1e-8
    assert abs((sv.relative_viscosity(eps, "einstein") - 1.0) / eps - 2.5) < 1e-8


def test_batchelor_second_order_coefficient():
    phi = 0.1
    assert sv.relative_viscosity(phi, "batchelor") == pytest.approx(
        1.0 + 2.5 * phi + 6.2 * phi**2, rel=0, abs=1e-14)


def test_krieger_dougherty_properties():
    pm = 0.63
    assert sv.relative_viscosity(0.0, phi_max=pm) == pytest.approx(1.0, abs=1e-15)
    # Einstein slope preserved at phi -> 0
    eps = 1e-7
    slope = (sv.relative_viscosity(eps, phi_max=pm) - 1.0) / eps
    assert slope == pytest.approx(2.5, rel=1e-5)
    # monotone divergence toward phi_max
    eta = sv.relative_viscosity([0.2, 0.4, 0.6], phi_max=pm)
    assert eta[0] < eta[1] < eta[2]
    assert eta[2] > 20.0
    with pytest.raises(ValueError):
        sv.relative_viscosity(0.7, phi_max=pm)


def test_model_keyword_and_bad_inputs():
    with pytest.raises(ValueError):
        sv.relative_viscosity(0.1, "power_law")
    with pytest.raises(ValueError):
        sv.relative_viscosity(-0.01)
    with pytest.raises(ValueError):
        sv.relative_viscosity(0.1, phi_max=0.0)


def test_effective_viscosity_scales_with_carrier():
    mu0 = 1.8e-3  # water at 0 C, Pa s
    assert sv.effective_viscosity(0.1, mu0, "einstein") == pytest.approx(
        mu0 * 1.25, rel=1e-15)


def test_inverse_roundtrip_all_models():
    phi = np.array([0.01, 0.1, 0.25, 0.4])
    pm = 0.62
    for model in ("einstein", "batchelor"):
        back = sv.inverse_relative_viscosity(
            sv.relative_viscosity(phi, model), model)
        assert np.allclose(back, phi, rtol=1e-10, atol=1e-12)
    eta = sv.relative_viscosity(phi, phi_max=pm)
    back = sv.inverse_relative_viscosity(eta, phi_max=pm)
    assert np.allclose(back, phi, rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError):
        sv.inverse_relative_viscosity(0.5)


# ---------------------------------------------------------------------------
# calibration fits
# ---------------------------------------------------------------------------
def test_intrinsic_viscosity_recovers_einstein_from_noisy_data():
    phi = np.array([0.02, 0.02, 0.05, 0.05])
    clean = sv.relative_viscosity(phi, "einstein")
    semi = np.array([0.004, -0.003, 0.005, -0.006])   # 2 x SEM, 2 seeds
    fit = sv.intrinsic_viscosity(phi, clean + semi, weights=[1 / 0.004**2] * 4)
    assert abs(fit["slope"] - 2.5) < 0.05
    assert fit["n_points"] == 4
    assert 0.0 <= fit["r2"] <= 1.0
    assert fit["slope_se"] >= 0.0
    with pytest.raises(ValueError):
        sv.intrinsic_viscosity([0.02], [1.05])
    with pytest.raises(ValueError):
        sv.intrinsic_viscosity([0.02, 0.05], [1.05, 1.1], weights=[1.0, -1.0])


def test_intrinsic_viscosity_detects_a_volume_convention_bias():
    """The campaign's central diagnostic: a (a_h/a_eff)^3 < 1 ratio lowers the slope."""
    phi_mask = np.array([0.02, 0.05])
    ratio = 0.75
    eta = sv.relative_viscosity(phi_mask * ratio, "einstein")
    mask_fit = sv.intrinsic_viscosity(phi_mask, eta)
    hydro_fit = sv.intrinsic_viscosity(phi_mask * ratio, eta)
    assert mask_fit["slope"] == pytest.approx(2.5 * ratio, rel=1e-10)
    assert hydro_fit["slope"] == pytest.approx(2.5, rel=1e-10)
    assert not mask_fit["pass_slope_5pct"] and hydro_fit["pass_slope_5pct"]


def test_fit_phi_max_recovers_ground_truth():
    pm_true = 0.615
    phi = np.linspace(0.02, 0.45, 10)
    eta = sv.relative_viscosity(phi, phi_max=pm_true)
    fit = sv.fit_phi_max(phi, eta)
    assert fit["phi_max"] == pytest.approx(pm_true, abs=2e-3)
    assert fit["log_r2"] > 0.9999
    with pytest.raises(ValueError):
        sv.fit_phi_max(phi[:2], eta[:2])


def test_fit_quality_flags_a_wrong_phi_max():
    phi = np.linspace(0.02, 0.45, 8)
    eta = sv.relative_viscosity(phi, phi_max=0.62)
    good = sv.fit_quality(phi, eta, phi_max=0.62)
    bad = sv.fit_quality(phi, eta, phi_max=0.72)
    assert good["r2"] > 0.999 and good["pass_r2_0p95"]
    assert bad["r2"] < good["r2"]
    assert bad["max_rel_err"] > good["max_rel_err"]


def test_divergence_exponent_matches_krieger_dougherty():
    pm = 0.63
    phi = np.linspace(0.05, 0.45, 9)
    eta = sv.relative_viscosity(phi, phi_max=pm)
    d = sv.divergence_exponent(phi, eta, pm)
    assert d["slope"] == pytest.approx(-2.5 * pm, rel=1e-6)
    assert d["pass_divergence_10pct"]


# ---------------------------------------------------------------------------
# volume-fraction convention
# ---------------------------------------------------------------------------
def test_hydrodynamic_radius_and_ratio():
    mu, u, a = 0.1, 0.002, 4.0
    # unconfined Stokes drag of a sphere of radius a
    F = 6.0 * math.pi * mu * a * u
    assert sv.hydrodynamic_radius(F, mu, u) == pytest.approx(a, rel=1e-12)
    # confinement factor > 1 inflates the drag of the same sphere
    K = 1.35
    assert sv.hydrodynamic_radius(F * K, mu, u, K) == pytest.approx(a, rel=1e-12)
    assert sv.hydrodynamic_volume_ratio(a, a) == pytest.approx(1.0)
    assert sv.hydrodynamic_volume_ratio(0.9 * a, a) == pytest.approx(0.729)
    with pytest.raises(ValueError):
        sv.hydrodynamic_radius(0.0, mu, u)


def test_volume_fraction_from_mask_is_linear_and_vectorised():
    ratio = 0.8
    a_eff = 5.0
    a_h = a_eff * ratio ** (1.0 / 3.0)
    phi = np.array([0.02, 0.1, 0.4])
    out = sv.volume_fraction_from_mask(phi, a_h, a_eff)
    assert np.allclose(out, phi * ratio, rtol=1e-12)
    assert isinstance(sv.volume_fraction_from_mask(0.1, a_h, a_eff), float)


def test_aggregate_by_fraction_mean_sem():
    phi = [0.02, 0.02, 0.05, 0.05, 0.02]
    eta = [1.04, 1.06, 1.12, 1.08, 1.05]
    rows = sv.aggregate_by_fraction(phi, eta, seeds=[1, 2, 1, 2, 3])
    assert len(rows) == 2
    assert rows[0]["phi"] == pytest.approx(0.02)
    assert rows[0]["n"] == 3
    assert rows[0]["eta_r"] == pytest.approx(np.mean([1.04, 1.06, 1.05]))
    assert rows[0]["sem"] == pytest.approx(np.std([1.04, 1.06, 1.05], ddof=1) / math.sqrt(3))


def test_extrapolate_resolution_recovers_limit():
    v_inf, c = 2.5, 1.8
    h = np.array([3.5, 5.0, 6.5])
    values = v_inf - c / h          # linear in 1/h
    out = sv.extrapolate_resolution(h, values, order=1.0)
    assert out["v_inf"] == pytest.approx(v_inf, abs=1e-10)
    assert out["r2"] > 0.9999
    with pytest.raises(ValueError):
        sv.extrapolate_resolution(h[:2], values[:2])


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
def _verdict(phi_h, eta_h, nc_eta=None, nc_mu=1.0, convention="hydrodynamic",
             dilute_max=0.011):
    if nc_eta is None:   # consistent limit: the smallest fraction follows Einstein
        nc_eta = float(sv.relative_viscosity(0.01, "einstein"))
    dil_m = phi_h <= dilute_max
    dil = sv.intrinsic_viscosity(phi_h[dil_m], eta_h[dil_m],
                                 weights=np.ones(int(dil_m.sum())))
    fit = sv.fit_phi_max(phi_h, eta_h)
    qual = sv.fit_quality(phi_h, eta_h, phi_max=fit["phi_max"])
    div = sv.divergence_exponent(phi_h, eta_h, fit["phi_max"])
    nc = {"phi": 0.01, "eta_r": nc_eta, "mu_eff_over_mu0": nc_mu}
    return sv.evaluate_acceptance(dil, fit, qual, div, nc, volume_convention=convention)


def test_acceptance_report_passes_on_the_closed_form():
    # dilute window phi <= 0.01: the Krieger-Dougherty *chord* slope there is
    # 2.53 (1.2 % above 2.5), i.e. the curvature bias is well inside the 5 %
    # criterion -- this is the window declared in the campaign design.
    phi = np.array([0.005, 0.008, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45])
    eta = sv.relative_viscosity(phi, phi_max=0.63)
    rep = _verdict(phi, eta)
    assert rep["all_pass"], rep["criteria"]
    assert rep["n_failed"] == 0
    assert rep["volume_convention"] == "hydrodynamic"


def test_acceptance_report_fails_a_biased_slope_and_a_bad_control():
    phi = np.array([0.005, 0.008, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45])
    eta = sv.relative_viscosity(phi, phi_max=0.63)
    eta = 1.0 + (eta - 1.0) * 0.7            # 30 % low viscosity increment
    rep = _verdict(phi, eta, nc_eta=1.05, nc_mu=1.02, convention="mask")
    assert not rep["all_pass"]
    assert not rep["criteria"]["a_einstein_slope"]["pass"]
    assert not rep["criteria"]["d_limit_increment"]["pass"]
    assert rep["n_failed"] >= 2
    assert rep["volume_convention"] == "mask"


# ---------------------------------------------------------------------------
# plug-in closure object
# ---------------------------------------------------------------------------
def test_closure_object_derivative_matches_finite_difference():
    for model in ("einstein", "batchelor", "krieger_dougherty"):
        c = sv.SuspensionViscosityClosure(mu0=1.8e-3, phi_max=0.62, model=model)
        phi = 0.2
        h = 1e-6
        fd = (c.mu_eff(phi + h) - c.mu_eff(phi - h)) / (2 * h)
        assert c.d_mu_dphi(phi) == pytest.approx(fd, rel=1e-6)


def test_closure_object_roundtrip_and_mean():
    c = sv.SuspensionViscosityClosure(mu0=1.0, phi_max=0.63)
    eta = c.relative(0.3)
    assert c.phi_from_relative(eta) == pytest.approx(0.3, rel=1e-12)
    field = np.array([[0.0, 0.1], [0.2, 0.3]])
    assert c.mean_viscosity(field) == pytest.approx(np.mean(c.relative(field)))
    with pytest.raises(ValueError):
        sv.SuspensionViscosityClosure(model="herschel_bulkley")
    with pytest.raises(ValueError):
        sv.SuspensionViscosityClosure(phi_max=0.0)


# ---------------------------------------------------------------------------
# consistency with the pre-existing ice-suspension closures (no drift)
# ---------------------------------------------------------------------------
def test_matches_ice_suspension_rheology_closures():
    ice = pytest.importorskip("tensorlbm.ice_suspension_rheology.closures")
    phi = np.linspace(0.0, 0.5, 11)
    assert np.allclose(sv.relative_viscosity(phi, "einstein"),
                       ice.einstein_relative_viscosity(phi), rtol=0, atol=1e-14)
    assert np.allclose(sv.relative_viscosity(phi, "batchelor"),
                       ice.batchelor_relative_viscosity(phi), rtol=0, atol=1e-12)
    pm = 0.63
    assert np.allclose(sv.relative_viscosity(phi, phi_max=pm),
                       ice.krieger_dougherty_relative_viscosity(phi, pm),
                       rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# campaign artefact (only when the LBM calibration has been run)
# ---------------------------------------------------------------------------
def test_campaign_analysis_artefact_is_self_consistent():
    if not os.path.exists(ANALYSIS):
        pytest.skip("benchmarks/suspension/artifacts/analysis.json not generated here")
    with open(ANALYSIS) as fh:
        a = json.load(fh)
    for conv in ("mask", "hydrodynamic"):
        blk = a["conventions"][conv]
        n = blk["dilute"]["n_points"]
        assert n >= 3
        assert 0.0 <= blk["kd_fit"]["phi_max"] <= 1.0
        assert blk["kd_quality"]["r2"] <= 1.0 + 1e-9
        assert set(blk["acceptance"]["criteria"]) >= {
            "a_einstein_slope", "b_divergence_exponent", "c_full_range_r2"}
    radii = a["design"]["radii"]
    assert len(radii) >= 3 and radii == sorted(radii), "resolution ladder must be >= 3 increasing radii"

# ---------------------------------------------------------------------------
# campaign driver integrity (no LBM: pure case-table + aggregation logic)
# ---------------------------------------------------------------------------
def test_campaign_case_table_matches_the_predeclared_design():
    import importlib.util
    path = os.path.join(REPO, "benchmarks", "suspension", "run_lbm_calibration.py")
    spec = importlib.util.spec_from_file_location("run_lbm_calibration", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = mod.build_cases()
    tags = [c["tag"] for c in cases]
    assert len(tags) == len(set(tags)), "duplicate case tag"
    assert sum(c["mode"] == "drag" for c in cases) == len(mod.RADII)
    assert any(c["tag"] == "pure_control" for c in cases)
    dilute = [c for c in cases if c["tag"].startswith("dilute_")]
    dense = [c for c in cases if c["tag"].startswith("dense_")]
    assert len(dilute) == len(mod.RADII) * len(mod.DILUTE_PHI) * len(mod.SEEDS)
    assert len(dense) == len(mod.DENSE_PHI) * len(mod.SEEDS)
    # the primary dilute window must be covered by at least two fractions
    assert sum(p <= 0.011 for p in mod.DILUTE_PHI) >= 2


def test_fit_pipeline_recovers_the_slope_under_the_hydrodynamic_convention():
    """Feed the fit script a synthetic campaign and check the mechanism it claims."""
    import importlib.util
    path = os.path.join(REPO, "benchmarks", "suspension", "fit_calibration.py")
    spec = importlib.util.spec_from_file_location("fit_calibration", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    nu, g0, tau0, ratio = 0.1, 0.001739, 0.1 * 0.001739, 0.8
    cases = [dict(tag="pure_control", mode="shear", nu=nu, tau_corrected=tau0,
                  gamma_dot_measured=g0, gamma_dot_nominal=g0, eta_eff=nu,
                  tau_wall_mem=tau0, profile_r2=1.0, phi_lattice=0.0, radius=5.0,
                  phi_target=0.0, seed=1, n_particles=0, re_p=0.0)]
    a_eff = 5.0
    K = 1.2
    for r in (3.5, 5.0):
        u = 0.1 * nu / (2 * r)
        f = 6 * math.pi * nu * a_eff * ratio ** (1 / 3) * u * K
        cases.append(dict(tag=f"drag_r{r:g}", mode="drag", radius=r, nu=nu, u_stream=u,
                          a_eff=a_eff, drag=f, beta=0.2, haberman_sayre_K=K,
                          err_pct=5.0, err_pct_corrected=0.0, profile_r2=1.0))
    for r in (3.5, 5.0):
        for phi in (0.005, 0.01, 0.05, 0.2):
            for s in (1, 2):
                eta = float(sv.relative_viscosity(phi * ratio, phi_max=0.63))
                cases.append(dict(tag=f"c_{r}_{phi}_{s}", mode="shear", nu=nu,
                                  tau_corrected=tau0 * eta, gamma_dot_measured=g0,
                                  gamma_dot_nominal=g0, eta_eff=nu * eta,
                                  tau_wall_mem=tau0 * eta, profile_r2=0.9999,
                                  phi_lattice=phi, radius=r, phi_target=phi, seed=s,
                                  n_particles=2, re_p=0.01))
    ds = mod.build_dataset({"cases": cases, "design": {"radii": [3.5, 5.0]}})
    mask = mod.analyse_dataset(ds, "phi_mask", "mask")
    hyd = mod.analyse_dataset(ds, "phi_hydro", "hydrodynamic")
    # the two conventions must differ by exactly the volume ratio (same records,
    # same weights); the residual O(phi) curvature of the closed form is shared
    assert mask["dilute"]["slope"] == pytest.approx(hyd["dilute"]["slope"] * ratio,
                                                    rel=1e-6)
    # on the wide dilute window the chord slope is curvature-biased, so the
    # curvature-consistent estimator is the one that must recover Einstein
    assert hyd["dilute"]["curvature_fit"]["eta_intrinsic"] == pytest.approx(2.5, rel=1e-6)
    assert hyd["acceptance_curvature_estimator"]["criteria"]["a_einstein_slope"]["pass"]
    assert not mask["acceptance_curvature_estimator"]["criteria"]["a_einstein_slope"]["pass"]
    assert hyd["kd_fit"]["phi_max"] == pytest.approx(0.63, abs=5e-3)


def test_curvature_consistent_intrinsic_viscosity_removes_chord_bias():
    """A chord slope over a wide dilute window is biased high by the curvature."""
    phi = np.array([0.02, 0.05, 0.10])
    eta = sv.relative_viscosity(phi, phi_max=0.63)
    chord = sv.intrinsic_viscosity(phi, eta)["slope"]
    fit = sv.fit_intrinsic_viscosity(phi, eta)
    assert chord > 3.0, "chord slope over [0.02, 0.1] must be strongly biased high"
    assert fit["eta_intrinsic"] == pytest.approx(2.5, abs=1e-9)
    assert fit["pass_5pct"]
    assert fit["log_r2"] > 0.999

    # noisy data: the estimate stays honest and reports a standard error
    rng = np.random.default_rng(11)
    eta_n = eta * (1.0 + rng.normal(0.0, 0.01, eta.size))
    fit_n = sv.fit_intrinsic_viscosity(phi, eta_n)
    assert abs(fit_n["eta_intrinsic"] - 2.5) < 0.25
    assert fit_n["eta_intrinsic_se"] > 0

    with pytest.raises(ValueError):
        sv.fit_intrinsic_viscosity([0.02], [1.05])
    with pytest.raises(ValueError):
        sv.fit_intrinsic_viscosity([0.02, 0.7], [1.05, 9.0], phi_max=0.63)
