"""Tests for the WMLES Reynolds-stress peak diagnostic and grid-error split.

The diagnostic is the measurement layer of P372W2-N01, so its own errors must be
bounded before any peak offset is reported:

* the sub-sample peak finder must be accurate on a profile whose maximum is
  known in closed form (``y+ exp(-y+/A)`` peaks exactly at ``A``), including the
  case where the peak falls *between* samples;
* the DNS reference chain must reproduce the published MKM value
  (``-<u'v'>+ peak ~ 0.72 at y+ ~ 32`` for Re_tau = 180) and the exact channel
  balance endpoint identities;
* the Richardson/GCI routine must recover a synthetic order of accuracy;
* the correction map must interpolate/validate, and the exchange-location knob
  must move the wall-model sampling row it claims to move.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.wall_model_les.channel import (
    ChannelWMLESConfig,
    _blend_target,
    stability_diagnostics,
)
from tensorlbm.wall_model_les.peak_diagnostic import (
    analytic_stress_profile,
    find_stress_peak,
    fit_offset_correction,
    load_channel_means_table,
    reynolds_stress_from_mean_profile,
    richardson_gci,
    spacing_for_tolerance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DNS_DIR = REPO_ROOT / "benchmarks" / "pending" / "channel_turb_3d" / "dns_ref"
MKM180 = DNS_DIR / "mkm180.csv"
MKM590 = DNS_DIR / "mkm590.means"
MKM590_SHA256 = "878bc951794516b6136912f8f098cd6f2f96c4bbe7f21a8cea657ffb270d419d"


# ---------------------------------------------------------------------------
# peak finder
# ---------------------------------------------------------------------------
def test_analytic_peak_is_recovered_between_samples():
    y = np.linspace(0.5, 200.0, 60)
    exact = 32.0
    values = analytic_stress_profile(y, a_plus=exact)
    est = find_stress_peak(y, values)
    # the discretisation step is ~3.4 wall units; both estimators must land
    # inside a quarter of a step of the closed-form maximum
    assert abs(est.y_peak - exact) <= 0.25 * est.spacing
    assert abs(est.y_peak_parabola - exact) <= 0.5 * est.spacing
    assert est.n_samples == y.size
    assert est.bracket[0] <= est.y_peak <= est.bracket[1]
    assert est.uncertainty <= 0.5 * est.spacing


def test_analytic_peak_on_a_coarse_three_point_bracket():
    """Three samples only (the wall-modelled extreme): the peak lies in between."""
    y = np.array([7.5, 22.5, 37.5])
    exact = 22.5
    values = analytic_stress_profile(y, a_plus=exact)
    est = find_stress_peak(y, values)
    assert est.n_samples == 3
    assert abs(est.y_peak - exact) <= 0.35 * est.spacing


@pytest.mark.parametrize("method", ["cubic_spline", "parabola"])
def test_peak_methods_agree_on_a_well_resolved_profile(method):
    y = np.linspace(0.5, 600.0, 200)
    values = analytic_stress_profile(y, a_plus=45.0)
    est = find_stress_peak(y, values, method=method)
    assert abs(est.y_peak - 45.0) <= 0.1 * est.spacing


def test_peak_search_can_be_truncated_at_the_half_channel():
    """A symmetric two-peak profile: the truncation must pick the lower peak."""
    y = np.linspace(0.5, 590.0, 300)
    first = analytic_stress_profile(y, a_plus=45.0)
    mirrored = analytic_stress_profile(590.0 - y, a_plus=45.0)
    both = np.maximum(first, mirrored)
    est_half = find_stress_peak(y, both, y_plus_max=295.0)
    assert est_half.y_peak < 295.0
    assert abs(est_half.y_peak - 45.0) <= 0.5


def test_peak_at_the_edge_of_the_search_band_falls_back_to_the_sample():
    """A monotone profile inside the band has its maximum on the last sample."""
    y = np.array([10.0, 20.0, 30.0, 40.0])
    v = np.array([0.2, 0.4, 0.6, 0.8])
    est = find_stress_peak(y, v)
    assert est.y_peak == pytest.approx(40.0)
    assert est.uncertainty == 0.0


def test_non_finite_samples_are_dropped_and_short_profiles_rejected():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v = np.array([0.1, np.nan, 0.5, 0.4, 0.2])
    est = find_stress_peak(y, v)
    assert est.n_samples == 4
    with pytest.raises(ValueError):
        find_stress_peak(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        find_stress_peak(y, v, method="quintic")


# ---------------------------------------------------------------------------
# DNS reference chain
# ---------------------------------------------------------------------------
def test_mkm590_table_matches_its_provenance_sha256():
    assert hashlib.sha256(MKM590.read_bytes()).hexdigest() == MKM590_SHA256


def test_mkm_channel_csv_reproduces_the_published_peak():
    """MKM-180: -<u'v'>+ peaks at ~0.72 near y+ = 32 (published values)."""
    data = np.genfromtxt(MKM180, delimiter=",", names=True)
    y_plus = np.asarray(data["y_plus"], dtype=float)
    u_plus = np.asarray(data["u_plus"], dtype=float)
    re_tau = 180.0
    y_over_h = y_plus / y_plus[-1]
    du = np.gradient(u_plus, y_plus, edge_order=2)
    uv_plus = 1.0 - y_over_h - du
    peak = find_stress_peak(y_plus, uv_plus, y_plus_max=re_tau / 2)
    assert 25.0 <= peak.y_peak <= 40.0
    assert 0.65 <= peak.value_peak <= 0.80


def test_means_table_balance_identities_hold():
    table = load_channel_means_table(MKM590)
    stress = reynolds_stress_from_mean_profile(table)
    gates = stress["gates"]
    # exact identities: no Reynolds stress at the wall (viscous) nor at the centre
    assert gates["wall_value"] == pytest.approx(0.0, abs=1e-6)
    assert gates["centre_value"] == pytest.approx(0.0, abs=1e-6)
    assert gates["wall_ok"] and gates["centre_ok"]
    assert 570.0 <= table["re_tau"] <= 600.0
    peak = find_stress_peak(stress["y_plus"], stress["uv_plus"], y_plus_max=table["re_tau"] / 2)
    assert 30.0 <= peak.y_peak <= 60.0
    assert 0.75 <= peak.value_peak <= 1.0


# ---------------------------------------------------------------------------
# Richardson / GCI
# ---------------------------------------------------------------------------
def test_richardson_recovers_a_synthetic_second_order_case():
    exact, c = 2.0, 3.0
    h = [1.0, 2.0, 4.0]
    values = [exact + c * hi**2 for hi in h]
    gci = richardson_gci(values, h, reference=exact)
    assert gci.observed_order == pytest.approx(2.0, abs=0.02)
    assert gci.value_extrapolated == pytest.approx(exact, rel=1e-6)
    assert gci.converged and gci.monotone
    # the fine grid carries the discretisation error c * h1**2 = 3.0
    assert gci.grid_error_fine == pytest.approx(c * h[0] ** 2, rel=1e-6)
    assert gci.gci_fine == pytest.approx(1.25 * abs((values[0] - values[1]) / values[0]) / (2.0**2 - 1.0), rel=1e-9)


def test_richardson_handles_unequal_refinement_ratios():
    exact, c, p = 1.0, 0.5, 1.5
    h = [1.0, 1.6, 3.2]
    values = [exact + c * hi**p for hi in h]
    gci = richardson_gci(values, h)
    assert gci.observed_order == pytest.approx(p, abs=0.02)
    assert gci.value_extrapolated == pytest.approx(exact, rel=1e-5)


def test_richardson_flags_a_non_monotone_triple():
    gci = richardson_gci([1.0, 0.5, 0.8], [1.0, 2.0, 4.0])
    assert not gci.converged
    assert "non-monotone" in gci.note
    assert math.isfinite(gci.value_extrapolated)


# ---------------------------------------------------------------------------
# correction map
# ---------------------------------------------------------------------------
def test_offset_correction_selects_the_inverse_map_and_predicts_the_holdout():
    dy_plus = np.array([7.5, 15.0, 30.0])
    offsets = 0.02 + 0.9 / dy_plus
    corr = fit_offset_correction(dy_plus, offsets, holdout=1, kind="inverse")
    assert corr["kind"] == "inverse"
    assert corr["holdout"]["abs_error"] < 1e-9
    assert corr["max_abs_residual"] < 1e-9


def test_offset_correction_requires_an_explicit_family_with_a_holdout():
    with pytest.raises(ValueError):
        fit_offset_correction([7.5, 15.0, 30.0], [0.14, 0.08, 0.05], holdout=1)


def test_offset_correction_linear_case_and_tolerance_crossing():
    dy_plus = np.array([7.5, 15.0, 30.0])
    offsets = -0.1 + 0.01 * dy_plus
    corr = fit_offset_correction(dy_plus, offsets)
    assert corr["kind"] == "linear"
    assert corr["rmse"] < 1e-9
    # the fitted tolerance crossing must satisfy the map at the returned spacing
    tol = 0.2
    x = spacing_for_tolerance(corr, tol)
    a, b = corr["coefficients"]
    assert x is not None and a + b * x == pytest.approx(tol, abs=1e-6)


def test_offset_correction_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_offset_correction([7.5, 15.0], [0.1, 0.05])


# ---------------------------------------------------------------------------
# exchange location and stability diagnostics (channel module)
# ---------------------------------------------------------------------------
def test_exchange_offset_moves_the_sampling_row():
    base = ChannelWMLESConfig(ny=49, re_tau=590.0)
    moved = ChannelWMLESConfig(ny=49, re_tau=590.0, exchange_offset=2)
    assert moved.j_exchange == base.j_exchange + 1
    assert moved.y_of_node(moved.j_exchange) > base.y_of_node(base.j_exchange)
    with pytest.raises(ValueError):
        ChannelWMLESConfig(ny=49, exchange_offset=0)


def test_wall_relaxation_validation_and_blending():
    ChannelWMLESConfig(ny=49, wall_relaxation=1.0)
    ChannelWMLESConfig(ny=49, wall_relaxation=0.3)
    with pytest.raises(ValueError):
        ChannelWMLESConfig(ny=49, wall_relaxation=0.0)
    with pytest.raises(ValueError):
        ChannelWMLESConfig(ny=49, wall_relaxation=1.5)

    current = torch.full((2, 3), 0.4, dtype=torch.float64)
    zero = torch.zeros_like(current)
    # beta = 1 reproduces the explicit (un-relaxed) wall-model target
    ux1, uz1 = _blend_target(current, current, 0.1, 0.0, 1.0)
    assert torch.allclose(ux1, torch.full_like(current, 0.1))
    assert torch.allclose(uz1, zero)
    # beta = 0.5 halves the correction
    ux2, _ = _blend_target(current, current, 0.1, 0.0, 0.5)
    assert torch.allclose(ux2, torch.full_like(current, 0.25))
    # beta -> 0 leaves the current state untouched
    ux3, _ = _blend_target(current, current, 0.1, 0.0, 1e-12)
    assert torch.allclose(ux3, current, atol=1e-9)


def test_stability_diagnostics_report_the_relaxation_margin():
    """A uniform state has no strain: nu_sgs = 0 and the margin is 3 * nu."""
    ny, nx, nz = 9, 4, 4
    rho = torch.ones((nz, ny, nx), dtype=torch.float64)
    u = torch.zeros_like(rho)
    from tensorlbm.d3q19 import equilibrium3d

    f = equilibrium3d(rho, u, u, u)
    nu = 1.0e-3
    diag = stability_diagnostics(f, nu=nu, cs=0.1)
    assert diag["nu_sgs_max"] == pytest.approx(0.0, abs=1e-12)
    assert diag["tau_minus_half_min"] == pytest.approx(3.0 * nu, rel=1e-9)
    assert diag["max_speed"] == pytest.approx(0.0, abs=1e-12)
    assert diag["rho_min"] == pytest.approx(1.0, rel=1e-12)
    assert json.dumps(diag)  # JSON-serialisable for the run artifacts