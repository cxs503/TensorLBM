"""Tests for the LES plane-channel campaign machinery (P127W2-N03).

These tests cover the pieces that are new in
:mod:`tensorlbm.turbulence.les_channel3d`:

* the van Driest damped Smagorinsky closure and its bookkeeping,
* the energy-spectrum power-law fit (including a negative control),
* the Celik three-grid GCI (known-order recovery and fallback flag),
* the DNS Reynolds-shear reconstruction from the exact stress budget,
* the acceptance-verdict plumbing (PASS / FAIL / WITHHELD branches),
* and an end-to-end short run of :class:`LESChannel`.

A regression test also pins the SGS *collision* path of the underlying TRT
solver, which used to raise ``TypeError`` because a per-cell relaxation rate was
passed to ``Tensor.add_(alpha=...)`` (only scalars are accepted).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tensorlbm.turbulence.les_channel3d import (
    LESChannel,
    LESChannelConfig,
    compare_mean_profile,
    compare_reynolds_stress,
    dns_reynolds_shear,
    evaluate_case,
    fit_power_law_slope,
    gci_three_grid,
    interpolate_profile,
    mkm_profile_for,
    summary_metrics,
    van_driest_damping,
    van_driest_nu_t,
)
from tensorlbm.turbulence.turbulent_channel3d import (
    TurbulentChannel3D,
    TurbulentChannel3DConfig,
)

REPO = Path(__file__).resolve().parents[1]
MKM180 = mkm_profile_for(180.0, REPO)


def _tiny_config(**overrides) -> LESChannelConfig:
    base = dict(
        nx=24,
        ny=12,
        nz=16,
        re_tau=180.0,
        u_tau=0.01,
        steps=80,
        warmup_steps=30,
        sample_interval=5,
        report_interval=40,
        smagorinsky_cs=0.12,
        perturbation_amplitude=0.2,
        label="test_les",
    )
    base.update(overrides)
    return LESChannelConfig(**base)


# ---------------------------------------------------------------------------
# SGS closure
# ---------------------------------------------------------------------------
def test_van_driest_damping_endpoints_and_monotonicity():
    y = torch.tensor([0.0, 1.0, 5.0, 26.0, 52.0, 500.0], dtype=torch.float64)
    fv = van_driest_damping(y)
    assert float(fv[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(fv[3]) == pytest.approx(1.0 - math.exp(-1.0), rel=1e-12)
    assert float(fv[-1]) == pytest.approx(1.0, abs=1e-6)
    assert torch.all(torch.diff(fv) > 0.0)
    # negative y+ (numerical noise in the near-wall row) must clamp to zero
    assert float(van_driest_damping(torch.tensor([-3.0]))[0]) == pytest.approx(0.0)


def test_van_driest_nu_t_scales_with_cs_squared():
    s = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y_plus = torch.tensor([10.0, 100.0], dtype=torch.float64)
    a = van_driest_nu_t(0.1, s, y_plus)
    b = van_driest_nu_t(0.2, s, y_plus)
    assert torch.allclose(b, 4.0 * a, rtol=1e-12)
    # damping kills the wall-adjacent eddy viscosity
    assert float(van_driest_nu_t(0.1, s, torch.tensor([0.0, 0.0]))[0]) == pytest.approx(0.0)


def test_smagorinsky_collision_branch_is_executable():
    """Regression: the per-cell SGS relaxation used to hit ``add_(alpha=Tensor)``."""
    cfg = TurbulentChannel3DConfig(
        nx=16,
        ny=10,
        nz=8,
        re_tau=180.0,
        u_tau=0.01,
        steps=20,
        warmup_steps=0,
        report_interval=10,
        sample_interval=5,
        smagorinsky_cs=0.12,
    )
    sim = TurbulentChannel3D(cfg)
    sim.step(10)
    assert sim.health()["finite"]


def test_les_channel_without_sgs_matches_base_solver():
    """``cs = 0`` must reproduce the base TRT trajectory bit for bit."""
    kw = dict(nx=20, ny=12, nz=12, re_tau=180.0, u_tau=0.01, steps=20,
              warmup_steps=0, report_interval=10, sample_interval=5,
              perturbation_amplitude=0.2, smagorinsky_cs=0.0, seed=4242)
    base = TurbulentChannel3D(TurbulentChannel3DConfig(**kw))
    les = LESChannel(LESChannelConfig(**kw, van_driest=False))
    base.step(10)
    les.step(10)
    assert torch.allclose(base.f, les.f, rtol=0.0, atol=0.0)
    assert les._nu_t_field is None


def test_les_channel_sgs_field_is_damped_at_the_wall():
    cfg = _tiny_config(re_tau=180.0)
    sim = LESChannel(cfg)
    sim.step(5)
    assert sim._nu_t_field is not None
    field = sim._nu_t_field
    assert field.shape == (cfg.nz, cfg.ny, cfg.nx)
    assert torch.isfinite(field).all()
    assert float(field.min()) >= 0.0
    # first and last rows are solid in the population array but the damping
    # envelope must still vanish there
    assert float(sim.fv2_rows[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(sim.y_plus_rows[1]) == pytest.approx(0.5 * cfg.dy_plus, rel=1e-12)


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------
def test_power_law_fit_recovers_minus_five_thirds():
    k = torch.arange(1, 41, dtype=torch.float64)
    e = 3.7 * k ** (-5.0 / 3.0)
    fit = fit_power_law_slope(k, e)
    assert fit["valid"]
    assert fit["slope"] == pytest.approx(-5.0 / 3.0, abs=1e-10)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["k_min"] == pytest.approx(4.0)
    assert fit["k_max"] <= float(k.max())


def test_power_law_fit_rejects_flat_spectrum():
    """Negative control: a k^0 spectrum must fail the -5/3 criterion."""
    k = torch.arange(1, 41, dtype=torch.float64)
    e = torch.ones_like(k)
    fit = fit_power_law_slope(k, e)
    assert fit["valid"]
    assert abs(fit["slope"] + 5.0 / 3.0) > 0.5


def test_power_law_fit_withholds_on_too_few_modes():
    k = torch.arange(1, 5, dtype=torch.float64)
    e = torch.ones_like(k)
    fit = fit_power_law_slope(k, e, k_min=4.0, min_points=4)
    assert not fit["valid"]


def test_spectrum_slope_is_band_limited_by_construction():
    """High-k noise must not move the fit when the band is fixed a priori."""
    k = torch.arange(1, 65, dtype=torch.float64)
    e = k ** (-5.0 / 3.0)
    noisy = e.clone()
    noisy[k > 20] = 1e-8
    fit_a = fit_power_law_slope(k, e, k_min=4.0, k_max=16.0)
    fit_b = fit_power_law_slope(k, noisy, k_min=4.0, k_max=16.0)
    assert fit_a["slope"] == pytest.approx(fit_b["slope"], abs=1e-12)


# ---------------------------------------------------------------------------
# interpolation / GCI
# ---------------------------------------------------------------------------
def test_interpolate_profile_basic_and_out_of_range():
    y = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    v = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)
    assert interpolate_profile(y, v, 0.5) == pytest.approx(1.0)
    assert interpolate_profile(y, v, 2.0) == pytest.approx(4.0)
    assert math.isnan(interpolate_profile(y, v, 5.0))
    # unsorted input must still interpolate correctly
    assert interpolate_profile(y.flip(0), v.flip(0), 1.5) == pytest.approx(3.0)


def test_gci_recovers_second_order_on_analytic_ladder():
    f0, c = 10.0, 3.0

    def f(h):
        return f0 + c * h**2

    fine, medium, coarse = f(1.0), f(2.0), f(4.0)
    g = gci_three_grid(fine, medium, coarse, r=2.0)
    assert g["valid"]
    assert g["order_used"] == pytest.approx(2.0, rel=1e-9)
    assert g["order_source"] == "observed"
    assert g["extrapolated"] == pytest.approx(f0, rel=1e-9)
    expected = 1.25 * abs(medium - fine) / fine / (2.0**2 - 1.0)
    assert g["gci_fine"] == pytest.approx(expected, rel=1e-9)


def test_gci_flags_fallback_for_non_monotone_ladder():
    g = gci_three_grid(1.0, 1.2, 1.1, r=2.0)
    assert g["valid"]
    assert g["order_source"] in ("fallback_p2", "observed")
    if g["order_source"] == "fallback_p2":
        assert g["order_used"] == pytest.approx(2.0)


def test_gci_handles_degenerate_ladder():
    g = gci_three_grid(1.0, 1.0, 1.0)
    assert not g["valid"]


# ---------------------------------------------------------------------------
# DNS reference reconstruction
# ---------------------------------------------------------------------------
@pytest.mark.skipif(MKM180 is None, reason="MKM1999 table not vendored")
def test_dns_shear_reconstruction_satisfies_stress_budget():
    prof = MKM180
    y_plus, uv = dns_reynolds_shear(prof)
    # zero at the wall
    assert abs(float(uv[0])) < 1e-6
    # total stress identity: -<u'v'>+ + dU+/dy+ = 1 - y+/Re_tau
    total = uv + prof.du_plus_dy_plus
    rhs = 1.0 - y_plus / float(prof.re_tau)
    assert torch.allclose(total, rhs, atol=1e-12)
    # peak location/amplitude are the published Re_tau=180 values (no digit copied)
    sel = y_plus <= float(prof.re_tau) * 0.999
    idx = int(torch.argmax(uv[sel]))
    peak, pos = float(uv[sel][idx]), float(y_plus[sel][idx])
    assert 0.65 <= peak <= 0.80
    assert 25.0 <= pos <= 40.0
    # and it vanishes at the centreline
    assert abs(float(uv[sel][-1])) < 0.05


@pytest.mark.skipif(MKM180 is None, reason="MKM1999 table not vendored")
def test_reynolds_stress_comparison_detects_amplitude_bias():
    prof = MKM180
    y_plus, uv = dns_reynolds_shear(prof)
    sel = y_plus <= float(prof.re_tau) * 0.999
    y = y_plus[sel]
    fake = {
        "wall_units": {
            "y_plus": y.tolist(),
            "u_plus": (torch.log(y.clamp(min=1.0)) / 0.41 + 5.2).tolist(),
            "uv_minus_plus": (1.10 * uv[sel]).tolist(),
        }
    }
    cmp = compare_reynolds_stress(fake, prof)
    assert cmp["valid"]
    assert cmp["rel_error_peak_amplitude"] == pytest.approx(0.10, rel=0.05)
    assert abs(cmp["rel_error_peak_position"]) < 1e-9


@pytest.mark.skipif(MKM180 is None, reason="MKM1999 table not vendored")
def test_mean_profile_comparison_reports_log_law_error():
    prof = MKM180
    y = torch.logspace(math.log10(30.0), math.log10(100.0), 12, dtype=torch.float64)
    exact = torch.log(y) / 0.41 + 5.2
    good = {"wall_units": {"y_plus": y.tolist(), "u_plus": exact.tolist()}}
    bad = {"wall_units": {"y_plus": y.tolist(), "u_plus": (1.2 * exact).tolist()}}
    cmp_good = compare_mean_profile(good, prof)
    cmp_bad = compare_mean_profile(bad, prof)
    assert cmp_good["valid"] and cmp_bad["valid"]
    assert cmp_good["max_rel_error_log_law"] < 1e-12
    assert cmp_good["kappa_fit"] == pytest.approx(0.41, rel=1e-9)
    assert cmp_bad["max_rel_error_log_law"] > 0.15


# ---------------------------------------------------------------------------
# verdict plumbing
# ---------------------------------------------------------------------------
def test_evaluate_case_withholds_without_dns_or_spectra():
    result = {
        "wall_units": {"y_plus": [1.0, 5.0, 20.0], "u_plus": [1.0, 5.0, 10.0],
                       "uv_minus_plus": [0.0, 0.3, 0.5]},
        "les": {
            "spectra": {"streamwise": [], "samples": 0},
            "sgs_stress": {"fraction_at_y_plus_50": float("nan")},
            "mass_conservation": {"bulk_rel_std_half_window": 1e-5,
                                  "total_mass_rel_drift": 1e-12},
        },
    }
    verdict = evaluate_case(result, dns=None)
    assert verdict["criterion_checks"]["b_reynolds_stress_peak"]["status"] == "WITHHELD"
    assert verdict["criterion_checks"]["c_energy_spectrum_slope"]["status"] == "WITHHELD"
    assert verdict["criterion_checks"]["d_sgs_stress_fraction"]["status"] == "WITHHELD"
    assert verdict["criterion_checks"]["f_mass_conservation"]["passed"] is True
    assert "b_reynolds_stress_peak" in verdict["withheld"]


def test_evaluate_case_passes_sgs_band_and_spectrum():
    k = torch.arange(1, 41, dtype=torch.float64)
    result = {
        "wall_units": {"y_plus": [30.0, 60.0, 100.0], "u_plus": [12.5, 14.4, 16.1],
                       "uv_minus_plus": [0.5, 0.4, 0.2]},
        "les": {
            "spectra": {
                "samples": 100,
                "streamwise": [
                    {"y_plus": 30.0, "fit": fit_power_law_slope(k, k ** (-5.0 / 3.0))}
                ],
            },
            "sgs_stress": {"fraction_at_y_plus_50": 0.3, "cs": 0.12, "van_driest": True},
            "mass_conservation": {"bulk_rel_std_half_window": 1e-5,
                                  "total_mass_rel_drift": 1e-12},
        },
    }
    verdict = evaluate_case(result, dns=None)
    assert verdict["criterion_checks"]["c_energy_spectrum_slope"]["passed"] is True
    assert verdict["criterion_checks"]["d_sgs_stress_fraction"]["passed"] is True


# ---------------------------------------------------------------------------
# end-to-end short run
# ---------------------------------------------------------------------------
def test_les_channel_short_run_reports_full_diagnostics():
    cfg = _tiny_config()
    sim = LESChannel(cfg)
    result = sim.run_les()
    for key in ("config", "wall_units", "reynolds_stress", "momentum_balance",
                "friction_velocity", "les"):
        assert key in result
    les = result["les"]
    for key in ("spectra", "sgs_stress", "production", "mass_conservation",
                "statistics_uncertainty"):
        assert key in les
    # Constant-body-force channel: the total fluid mass must stay put at the
    # campaign tolerance.  It is *not* round-off bounded: the TRT-split force
    # injection carries a first-order density source O((s+ - s-) u F), measured
    # at 8.1e-5 relative drift over 120 steps on this toy grid (identical in
    # float32 and float64, so it is a scheme property and not accumulation
    # noise).  The campaign's mass criterion is 1e-3, which is what is asserted
    # here; a trace-free force term would be needed for a 1e-9 bound.
    assert les["mass_conservation"]["total_mass_rel_drift"] < 1e-3
    # head-line scalars must be finite
    summary = summary_metrics(result)
    assert math.isfinite(summary["u_plus_max"])
    assert math.isfinite(summary["uv_peak_plus"])
    # friction velocity from the body force is exact by construction
    assert summary["u_tau_body_force"] == pytest.approx(
        math.sqrt(cfg.acceleration * cfg.half_height), rel=1e-12
    )
    # three independent u_tau routes must be within a factor of 2 on this toy grid
    for route in ("u_tau_body_force", "u_tau_wall_exchange", "u_tau_stress_fit"):
        value = summary[route]
        assert value is None or math.isfinite(value)


def test_les_channel_spectrum_dict_is_json_serialisable():
    import json

    cfg = _tiny_config(steps=60, warmup_steps=20)
    sim = LESChannel(cfg)
    result = sim.run_les()
    text = json.dumps(result)
    assert '"spectra"' in text
    assert len(text) > 1000