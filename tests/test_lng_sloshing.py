"""Tests for the LNG-tank partial-fill sloshing module (campaign node IN143-N07).

Analytic anchors, the free-surface tank driver, the signal estimators and the
campaign judges/report path.  The LBM runs use a deliberately small tank so the
whole file stays within a couple of minutes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.free_surface_lbm_2d import (
    SOLID,
    count_direct_liquid_gas_links_2d,
)
from tensorlbm.lng_sloshing import (
    TankConfig,
    abramson_angular_frequency,
    abramson_hydrostatic_wave_speed,
    abramson_mode_ratio,
    abramson_mode_shape,
    abramson_natural_frequency,
    abramson_wave_number,
    build_tank,
    column_liquid_height,
    decay_damping,
    fill_ratio_to_depth,
    harmonic_amplitudes,
    linear_forced_modal_amplitude,
    linear_wall_pressure_amplitude,
    modal_projection,
    pressure_peak,
    refined_frequency,
    simulate_tank,
    stagnation_impact_pressure,
    wall_pressure_profile,
)
from tensorlbm.lng_sloshing import campaign as C

# Small-grid LBM steps are overhead bound; on a busy many-core host the default
# torch thread pool makes them an order of magnitude slower, so cap it here.
torch.set_num_threads(min(4, torch.get_num_threads()))

SMALL = TankConfig(length=32, depth=16.0, headspace=0.35, g=2.0e-3, tau=0.6)


# ---------------------------------------------------------------------------
# 1-6: Abramson (1966) analytic anchors
# ---------------------------------------------------------------------------


def test_abramson_wave_number_and_mode_shape():
    L = 10.0
    assert abramson_wave_number(1, L) == pytest.approx(math.pi / L)
    assert abramson_wave_number(2, L) == pytest.approx(2 * math.pi / L)
    # odd/even mode shapes at the walls
    assert abramson_mode_shape(1, 0.0, L) == pytest.approx(1.0)
    assert abramson_mode_shape(1, L, L) == pytest.approx(-1.0)
    assert abramson_mode_shape(2, L / 2.0, L) == pytest.approx(-1.0)
    assert abramson_mode_shape(1, L / 2.0, L) == pytest.approx(0.0, abs=1e-12)


def test_abramson_shallow_water_limit():
    """k h -> 0: omega -> k sqrt(g h), i.e. f1 -> sqrt(g h) / (2 L)."""
    L, h, g = 100.0, 0.05, 9.81
    f1 = abramson_natural_frequency(1, L, h, g)
    limit = abramson_hydrostatic_wave_speed(h, g) / (2.0 * L)
    assert f1 == pytest.approx(limit, rel=2e-3)


def test_abramson_deep_water_limit():
    """k h -> inf: omega -> sqrt(g k)."""
    L, g = 1.0, 9.81
    h = 200.0
    w1 = abramson_angular_frequency(1, L, h, g)
    assert w1 == pytest.approx(math.sqrt(g * math.pi / L), rel=1e-4)


def test_abramson_matches_legacy_faltinsen_helper():
    """Same closed form as the pre-existing two-phase module's helper."""
    from tensorlbm import faltinsen_natural_frequency

    for L, h, g in [(100, 30, 9.81), (80, 80, 1.25e-3), (64, 12, 1e-2)]:
        legacy = faltinsen_natural_frequency(L, h, g)  # returns omega_n
        assert abramson_angular_frequency(1, L, h, g) == pytest.approx(legacy, rel=1e-12)


def test_abramson_mode_ratio_tends_to_two_in_deep_water():
    L = 1.0
    w1 = abramson_angular_frequency(1, L, 50.0)
    ratio = abramson_mode_ratio(w1, 50.0, L)
    assert ratio == pytest.approx(math.sqrt(2.0), rel=1e-3)
    # shallow water: omega ~ k sqrt(g h), so the ratio approaches 2
    w1s = abramson_angular_frequency(1, L, 1e-3)
    assert abramson_mode_ratio(w1s, 1e-3, L) == pytest.approx(2.0, rel=1e-2)


def test_abramson_rejects_bad_input():
    with pytest.raises(ValueError):
        abramson_wave_number(0, 1.0)
    with pytest.raises(ValueError):
        abramson_wave_number(1, 0.0)
    with pytest.raises(ValueError):
        fill_ratio_to_depth(0.0, 80)
    with pytest.raises(ValueError):
        fill_ratio_to_depth(2.0, 80)


# ---------------------------------------------------------------------------
# 7-13: linear forced-response / pressure theory
# ---------------------------------------------------------------------------


def test_linear_modal_amplitude_static_limit():
    """omega -> 0: the modal amplitude is the projection of the rigid tilt."""
    L, h, g, x0 = 1.0, 0.4, 9.81, 1e-3
    f1 = abramson_natural_frequency(1, L, h, g)
    omega = 2.0 * math.pi * f1 * 1e-3
    eta = linear_forced_modal_amplitude(1, L, h, x0, omega, g_g := g, zeta=0.0)
    a_x = x0 * omega**2
    assert eta == pytest.approx(a_x / g * 4.0 / math.pi**2, rel=2e-3)
    assert g_g == g


def test_linear_modal_amplitude_even_mode_not_excited():
    L, h, g = 1.0, 0.5, 9.81
    omega = 2.0 * math.pi * abramson_natural_frequency(1, L, h, g)
    assert linear_forced_modal_amplitude(2, L, h, 1e-2, omega, g) == 0.0


def test_linear_modal_amplitude_resonance_scales_with_damping():
    """At omega = omega_1 the response is X0 * beta_1 / (2 zeta)."""
    L, h, g, x0, zeta = 1.0, 0.5, 9.81, 1e-3, 0.05
    w1 = abramson_angular_frequency(1, L, h, g)
    eta = linear_forced_modal_amplitude(1, L, h, x0, w1, g, zeta=zeta)
    beta1 = 4.0 * math.tanh(math.pi * h / L) / math.pi
    assert eta == pytest.approx(x0 * beta1 / (2.0 * zeta), rel=1e-12)
    # off-resonance is smaller than the resonant value
    assert linear_forced_modal_amplitude(1, L, h, x0, 0.5 * w1, g, zeta=zeta) < eta


def test_linear_wall_pressure_inertial_tilt_term():
    """omega -> 0 with fixed acceleration: the wall head is the static tilt."""
    L, h, g, x0 = 80.0, 40.0, 5e-4, 0.4
    w1 = abramson_angular_frequency(1, L, h, g)
    omega = 1e-3 * w1                      # quasi-static
    eta1 = linear_forced_modal_amplitude(1, L, h, x0, omega, g)
    head = linear_wall_pressure_amplitude(0.0, L, h, eta1, omega, g, sway_amplitude=x0)
    a_x = x0 * omega**2                    # the horizontal acceleration amplitude
    assert head == pytest.approx(a_x * L / (2.0 * g), rel=1e-4)
    # the tilt term is depth independent, the modal term is not
    t0 = abs(x0) * omega**2 * L / (2 * g)
    for y in (0.0, -h):
        assert linear_wall_pressure_amplitude(y, L, h, 0.0, omega, g, sway_amplitude=x0) == pytest.approx(t0, rel=1e-12)


def test_linear_wall_pressure_decays_with_depth():
    L, h, g, omega = 1.0, 0.5, 9.81, 0.9 * abramson_angular_frequency(1, 1.0, 0.5, 9.81)
    eta = 0.05
    p0 = linear_wall_pressure_amplitude(0.0, L, h, eta, omega, g)
    p_bottom = linear_wall_pressure_amplitude(-h, L, h, eta, omega, g)
    assert p0 > p_bottom > 0.0
    k = math.pi / L
    # the depth profile is cosh(k (y+h)) / cosh(k h)
    assert p_bottom / p0 == pytest.approx(1.0 / math.cosh(k * h), rel=1e-12)


def test_linear_wall_pressure_scales_quadratically_with_frequency():
    L, h, g, eta = 1.0, 0.5, 9.81, 0.05
    w1 = abramson_angular_frequency(1, L, h, g)
    p1 = linear_wall_pressure_amplitude(0.0, L, h, eta, 0.5 * w1, g)
    p2 = linear_wall_pressure_amplitude(0.0, L, h, eta, w1, g)
    assert p2 / p1 == pytest.approx(4.0, rel=1e-12)


def test_linear_wall_pressure_phase_reverses_above_resonance():
    """Above resonance the modal term is anti-phase with the inertial tilt."""
    L, h, g, x0, zeta = 80.0, 40.0, 5e-4, 0.8, 0.058
    w1 = abramson_angular_frequency(1, L, h, g)
    w = 1.2 * w1
    eta1 = linear_forced_modal_amplitude(1, L, h, x0, w, g, zeta=zeta)
    head = linear_wall_pressure_amplitude(-4.0, L, h, eta1, w, g, sway_amplitude=x0, zeta=zeta)
    modal = linear_wall_pressure_amplitude(-4.0, L, h, eta1, w, g, sway_amplitude=0.0, zeta=zeta)
    tilt = linear_wall_pressure_amplitude(-4.0, L, h, 0.0, w, g, sway_amplitude=x0, zeta=zeta)
    # in-phase sum would be modal + tilt; the true value must be smaller here
    assert head < modal + tilt - 1e-6
    assert head > abs(modal - tilt) - 1e-6
    # below resonance the two terms add (weakly) in phase
    w02 = 0.6 * w1
    e02 = linear_forced_modal_amplitude(1, L, h, x0, w02, g, zeta=zeta)
    h02 = linear_wall_pressure_amplitude(-4.0, L, h, e02, w02, g, sway_amplitude=x0, zeta=zeta)
    m02 = linear_wall_pressure_amplitude(-4.0, L, h, e02, w02, g, sway_amplitude=0.0, zeta=zeta)
    t02 = linear_wall_pressure_amplitude(-4.0, L, h, 0.0, w02, g, sway_amplitude=x0, zeta=zeta)
    assert h02 > max(m02, t02)
    assert h02 <= m02 + t02 + 1e-9


def test_stagnation_pressure_bound():
    assert stagnation_impact_pressure(2.0, rho=1.0) == pytest.approx(2.0)
    assert stagnation_impact_pressure(3.0, rho=0.5, cp=2.0) == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# 14-19: tank geometry, ledger, driver
# ---------------------------------------------------------------------------


def test_tank_initial_state_is_consistent():
    st = build_tank(SMALL)
    cfg = SMALL
    content = column_liquid_height(st)
    interior = content[1:-1]
    assert interior.numel() == cfg.length
    # the column inventory equals the still-water depth (within the interface layer)
    assert float(interior.mean()) == pytest.approx(cfg.depth, abs=0.02)
    # no direct liquid-gas link (the Körner envelope guarantee)
    assert count_direct_liquid_gas_links_2d(st.flags) == 0
    assert int((st.flags[0, :] == SOLID).sum()) == cfg.nx  # SOLID bottom row
    assert int((st.flags[:, 0] == SOLID).sum()) == cfg.ny
    assert float(st.inventory0) > 0.0


def test_fill_ratio_maps_to_depth():
    assert fill_ratio_to_depth(0.5, 80) == 80 * 0.5
    cfg = TankConfig(length=80, depth=fill_ratio_to_depth(0.25, 80), g=5e-4)
    assert cfg.fill_ratio == pytest.approx(0.25)


def test_modal_projection_of_pure_first_mode():
    cfg = SMALL
    n = cfg.length
    idx = np.arange(1, n + 1) - 0.5
    k = math.pi / n
    import torch

    eta = torch.tensor(0.75 * np.cos(k * idx), dtype=torch.float64)
    assert modal_projection(eta, 1) == pytest.approx(0.75, rel=1e-10)
    # a pure second mode has (numerically) zero first-mode projection
    eta2 = torch.tensor(0.4 * np.cos(2 * k * idx), dtype=torch.float64)
    assert abs(modal_projection(eta2, 1)) < 1e-10
    assert modal_projection(eta2, 2) == pytest.approx(0.4, rel=1e-10)


def test_mass_is_conserved_in_a_short_run():
    out = simulate_tank(SMALL, 220, tilt_amplitude=0.6, record_every=20, n_modes=1)
    assert abs(out["mass_drift"]) < 2e-3
    inv = out["inventory"].numpy()
    assert np.all(np.isfinite(inv))
    assert out["eta"].shape == (len(out["t"]), SMALL.length)


def test_hydrostatic_preconditioning_leaves_no_wall_pressure():
    """A resting tank must show (almost) zero dynamic wall pressure."""
    import torch

    st = build_tank(SMALL)
    from tensorlbm.free_surface_lbm_2d import koerner_step_2d

    for _ in range(120):
        st.f, st.mass, st.flags = koerner_step_2d(
            st.f, st.mass, st.flags, st.solid, tau=SMALL.tau, gy=-SMALL.g
        )
    prof = wall_pressure_profile(st)
    dy = float(torch.nan_to_num(prof).abs().max())
    hydro = SMALL.g * SMALL.depth  # rho g h in lattice units
    assert dy < 0.10 * hydro


def test_free_decay_frequency_tracks_abramson_at_coarse_resolution():
    """Cheapest end-to-end frequency check (loose tolerance, small tank)."""
    f1 = abramson_natural_frequency(1, SMALL.length, SMALL.depth, SMALL.g)
    period = 1.0 / f1
    out = simulate_tank(
        SMALL, int(3.0 * period), tilt_amplitude=0.6, record_every=5, n_modes=1
    )
    est = refined_frequency(out["t"].numpy(), out["x_centroid"].numpy())
    assert est["f"] / f1 == pytest.approx(1.0, abs=0.10)
    assert est["snr"] > 1.0


def test_forced_run_produces_oscillating_response():
    f1 = abramson_natural_frequency(1, SMALL.length, SMALL.depth, SMALL.g)
    omega = 2.0 * math.pi * 0.9 * f1
    out = simulate_tank(
        SMALL,
        int(4.0 / f1),
        sway_amplitude=0.05 * SMALL.length,
        omega=omega,
        record_every=5,
        n_modes=1,
    )
    xcm = out["x_centroid"].numpy() - out["x_centroid"].numpy().mean()
    assert np.ptp(xcm) > 1e-3
    est = refined_frequency(out["t"].numpy(), xcm)
    assert est["f"] == pytest.approx(omega / (2 * math.pi), rel=0.05)


# ---------------------------------------------------------------------------
# 20-24: signal estimators
# ---------------------------------------------------------------------------


def test_refined_frequency_recovers_damped_sinusoid_with_drift():
    t = np.arange(0.0, 4000.0, 2.0)
    f0, zeta = 0.0042, 0.03
    rng = np.random.default_rng(1)
    y = (
        0.3 * np.exp(-zeta * 2 * math.pi * f0 * t) * np.cos(2 * math.pi * f0 * t + 0.7)
        + 0.02 * (t / t.max()) ** 2
        + 0.002 * rng.standard_normal(t.size)
    )
    est = refined_frequency(t, y)
    assert est["f"] == pytest.approx(f0, rel=5e-3)
    assert est["snr"] > 1.0


def test_decay_damping_recovers_zeta():
    t = np.arange(0.0, 8000.0, 2.0)
    f0, zeta = 0.003, 0.06
    y = 0.5 * np.exp(-zeta * 2 * math.pi * f0 * t) * np.cos(2 * math.pi * f0 * t)
    dm = decay_damping(t, y, f0)
    assert dm["zeta"] == pytest.approx(zeta, rel=0.10)
    assert dm["r2"] > 0.95
    # and with a slow quadratic drift superimposed on the same decay
    dm2 = decay_damping(t, y + 0.02 * (t / t.max()) ** 2, f0)
    assert dm2["zeta"] == pytest.approx(zeta, rel=0.15)


def test_harmonic_amplitudes_recovers_two_harmonics():
    t = np.arange(0.0, 3000.0, 1.0)
    f1 = 0.01
    y = 1.0 * np.cos(2 * math.pi * f1 * t) + 0.12 * np.cos(4 * math.pi * f1 * t + 0.3) + 0.5
    h = harmonic_amplitudes(t, y, f1, n_harm=3)
    assert h["A1"] == pytest.approx(1.0, rel=1e-6)
    assert h["A2"] == pytest.approx(0.12, rel=1e-6)
    assert h["ratios"]["A2/A1"] == pytest.approx(0.12, rel=1e-6)
    assert h["dc"] == pytest.approx(0.5, rel=1e-6)


def test_pressure_peak_and_window():
    t = np.arange(0.0, 100.0, 1.0)
    p = np.zeros_like(t)
    p[60] = 3.0
    p[70] = -4.0
    pk = pressure_peak(p, t)
    assert pk["p_max"] == pytest.approx(4.0)
    assert pk["t_peak"] == pytest.approx(70.0)
    pk2 = pressure_peak(p, t, t_start=65.0)
    assert pk2["p_max"] == pytest.approx(4.0)


def test_reconstruction_error_metrics():
    from tensorlbm.lng_sloshing.analysis import reconstruction_error

    ref = np.sin(np.linspace(0, 6, 200))
    err = reconstruction_error(ref * 1.1, ref)
    assert err["rel_max"] == pytest.approx(0.1, rel=1e-6)
    assert err["rel_l2"] == pytest.approx(0.1, rel=1e-6)


# ---------------------------------------------------------------------------
# 25-28: campaign judges and report path
# ---------------------------------------------------------------------------


def test_case_list_is_complete_and_unique():
    spec = C.CampaignSpec()
    cases = C.case_list(spec)
    n_conv = len([L for L in spec.conv_lengths if L != spec.length])
    assert len(cases) == len(spec.fill_ratios) + len(spec.pressure_ratios) + n_conv + 1
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == len(ids)
    assert sum(1 for c in cases if c["kind"] == "decay") == 10 + n_conv
    assert sum(1 for c in cases if c["kind"] == "forced") == len(spec.pressure_ratios)
    assert sum(1 for c in cases if c["kind"] == "waveform") == 1
    assert sum(1 for c in cases if c["kind"] == "decay" and c.get("length")) == n_conv


def test_frequency_judge_pass_and_fail():
    spec = C.CampaignSpec()
    good = []
    for fr in spec.fill_ratios:
        depth = fr * spec.length
        f1 = abramson_natural_frequency(1, spec.length, depth, spec.g)
        good.append(
            {
                "kind": "decay",
                "fill_ratio": fr,
                "depth": depth,
                "length": spec.length,
                "g": spec.g,
                "f1_abramson": f1,
                "f1_measured": f1 * 1.02,
                "rel_error": 0.02,
                "zeta": 0.05,
                "three_g_h": 3 * spec.g * depth,
            }
        )
    judge = C._judge_frequency(good, 0.05)
    assert judge["passed"] and judge["n_within_tolerance"] == 10
    bad = [dict(row) for row in good]
    bad[3] = {**bad[3], "f1_measured": bad[3]["f1_abramson"] * 1.11, "rel_error": 0.11}
    judge_bad = C._judge_frequency(bad, 0.05)
    assert not judge_bad["passed"]
    assert judge_bad["n_within_tolerance"] == 9


def test_pressure_and_waveform_judges():
    def gauge(peak, lin, theo, pred, yrel):
        return {
            "y_relative": yrel,
            "y_index": 40,
            "P_star_max": peak,
            "P_star_linear": lin,
            "P_star_theory": theo,
            "P_star_theory_pred": pred,
            "n_samples": 1000,
        }

    forced = [
        {
            "kind": "forced",
            "id": f"forced_r{r:.2f}",
            "ratio": r,
            "fill_ratio": 0.5,
            "depth": 40.0,
            "gauge_rows": [
                gauge(0.35, 0.20, 0.30, 0.30, 0.0),
                gauge(0.30 * (1 + 0.05 * (r - 1)), 0.29, 0.30, 0.30, -4.0),
                gauge(0.25, 0.24, 0.25, 0.25, -12.0),
            ],
            "eta1_measured": 2.0,
            "eta1_theory": 2.1,
            "eta2_over_eta1": 0.05,
            "steady_state": True,
            "p_max_over_half_rho_v2": 1.4,
        }
        for r in (0.8, 1.1, 1.2)
    ]
    pj = C._judge_pressure(forced, 0.20)
    assert pj["n_ratio_points"] == 3
    assert pj["experimental_comparison"] == "WITHHELD_NO_DATASET"
    assert pj["n_steady_points"] == 3
    assert pj["passed_vs_linear_theory"]
    wave = [
        {
            "kind": "waveform",
            "ratio": 0.8,
            "fill_ratio": 0.5,
            "waveform_rel_l2": 0.08,
            "waveform_rel_max": 0.2,
            "waveform_amp_fit": 2.0,
            "eta1_theory": 2.05,
        }
    ]
    wj = C._judge_waveform(wave, 0.15)
    assert wj["passed"]
    wave[0]["waveform_rel_l2"] = 0.30
    assert not C._judge_waveform(wave, 0.15)["passed"]


def test_summarize_writes_report_and_three_figures(tmp_path: Path):
    """End-to-end report path on synthetic case JSONs (no LBM run needed)."""
    spec = C.CampaignSpec()
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    for fr in spec.fill_ratios:
        depth = fr * spec.length
        f1 = abramson_natural_frequency(1, spec.length, depth, spec.g)
        (case_dir / f"decay_f{fr:.2f}.json").write_text(
            json.dumps(
                {
                    "id": f"decay_f{fr:.2f}",
                    "kind": "decay",
                    "fill_ratio": fr,
                    "depth": depth,
                    "length": spec.length,
                    "g": spec.g,
                    "f1_abramson": f1,
                    "f1_measured": f1 * 1.01,
                    "rel_error": 0.01,
                    "zeta": 0.05,
                    "three_g_h": 3 * spec.g * depth,
                }
            )
        )
    for r in (0.8, 1.0, 1.2):
        (case_dir / f"forced_r{r:.2f}.json").write_text(
            json.dumps(
                {
                    "id": f"forced_r{r:.2f}",
                    "kind": "forced",
                    "fill_ratio": 0.5,
                    "ratio": r,
                    "depth": 40.0,
                    "gauge_rows": [
                        {"y_relative": 0.0, "P_star_max": 0.35, "P_star_linear": 0.2,
                         "P_star_theory": 0.3, "P_star_theory_pred": 0.3, "n_samples": 1000},
                        {"y_relative": -4.0, "P_star_max": 0.3, "P_star_linear": 0.29,
                         "P_star_theory": 0.3, "P_star_theory_pred": 0.3, "n_samples": 1000},
                    ],
                    "steady_state": True,
                    "omega_over_omega1": r,
                    "eta1_measured": 2.0,
                    "eta1_theory": 2.1,
                    "eta2_over_eta1": 0.04,
                    "p_max_over_half_rho_v2": 1.5,
                }
            )
        )
    nt, nx = 6, 20
    omega = 2 * math.pi * 0.01
    (case_dir / "waveform_f0.50_r0.80.json").write_text(
        json.dumps(
            {
                "id": "waveform_f0.50_r0.80",
                "kind": "waveform",
                "ratio": 0.8,
                "fill_ratio": 0.5,
                "omega": omega,
                "eta1_theory": 2.0,
                "waveform_rel_l2": 0.05,
                "waveform_rel_max": 0.1,
                "waveform_amp_fit": 2.0,
                "waveform_x": list(np.arange(1, nx + 1) - 0.5),
                "waveform_model": (np.ones((nt, nx)) * 0.1).tolist(),
                "waveform_measured": (np.ones((nt, nx)) * 0.1).tolist(),
                "waveform_t": list(np.arange(nt) * 10.0),
            }
        )
    )
    report = C.summarize(case_dir, tmp_path / "out", spec)
    assert (tmp_path / "out" / "campaign_report.json").exists()
    for key in ("fig1_frequency_vs_fill", "fig2_pressure_vs_frequency_ratio", "fig3_free_surface_waveform"):
        assert key in report["figures"], report["figures"]
        assert (tmp_path / "out" / report["figures"][key]).exists()
    assert report["checks"]["criterion_1_frequency_5pct"]
    assert report["checks"]["criterion_5_figures"]
    assert report["provenance"]["n_cases"] == 14


def test_tank_config_rejects_incompressible_violation():
    with pytest.raises(ValueError):
        TankConfig(length=80, depth=80.0, g=5e-3).check()
    with pytest.raises(ValueError):
        TankConfig(length=8, depth=4.0, g=5e-4).check()


def test_campaign_guard_records_errors_instead_of_raising():
    spec = C.CampaignSpec()
    bad = {"kind": "decay", "fill_ratio": -1.0, "id": "bad"}
    out = C.run_case(spec, bad)
    assert "error" in out and out["id"] == "bad"