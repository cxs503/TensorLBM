"""Acceptance tests for the LBM-LES turbulence verification surface (p84w3-n05).

Coverage map (campaign criteria in brackets):

* ``test_mean_profile_*``            -> (1) U+(y+) log-region vs MKM1999 <= 5 %
* ``test_reynolds_stress_*``         -> (2) -<u'v'>+ peak position <= 5 wall units
* ``test_spectrum_*``                -> (3) E(k) slope == -5/3 with R^2 > 0.95
* ``test_friction_velocity_*``       -> (4) u_tau vs DNS <= 3 %
* ``test_completed_case_*``          -> the committed campaign artifact, re-evaluated
                                        from its own stored profiles (anti-fabrication)
* ``test_les_channel_short_run_*``   -> the LES path still runs end-to-end and the
                                        criteria are attached to its output

The synthetic-data tests are the real guards: they build a "perfect" result from
the DNS table itself (must PASS) and perturbed copies (must FAIL), so the
criteria cannot be satisfied by an implementation that always returns true.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorlbm.turbulence.les_channel3d import (
    LESChannel,
    LESChannelConfig,
    fit_power_law_slope,
)
from tensorlbm.turbulence.les_channel_verification import (
    CRITERIA_KEYS,
    LOG_LAW_MAX_REL_ERROR,
    SPECTRUM_R2_MIN,
    SPECTRUM_SLOPE_TOL,
    U_TAU_MAX_REL_ERROR,
    UV_PEAK_MAX_DY_PLUS,
    criteria_from_artifact,
    evaluate_criteria,
    friction_velocity_error,
    load_dns_reference,
    mean_profile_error,
    reynolds_stress_peak_error,
    run_case,
    spectrum_slope_check,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "artifacts" / "p84w3_n05_les_channel" / "case_re180_h10_forward_gradient_cs0.16.json"


# ---------------------------------------------------------------------------
# fixtures: synthetic results built from the DNS table itself
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dns():
    return load_dns_reference()


def _synthetic_result(dns, *, u_scale=1.0, y_shift=0.0, slope=-5.0 / 3.0, noise=0.0):
    """A result dict whose statistics are the DNS data (optionally perturbed)."""
    y = dns.y_plus
    u = dns.u_plus * u_scale
    uv = dns.uv_minus_plus
    if y_shift:
        # shift the stress peak: interpolate the profile onto y + shift
        y_src = y
        shifted = torch.zeros_like(uv)
        for i, yy in enumerate(y + y_shift):
            lo = int(torch.searchsorted(y_src, yy).clamp(1, y_src.numel() - 1))
            y0, y1 = float(y_src[lo - 1]), float(y_src[lo])
            t = (float(yy) - y0) / (y1 - y0) if y1 > y0 else 0.0
            shifted[i] = (1 - t) * uv[lo - 1] + t * uv[lo]
        uv = shifted
    k = torch.arange(1, y.numel() // 2 + 1, dtype=torch.float64)
    e = k**slope
    return {
        "label": "synthetic",
        "config": {"re_tau_target": dns.re_tau, "dy_plus": float(y[1] - y[0])},
        "wall_units": {
            "y_plus": y.tolist(),
            "u_plus": u.tolist(),
            "uv_minus_plus": uv.tolist(),
        },
        "reynolds_stress": {},
        "friction_velocity": {"target": 0.01, "wall_exchange": 0.01, "body_force": 0.01},
        "les": {
            "spectra": {
                "samples": 10,
                "streamwise": [
                    {
                        "y_plus": 30.0,
                        "k": k.tolist(),
                        "E": e.tolist(),
                        "fit": fit_power_law_slope(k, e),
                    }
                ],
            }
        },
    }


def _codes(criteria):
    return {k: v["status"] for k, v in criteria["checks"].items()}


# ---------------------------------------------------------------------------
# SGS strain estimation: the grid-scale (checkerboard) blindness
# ---------------------------------------------------------------------------
def _channel_for_strain(nx=16, ny=10, nz=8):
    cfg = LESChannelConfig(
        nx=nx, ny=ny, nz=nz, re_tau=180.0, u_tau=0.01, steps=2, warmup_steps=1,
        smagorinsky_cs=0.12, strain_source="forward_gradient",
    )
    return LESChannel(cfg)


def _set_velocity(les, u):
    les.b_u = u.reshape(3, les.N).to(les.b_u.dtype).clone()


def _strain(les, name):
    fn = getattr(les, f"_strain_magnitude_{name}")
    return fn().view(les.nz, les.ny, les.nx)[les.nz // 2, les.ny // 2, les.nx // 2]


def test_central_difference_strain_is_blind_to_the_grid_scale_mode():
    """The documented defect: a checkerboard mode has *zero* central-difference strain."""
    les = _channel_for_strain()
    nz, ny, nx = les.nz, les.ny, les.nx
    i = torch.arange(nx, dtype=torch.float64)
    u = torch.zeros(3, nz, ny, nx, dtype=torch.float64)
    u[0] = 0.01 * ((-1.0) ** i).view(1, 1, nx)
    _set_velocity(les, u)
    assert abs(float(_strain(les, "velocity_gradient"))) < 1e-12
    # the Nyquist-aware estimators do see it
    assert float(_strain(les, "forward_gradient")) == pytest.approx(2.0 * 0.01 * math.sqrt(2.0), rel=1e-5)
    assert float(_strain(les, "structure_function")) > 0.0


def test_forward_gradient_strain_keeps_the_smagorinsky_calibration():
    """One-sided gradients: exact for linear shear, exactly zero for rotation."""
    les = _channel_for_strain()
    nz, ny, nx = les.nz, les.ny, les.nx
    y = torch.arange(ny, dtype=torch.float64) - 0.5
    g = 0.1
    u = torch.zeros(3, nz, ny, nx, dtype=torch.float64)
    u[0] = g * y.view(1, ny, 1)
    _set_velocity(les, u)
    assert float(_strain(les, "forward_gradient")) == pytest.approx(g, rel=1e-5)
    # solid-body rotation: only the antisymmetric part, |S| = 0
    x = torch.arange(nx, dtype=torch.float64)
    u = torch.zeros(3, nz, ny, nx, dtype=torch.float64)
    u[0] = 0.05 * y.view(1, ny, 1)
    u[1] = -0.05 * x.view(1, 1, nx)
    _set_velocity(les, u)
    assert abs(float(_strain(les, "forward_gradient"))) < 1e-6


def test_structure_function_model_is_rotation_blind():
    """Documented limitation of the Metais-Lesieur estimator (kept as an audited variant).

    Its structure function cannot tell rotation from strain, so it returns a
    finite ``|S|`` for solid-body rotation.  That is exactly why the campaign
    run uses ``forward_gradient`` instead.
    """
    les = _channel_for_strain()
    nz, ny, nx = les.nz, les.ny, les.nx
    y = torch.arange(ny, dtype=torch.float64) - 0.5
    x = torch.arange(nx, dtype=torch.float64)
    u = torch.zeros(3, nz, ny, nx, dtype=torch.float64)
    u[0] = 0.05 * y.view(1, ny, 1)
    u[1] = -0.05 * x.view(1, 1, nx)
    _set_velocity(les, u)
    assert float(_strain(les, "structure_function")) > 1e-3


def test_trt_antisymmetric_time_is_pathological_at_low_viscosity():
    """Re_tau=180 in lattice units puts TRT's shear mode at tau- ~ 1e2.

    Documented here because it is the numerical context of the campaign: with
    the magic parameter Lambda = 3/16 the antisymmetric relaxation time follows
    ``tau_minus = 0.5 + Lambda/(tau_plus - 0.5)``, so the vanishing molecular
    viscosity of the wall-resolved Reynolds number in lattice units
    (``tau_plus ~ 0.5013``) forces ``tau_minus ~ 141`` -- a mode relaxing at
    7e-3 per step.  ``tau_minus_max`` can clamp it; the campaign measured that
    the clamp alone does *not* remove the divergence of the grid-scale mode
    (see docs/p84w3_n05_les_channel.md), so it is kept as a diagnostic knob.
    """
    base = dict(nx=38, ny=18, nz=19, re_tau=180.0, u_tau=0.01, steps=2, warmup_steps=1,
                smagorinsky_cs=0.12, strain_source="forward_gradient")
    cfg = LESChannelConfig(**base)
    assert cfg.tau_minus > 100.0
    unclamped = LESChannel(cfg)
    assert float(unclamped.sm) < 0.01
    clamped = LESChannel(LESChannelConfig(**base, tau_minus_max=2.0))
    assert float(clamped.sm) == pytest.approx(0.5, rel=1e-12)
    with pytest.raises(ValueError):
        LESChannelConfig(**base, tau_minus_max=0.4).validate()


def test_forward_gradient_branch_runs_end_to_end():
    cfg = LESChannelConfig(
        nx=24, ny=12, nz=16, re_tau=180.0, u_tau=0.01, steps=40, warmup_steps=10,
        sample_interval=5, report_interval=20, smagorinsky_cs=0.12,
        strain_source="forward_gradient", label="fg_smoke",
    )
    result = LESChannel(cfg).run_les()
    assert result["config"]["strain_source"] == "forward_gradient"
    sgs = result["les"]["sgs_stress"]
    assert math.isfinite(float(sgs["nu_t_over_nu_max"]))
    assert float(sgs["nu_t_over_nu_max"]) >= 0.0


# ---------------------------------------------------------------------------
# criterion (1): mean profile
# ---------------------------------------------------------------------------
def test_mean_profile_of_the_dns_table_itself_is_exact(dns):
    cmp = mean_profile_error(_synthetic_result(dns), dns)
    assert cmp["valid"] and cmp["n_points"] >= 4
    assert cmp["max_rel_error"] < 1e-9
    assert cmp["requested_band"][0] == pytest.approx(30.0)
    assert cmp["requested_band"][1] == pytest.approx(min(100.0, 0.3 * dns.re_tau))


def test_mean_profile_detects_a_five_percent_bias(dns):
    ok = mean_profile_error(_synthetic_result(dns, u_scale=1.02), dns)
    bad = mean_profile_error(_synthetic_result(dns, u_scale=1.10), dns)
    assert ok["max_rel_error"] <= LOG_LAW_MAX_REL_ERROR
    assert bad["max_rel_error"] > LOG_LAW_MAX_REL_ERROR


def test_mean_profile_withholds_when_the_band_has_no_node(dns):
    coarse = _synthetic_result(dns)
    coarse["wall_units"]["y_plus"] = [0.0, 60.0, 120.0]
    coarse["wall_units"]["u_plus"] = [0.0, 20.0, 24.0]
    coarse["wall_units"]["uv_minus_plus"] = [0.0, 0.7, 0.0]
    cmp = mean_profile_error(coarse, dns)
    assert cmp["valid"] is False
    assert cmp["n_points"] == 0


# ---------------------------------------------------------------------------
# criterion (2): Reynolds-stress peak position
# ---------------------------------------------------------------------------
def test_reynolds_stress_peak_matches_the_dns_budget(dns):
    cmp = reynolds_stress_peak_error(_synthetic_result(dns), dns)
    assert cmp["valid"]
    assert cmp["abs_delta_y_plus"] < 1e-9
    y_peak, uv_peak = dns.uv_peak()
    assert 20.0 < y_peak < 50.0  # MKM1999 Re_tau=180 inner peak
    assert 0.6 < uv_peak < 0.9


def test_reynolds_stress_peak_position_is_gated_on_wall_units(dns):
    small = reynolds_stress_peak_error(_synthetic_result(dns, y_shift=2.0), dns)
    large = reynolds_stress_peak_error(_synthetic_result(dns, y_shift=12.0), dns)
    assert small["abs_delta_y_plus"] <= UV_PEAK_MAX_DY_PLUS
    assert large["abs_delta_y_plus"] > UV_PEAK_MAX_DY_PLUS
    # amplitude error is reported separately and is not what gates the check
    assert "rel_error_amplitude" in large


# ---------------------------------------------------------------------------
# criterion (3): spectrum slope
# ---------------------------------------------------------------------------
def test_spectrum_slope_recovers_minus_five_thirds(dns):
    spec = spectrum_slope_check(_synthetic_result(dns, slope=-5.0 / 3.0))
    assert spec["valid"] and spec["passed"]
    assert spec["slope"] == pytest.approx(-5.0 / 3.0, abs=1e-6)
    assert spec["r2"] > SPECTRUM_R2_MIN


def test_spectrum_slope_rejects_a_wrong_inertial_slope(dns):
    spec = spectrum_slope_check(_synthetic_result(dns, slope=-1.0))
    assert spec["valid"] and not spec["passed"]
    assert spec["slope"] + 5.0 / 3.0 > SPECTRUM_SLOPE_TOL


def test_spectrum_check_withholds_without_spectra(dns):
    res = _synthetic_result(dns)
    res["les"] = {}
    spec = spectrum_slope_check(res)
    assert spec["valid"] is False


# ---------------------------------------------------------------------------
# criterion (4): friction velocity
# ---------------------------------------------------------------------------
def test_friction_velocity_error_is_relative_to_the_target(dns):
    res = _synthetic_result(dns)
    good = friction_velocity_error(res, dns)
    assert good["valid"] and good["abs_rel_error"] == pytest.approx(0.0)
    res["friction_velocity"]["wall_exchange"] = 0.01 * (1.0 + 1.5 * U_TAU_MAX_REL_ERROR)
    bad = friction_velocity_error(res, dns)
    assert bad["abs_rel_error"] > U_TAU_MAX_REL_ERROR


def test_friction_velocity_withholds_without_the_wall_flux(dns):
    res = _synthetic_result(dns)
    res["friction_velocity"] = {"target": 0.01}
    out = friction_velocity_error(res, dns)
    assert out["valid"] is False


# ---------------------------------------------------------------------------
# verdict assembly
# ---------------------------------------------------------------------------
def test_all_four_criteria_pass_on_the_dns_itself(dns):
    criteria = evaluate_criteria(_synthetic_result(dns), dns)
    assert set(criteria["checks"]) == set(CRITERIA_KEYS)
    assert criteria["all_passed"], criteria["failed"]
    assert criteria["withheld"] == []


def test_empty_result_withholds_everything_instead_of_passing(dns):
    criteria = evaluate_criteria({"config": {}}, dns)
    assert criteria["passed"] == []
    assert len(criteria["withheld"]) == 4
    assert criteria["all_passed"] is False


def test_criteria_are_json_serialisable(dns):
    criteria = evaluate_criteria(_synthetic_result(dns), dns)
    blob = json.dumps(criteria)
    assert "1_mean_profile_log_region" in blob


# ---------------------------------------------------------------------------
# LES path still runs end-to-end
# ---------------------------------------------------------------------------
def test_les_channel_short_run_carries_the_criteria():
    cfg = LESChannelConfig(
        nx=24,
        ny=12,
        nz=16,
        re_tau=180.0,
        u_tau=0.01,
        steps=60,
        warmup_steps=20,
        sample_interval=5,
        report_interval=30,
        smagorinsky_cs=0.12,
        label="les_verify_short",
    )
    result = LESChannel(cfg).run_les()
    for key in ("wall_units", "reynolds_stress", "friction_velocity", "les"):
        assert key in result
    criteria = evaluate_criteria(result)
    assert set(criteria["checks"]) == set(CRITERIA_KEYS)
    # a 60-step laminar-ish run cannot meet the campaign: each criterion must
    # be either FAIL or WITHHELD with the offending number attached.
    for name, check in criteria["checks"].items():
        assert check["status"] in ("FAIL", "WITHHELD"), (name, check["status"])
        assert check.get("criterion")


def test_run_case_returns_artifact_shape():
    art = run_case(half_height=4, steps=60, warmup_steps=20, sample_interval=5, report_interval=30, label="tiny")
    assert art["campaign"] == "p84w3-n05-lbm-les-turbulence"
    assert set(art["criteria"]["checks"]) == set(CRITERIA_KEYS)
    assert art["summary"]["cells"] == int(round(2 * math.pi * 4)) * (2 * 4 + 2) * int(round(math.pi * 4))
    json.dumps(art)


# ---------------------------------------------------------------------------
# committed campaign artifact: re-evaluate from its own stored numbers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT.exists():
        pytest.fail(f"missing campaign artifact {ARTIFACT}; run benchmarks/p84w3_n05_les_channel/run_case.py")
    return json.loads(ARTIFACT.read_text())


def test_completed_case_artifact_is_reproducible_from_its_own_data(artifact):
    stored = artifact["criteria"]
    recomputed = criteria_from_artifact(artifact)
    assert set(stored["checks"]) == set(recomputed["checks"])
    for name, check in stored["checks"].items():
        again = recomputed["checks"][name]
        assert check["status"] == again["status"], name
        # the stored metric must match the value recomputed from the raw arrays
        if isinstance(check.get("metric"), (int, float)) and math.isfinite(float(check["metric"])):
            assert float(check["metric"]) == pytest.approx(float(again["metric"]), rel=1e-6, abs=1e-9)


def test_completed_case_records_its_resolution_and_window(artifact):
    """The artifact must state the grid/window it was obtained on.

    The campaign's honest outcome is a FAIL, so this test does *not* assert that
    the statistics are converged: it asserts that the resolution, the sampling
    window and the resolved-flow diagnostics are recorded, i.e. that the verdict
    can be read against the grid it came from.
    """
    result = artifact["result"]
    cfg = result["config"]
    assert cfg["dy_plus"] <= 20.0  # gray-zone grid, stated in the report
    assert cfg["eddy_turnovers_averaged"] >= 5.0
    assert cfg["samples"] >= 200
    wall = result["wall_units"]
    assert max(wall["u_plus"]) > 5.0  # a channel flow, not a still fluid
    assert len(result.get("history", [])) >= 2
    assert result["reynolds_stress"]["uv_minus_peak_plus"] >= 0.0


def test_completed_case_criteria_are_evaluated_honestly(artifact):
    criteria = artifact["criteria"]
    assert set(criteria["checks"]) == set(CRITERIA_KEYS)
    for name, check in criteria["checks"].items():
        assert check["status"] in ("PASS", "FAIL", "WITHHELD"), (name, check["status"])
        assert check.get("criterion")
        assert check.get("numbers")
    # every criterion that is claimed PASS must carry its number inside tolerance
    th = criteria["thresholds"]
    if criteria["checks"]["1_mean_profile_log_region"]["status"] == "PASS":
        assert criteria["checks"]["1_mean_profile_log_region"]["metric"] <= th["log_law_max_rel_error"]
    if criteria["checks"]["2_reynolds_stress_peak"]["status"] == "PASS":
        assert criteria["checks"]["2_reynolds_stress_peak"]["metric"] <= th["uv_peak_max_dy_plus"]
    if criteria["checks"]["3_energy_spectrum_slope"]["status"] == "PASS":
        assert abs(criteria["checks"]["3_energy_spectrum_slope"]["metric"]["slope"] + 5.0 / 3.0) <= th["spectrum_slope_tol"]
        assert criteria["checks"]["3_energy_spectrum_slope"]["metric"]["r2"] >= th["spectrum_r2_min"]
    if criteria["checks"]["4_friction_velocity"]["status"] == "PASS":
        assert criteria["checks"]["4_friction_velocity"]["metric"] <= th["u_tau_max_rel_error"]