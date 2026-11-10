"""p223w3-n02: turbulent channel DNS verification face (Re_tau = 180/395/590).

The suite is split in two halves:

* **pure-function tests** on the reduction utilities
  (``benchmarks/p223w3_n02_channel_dns/analysis.py``) - analytic anchors, no
  solver involved;
* **kernel tests** on the native D3Q19 TRT kernel
  (``src/tensorlbm/turbulence/les_chan.c``) on deliberately tiny grids - they
  only assert invariants that must hold on *any* grid (mass conservation,
  fail-closed divergence, laminar Poiseuille recovery, determinism), so the
  suite stays under a minute and never duplicates the production campaign.

Plus **gate-logic tests** that feed synthetic records to
``benchmarks/p223w3_n02_channel_dns/gates.py`` so the pass/fail wiring is
itself covered.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_BENCH = _REPO / "benchmarks" / "p223w3_n02_channel_dns"
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_BENCH))

import analysis as an  # noqa: E402
import gates as gt  # noqa: E402
from tensorlbm.turbulence.les_channel import (  # noqa: E402
    LesChannelConfig,
    run_native,
)


def _cfg(**kw) -> LesChannelConfig:
    base = dict(nx=16, ny=18, nz=16, steps=200, warmup=60, sample=20,
                re_tau=10.0, u_tau=0.0125, cs=0.0, perturb=0.0,
                perturb_modes=1, seed=11, threads=4, regularized=0)
    base.update(kw)
    return LesChannelConfig(**base)


# ---------------------------------------------------------------------------
# analysis: law of the wall
# ---------------------------------------------------------------------------

def test_log_law_matches_definition() -> None:
    y = np.array([30.0, 50.0, 100.0, 200.0])
    up = an.log_law(y)
    assert np.allclose(up, np.log(y) / 0.41 + 5.2)


def test_log_law_errors_are_zero_on_its_own_profile() -> None:
    yp = np.geomspace(30.0, 200.0, 40)

    class P:
        y_plus = yp
        u_plus = an.log_law(yp)

    err = an.log_law_error(P(), 30.0, 200.0)
    assert err["n_points"] == 40
    assert err["rms_rel_error"] < 1e-12
    assert abs(err["k_fit"] - 0.41) < 1e-8
    assert abs(err["b_fit"] - 5.2) < 1e-8


def test_log_law_error_detects_an_offset_profile() -> None:
    yp = np.geomspace(30.0, 200.0, 40)
    klass = type("P", (), {})
    p = klass()
    p.y_plus = yp
    p.u_plus = an.log_law(yp) * 1.06          # 6 % high
    err = an.log_law_error(p, 30.0, 200.0)
    assert 0.05 < err["rms_rel_error"] < 0.07


def test_reichardt_limits() -> None:
    # viscous sublayer U+ -> y+ ; log layer U+ -> ln(y+)/k + B-ish
    assert abs(float(an.reichardt(0.1)) - 0.1) < 0.01
    assert abs(float(an.reichardt(1.0)) - 1.0) < 0.02
    assert an.reichardt(30.0) > 12.0


def test_reichardt_beats_log_law_in_the_buffer_layer() -> None:
    # sanity: the composite law must lie *below* the log law near y+ = 15
    assert float(an.reichardt(15.0)) < float(an.log_law(15.0))


def test_wall_units_use_the_half_way_wall() -> None:
    yp = an.wall_units(np.array([1.0, 2.0, 3.0]), u_tau=0.5, nu=0.05)
    assert np.allclose(yp, np.array([0.5, 1.5, 2.5]) * 10.0)


# ---------------------------------------------------------------------------
# analysis: friction velocity + stresses
# ---------------------------------------------------------------------------

def _synthetic_channel(n=60, u_tau=0.4, nu=0.02, h=20.0):
    """Total-stress budget of a force-driven channel, exactly."""
    y = np.arange(1, n + 1, dtype=float)
    yw = y - 0.5
    # u' = -u_tau^2 (1 - y/h) + nu dU/dy  =>  integrate the mean profile
    # pick U(y) from a Reichardt-like profile scaled to the wall units
    yp = yw * u_tau / nu
    u = u_tau * an.reichardt(yp)
    dudy = np.gradient(u, yw)
    uv = nu * dudy - u_tau**2 * (1.0 - yw / h)
    return y, u, uv, dudy


def test_stress_fit_recovers_the_imposed_friction_velocity() -> None:
    y, u, uv, _ = _synthetic_channel()
    fit, total = an.stress_fit_u_tau(y, u, uv, nu=0.02, h=20.0)
    assert abs(fit - 0.4) / 0.4 < 1e-6
    assert total.shape == u.shape


def test_stress_fit_rejects_too_few_rows() -> None:
    y = np.array([1.0, 2.0])
    u = np.array([0.0, 0.1])
    uv = np.zeros(2)
    with pytest.raises(ValueError):
        an.stress_fit_u_tau(y, u, uv, nu=0.02, h=2.0)


def test_total_stress_is_linear_in_y_over_h() -> None:
    y, u, uv, dudy = _synthetic_channel()
    total = 0.02 * dudy - uv
    h = 20.0
    ref = 0.4**2 * (1.0 - (y - 0.5) / h)
    assert np.max(np.abs(total - ref)) < 1e-9


def test_peak_reports_location_and_value() -> None:
    yp = np.array([5.0, 10.0, 15.0, 20.0])
    v = np.array([1.0, 2.0, 3.0, 2.5])
    p = an.peak(v, yp)
    assert p["value"] == 3.0 and p["y_plus"] == 15.0


# ---------------------------------------------------------------------------
# analysis: friction coefficient
# ---------------------------------------------------------------------------

def test_cf_definition_and_dean_consistency() -> None:
    # DNS MKM-1999 Re_tau=180: U_b+ = 15.68 -> Cf = 2/15.68^2
    assert abs(an.cf_from_bulk(1.0, 15.68) - 2.0 / 15.68**2) < 1e-15
    # Dean at Re_b = 5600 reproduces the DNS friction within 5 %
    cf_dean = an.dean_cf(5600.0)
    assert abs(cf_dean - 0.00814) / 0.00814 < 0.05


def test_dean_scaling_exponent() -> None:
    a = an.dean_cf(1000.0)
    b = an.dean_cf(16000.0)
    assert abs(math.log(a / b) / math.log(16.0) - 0.25) < 1e-12


def test_bulk_reynolds_uses_the_full_height() -> None:
    assert an.bulk_reynolds(1.0, 40.0, 0.5) == 80.0


# ---------------------------------------------------------------------------
# analysis: spectra
# ---------------------------------------------------------------------------

def _power_law_plane(nz=400, nx=256, slope=-5.0 / 3.0, seed=3):
    """Random-phase field whose span-averaged spectrum is exactly k**slope."""
    rng = np.random.default_rng(seed)
    k = np.fft.rfftfreq(nx)[1:]
    amp = k ** (slope / 2.0)
    spec = np.zeros((nz, nx // 2 + 1), dtype=complex)
    spec[:, 1:] = amp * np.exp(2j * np.pi * rng.random((nz, k.size)))
    return np.fft.irfft(spec, n=nx)


def test_spectrum_recovers_a_known_slope() -> None:
    plane = _power_law_plane()
    k, e = an.streamwise_spectrum(plane, lx=2.0 * np.pi)
    fit = an.power_law_slope(k, e, 3.0, 40.0)
    assert abs(fit["slope"] + 5.0 / 3.0) < 0.05


def test_spectrum_normalisation_conserves_variance() -> None:
    rng = np.random.default_rng(5)
    plane = rng.normal(size=(6, 96))
    k, e = an.streamwise_spectrum(plane, lx=2.0 * np.pi)
    dk = k[1] - k[0]
    var = np.mean((plane - plane.mean(axis=1, keepdims=True)) ** 2)
    assert abs(np.sum(e) * dk - var) / var < 0.35   # Hann window leakage


def test_spectrum_slope_is_robust_to_a_negative_control() -> None:
    # a spectrum with slope -3 must NOT be accepted as -5/3
    plane = _power_law_plane(slope=-3.0)
    k, e = an.streamwise_spectrum(plane, lx=2.0 * np.pi)
    fit = an.power_law_slope(k, e, 3.0, 40.0)
    assert abs(fit["slope"] + 5.0 / 3.0) > 0.5


# ---------------------------------------------------------------------------
# analysis: laminar reference
# ---------------------------------------------------------------------------

def test_poiseuille_satisfies_the_momentum_balance() -> None:
    y = np.arange(1, 41, dtype=float)
    height, accel, nu = 40.0, 1e-5, 0.01
    u = an.poiseuille_exact(y, height, accel, nu)
    d2u = np.gradient(np.gradient(u, 1.0), 1.0)
    assert np.allclose(nu * d2u[2:-2], -accel, atol=1e-12)
    # zero at both walls (wall planes are y = 0.5 and H + 0.5)
    assert np.allclose(an.poiseuille_exact([0.5, 40.5], height, accel, nu), 0.0)


def test_poiseuille_bulk_is_two_thirds_of_the_peak() -> None:
    y = np.arange(1, 81, dtype=float)
    u = an.poiseuille_exact(y, 80.0, 1e-5, 0.02)
    assert abs(u.mean() / u.max() - 2.0 / 3.0) < 0.01


def test_relative_norms() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert an.relative_l2(a, a) == 0.0
    assert abs(an.relative_max(a * 1.1, a) - 0.1) < 1e-12


# ---------------------------------------------------------------------------
# kernel: configuration plumbing
# ---------------------------------------------------------------------------

def test_kernel_config_passes_the_collision_switch() -> None:
    args = _cfg(regularized=0).cli_args("/tmp/x")
    assert "--reg" in args
    assert args[args.index("--reg") + 1] == "0"


def test_regularized_projection_is_off_by_default() -> None:
    assert LesChannelConfig().regularized == 0


def test_kernel_config_rejects_a_supersonic_centreline() -> None:
    with pytest.raises(ValueError):
        _cfg(u_tau=0.05).validate()


def test_kernel_config_rejects_unresolved_relaxation_time() -> None:
    with pytest.raises(ValueError):
        _cfg(re_tau=1.0e6).validate()


# ---------------------------------------------------------------------------
# kernel: physics invariants on a tiny grid
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_laminar(tmp_path_factory):
    out = tmp_path_factory.mktemp("laminar")
    # H=16, nu=0.01 -> viscous time H^2/(pi^2 nu) ~ 2.6e3 steps; 24000 steps
    # is ~9 viscous times, so the initial law-of-the-wall transient is gone.
    cfg = _cfg(nx=16, ny=18, nz=16, steps=24000, warmup=20000, sample=200,
               re_tau=10.0, u_tau=0.0125, cs=0.0, perturb=0.0)
    return run_native(cfg, out, "lam")


def test_kernel_reports_ok_status_and_schema(tiny_laminar) -> None:
    payload = tiny_laminar
    assert payload["status"] == "ok"
    for key in ("configuration", "derived", "diagnostics", "profiles",
                "bulk_velocity", "bulk_velocity_series"):
        assert key in payload
    for key in ("y", "mean_ux", "mean_uy", "mean_uz", "r_uu", "r_vv", "r_ww", "r_uv"):
        assert key in payload["profiles"]


def test_kernel_conserves_mass(tiny_laminar) -> None:
    drift = tiny_laminar["diagnostics"]["mass_drift_fraction"]
    assert drift < 1e-6


def test_kernel_wall_normal_velocity_vanishes_in_a_laminar_run(tiny_laminar) -> None:
    uy = np.asarray(tiny_laminar["profiles"]["mean_uy"])
    assert np.max(np.abs(uy)) < 1e-3


def test_kernel_recovers_the_poiseuille_parabola(tiny_laminar) -> None:
    payload = tiny_laminar
    derived = payload["derived"]
    y = np.asarray(payload["profiles"]["y"], dtype=float)
    u = np.asarray(payload["profiles"]["mean_ux"], dtype=float)
    exact = an.poiseuille_exact(y, derived["height"], derived["accel"], derived["nu"])
    assert an.relative_l2(u, exact) < 0.01     # gate (e) threshold
    assert an.relative_max(u, exact) < 0.02


def test_kernel_body_force_route_is_exact(tiny_laminar) -> None:
    derived = tiny_laminar["derived"]
    # a = 2 u_tau^2 / H  =>  sqrt(a h) == u_tau exactly
    assert abs(math.sqrt(derived["accel"] * derived["half_height"])
               - tiny_laminar["configuration"]["u_tau"]) < 1e-12


def test_kernel_is_deterministic_for_a_fixed_seed(tmp_path) -> None:
    cfg = _cfg(steps=300, warmup=100, sample=25)
    a = run_native(cfg, tmp_path / "a", "run")
    b = run_native(cfg, tmp_path / "b", "run")
    assert a["bulk_velocity"] == b["bulk_velocity"]
    assert a["profiles"]["r_uu"] == b["profiles"]["r_uu"]


def test_kernel_divergence_fails_closed(tmp_path) -> None:
    """An under-resolved, highly turbulent case must raise - not emit NaN JSON."""
    # Re_tau=400 on 18 wall-normal rows => tau0 = 0.5007, which is below the
    # stability envelope of the scheme; the run must fail *loudly*.
    cfg = _cfg(nx=16, ny=38, nz=16, steps=600, warmup=100, sample=50,
               re_tau=400.0, u_tau=0.0125, cs=0.1, perturb=0.05,
               perturb_modes=6)
    with pytest.raises(RuntimeError):
        run_native(cfg, tmp_path, "boom")
    # nothing unreadable must be left behind
    for json_file in tmp_path.glob("*.json"):
        json.loads(json_file.read_text())


# ---------------------------------------------------------------------------
# gate wiring (synthetic records - no solver)
# ---------------------------------------------------------------------------

def _fake_case(name="R180_dy3", re_tau=180.0, rms=0.03, uu_peak=7.0,
               uu_y=15.0, slope=-1.66, cf=0.00814):
    cf_dean = 0.00842
    return {
        "case": name,
        "note": "",
        "geometry": {"re_tau": re_tau, "dy_plus": 3.0, "ny": 122, "nx": 120,
                     "nz": 60, "cells": 1, "half_height": 60.0, "u_tau": 0.0125,
                     "nu": 0.0042, "tau0": 0.5125, "steps": 1, "warmup": 0,
                     "stat_eddy_turnovers": 12.0},
        "mean_flow": {"u_bulk": 0.2, "re_bulk": 5600.0,
                      "u_tau_routes": {"body_force": 0.0125, "stress_fit": 0.0125,
                                       "imposed": 0.0125, "spread_rel": 0.0},
                      "cf": cf, "cf_dean": cf_dean,
                      "cf_error_vs_dean": (cf - cf_dean) / cf_dean,
                      "re_tau_achieved": re_tau},
        "profiles": {"y_plus": [1.5, 10.0, uu_y, 40.0],
                     "u_plus": [1.5, 10.0, 15.0, 20.0],
                     "uu_plus": [0.2, 5.0, uu_peak, 2.0],
                     "vv_plus": [0.1] * 4, "ww_plus": [0.1] * 4,
                     "uv_plus": [-0.1, -0.4, -0.6, -0.4],
                     "y_over_h": [0.02, 0.1, 0.25, 0.6]},
        "diagnostics": {"log_law": {"n_points": 30, "rms_rel_error": rms,
                                    "max_rel_error": rms * 1.4,
                                    "k_fit": 0.41, "b_fit": 5.2},
                        "reichardt": {"n_points": 30, "max_rel_error": 0.04},
                        "uu_peak": {"value": uu_peak, "y_plus": uu_y},
                        "uv_minus_peak": {"value": 0.7, "y_plus": 30.0},
                        "mass_drift_fraction": 1e-9, "max_speed": 0.25,
                        "bulk_velocity_series": [0.2]},
        "spectrum": {"k_plus": [0.1, 0.2], "e": [1.0, 0.5],
                     "fit": {"slope": slope, "n": 15, "k_min": 0.05,
                             "k_max": 0.30, "intercept": 0.0}},
        "provenance": {},
    }


def test_gate_log_law_passes_below_five_percent() -> None:
    g = gt.gate_log_law([_fake_case(rms=0.03)], required=(180.0,))
    assert g["passed"] is True


def test_gate_log_law_fails_above_five_percent() -> None:
    g = gt.gate_log_law([_fake_case(rms=0.07)], required=(180.0,))
    assert g["passed"] is False
    assert g["problems"]


def test_gate_log_law_withholds_missing_reynolds_numbers() -> None:
    # a green log law at Re_tau=180 must not certify the gate when 395/590 are absent
    g = gt.gate_log_law([_fake_case(rms=0.01)])
    assert g["status"] == "WITHHELD"
    assert g["passed"] is False
    assert len(g["withheld"]) == 2


def test_gate_reynolds_stress_accepts_a_good_peak() -> None:
    rec = _fake_case(uu_peak=7.0, uu_y=15.0)
    g = gt.gate_reynolds_stress(rec)
    assert g["evidence"]["position_ok"] is True
    assert g["evidence"]["magnitude_ok"] is True


def test_gate_reynolds_stress_rejects_a_shifted_peak() -> None:
    g = gt.gate_reynolds_stress(_fake_case(uu_peak=7.0, uu_y=45.0))
    assert g["evidence"]["position_ok"] is False


def test_gate_spectrum_accepts_minus_five_thirds() -> None:
    assert gt.gate_spectrum([_fake_case(slope=-5.0 / 3.0)])["passed"] is True


def test_gate_spectrum_rejects_a_steep_slope() -> None:
    assert gt.gate_spectrum([_fake_case(slope=-2.5)])["passed"] is False


def test_gate_friction_accepts_a_dean_consistent_cf() -> None:
    assert gt.gate_friction([_fake_case(cf=0.00842)], required=(180.0,))["passed"] is True


def test_gate_friction_rejects_a_bad_cf() -> None:
    assert gt.gate_friction([_fake_case(cf=0.0110)], required=(180.0,))["passed"] is False


def test_gate_grid_convergence_detects_spread() -> None:
    good = [_fake_case(name=n) for n in ("a", "b", "c")]
    for i, rec in enumerate(good):
        rec["geometry"]["dy_plus"] = 3.0 + i
    bad = [_fake_case(name="a"), _fake_case(name="b"), _fake_case(name="c")]
    for i, rec in enumerate(bad):
        rec["geometry"]["dy_plus"] = 3.0 + i
        rec["mean_flow"]["u_tau_routes"]["stress_fit"] = 0.0125 * (1 + 0.05 * i)
    assert gt.gate_grid_convergence(good)["passed"] is True
    assert gt.gate_grid_convergence(bad)["passed"] is False
    # a single completed grid can never certify convergence
    assert gt.gate_grid_convergence([good[0]])["status"] == "WITHHELD"


def test_gate_laminar_threshold() -> None:
    rec = {"geometry": {"re_laminar": 10.0, "steps": 10},
           "norm": {"l2_rel_error": 0.004, "max_rel_error": 0.02,
                    "u_max_sim": 1.0, "u_max_exact": 1.0}}
    assert gt.gate_laminar(rec)["passed"] is True
    rec["norm"]["l2_rel_error"] = 0.02
    assert gt.gate_laminar(rec)["passed"] is False


# ---------------------------------------------------------------------------
# DNS reference tables shipped with the repo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("re_tau", [180, 395, 590])
def test_dns_reference_tables_are_usable(re_tau: int) -> None:
    dns = gt.load_dns(re_tau)
    for key in ("y_plus", "u_plus", "uu_plus", "uv_plus"):
        assert key in dns and dns[key].size > 10
    assert np.all(np.diff(dns["y_plus"]) > 0)
    assert np.all(dns["uu_plus"] >= 0.0)


def test_dns_180_has_its_buffer_layer_peak_where_the_gate_expects_it() -> None:
    dns = gt.load_dns(180)
    i = int(np.argmax(dns["uu_plus"]))
    assert 12.0 < dns["y_plus"][i] < 20.0
    assert 6.0 < dns["uu_plus"][i] < 8.5


def test_dns_mean_velocity_grows_with_reynolds_number() -> None:
    peaks = [float(gt.load_dns(r)["u_plus"].max()) for r in (180, 395, 590)]
    assert peaks[0] < peaks[1] < peaks[2]