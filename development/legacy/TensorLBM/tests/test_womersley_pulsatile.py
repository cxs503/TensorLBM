"""Tests for the Womersley pulsatile-flow verification module.

Covers: analytic reference values (alpha=5), quasi-steady limits, the
whole-period phase-averaging/phasor estimator on synthetic signals, the
acceptance gates, and short end-to-end solver smoke runs for both geometries
(2D channel and 3D pipe with BFL curved-wall bounce-back).
"""

from __future__ import annotations

import math

import numpy as np

from tensorlbm.verification.womersley_pulsatile import analytical, gates, metrics
from tensorlbm.verification.womersley_pulsatile.channel2d import run_case as run_channel
from tensorlbm.verification.womersley_pulsatile.pipe3d import run_case as run_pipe


# ---------------------------------------------------------------------------
# analytic reference
# ---------------------------------------------------------------------------
def test_reference_alpha5_channel_and_pipe():
    for geom in ("channel", "pipe"):
        ref = analytical.REFERENCE_ALPHA5[geom]
        assert abs(analytical.centerline_amplitude_unit(geom, 5.0) - ref["abs_U0_over_a0_w"]) < 1e-12
        assert abs(analytical.centerline_phase_lag_deg(geom, 5.0) - ref["phase_lag_deg"]) < 1e-10


def test_alpha_omega_roundtrip():
    for geom, L in (("channel", 29.0), ("pipe", 20.5)):
        om = analytical.omega_from_alpha(geom, 5.0, 0.1, L)
        assert abs(analytical.alpha_from_omega(geom, om, 0.1, L) - 5.0) < 1e-12


def test_quasi_steady_limit_recovers_poiseuille():
    """alpha -> 0: U(0) -> a0 L^2/(8 nu) (channel) / a0 R^2/(4 nu) (pipe)."""
    nu, alpha = 0.1, 0.01
    for geom, L in (("channel", 40.0), ("pipe", 20.0)):
        om = analytical.omega_from_alpha(geom, alpha, nu, L)
        a0 = 1e-4
        z = analytical.centerline_phasor(geom, L, alpha, a0, om)
        u_ana = analytical.poiseuille_centerline(geom, L, a0, nu)
        assert abs(z - u_ana) / u_ana < 1e-3
        assert abs(-math.degrees(math.atan2(z.imag, z.real))) < 0.1


def test_force_scaling_gives_requested_peak():
    for geom, L in (("channel", 59.0), ("pipe", 20.5)):
        om = analytical.omega_from_alpha(geom, 5.0, 0.1, L)
        a0 = analytical.force_amplitude_for_peak(geom, L, 5.0, om, 0.03)
        z = analytical.centerline_phasor(geom, L, 5.0, a0, om)
        assert abs(abs(z) - 0.03) < 1e-12


# ---------------------------------------------------------------------------
# phase averaging / phasor estimator
# ---------------------------------------------------------------------------
def test_phasor_fit_recovers_synthetic_signal():
    om = 2.0 * math.pi / 1013.0
    t = np.arange(0, 6000, dtype=float)
    amp, phase0, off = 1.3, 0.4, 0.01
    u = amp * np.cos(om * t + phase0) + off
    pc, um, counts = metrics.fold_phase(t, u, om, 64)
    assert counts.sum() == t.size and (counts > 0).all()
    fit = metrics.fit_phasor(pc / om, um, om)
    assert abs(abs(fit["Z"]) - amp) < 1e-3
    assert abs(math.remainder(np.angle(fit["Z"]) - phase0, 2 * math.pi)) < 1e-3
    assert abs(fit["offset"] - off) < 1e-3
    # bin averaging attenuates by sinc(dphi/2) = (2*pi/n_bins) -> ~0.04 % at
    # n_bins=64; the estimator is unbiased below that.
    m = metrics.harmonic_metrics(fit["Z"], amp * np.exp(1j * phase0))
    assert m["amplitude_rel_pct"] < 0.1
    assert m["phase_rel_pct"] < 0.5


def test_fold_phase_averages_noise_out():
    om = 2.0 * math.pi / 97.0
    t = np.arange(0, 97 * 40, dtype=float)
    rng = np.random.default_rng(0)
    u = 2.0 * np.cos(om * t) + rng.normal(0.0, 0.5, t.size)
    pc, um, _ = metrics.fold_phase(t, u, om, 32)
    fit = metrics.fit_phasor(pc / om, um, om)
    assert abs(abs(fit["Z"]) - 2.0) < 0.05


def test_gate_verdict_semantics():
    good = {"amplitude_rel_pct": 0.1, "phase_rel_pct": 0.2,
            "profile_l2_pct": 0.3, "profile_l2_inner_pct": 0.2}
    bad = dict(good, phase_rel_pct=6.0)
    assert gates.evaluate_case(good)["passed"]
    v = gates.evaluate_case(bad)
    assert not v["passed"]
    conv = gates.convergence_verdict([
        {"resolution": 10, "metrics": dict(good, profile_l2_pct=2.0, profile_l2_inner_pct=1.0)},
        {"resolution": 20, "metrics": good},
    ])
    assert conv["passed"]


# ---------------------------------------------------------------------------
# end-to-end solver smoke runs (short, coarse)
# ---------------------------------------------------------------------------
def test_channel_smoke_alpha5():
    r = run_channel(H=15, alpha=5.0, nx=8, skip_periods=8)
    assert r["finite"]
    assert abs(r["mass_drift_pct"]) < 1e-3
    m = r["metrics"]
    assert m["amplitude_rel_pct"] < 5.0
    assert m["phase_rel_pct"] < 5.0
    assert r["gate"]["passed"]


def test_pipe_steady_poiseuille_wall_treatment():
    r = run_pipe(R=8.5, alpha=5.0, steady=True)
    assert r["finite"]
    assert r["u_center_rel_pct"] < 6.0
    assert r["profile_l2_pct"] < 8.0


def test_pipe_smoke_alpha5():
    r = run_pipe(R=10.5, alpha=5.0, nx=2, skip_periods=8)
    assert r["finite"]
    assert abs(r["mass_drift_pct"]) < 1.0
    m = r["metrics"]
    assert m["amplitude_rel_pct"] < 8.0
    assert m["phase_rel_pct"] < 8.0
