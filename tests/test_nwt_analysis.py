"""Post-processing estimators: harmonic fit, phase speed, travel time, budgets."""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.nwt.analysis import (
    fit_harmonics,
    inventory_drift,
    lag_from_cross_correlation,
    wavenumber_from_probes,
)
from tensorlbm.nwt.stokes2 import Stokes2Wave

DT = 1.0


def test_fit_harmonics_recovers_stokes_series():
    omega = 2.0 * math.pi / 500.0
    w = Stokes2Wave.from_steepness(0.08, 0.1, 20.0, 9.81)
    t = torch.arange(2600, dtype=torch.float64) * DT
    eta = w.amplitude * torch.cos(omega * t) + w.second_harmonic_amplitude * torch.cos(2 * omega * t) + 0.013
    fit = fit_harmonics(eta, DT, omega)
    assert fit["A1"] == pytest.approx(w.amplitude, rel=1e-9)
    assert fit["A2"] == pytest.approx(abs(w.second_harmonic_amplitude), rel=1e-8)
    assert fit["mean_level"] == pytest.approx(0.013, abs=1e-9)
    assert fit["residual_rms"] < 1e-12


def test_fit_harmonics_residual_flags_non_stokes_content():
    omega = 2.0 * math.pi / 500.0
    t = torch.arange(2600, dtype=torch.float64) * DT
    clean = torch.cos(omega * t)
    dirty = clean + 0.2 * torch.cos(3.0 * omega * t)
    assert fit_harmonics(clean, DT, omega)["residual_rms"] < 1e-12
    assert fit_harmonics(dirty, DT, omega)["residual_rms"] > 0.05


@pytest.mark.parametrize("k_true", [2.0 * math.pi / 60.0, 2.0 * math.pi / 145.0])
def test_wavenumber_from_probes_recovers_k(k_true):
    omega = 0.01
    t = torch.arange(3000, dtype=torch.float64) * DT
    xs = [0.0, 15.0, 30.0, 45.0]
    probes = [torch.cos(omega * t - k_true * x) for x in xs]
    out = wavenumber_from_probes(probes, xs, DT, omega)
    assert out["k_measured"] == pytest.approx(k_true, rel=1e-9)
    assert out["k_spread"] < 1e-9
    assert out["c_measured"] == pytest.approx(omega / k_true, rel=1e-9)


def test_wavenumber_requires_two_probes():
    t = torch.arange(100, dtype=torch.float64)
    with pytest.raises(ValueError):
        wavenumber_from_probes([torch.cos(0.01 * t)], [0.0], DT, 0.01)


def test_cross_correlation_lag_matches_known_delay():
    omega = 2.0 * math.pi / 400.0
    t = torch.arange(4000, dtype=torch.float64) * DT
    lag = 37.0
    a = torch.cos(omega * t)
    b = torch.cos(omega * (t - lag))
    out = lag_from_cross_correlation(a, b, DT)
    assert out["lag_samples"] == pytest.approx(lag, abs=0.1)
    assert out["peak_correlation"] > 0.95  # full-record normalisation, ~1-|lag|/n
    assert out["lag_time"] == pytest.approx(lag * DT, abs=0.1)


def test_cross_correlation_on_integer_shift_is_exact():
    t = torch.arange(2000, dtype=torch.float64)
    a = torch.sin(0.02 * t) + 0.3 * torch.sin(0.05 * t)
    b = torch.roll(a, 11)
    out = lag_from_cross_correlation(a, b, DT)
    assert out["lag_samples"] == pytest.approx(11.0, abs=0.1)


def test_inventory_drift_detects_linear_leak():
    n_per = 100
    inv = [1.0e4 * (1.0 + 0.001 * i / n_per) for i in range(5 * n_per + 1)]
    out = inventory_drift(inv, n_per)
    assert out["max_per_period_drift"] == pytest.approx(0.001, rel=1e-6)
    assert out["n_periods"] == 5
    assert out["final_relative_drift"] == pytest.approx(0.005, rel=1e-6)


def test_inventory_drift_zero_for_constant_record():
    out = inventory_drift([5000.0] * 401, 100)
    assert out["max_relative_drift"] == 0.0
    assert out["max_per_period_drift"] == 0.0