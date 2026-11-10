"""Capillary-wave dispersion verification for the CAC phase-field LBM.

The measurement chain (interface tracker -> modal amplitude -> damped-sinusoid
estimator) is tested exactly against synthetic signals, and the solver itself is
pinned at the reference capillary-wave case.

Physics scope
-------------
``tensorlbm.cac_lbm`` takes ``sigma`` as an input parameter.  The static Laplace
law of a cylindrical drop recovers that number to within 12%.  The *dynamic*
response does not follow the sharp-interface capillary-wave dispersion of a
two-fluid system (``omega^2 = sigma k^3 / (rho_l + rho_g)``): at the reference
case the measured frequency is ~19% low and the amplitude decay rate is ~40% low
with respect to ``gamma = 2 nu k^2``.  See
``docs/capillary_wave_dispersion_20261109.md`` and
``artifacts/capillary_wave_20261109/results.json``.

``test_capillary_wave_reference_deficit_is_pinned`` therefore asserts the
*measured, documented deficit band*.  It is a regression pin (any change of the
model or of the measurement chain must be re-adjudicated), **not** a validation
claim: the acceptance bar is 15% and the reference case does not meet it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.capillary_wave import (
    CapillaryWaveConfig,
    damped_sinusoid_fit,
    damping_theory,
    dispersion_theory,
    flat_interface_phi,
    interface_height,
    modal_amplitude,
    perturbed_layer_phi,
    run_capillary_wave,
    run_flat_interface,
    run_laplace_drop,
    sigma_from_profile,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Theory helpers
# ---------------------------------------------------------------------------


class TestTheoryHelpers:
    def test_dispersion_scaling_is_k_cubed(self) -> None:
        a = dispersion_theory(0.05, 0.1, 1.0, 1.0)
        b = dispersion_theory(0.05, 0.2, 1.0, 1.0)
        assert b / a == pytest.approx(2.0**1.5, rel=1e-12)

    def test_dispersion_inertia_is_the_density_sum(self) -> None:
        heavy = dispersion_theory(0.05, 0.2, 1.9, 0.1)
        light = dispersion_theory(0.05, 0.2, 1.0, 1.0)
        assert heavy == pytest.approx(light, rel=1e-12)  # same rho_l + rho_g

    def test_damping_scales_with_viscosity(self) -> None:
        assert damping_theory(0.04, 0.2) == pytest.approx(2.0 * 0.04 * 0.04, rel=1e-12)


# ---------------------------------------------------------------------------
# Configuration contract
# ---------------------------------------------------------------------------


class TestConfigContract:
    def test_default_config_is_valid(self) -> None:
        CapillaryWaveConfig().validate()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"amplitude": 2.0}, "steepness"),
            ({"sigma": 0.0}, "sigma"),
            ({"nu": 0.0}, "nu"),
            ({"mode": 0}, "mode"),
            ({"width": 1.0}, "width"),
            ({"ny": 40, "width": 11.0}, "width"),  # wider than ny/4
        ],
    )
    def test_invalid_config_is_rejected(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            CapillaryWaveConfig(**kwargs).validate()

    def test_steepness_is_amplitude_times_wavenumber(self) -> None:
        cfg = CapillaryWaveConfig(nx=64, mode=2, amplitude=0.5)
        assert cfg.steepness == pytest.approx(0.5 * 2.0 * math.pi * 2 / 64, rel=1e-12)


# ---------------------------------------------------------------------------
# Measurement chain vs synthetic ground truth
# ---------------------------------------------------------------------------


class TestMeasurementChain:
    def test_interface_height_recovers_prescribed_contour(self) -> None:
        """Absolute contour error of the linear-interpolation tracker must stay < 0.012 cells.

        The tracker interpolates ``phi`` linearly between the two rows straddling
        ``phi = 0.5``; on a W=4 tanh profile that leaves a systematic O(1%) bias
        which is *proportional to the amplitude*, so it scales the measured
        amplitude but does not bias the frequency.
        """
        cfg = CapillaryWaveConfig(ny=48, nx=64, nz=2, mode=2, amplitude=0.4, width=4.0)
        phi = perturbed_layer_phi(cfg)
        h = interface_height(phi)
        xs = torch.arange(cfg.nx, dtype=h.dtype)
        want = 0.5 * cfg.ny + cfg.amplitude * torch.cos(cfg.k * xs)
        assert float((h - want).abs().max()) < 1.2e-2

    def test_modal_amplitude_recovers_prescribed_amplitude(self) -> None:
        """Amplitude bias of the linear tracker is < 3% at W=4 (measured: +1.8%)."""
        cfg = CapillaryWaveConfig(ny=48, nx=64, nz=2, mode=2, amplitude=0.4, width=4.0)
        a = modal_amplitude(perturbed_layer_phi(cfg), cfg.k)
        assert a == pytest.approx(0.4, rel=0.03)

    def test_modal_amplitude_is_zero_for_a_flat_interface(self) -> None:
        cfg = CapillaryWaveConfig(ny=48, nx=64, nz=2, mode=2, amplitude=0.4, width=4.0)
        flat = perturbed_layer_phi(CapillaryWaveConfig(**{**cfg.__dict__, "amplitude": 1e-9}))
        assert abs(modal_amplitude(flat, cfg.k)) < 1e-6

    def test_interface_height_rejects_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match="nz, ny, nx"):
            interface_height(torch.zeros(8, 8))


class TestDampedSinusoidEstimator:
    @staticmethod
    def _signal(omega: float, gamma: float, *, amp: float = 0.5, n: int = 300, dt: float = 10.0):
        t = np.arange(n, dtype=float) * dt
        phase = 0.7
        a = amp * np.exp(-gamma * t) * np.cos(omega * t + phase)
        rng = np.random.default_rng(1234)
        return t, a + 1e-6 * rng.standard_normal(t.size)

    def test_recovers_known_parameters(self) -> None:
        t, a = self._signal(0.05, 0.002)
        fit = damped_sinusoid_fit(t, a, omega_bounds=(0.02, 0.1), gamma_bounds=(0.0, 0.01))
        assert fit["omega"] == pytest.approx(0.05, rel=0.005)
        assert fit["gamma"] == pytest.approx(0.002, rel=0.05)
        assert fit["amplitude"] == pytest.approx(0.5, rel=0.02)
        assert fit["r2"] > 0.999
        assert fit["n_samples"] == t.size

    def test_two_frequencies_are_distinguished(self) -> None:
        """Negative control: the estimator must not collapse both signals onto one value."""
        t, a1 = self._signal(0.05, 0.001)
        _, a2 = self._signal(0.075, 0.001)
        f1 = damped_sinusoid_fit(t, a1, omega_bounds=(0.02, 0.12), gamma_bounds=(0.0, 0.01))
        f2 = damped_sinusoid_fit(t, a2, omega_bounds=(0.02, 0.12), gamma_bounds=(0.0, 0.01))
        assert f1["omega"] == pytest.approx(0.05, rel=0.01)
        assert f2["omega"] == pytest.approx(0.075, rel=0.01)
        assert f2["omega"] / f1["omega"] == pytest.approx(1.5, rel=0.02)

    def test_exponential_without_oscillation_prefers_bound_and_flags_bad_fit(self) -> None:
        t = np.arange(300, dtype=float) * 10.0
        a = 0.5 * np.exp(-0.002 * t)
        fit = damped_sinusoid_fit(t, a, omega_bounds=(0.01, 0.1), gamma_bounds=(0.0, 0.01))
        # no oscillation to find: the record is not a decaying sinusoid, R^2 is poor
        assert fit["r2"] < 0.9

    def test_rejects_short_record(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            damped_sinusoid_fit([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], omega_bounds=(0.1, 1.0))

    def test_ignores_non_finite_samples(self) -> None:
        t, a = self._signal(0.05, 0.002)
        a = a.copy()
        a[::7] = np.nan
        fit = damped_sinusoid_fit(t, a, omega_bounds=(0.02, 0.1), gamma_bounds=(0.0, 0.01))
        assert fit["omega"] == pytest.approx(0.05, rel=0.01)


# ---------------------------------------------------------------------------
# Static surface tension (Laplace law) -- the calibration the wave is read against
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_static_laplace_sigma_within_25pct_at_ci_scale() -> None:
    """Cheap cylindrical-drop Laplace law: ``Delta p R`` must recover the input sigma.

    The CI-scale box (32^3 with only two z-layers, R=6, W=3) is coarse; the
    production calibration in ``artifacts/capillary_wave_20261109/laplace.json``
    uses R=8..16 at W=4 and lands within 12%.  The 25% band here only guards the
    sign and order of magnitude of the implemented Korteweg stress.
    """
    torch.set_num_threads(1)
    res = run_laplace_drop(6.0, width=3.0, sigma=0.05, nu=0.05, n_steps=2000, ny=32, nx=32)
    assert res["phi_range"][0] > -0.05
    assert res["phi_range"][1] < 1.05
    assert res["sigma_ratio"] == pytest.approx(1.0, abs=0.25)
    assert res["spurious_u_max"] < 0.05


# ---------------------------------------------------------------------------
# Dynamic response -- the open verification debt
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.physics
def test_capillary_wave_reference_deficit_is_pinned() -> None:
    """Regression pin of the *measured* dynamic deficit at the reference case.

    Reference: ``nx=64, ny=32, nz=2, mode=2 (k=0.196), W=4, nu=0.05, M=0.05,
    A=0.5``.  Measured (artifacts/capillary_wave_20261109): ``omega`` is 18-19%
    below ``sqrt(sigma k^3/(rho_l+rho_g))`` and ``gamma`` is ~0.58 x ``2 nu k^2``.
    The 15% acceptance bar is *not* met -- this test pins the deficit so that a
    model change or a measurement regression cannot pass unnoticed.
    """
    torch.set_num_threads(1)
    cfg = CapillaryWaveConfig(
        ny=32,
        nx=64,
        nz=2,
        mode=2,
        amplitude=0.5,
        sigma=0.05,
        nu=0.05,
        width=4.0,
        n_steps=2000,
        sample_every=10,
    )
    res = run_capillary_wave(cfg)
    assert res["blow_up_step"] is None
    fit = res["fit"]
    assert fit is not None
    assert fit["r2"] > 0.98
    # documented amplitude decay (stable run, no growth)
    assert abs(res["amplitudes"][-1]) < abs(res["amplitudes"][0])
    # pinned deficit band: 10% .. 25% low, i.e. outside the 15% acceptance bar
    assert -0.25 < res["rel_err_omega"] < -0.10
    # the decay rate is ~0.4-0.6 x the two-fluid viscous prediction 2 nu k^2
    assert 0.35 < res["gamma_meas"] / res["gamma_theory"] < 0.80
    # the phi field must stay bounded (no phase-field blow-up)
    assert res["phi_range"][0] > -0.1
    assert res["phi_range"][1] < 1.15


# ---------------------------------------------------------------------------
# Static surface tension from the interface energy integral
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_sigma_from_profile_recovers_input_on_analytic_profile() -> None:
    """On the analytic tanh profile the only error is the central-difference truncation."""
    phi = flat_interface_phi(ny=48, nx=8, nz=2, width=4.0).to(torch.float32)
    res = sigma_from_profile(phi, sigma_input=0.05, width=4.0)
    # measured 0.939: the central difference under-integrates (dphi/dn)^2 by ~6%
    assert res["sigma_ratio"] == pytest.approx(0.94, abs=0.05)
    assert res["width_measured"] == pytest.approx(2.16, abs=0.05)


@pytest.mark.physics
def test_sigma_from_profile_excludes_the_periodic_wrap() -> None:
    """Regression guard for the 3.9x bias a wrap-inclusive difference produces.

    In a periodic box the two rows straddling the wrap differ by ~1 in ``phi``.
    Including them in ``sum (dphi/dy)^2`` inflates the integral to 3.9x the analytic
    value (0.657 vs 0.167) -- a bug that would have made the static surface tension
    look 4x too large.  The function must stay within 10% of the truncation-only value.
    """
    phi = flat_interface_phi(ny=48, nx=8, nz=2, width=4.0).to(torch.float32)
    d_full = 0.5 * (torch.roll(phi, -1, dims=1) - torch.roll(phi, 1, dims=1))
    naive_integral = float((d_full * d_full).sum(dim=1).mean())
    res = sigma_from_profile(phi, sigma_input=0.05, width=4.0)
    assert naive_integral / res["integral"] > 3.0  # the artifact is real ...
    assert res["sigma_ratio"] < 1.05  # ... and the function must not contain it


def test_sigma_from_profile_rejects_thin_domain() -> None:
    with pytest.raises(ValueError, match="ny >= 5"):
        sigma_from_profile(torch.zeros(2, 4, 8), sigma_input=0.05, width=4.0)


@pytest.mark.slow
@pytest.mark.physics
def test_flat_interface_sigma_ratio_is_pinned() -> None:
    """Regression pin: the relaxed CAC interface carries ~1.30x the input sigma.

    The relaxed profile is 11.5% wider than ``tanh(2(y0-y)/W)`` at half maximum, so
    the interface energy integral exceeds ``sigma`` by 30%.  This is a documented
    deficit (see docs/capillary_wave_dispersion_20261109.md), not a validation: the
    acceptance bar is 15%.
    """
    torch.set_num_threads(1)
    res = run_flat_interface(sigma=0.05, width=4.0, nu=0.05, n_steps=3000)
    assert 1.10 < res["relaxed"]["sigma_ratio"] < 1.50
    assert res["relaxed"]["width_measured"] > res["relaxed"]["width_analytic"]
    assert abs(res["initial"]["sigma_ratio"] - 0.94) < 0.05
