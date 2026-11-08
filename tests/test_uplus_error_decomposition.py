"""Tests for the ``u+`` statistical error budget and AMR conservation ledgers.

The estimators are checked against closed-form answers for a first-order
autoregressive (AR(1)) process, which has ``rho(k) = phi**k``,
``T_int = 0.5 (1 + phi) / (1 - phi)`` and sample-mean variance
``sigma**2 (1 + phi) / (n (1 - phi))``.  The AMR ledger is checked against an
exactly conserved case (uniform flow) and against the analytical 2-D
Taylor-Green vortex.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorlbm.statistics_error_budget import (
    autocorrelation_function,
    batch_means_estimate,
    integrated_autocorrelation_time,
    peak_relative_ci_width,
    profile_statistics,
    wall_units,
)
from tensorlbm.taylor_green_amr import (
    TaylorGreenAMRConfig,
    TaylorGreenExact,
    initial_populations,
    run_taylor_green_amr,
    uniform_fine_initial,
)
from tensorlbm.uplus_error_decomposition import (
    ErrorDecompositionReport,
    arbitrate_domain_length,
    band_max_relative_difference,
    decompose_error_budget,
    load_error_decomposition,
    log_law_band,
    sgs_negative_control,
    statistical_error_verdict,
    write_error_decomposition,
)

PHI = 0.85


def ar1_series(n: int, *, phi: float = PHI, seed: int = 0, sigma: float = 1.0) -> np.ndarray:
    """Stationary AR(1) realisation with unit stationary variance."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) * sigma * math.sqrt(1.0 - phi**2)
    out = np.empty(n)
    out[0] = rng.standard_normal()
    for index in range(1, n):
        out[index] = phi * out[index - 1] + noise[index]
    return out


class TestAutocorrelation:
    def test_zero_lag_is_unity(self) -> None:
        rho = autocorrelation_function(ar1_series(4000))
        assert rho[0] == pytest.approx(1.0, abs=1e-12)

    def test_ar1_matches_geometric_decay(self) -> None:
        rho = autocorrelation_function(ar1_series(40000, seed=3), max_lag=12)
        for lag in (1, 2, 4, 8):
            assert rho[lag] == pytest.approx(PHI**lag, abs=0.06)

    def test_white_noise_decorrelates(self) -> None:
        rng = np.random.default_rng(7)
        rho = autocorrelation_function(rng.standard_normal((20000, 1)))
        assert abs(rho[1, 0]) < 0.03
        assert abs(rho[5, 0]) < 0.03

    def test_constant_column_rejected(self) -> None:
        with pytest.raises(ValueError):
            autocorrelation_function(np.ones((100, 1)))


class TestIntegratedTime:
    def test_ar1_theory(self) -> None:
        expected = 0.5 * (1.0 + PHI) / (1.0 - PHI)
        estimate = integrated_autocorrelation_time(ar1_series(200000, seed=11))
        assert estimate.t_int == pytest.approx(expected, rel=0.25)
        assert estimate.converged is True
        assert estimate.n_effective < 200000 / (2 * expected) * 2.0

    def test_ratio_scales_with_record_length(self) -> None:
        short = integrated_autocorrelation_time(ar1_series(4000, seed=5))
        long = integrated_autocorrelation_time(ar1_series(40000, seed=5))
        assert long.ratio > 5.0 * short.ratio

    def test_short_record_flags_non_convergence(self) -> None:
        # phi ~ 0.995 => T_int ~ 200 samples, far longer than the record.
        estimate = integrated_autocorrelation_time(ar1_series(300, phi=0.995, seed=2))
        assert estimate.converged is False
        assert "lower bound" in estimate.notes

    def test_initial_positive_method_truncates(self) -> None:
        estimate = integrated_autocorrelation_time(
            ar1_series(50000, seed=9),
            method="initial_positive",
        )
        assert estimate.method == "initial_positive"
        assert estimate.t_int > 0.5
        assert estimate.converged is True

    def test_invalid_method_rejected(self) -> None:
        with pytest.raises(ValueError):
            integrated_autocorrelation_time(ar1_series(100), method="nope")


class TestBatchMeans:
    def test_correlation_inflates_error(self) -> None:
        estimate = batch_means_estimate(ar1_series(40000, seed=13), n_batches=20)
        assert estimate.variance_inflation is not None
        # sqrt(2 T_int) = sqrt((1+phi)/(1-phi)) ~ 3.5 for phi = 0.85
        assert estimate.variance_inflation == pytest.approx(
            math.sqrt((1 + PHI) / (1 - PHI)),
            rel=0.35,
        )

    def test_interval_coverage(self) -> None:
        """The 95% interval must cover the true mean in most replicates."""
        covered = 0
        trials = 60
        for seed in range(trials):
            estimate = batch_means_estimate(ar1_series(6000, seed=seed, phi=0.6))
            low, high = estimate.ci
            covered += int(low <= 0.0 <= high)
        assert covered / trials >= 0.85

    def test_mean_and_dof(self) -> None:
        estimate = batch_means_estimate(ar1_series(4000, seed=4), n_batches=10)
        assert estimate.n_batches == 10
        assert estimate.dof == 9
        assert estimate.batch_length == 400
        assert estimate.ci_half_width > 0.0
        assert estimate.ci[0] < estimate.mean < estimate.ci[1]

    def test_rejects_too_many_batches(self) -> None:
        with pytest.raises(ValueError):
            batch_means_estimate(ar1_series(50), n_batches=100)

    def test_zero_mean_gives_infinite_relative_width(self) -> None:
        estimate = batch_means_estimate(np.zeros((100, 1)))
        assert math.isinf(estimate.relative_ci_width)


class TestProfileStatistics:
    def test_columns_match_batch_means(self) -> None:
        rng = np.random.default_rng(21)
        samples = rng.standard_normal((4000, 3))
        stats = profile_statistics(samples, n_batches=8)
        for column in range(3):
            reference = batch_means_estimate(samples[:, column : column + 1], n_batches=8)
            assert stats.mean[column] == pytest.approx(reference.mean)
            assert stats.ci_half_width[column] == pytest.approx(reference.ci_half_width)

    def test_ledger_rows_and_json(self) -> None:
        rng = np.random.default_rng(22)
        stats = profile_statistics(rng.standard_normal((2000, 2)), n_batches=10)
        rows = stats.ledger(labels=["a", "b"])
        assert len(rows) == 2
        assert rows[0]["label"] == "a"
        assert set(("t_int", "t_obs", "obs_over_t_int", "converged")) <= set(rows[0])
        payload = stats.to_dict()
        assert json.loads(json.dumps(payload))["n_samples"] == 2000

    def test_peak_relative_ci_width_mask(self) -> None:
        uplus = np.array([1.0, 5.0, 10.0, 20.0])
        half = np.array([0.1, 0.5, 0.1, 0.1])
        assert peak_relative_ci_width(uplus, half) == pytest.approx(0.2)
        assert peak_relative_ci_width(uplus, half, mask=np.array([True, False, False, False])) == pytest.approx(0.2)


class TestWallUnits:
    def test_conversion(self) -> None:
        u_plus, y_plus = wall_units(
            np.array([0.1, 0.2]),
            np.array([0.5, 1.5]),
            u_tau=0.02,
            nu=0.001,
        )
        assert y_plus.tolist() == pytest.approx([10.0, 30.0])
        assert u_plus.tolist() == pytest.approx([5.0, 10.0])

    def test_rejects_non_positive_viscosity(self) -> None:
        with pytest.raises(ValueError):
            wall_units(np.zeros(2), np.zeros(2), u_tau=0.02, nu=0.0)

    def test_rejects_negative_distance(self) -> None:
        with pytest.raises(ValueError):
            wall_units(np.zeros(2), np.array([-1.0, 1.0]), u_tau=0.02, nu=0.001)


class TestLogLawBand:
    def test_default_band(self) -> None:
        y_plus = np.array([5.0, 30.0, 60.0, 90.0, 120.0])
        mask = log_law_band(y_plus, 180.0)
        assert mask.tolist() == [False, True, True, True, False]

    def test_empty_band_rejected(self) -> None:
        with pytest.raises(ValueError):
            log_law_band(np.array([1.0, 2.0]), 20.0)

    def test_relative_difference_numbers(self) -> None:
        y_plus = np.array([40.0, 60.0, 80.0])
        reference = np.array([13.0, 14.0, 15.0])
        candidate = reference * 1.05
        mask = np.ones(3, dtype=bool)
        stats = band_max_relative_difference(reference, candidate, y_plus, mask)
        assert stats["max_relative_difference"] == pytest.approx(0.05)
        assert stats["band_points"] == 3


class TestVerdicts:
    def _profiles(self, scale_short: float, scale_control: float) -> tuple[np.ndarray, ...]:
        y_plus = np.linspace(40.0, 90.0, 11)
        long_profile = 2.44 * np.log(y_plus) + 5.2
        short_profile = long_profile * scale_short
        control = long_profile * scale_control
        return y_plus, long_profile, short_profile, control

    def test_domain_arbitration_pass_and_fail(self) -> None:
        y_plus, long_profile, short_profile, _ = self._profiles(1.005, 1.0)
        passed = arbitrate_domain_length(short_profile, long_profile, y_plus, 180.0)
        assert passed.passed is True
        assert passed.value < 0.01
        _, long_bad, short_bad, _ = self._profiles(1.05, 1.0)
        failed = arbitrate_domain_length(short_bad, long_bad, y_plus, 180.0)
        assert failed.passed is False
        assert failed.value == pytest.approx(0.05, rel=0.02)

    def test_sgs_control_requires_measurable_effect(self) -> None:
        y_plus, long_profile, _, _ = self._profiles(1.0, 1.0)
        no_effect = sgs_negative_control(long_profile, long_profile, y_plus, 180.0)
        assert no_effect.passed is False
        y_plus, long_profile, _, control = self._profiles(1.0, 1.2)
        effect = sgs_negative_control(control, long_profile, y_plus, 180.0)
        assert effect.passed is True
        assert effect.value == pytest.approx(0.2, rel=0.02)

    def test_statistical_verdict_needs_ratio(self) -> None:
        y_plus = np.linspace(40.0, 90.0, 11)
        u_plus = 2.44 * np.log(y_plus) + 5.2
        half = 0.001 * u_plus
        ok = statistical_error_verdict(u_plus, half, y_plus, 180.0, obs_over_t_int=60.0)
        assert ok.passed is True
        bad_ratio = statistical_error_verdict(u_plus, half, y_plus, 180.0, obs_over_t_int=5.0)
        assert bad_ratio.passed is False
        assert bad_ratio.detail["ratio_satisfied"] is False


class TestErrorBudget:
    def test_quadrature_combination(self) -> None:
        budget = decompose_error_budget({"a": 0.03, "b": 0.04, "c": None})
        assert budget["quadrature_total"] == pytest.approx(0.05)
        assert budget["unmeasured_components"] == ["c"]
        assert budget["dominant_component"] == "b"
        assert budget["dominant_share"] == pytest.approx(0.64)

    def test_report_roundtrip(self, tmp_path) -> None:
        report = ErrorDecompositionReport(case={"re_tau": 180.0})
        y_plus = np.linspace(40.0, 90.0, 11)
        u_plus = 2.44 * np.log(y_plus) + 5.2
        report.add_verdict(
            statistical_error_verdict(u_plus, 0.001 * u_plus, y_plus, 180.0, obs_over_t_int=80.0),
        )
        report.add_verdict(
            arbitrate_domain_length(u_plus * 1.004, u_plus, y_plus, 180.0),
        )
        report.rebuild_budget()
        payload = report.to_dict()
        assert payload["all_passed"] is True
        target = write_error_decomposition(tmp_path / "u+_error_decomposition.json", payload)
        reloaded = load_error_decomposition(target)
        assert reloaded["budget"]["quadrature_total"] == pytest.approx(
            payload["budget"]["quadrature_total"],
        )
        assert len(reloaded["verdicts"]) == 2


class TestTaylorGreenExact:
    def test_decay_factor(self) -> None:
        exact = TaylorGreenExact(nx=32, ny=32, amplitude=0.03, nu=0.02)
        k2 = exact.kx**2 + exact.ky**2
        assert exact.decay(5.0) == pytest.approx(math.exp(-exact.nu * k2 * 5.0))

    def test_velocity_is_incompressible(self) -> None:
        exact = TaylorGreenExact(nx=32, ny=32, amplitude=0.03, nu=0.02)
        ux, uy = exact.velocity(0.0)
        dux_dx = np.gradient(ux.numpy(), axis=1)
        duy_dy = np.gradient(uy.numpy(), axis=0)
        periodic = np.roll(ux.numpy(), -1, axis=1) - ux.numpy()
        assert np.abs(dux_dx + duy_dy).max() < 5e-3
        assert np.abs(periodic).max() < 0.02

    def test_initial_field_has_exact_amplitude(self) -> None:
        config = TaylorGreenAMRConfig(steps=1, amplitude=0.02)
        populations = initial_populations(config)
        assert populations.dtype.is_floating_point
        from tensorlbm.d3q19 import macroscopic3d

        rho, ux, uy, uz = macroscopic3d(populations)
        assert float(rho.min()) == pytest.approx(1.0)
        assert float(ux.abs().max()) == pytest.approx(0.02, rel=1e-6)
        assert float(uz.abs().max()) < 1e-15


@pytest.fixture(scope="module")
def short_run():  # noqa: ANN201
    config = TaylorGreenAMRConfig(steps=6, initialization="exact")
    return run_taylor_green_amr(config)


class TestAMRConservation:

    def test_mass_conserved_to_machine_precision(self, short_run) -> None:  # noqa: ANN001
        assert short_run.metrics["relative_mass_drift_max"] < 1e-12

    def test_interface_flux_balance_to_machine_precision(self, short_run) -> None:  # noqa: ANN001
        assert short_run.metrics["interface_residual_over_raw"] < 1e-13

    def test_interface_momentum_balance(self, short_run) -> None:  # noqa: ANN001
        assert short_run.metrics["interface_momentum_residual_relative"] < 1e-10
        assert short_run.metrics["interface_momentum_applied_ratio_min"] == pytest.approx(1.0)

    def test_no_limited_directions(self, short_run) -> None:  # noqa: ANN001
        assert short_run.metrics["interface_limited_directions_total"] == 0

    def test_uniform_flow_preserved(self) -> None:
        """A uniform flow must survive the interface untouched."""
        import torch

        from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
        from tensorlbm.static_block_amr import StaticBlockAMR3D, StaticBlockAMRConfig
        from tensorlbm.taylor_green_amr import _advance

        config = TaylorGreenAMRConfig(steps=12)
        nz, ny, nx = config.shape_zyx
        rho = torch.ones((nz, ny, nx), dtype=torch.float64)
        ux = torch.full_like(rho, 0.02)
        zero = torch.zeros_like(rho)
        populations = equilibrium3d(rho, ux, zero, zero)
        solver = StaticBlockAMR3D(
            populations.clone(),
            StaticBlockAMRConfig(config.box, tau_coarse=config.tau_coarse),
        )
        for _ in range(12):
            solver.step(_advance)
        _, ux_amr, uy_amr, _ = macroscopic3d(solver.coarse_f)
        assert float((ux_amr - 0.02).abs().max()) < 1e-14
        assert float(uy_amr.abs().max()) < 1e-14

    def test_reflux_off_leaks_mass(self) -> None:
        """The conservation ledger must be able to fail."""
        config = TaylorGreenAMRConfig(steps=6, initialization="exact", reflux=False)
        result = run_taylor_green_amr(config)
        assert result.metrics["relative_mass_drift_max"] > 1e-12

    def test_config_rejects_bad_box(self) -> None:
        with pytest.raises(ValueError):
            TaylorGreenAMRConfig(steps=2, box_x=(1, 31)).validate()

    def test_config_rejects_bad_initialization(self) -> None:
        with pytest.raises(ValueError):
            TaylorGreenAMRConfig(steps=2, initialization="magic").validate()

    def test_uniform_fine_initial_shape(self) -> None:
        config = TaylorGreenAMRConfig(steps=1)
        coarse = initial_populations(config)
        fine = uniform_fine_initial(config)
        assert fine.shape == (coarse.shape[0], 2 * coarse.shape[1], 2 * coarse.shape[2], 2 * coarse.shape[3])