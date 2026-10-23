"""Tests for ``tensorlbm.ai.uq_calibration`` — synthetic, CPU-only.

Correctness anchors:

* coverage / NLL / CRPS / z-normality on a *known* generative Gaussian
  (y = mu + sigma * N(0,1) with heterogeneous sigma);
* the closed-form temperature on the identity case (T = 1) and on an
  artificially mis-scaled sigma (T = the scale factor, coverage restored
  after ``apply_temperature``);
* guard ROC does not degrade: informative scores give AUC near 1,
  uninformative scores near 0.5, degenerate labels give ``nan`` without
  raising, and operating-point arithmetic is internally consistent;
* per-row verdicts are exactly the aggregate ``check`` on one row;
* protocol fields (``UQMETRIC_FIELDS`` / ``GUARDROC_FIELDS`` / confusion
  ``as_dict``) are stable and complete.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.ai.inference_service import EnvelopeMahalanobisGuardrail
from tensorlbm.ai.uq_calibration import (
    FLAG_ORDER,
    GUARDROC_FIELDS,
    UQMETRIC_FIELDS,
    apply_temperature,
    calibration_metrics,
    error_summary_by_flag,
    fit_temperature,
    grouped_calibration,
    guard_roc,
    p_error_below,
    roc_auc,
    row_verdicts,
    verdict_confusion,
)

RNG = np.random.default_rng(20260827)


def _hetero_gaussian(n: int, scale: float = 1.0, seed: int = 0) -> tuple[np.ndarray, ...]:
    """y = mu + scale * sigma * N(0,1) with heterogeneous mu/sigma."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.5, 5.0, size=n)
    sigma = rng.uniform(0.05, 0.5, size=n)
    y = mu + scale * sigma * rng.standard_normal(n)
    return y, mu, sigma


# ---------------------------------------------------------------------------
# calibration metrics on known Gaussians
# ---------------------------------------------------------------------------


class TestCalibrationMetrics:
    def test_perfectly_calibrated_coverage(self) -> None:
        y, mu, sigma = _hetero_gaussian(40000)
        m = calibration_metrics(y, mu, sigma)
        assert m.n == 40000
        assert m.coverage_68 == pytest.approx(0.6827, abs=0.012)
        assert m.coverage_95 == pytest.approx(0.95, abs=0.008)
        assert m.coverage_99 == pytest.approx(0.99, abs=0.004)
        # NLL of a true predictive density: 0.5*log(2*pi*e) in z units plus
        # the irreducible sigma term; per-point mean NLL - log(sigma) mean
        # must equal 0.5*log(2*pi*e) = 1.41894.
        z_nll = m.gaussian_nll - float(np.mean(np.log(sigma)))
        assert z_nll == pytest.approx(0.5 * math.log(2.0 * math.pi * math.e), abs=0.02)
        assert m.rms_z == pytest.approx(1.0, abs=0.03)
        assert m.mean_abs_z == pytest.approx(math.sqrt(2 / math.pi), abs=0.02)
        assert m.z_ks_pvalue > 0.05  # z is standard normal by construction
        assert abs(m.z_skew) < 0.08
        assert abs(m.z_excess_kurtosis) < 0.15

    def test_crps_matches_monte_carlo(self) -> None:
        # closed-form Gaussian CRPS vs E|X-y| - 0.5 E|X-X'| on fixed points
        rng = np.random.default_rng(7)
        mu = np.array([1.0, 2.0, -0.5, 3.25])
        sigma = np.array([0.3, 1.7, 0.55, 2.0])
        y = np.array([1.4, 0.2, -1.1, 4.0])
        m = calibration_metrics(y, mu, sigma)
        n = 400000
        x = rng.standard_normal((n, 4)) * sigma + mu
        term1 = np.mean(np.abs(x - y), axis=0)
        term2 = np.mean(np.abs(x - rng.standard_normal((n, 4)) * sigma - mu), axis=0) / 2.0
        assert m.crps_gauss == pytest.approx(float(np.mean(term1 - term2)), rel=0.02)

    def test_overconfident_sigma_loses_coverage(self) -> None:
        # sigma reported at half the true spread -> z ~ N(0, 4)
        y, mu, sigma = _hetero_gaussian(20000, scale=2.0, seed=1)
        m = calibration_metrics(y, mu, sigma)
        # true 95 % band needs 1.96*2 sigma: P(|N(0,1)| <= 0.98) ~= 0.673
        assert m.coverage_95 == pytest.approx(0.673, abs=0.01)
        assert m.rms_z == pytest.approx(2.0, abs=0.06)
        # z-space NLL = 0.5*log(2*pi) + 0.5*rms_z^2 ~= 2.92 (vs 1.42 calibrated)
        z_nll = m.gaussian_nll - float(np.mean(np.log(sigma)))
        assert z_nll == pytest.approx(0.5 * math.log(2.0 * math.pi) + 2.0, abs=0.06)

    def test_zero_sigma_rejected(self) -> None:
        y, mu, _ = _hetero_gaussian(10)
        with pytest.raises(ValueError, match="strictly positive"):
            calibration_metrics(y, mu, np.zeros_like(y))

    def test_protocol_fields(self) -> None:
        y, mu, sigma = _hetero_gaussian(50, seed=2)
        d = calibration_metrics(y, mu, sigma).as_dict()
        assert set(d) == set(UQMETRIC_FIELDS)
        assert all(math.isfinite(float(v)) for v in d.values())

    def test_grouped_matches_direct(self) -> None:
        y, mu, sigma = _hetero_gaussian(120, seed=3)
        idx = np.arange(120)
        groups = {"a": idx[:40], "b": idx[40:100], "c": idx[100:]}
        out = grouped_calibration(y, mu, sigma, groups)
        assert set(out) == {"a", "b", "c"}
        direct = calibration_metrics(y[groups["b"]], mu[groups["b"]], sigma[groups["b"]])
        assert out["b"].as_dict() == pytest.approx(direct.as_dict())

    def test_grouped_index_validation(self) -> None:
        y, mu, sigma = _hetero_gaussian(5, seed=4)
        with pytest.raises(ValueError, match="out of range"):
            grouped_calibration(y, mu, sigma, {"bad": [5]})


# ---------------------------------------------------------------------------
# temperature scaling
# ---------------------------------------------------------------------------


class TestTemperature:
    def test_identity_on_calibrated_data(self) -> None:
        y, mu, sigma = _hetero_gaussian(40000, seed=5)
        assert fit_temperature(y, mu, sigma) == pytest.approx(1.0, abs=0.02)
        # T = 1 must leave sigma (and hence every metric) unchanged
        s2 = apply_temperature(sigma, 1.0)
        np.testing.assert_array_equal(s2, sigma)

    def test_recovers_known_scale_factor(self) -> None:
        # true spread 3x the reported sigma -> closed-form T = 3
        y, mu, sigma = _hetero_gaussian(40000, scale=3.0, seed=6)
        t = fit_temperature(y, mu, sigma)
        assert t == pytest.approx(3.0, abs=0.06)
        fixed = apply_temperature(sigma, t)
        m = calibration_metrics(y, mu, fixed)
        assert m.coverage_95 == pytest.approx(0.95, abs=0.01)
        # and the NLL genuinely improved on this same half
        m_raw = calibration_metrics(y, mu, sigma)
        assert m.gaussian_nll < m_raw.gaussian_nll

    def test_fit_on_train_half_validates_on_heldout(self) -> None:
        y, mu, sigma = _hetero_gaussian(40000, scale=1.6, seed=7)
        half = y.size // 2
        t = fit_temperature(y[half:], mu[half:], sigma[half:])
        m = calibration_metrics(y[:half], mu[:half], apply_temperature(sigma[:half], t))
        assert t == pytest.approx(1.6, abs=0.06)
        assert m.coverage_95 == pytest.approx(0.95, abs=0.012)

    def test_invalid_temperature_rejected(self) -> None:
        y, mu, sigma = _hetero_gaussian(4, seed=8)
        with pytest.raises(ValueError, match="positive"):
            apply_temperature(sigma, 0.0)


# ---------------------------------------------------------------------------
# guard ROC
# ---------------------------------------------------------------------------


class TestGuardRoc:
    def test_informative_scores_auc_near_one(self) -> None:
        rng = np.random.default_rng(9)
        scores = rng.uniform(0, 1, 2000)
        errors = np.where(scores > 0.5, 0.2, 0.005) + rng.uniform(0, 0.002, 2000)
        reports = guard_roc(scores, errors, error_thresholds=[0.1])
        assert len(reports) == 1
        assert reports[0].auc > 0.99
        assert reports[0].n_large == int((errors > 0.1).sum())
        # operating sweep is monotone: lower score cut flags more points
        pts = reports[0].points
        assert pts[0].n_flagged >= pts[-1].n_flagged

    def test_uninformative_scores_auc_near_half(self) -> None:
        rng = np.random.default_rng(10)
        scores = rng.uniform(0, 1, 4000)
        labels = rng.random(4000) < 0.3
        errors = np.where(labels, 0.5, 0.001)
        assert roc_auc(scores, errors > 0.1) == pytest.approx(0.5, abs=0.05)

    def test_perfect_discriminator(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        errors = np.array([0.001, 0.002, 0.001, 0.5, 0.6, 0.4])
        rep = guard_roc(scores, errors, error_thresholds=[0.1], score_thresholds=[0.5])[0]
        assert rep.auc == pytest.approx(1.0)
        pt = rep.points[0]
        assert pt.tpr == pytest.approx(1.0)
        assert pt.fpr == pytest.approx(0.0)
        assert pt.precision == pytest.approx(1.0)

    def test_degenerate_labels_are_nan_not_crash(self) -> None:
        scores = np.array([0.1, 0.5, 0.9])
        errors = np.array([0.001, 0.002, 0.001])
        rep = guard_roc(scores, errors, error_thresholds=[0.1])[0]
        assert math.isnan(rep.auc)
        assert rep.n_large == 0
        assert math.isnan(rep.points[0].tpr)  # capture undefined, not fake zero

    def test_ties_get_midranks(self) -> None:
        # all-tied scores: AUC exactly 0.5 whatever the labels
        scores = np.full(8, 0.5)
        labels = np.array([True, False, True, False, True, False, True, False])
        assert roc_auc(scores, labels) == pytest.approx(0.5)
        # hand-computed tie case: pairs (pos,neg) = 2; one win + one tie
        scores2 = np.array([0.5, 0.5, 0.8])
        labels2 = np.array([False, True, True])
        assert roc_auc(scores2, labels2) == pytest.approx(0.75)

    def test_operating_point_arithmetic(self) -> None:
        rng = np.random.default_rng(11)
        scores = rng.uniform(0, 1, 500)
        errors = rng.uniform(0, 0.3, 500)
        rep = guard_roc(scores, errors, error_thresholds=[0.1])[0]
        for pt in rep.points:
            if pt.n_flagged and pt.n_large:
                # precision == tpr * n_large / n_flagged
                assert pt.precision == pytest.approx(pt.tpr * pt.n_large / pt.n_flagged)
            assert 0.0 <= pt.tpr <= 1.0 and 0.0 <= pt.fpr <= 1.0
        assert rep.n_large + rep.n_small == 500

    def test_protocol_fields(self) -> None:
        rep = guard_roc(
            np.array([0.1, 0.6, 0.9, 0.4]),
            np.array([0.01, 0.2, 0.3, 0.02]),
            error_thresholds=[0.1],
        )[0]
        d = rep.as_dict()
        assert set(d) == set(GUARDROC_FIELDS)
        assert d["n_large"] == 2 and d["n_small"] == 2


# ---------------------------------------------------------------------------
# verdict semantics
# ---------------------------------------------------------------------------


def _synthetic_guard() -> tuple[EnvelopeMahalanobisGuardrail, np.ndarray]:
    rng = np.random.default_rng(12)
    feats = rng.normal(0.0, 1.0, size=(200, 4))
    return EnvelopeMahalanobisGuardrail(feats), feats


class TestVerdicts:
    def test_row_verdicts_match_single_row_check(self) -> None:
        guard, feats = _synthetic_guard()
        rows = np.vstack(
            [feats[:20], feats.mean(axis=0), feats.mean(axis=0) + 25.0]
        )  # in-corpus rows + centre + far outlier
        v = row_verdicts(guard, rows)
        assert v.shape == (rows.shape[0],)
        for i in range(rows.shape[0]):
            assert v[i] == guard.check(rows[i : i + 1]).flag
        assert v[20] == "ok"  # corpus centre
        assert v[21] in ("review", "reject")  # 25-sigma outlier

    def test_verdict_confusion_partition_and_rows(self) -> None:
        flags = np.array(["ok", "ok", "ok", "review", "reject", "ok", "review", "ok"])
        errors = np.array([0.002, 0.005, 0.012, 0.06, 0.3, 0.001, 0.15, 0.004])
        conf = verdict_confusion(flags, errors, band_edges=(0.01, 0.05, 0.15))
        assert conf.n == 8
        assert conf.flags == tuple(f for f in FLAG_ORDER if f in set(flags.tolist()))
        assert conf.bands == ("<1%", "1%-5%", "5%-15%", ">=15%")
        ok_row = conf.row("ok")
        # ok errors 0.002/0.005/0.012/0.001/0.004: four below 1 %, one in 1-5 %
        assert ok_row["<1%"] == 4 and ok_row["1%-5%"] == 1
        assert sum(ok_row.values()) == 5
        assert sum(conf.row("review").values()) == 2
        assert sum(conf.row("reject").values()) == 1
        d = conf.as_dict()
        assert set(d) == {"flags", "bands", "counts", "n", "errors_by_flag"}
        assert np.asarray(d["counts"]).sum() == 8

    def test_band_edges_validated(self) -> None:
        flags = np.array(["ok", "ok"])
        errors = np.array([0.01, 0.02])
        with pytest.raises(ValueError, match="ascending"):
            verdict_confusion(flags, errors, band_edges=(0.05, 0.01))

    def test_unknown_flag_rejected(self) -> None:
        with pytest.raises(ValueError, match="ok/review/reject"):
            verdict_confusion(np.array(["maybe", "ok"]), np.array([0.01, 0.02]), band_edges=(0.05,))

    def test_p_error_below_and_summary(self) -> None:
        flags = np.array(["ok"] * 6 + ["review"] * 2)
        errors = np.array([0.001, 0.002, 0.004, 0.008, 0.02, 0.03, 0.2, 0.4])
        assert p_error_below(flags, errors, "ok", 0.005) == pytest.approx(3 / 6)
        assert p_error_below(flags, errors, "ok", 0.025) == pytest.approx(5 / 6)
        assert p_error_below(flags, errors, "review", 0.05) == pytest.approx(0.0)
        s = error_summary_by_flag(flags, errors)
        assert s["ok"]["n"] == 6.0
        assert s["review"]["q50"] == pytest.approx(0.3)
        with pytest.raises(ValueError, match="no points"):
            p_error_below(flags, errors, "reject", 0.05)
