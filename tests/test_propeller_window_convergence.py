"""Contracts for windowed dynamic-geometry propeller campaign evidence."""
from __future__ import annotations

import pytest

from tensorlbm.propeller_benchmark import _summarize_windows


def test_window_summary_discards_warmup_samples_and_compares_final_windows() -> None:
    samples = [
        {"step": 5, "azimuth_deg": 9.0, "j": 0.4, "kt": 0.20, "kq": 0.030},
        {"step": 6, "azimuth_deg": 18.0, "j": 0.4, "kt": 0.22, "kq": 0.032},
        {"step": 7, "azimuth_deg": 27.0, "j": 0.4, "kt": 0.24, "kq": 0.034},
        {"step": 8, "azimuth_deg": 36.0, "j": 0.4, "kt": 0.26, "kq": 0.036},
    ]

    report = _summarize_windows(
        samples, window_steps=2, transient_discard_steps=4, convergence_rel_tol=0.2,
    )

    assert report["discarded_transient_samples"] == 0
    assert [window["n_samples"] for window in report["windows"]] == [2, 2]
    assert report["windows"][0]["kt_mean"] == pytest.approx(0.21)
    assert report["windows"][1]["kq_mean"] == pytest.approx(0.035)
    assert report["convergence"]["available"] is True
    assert report["convergence"]["window_converged"] is True
    assert report["convergence"]["kt_last_window_rel_change"] == pytest.approx(0.04 / 0.25)


def test_window_convergence_requires_both_kt_and_kq_to_be_strictly_below_tolerance() -> None:
    samples = [
        {"step": 1, "j": 0.4, "kt": 1.0, "kq": 1.0},
        {"step": 2, "j": 0.4, "kt": 1.0, "kq": 1.0},
        {"step": 3, "j": 0.4, "kt": 1.1, "kq": 1.3},
        {"step": 4, "j": 0.4, "kt": 1.1, "kq": 1.3},
    ]

    report = _summarize_windows(
        samples, window_steps=2, transient_discard_steps=0, convergence_rel_tol=0.1,
    )

    convergence = report["convergence"]
    assert convergence["available"] is True
    assert convergence["kt_last_window_rel_change"] == pytest.approx(0.1 / 1.1)
    assert convergence["kq_last_window_rel_change"] == pytest.approx(0.3 / 1.3)
    assert convergence["window_converged"] is False


def test_window_summary_is_explicitly_unavailable_with_one_window() -> None:
    report = _summarize_windows(
        [{"step": 1, "azimuth_deg": 0.0, "j": 0.2, "kt": 0.1, "kq": 0.01}],
        window_steps=4,
        transient_discard_steps=0,
        convergence_rel_tol=0.1,
    )

    assert report["convergence"] == {
        "available": False,
        "window_converged": False,
        "reason": "fewer_than_two_complete_windows",
        "kt_rel_tol": 0.1,
        "kq_rel_tol": 0.1,
    }
