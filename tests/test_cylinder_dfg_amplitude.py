"""Regression tests for the DFG 2D-2 lift-amplitude estimator (mission m16827).

The published C_L-amplitude error of ``CASE-CYLINDER-DRAG-RE100`` used to be
``0.5*(max-min)`` over a window that (defect D3) covered the whole record,
start-up transient included.  These tests pin the two properties that make the
replacement estimator trustworthy and keep the old defects from coming back:

* a limit-cycle amplitude is recovered from a *contaminated* record (transient
  overshoot, single-sample spikes, white noise) where the peak-pick is inflated;
* an unsupported number is never emitted (too few periods -> NaN + status;
  fundamental/peak-pick disagreement -> ``unresolved``);
* ``CaseConfig`` refuses a run that is too short to host transient + window, so
  the averaging window can never again silently swallow the transient.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.validation import cylinder_dfg as cdfg

T = 1120.0  # shedding period in lattice steps (D=28, St=0.3, u_mean=0.0833)
AMP = 1.05  # limit-cycle amplitude to recover


def _limit_cycle(n_periods: float, amp: float = AMP, phase: float = 0.0) -> torch.Tensor:
    t = torch.arange(int(round(n_periods * T)), dtype=torch.float64)
    return amp * torch.sin(2.0 * math.pi * t / T + phase)


def test_pure_sinusoid_recovers_amplitude():
    sig = _limit_cycle(10.0)
    out = cdfg.limit_cycle_amplitude(sig, T, d_cells=28, u_mean_lb=0.0833)
    assert out["status"] == "ok"
    assert out["n_periods"] >= 9
    assert abs(out["amp"] - AMP) / AMP < 0.005
    assert abs(out["amp_peakpick_median"] - AMP) / AMP < 0.005
    assert abs(out["st_fit"] - 0.30) / 0.30 < 0.02


def test_transient_overshoot_inflates_peakpick_but_not_robust_estimator():
    """The m16827 root cause, in one assertion pair."""
    limit = _limit_cycle(12.0)
    # 3 periods of a 1.5x overshooting start-up, then the limit cycle
    transient = _limit_cycle(3.0, amp=AMP * 1.5)
    record = torch.cat([transient, limit])

    peakpick_full = 0.5 * float(record.max() - record.min())
    assert abs(peakpick_full - AMP) / AMP > 0.10  # old metric: >10 % too high

    out = cdfg.limit_cycle_amplitude(record[int(round(3 * T)):], T, d_cells=28,
                                     u_mean_lb=0.0833)
    assert out["status"] == "ok"
    assert abs(out["amp"] - AMP) / AMP < 0.005  # new metric: within 0.5 %


def test_single_spike_does_not_move_the_amplitude():
    sig = _limit_cycle(12.0)
    sig[int(5.25 * T)] += 0.5  # one contaminated sample, right on the crest
    peakpick = 0.5 * float(sig.max() - sig.min())
    out = cdfg.limit_cycle_amplitude(sig, T, d_cells=28, u_mean_lb=0.0833)
    assert peakpick > AMP * 1.05  # the old metric moves by >5 %
    assert abs(out["amp"] - AMP) / AMP < 0.005  # the reported amplitude does not
    assert out["status"] == "ok"


def test_noisy_record_amplitude_within_one_percent():
    """White noise biases every max-based statistic up by ~3 sigma; the
    fundamental fit must not move.  This is why the reported number is the
    fundamental, not the peak-pick."""
    g = torch.Generator().manual_seed(0)
    sig = _limit_cycle(14.0) + 0.03 * torch.randn(14 * int(T), generator=g, dtype=torch.float64)
    out = cdfg.limit_cycle_amplitude(sig, T, d_cells=28, u_mean_lb=0.0833)
    assert out["status"] == "ok"
    assert abs(out["amp"] - AMP) / AMP < 0.01
    assert out["amp_peakpick_median"] > AMP * 1.03   # peak statistic is biased up
    assert out["peakpick_excess_pct"] > 3.0
    assert out["iqr"] < 0.1


def test_insufficient_periods_never_emits_a_number():
    out = cdfg.limit_cycle_amplitude(_limit_cycle(3.5), T, d_cells=28, u_mean_lb=0.0833)
    assert out["status"] == "insufficient_periods"
    assert math.isnan(out["amp"])


def test_non_sinusoidal_record_raises_a_warning():
    """A strong 2nd harmonic makes the peak-pick exceed the fundamental by more
    than the declared 10 %: the value is still reported (the DFG quantity is the
    amplitude of that waveform) but never silently."""
    t = torch.arange(12 * int(T), dtype=torch.float64)
    sig = AMP * torch.sin(2 * math.pi * t / T) + 0.35 * AMP * torch.sin(4 * math.pi * t / T)
    out = cdfg.limit_cycle_amplitude(sig, T, d_cells=28, u_mean_lb=0.0833)
    assert out["peakpick_excess_pct"] > out["fundamental_tolerance_pct"]
    assert "warning" in out
    assert abs(out["amp"] - AMP) / AMP < 0.01


def test_config_rejects_run_shorter_than_transient_plus_window():
    """Defect D3 regression: 17 periods used to be accepted and silently turned
    the averaging window into 'the whole record'."""
    with pytest.raises(ValueError, match="transient"):
        cdfg.CaseConfig(D_cells=28, n_steps=19040)  # 17 periods < 5 + 12 needed


def test_config_window_bookkeeping_excludes_the_transient():
    cfg = cdfg.CaseConfig(D_cells=28, n_steps=30400)  # 19 periods
    assert cfg.transient_steps == int(round(5 * cfg.period_steps))
    assert cfg.avg_steps == int(round(12 * cfg.period_steps))
    assert cfg.window_start_step == cfg.n_steps - cfg.avg_steps + 1
    assert cfg.window_start_step > cfg.transient_steps


def test_frozen_series_rejudge_is_reproducible():
    """The documented re-judgment of the frozen D=28 record (artifacts/, no re-run)."""
    import json
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "benchmarks/verified/cylinder/re100_dfg/artifacts/rejudge_cl_amp.json")
    if not path.exists():
        pytest.skip("frozen rejudge artifact not present")
    d = json.loads(path.read_text())
    lvl = d["levels"]["D28"]
    assert lvl["cl_amp_new_status"] == "ok"
    assert lvl["cl_amp_n_periods"] >= 5
    # the published (peak-pick, transient-contaminated) value must be the larger one
    assert lvl["cl_amp_old_peakpick_all_window"] > lvl["cl_amp_new"]
    assert 1.05 < lvl["cl_amp_new"] < 1.30