"""Unit tests for the B6 (SUBOFF bare hull, Re=1000) acceptance gate.

The gate exists because shift m319 quoted a short/coarse transient
(``Cd_tot = 0.061495``, +46.43 %) as the benchmark result and blamed
``compile_status = eager_fallback``.  These tests pin the decision logic:

* the registered PASS configuration (L=80 / 12000 steps, window means
  0.041513 and 0.041428) must pass;
* the registered L=48 controls must be judged on accuracy only
  (-9.69 % standard formula -> FAIL_ACCURACY, i.e. a *real* accuracy
  failure is still reported as such);
* a coarse/short transient must be labelled COARSE_GRID / SHORT_RUN, i.e.
  "this number is not the benchmark result", never FAIL_ACCURACY;
* a drifting window mean on an eligible grid must be NOT_CONVERGED.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "suboff_re1000"
sys.path.insert(0, str(_BENCH_DIR))

from acceptance_gate import (  # noqa: E402
    CF_BLASIUS_RE1000,
    assess,
    banner,
    window_drift_pct,
)

# Values copied from benchmarks/verified/suboff_re1000/result.json (primary).
CD_TOT_L80_PASS = 0.04151330249437245
CD_TOT_L80_LAST500 = 0.04142754760397146
# m319 reported number.
CD_TOT_M319_TRANSIENT = 0.061495


def test_reference_value():
    assert CF_BLASIUS_RE1000 == pytest.approx(0.041995, abs=5e-7)


def test_registered_l80_passes():
    a = assess(
        L_cells=80,
        n_steps=12000,
        cd_total=CD_TOT_L80_PASS,
        cd_total_recent=CD_TOT_L80_LAST500,
    )
    assert a["verdict"] == "PASS"
    assert a["quotable_as_benchmark_result"] is True
    assert a["error_pct_vs_Blasius"] == pytest.approx(-1.147, abs=0.01)
    assert a["window_drift_pct"] < 1.0
    assert a["reasons"] == []


def test_registered_l48_standard_is_a_real_accuracy_failure():
    """Eligible grid/steps -> the -9.69 % control must surface as FAIL_ACCURACY."""
    a = assess(L_cells=48, n_steps=20000, cd_total=0.037925324, cd_total_recent=0.037925324)
    assert a["verdict"] == "FAIL_ACCURACY"
    assert a["error_pct_vs_Blasius"] == pytest.approx(-9.69, abs=0.05)
    assert a["quotable_as_benchmark_result"] is False
    assert any("> 3.0%" in r for r in a["reasons"])


def test_registered_l48_mix50_is_marginal_failure_not_convergence_error():
    a = assess(L_cells=48, n_steps=20000, cd_total=0.040663808, cd_total_recent=0.040663808)
    assert a["verdict"] == "FAIL_ACCURACY"  # -3.17 % is just outside 3 %
    assert a["error_pct_vs_Blasius"] == pytest.approx(-3.17, abs=0.05)


def test_m319_transient_is_never_reported_as_benchmark_result():
    """The +46 % number came from a coarse/short probe, not the verified config."""
    a = assess(L_cells=32, n_steps=1200, cd_total=CD_TOT_M319_TRANSIENT,
               cd_total_recent=CD_TOT_M319_TRANSIENT)
    assert a["verdict"] == "COARSE_GRID"
    assert a["quotable_as_benchmark_result"] is False
    assert a["error_pct_vs_Blasius"] > 40.0
    assert any("discretisation" in r for r in a["reasons"])


def test_short_run_on_fine_grid_is_flagged_as_transient():
    a = assess(L_cells=80, n_steps=800, cd_total=CD_TOT_M319_TRANSIENT,
               cd_total_recent=CD_TOT_M319_TRANSIENT)
    assert a["verdict"] == "SHORT_RUN"
    assert any("transient" in r for r in a["reasons"])


def test_drifting_window_on_eligible_grid_is_not_converged():
    a = assess(L_cells=80, n_steps=12000, cd_total=0.04468, cd_total_recent=0.041428)
    assert a["verdict"] == "NOT_CONVERGED"
    assert a["converged"] is False
    assert a["window_drift_pct"] > 1.0


def test_window_drift_formula():
    assert window_drift_pct(0.0415133, 0.0414275) == pytest.approx(0.2067, abs=1e-3)
    with pytest.raises(ValueError):
        window_drift_pct(0.0, 0.0414)


def test_banner_mentions_do_not_quote_for_ineligible_runs():
    a = assess(L_cells=32, n_steps=1200, cd_total=CD_TOT_M319_TRANSIENT)
    text = banner(a)
    assert "COARSE_GRID" in text
    assert "DO NOT quote" in text

    ok = assess(L_cells=80, n_steps=12000, cd_total=CD_TOT_L80_PASS,
                cd_total_recent=CD_TOT_L80_LAST500)
    assert "DO NOT quote" not in banner(ok)


def test_error_pct_sign_convention():
    """Negative error = under-prediction (verified run is -1.15 %)."""
    a = assess(L_cells=80, n_steps=12000, cd_total=CD_TOT_L80_PASS)
    assert a["error_pct_vs_Blasius"] < 0
    assert math.isclose(a["error_pct_vs_Blasius"], -1.1471467787904794, rel_tol=1e-6)