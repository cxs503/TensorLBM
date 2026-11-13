"""Tests for the XVAL vs OSS `vivsim` continuous-error-print-vs-DOI-refs surface.

The OSS practice (``examples/2d/flow_pass_cylinder.py:207-208``) prints
``err%`` against ``CD_REF/CL_REF`` but never asserts; the ported surface must be
machine-checkable, so the post-processing calibers and the gate are pinned here
against synthetic series with known answers, *independently of any solver run*.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "benchmarks" / "xval" / "xval_vivsim_cylinder_re200.py"


def _load():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("xval_vivsim_cyl", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


X = _load()


def _sine(t, amp, freq, phase=0.0, offset=0.0):
    return offset + amp * np.sin(2.0 * math.pi * freq * t + phase)


# ------------------------------------------------------------------ provenance
def test_reference_values_are_the_oss_ones():
    """The two reference numbers must be the OSS values, with the DOI lineage."""
    assert X.CD_REF == 1.389
    assert X.CL_REF == 0.718
    assert len(X.CD_REFS_DOI) == 3
    assert "vivsim" in X.VIVSIM_SRC and "fb84b12" in X.VIVSIM_SRC
    assert X.TOL_PCT == 15.0  # owner precision standard


# ------------------------------------------------------------ OSS calibers
def test_vivsim_calibers_selects_last_half_only():
    """cd_mean/cl_peak use the second half of the record, as the OSS does."""
    n = 400
    cd = np.full(n, 10.0)
    cl = np.full(n, 5.0)
    cd[: n // 2] = 1.0
    cl[: n // 2] = 1.0
    out = X.vivsim_calibers(cd, cl)
    assert out["cd_mean_last_half"] == pytest.approx(10.0)
    assert out["cl_peak_last_half"] == pytest.approx(5.0)
    assert out["window"] == [n // 2, n]


def test_vivsim_cl_peak_amplitude_on_a_pure_sine():
    """max|Cl| of a sine of amplitude A is A -- the caliber is peak amplitude."""
    t = np.arange(0, 200.0, 1.0)
    A, f = 0.718, 0.01
    cl = _sine(t, A, f)
    out = X.vivsim_calibers(np.full(t.size, 1.389), cl)
    assert out["cl_peak_last_half"] == pytest.approx(A, rel=5e-3)


def test_vivsim_cl_peak_is_single_sample_sensitive():
    """Documented defect of the OSS caliber: one glitch inflates max|Cl|.

    This is the reason our own surface reports a frequency-domain amplitude too;
    the assertion pins the defect so it cannot silently disappear.
    """
    t = np.arange(0, 200.0, 1.0)
    cl = _sine(t, 0.5, 0.01)
    cl[-5] = 1.3  # single-sample numerical glitch in the averaging window
    out = X.vivsim_calibers(np.full(t.size, 1.389), cl)
    assert out["cl_peak_last_half"] > 1.25  # 0.5 true amplitude -> >2.5x inflated
    amp, _ = X.harmonic_amplitude(t[100:], cl[100:])
    # least-squares amplitude is *bounded* by the glitch (bias ~O(1/N)); it does
    # not inherit the glitch the way max|Cl| does (measured bias here ~-19%)
    assert abs(amp - 0.5) < 0.15


# ------------------------------------------------------- our own calibers
def test_harmonic_amplitude_recovers_amplitude_and_frequency():
    rng = np.random.default_rng(0)
    t = np.arange(0, 2000.0, 10.0)
    A, f = 0.66, 0.0025
    cl = _sine(t, A, f) + rng.normal(0.0, 0.02, t.size)
    amp, f0 = X.harmonic_amplitude(t, cl)
    assert amp == pytest.approx(A, rel=2e-2)
    assert f0 == pytest.approx(f, rel=1e-3)


def test_harmonic_amplitude_matches_peak_of_clean_sine():
    t = np.arange(0, 5000.0, 5.0)  # 5 shedding periods
    cl = _sine(t, 0.718, 0.001)
    amp, _ = X.harmonic_amplitude(t, cl)
    peak = float(np.max(np.abs(cl)))
    assert amp == pytest.approx(peak, rel=1e-3)


def test_per_period_half_peak_to_peak():
    t = np.arange(0, 9000.0, 5.0)
    A, f = 0.4, 0.0005  # T = 2000 steps, 4 whole periods
    cl = _sine(t, A, f)
    med, iqr = X.per_period_half_pp(t, cl, f)
    assert med == pytest.approx(A, rel=1e-2)
    assert iqr == pytest.approx(0.0, abs=1e-2)


def test_st_from_ours_calibers():
    """St = f*D/U must come out of the fitted frequency."""
    t = np.arange(0, 8000.0, 10.0)
    D, u, st = 24, 0.1, 0.197
    f_lattice = st * u / D
    cl = _sine(t, 0.66, f_lattice)
    out = X.ours_calibers(t, np.full(t.size, 1.34), cl, u, D)
    assert out["st"] == pytest.approx(st, rel=2e-2)


# ------------------------------------------------------------------ gate
def _res(cd, cl_peak, cl_harm=None):
    return {
        "vivsim": {"cd_mean_last_half": cd, "cl_peak_last_half": cl_peak},
        "ours": {"cl_amp_harmonic": cl_peak if cl_harm is None else cl_harm, "st": 0.197},
    }


def test_gate_passes_inside_tolerance():
    v = X.verdict(_res(1.389 * 1.10, 0.718 * 0.90))
    assert v["pass"] and v["pass_cd"] and v["pass_cl"]
    assert v["err_pct_cd_mean"] == pytest.approx(10.0, abs=1e-6)


def test_gate_fails_outside_tolerance_and_at_the_boundary():
    v = X.verdict(_res(1.389 * 1.16, 0.718 * 1.01))
    assert not v["pass"] and not v["pass_cd"] and v["pass_cl"]
    # strict inequality: at/above the 15% boundary the gate must not pass
    ref = X.CD_REF
    above = ref * 1.15
    for _ in range(3):
        above = np.nextafter(above, ref * 2.0)
    v_above = X.verdict(_res(above, 0.718))
    assert v_above["err_pct_cd_mean"] > X.TOL_PCT and not v_above["pass_cd"]
    below = ref * 1.15
    for _ in range(3):
        below = np.nextafter(below, ref)
    v_below = X.verdict(_res(below, 0.718))
    assert v_below["err_pct_cd_mean"] < X.TOL_PCT and v_below["pass_cd"]


def test_gate_reports_harmonic_caliber_separately():
    """A caliber-driven disagreement must stay visible, not be averaged away."""
    v = X.verdict(_res(1.389, 0.718 * 1.30, cl_harm=0.718 * 0.98))
    assert not v["pass_cl"]
    assert abs(v["err_pct_cl_amp_harmonic_vs_CL_REF"]) < 15.0