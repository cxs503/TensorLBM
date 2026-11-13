"""Unit + smoke tests for tensorlbm.oscillatory_cylinder_inertia.

The physics gate is in docs/XVAL_LBM_HORIZONTAL_SURVEY_EMPTY.md (benchmark table);
these tests lock the *machinery*: geometry readings, harmonic orthogonality,
the coefficient normalisation, the same-mask potential reference, and that the
arm runs, is stable and produces two mutually consistent force estimators.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.oscillatory_cylinder_inertia import (
    CASES,
    circle_mask,
    geometry_readout,
    harmonic_decompose,
    inertia_coefficient,
    main,
    potential_reference,
    run_arm,
)


def test_geometry_readout_matches_circle():
    D = 16
    geo = geometry_readout(circle_mask(128, 128, D))
    assert abs(geo["D_area"] - D) / D < 0.03
    assert geo["D_area"] < geo["D_halfway"]
    perim_ideal = math.pi * D
    assert 0.75 < geo["perimeter_cells"] / perim_ideal < 1.30
    # halfway diameter must stay within one cell of the area diameter
    assert geo["D_halfway"] - geo["D_area"] < 1.5


def test_harmonic_decompose_recovers_synthetic():
    omega = 2 * math.pi / 100.0   # T = 100 steps -> integer-period windows
    t = np.arange(1, 401, dtype=float)
    y = 2.0 * np.cos(omega * t) + 0.5 * np.sin(omega * t) + 0.3
    d = harmonic_decompose(t, y, omega, t0=100.0, t1=400.0, nh=2)
    assert abs(d["cos"] - 2.0) < 1e-9
    assert abs(d["sin"] - 0.5) < 1e-9
    assert abs(d["dc"] - 0.3) < 1e-9
    assert d["resid_rel"] < 1e-9
    assert abs(d["amp"] - math.hypot(2.0, 0.5)) < 1e-9


def test_harmonic_window_orthogonality():
    """A large DC offset must not leak into the first harmonic (integer periods)."""
    omega = 2 * math.pi / 100.0
    t = np.arange(1, 501, dtype=float)
    y = 1.25 * np.cos(omega * t) - 0.75 * np.sin(omega * t) + 100.0
    d = harmonic_decompose(t, y, omega, t0=100.0, t1=400.0, nh=1)
    assert abs(d["cos"] - 1.25) < 1e-8
    assert abs(d["sin"] + 0.75) < 1e-8
    assert abs(d["dc"] - 100.0) < 1e-8


def test_harmonic_decompose_flags_nonlinear_residual():
    """Second-harmonic content must show up in the residual, not in C_I."""
    omega = 2 * math.pi / 100.0
    t = np.arange(1, 401, dtype=float)
    y = np.cos(omega * t) + 0.5 * np.cos(2 * omega * t)
    d1 = harmonic_decompose(t, y, omega, t0=100.0, t1=400.0, nh=1)
    d2 = harmonic_decompose(t, y, omega, t0=100.0, t1=400.0, nh=2)
    assert d1["resid_rel"] > 0.2
    assert d2["resid_rel"] < 1e-9
    assert abs(d1["cos"] - 1.0) < 1e-8


def test_inertia_coefficient_normalisation():
    # F_cos = 1, a0 = 1, D = 2 -> V = pi -> C_I = 1/pi
    assert inertia_coefficient(1.0, 1.0, 2.0) == pytest.approx(1.0 / math.pi)
    assert inertia_coefficient(2.0, 1.0, 2.0, rho=2.0) == pytest.approx(1.0 / math.pi)


def test_potential_reference_bounds_and_conventions():
    pot = potential_reference(64, 64, 8.0)
    # same-mask potential added mass: below the unbounded 1.0 up to a few %
    assert 0.85 < pot["c_a_area"] < 1.35
    assert pot["c_a_halfway"] < pot["c_a_area"]
    assert pot["added_mass"] > 0


def test_quick_arm_is_stable_and_self_consistent():
    result, trace = run_arm(CASES["quick"], threads=2)
    assert result["stability_ok"]
    assert len(trace["t"]) == result["numerics"]["n_steps"]
    # the box-mean acceleration must track the drive within 25 %
    assert 0.75 < result["achieved"]["a0_over_nominal"] < 1.25
    kpi = result["kpi"]
    for tag in ("E1_mem_halfway", "E1_mem_area", "E2_trac_halfway", "E2_trac_area"):
        assert math.isfinite(kpi[tag]["C_I"])
    assert kpi["E1_mem_area"]["C_I"] > 0.0
    # the two independent estimators must bracket each other within 60 %
    assert result["estimator_spread_pct"] < 0.6 * kpi["E1_mem_area"]["C_I"]
    # reference must be reported for the error budget
    assert result["reference"]["potential_same_mask_area"] > 0
    assert math.isfinite(result["err_inphase_pct_area"])


def test_cli_list_runs(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "base" in out and "quick" in out
