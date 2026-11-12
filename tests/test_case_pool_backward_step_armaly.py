"""Tests for the CASE-BACKWARD-STEP-LBM pool case (TensorLBM).

Fast (unit) tests pin the reusable pieces in :mod:`tensorlbm.bfs_armaly`; the
last test grades the committed run artefacts against ``refs.json`` and is
skipped when the artefacts are absent (artefacts are produced by
``run_case.py`` and committed under ``artifacts/case_backward_step_armaly``).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.bfs_armaly import (  # noqa: E402
    ARMALY_EXPANSION_RATIO,
    ArmalyConfig,
    armaly_geometry,
    make_armaly_solid_mask,
    parabolic_inlet_profile,
    reattachment_from_wall_shear,
    run_armaly_case,
    wall_shear_from_near_wall,
    zero_crossing,
)

ART = REPO / "artifacts" / "case_backward_step_armaly"
REFS = json.loads((REPO / "benchmarks/case_pool/backward_step_armaly/refs.json").read_text())


def test_geometry_expansion_ratio_matches_experiment():
    """101/52 must equal the measured 10.1 mm / 5.2 mm to machine precision."""
    hi, s = armaly_geometry(1)
    assert (hi, s) == (52, 49)
    assert hi + s == 101
    assert abs((hi + s) / hi - ARMALY_EXPANSION_RATIO) < 1e-12
    # scale 2 keeps the ratio (grid-refinement ladder is geometrically similar)
    hi2, s2 = armaly_geometry(2)
    assert (hi2, s2) == (104, 98)
    assert abs((hi2 + s2) / hi2 - ARMALY_EXPANSION_RATIO) < 1e-12


def test_solid_mask_counts_walls_and_step():
    hi, s = 26, 25
    cfg = ArmalyConfig(re=100.0, hi=hi, s=s, u_bar=0.1)
    mask = make_armaly_solid_mask(cfg.ny, cfg.nx, hi, s, cfg.x_step)
    assert bool(mask[-1].all())                      # top wall
    assert bool(mask[0, cfg.x_step:].all())          # downstream lower wall
    assert bool(mask[: s + 1, : cfg.x_step].all())   # step block
    # fluid column heights: inlet h_in, downstream H
    col_in = int((~mask[:, 0]).sum())
    col_out = int((~mask[:, cfg.nx - 1]).sum())
    assert col_in == hi
    assert col_out == hi + s


def test_parabolic_profile_mean_and_no_slip():
    prof = parabolic_inlet_profile(52, 0.1).numpy()
    assert prof.shape == (52,)
    assert abs(prof.mean() - 0.1) < 1e-7        # discrete row mean is exact
    assert prof[0] < prof[26] and prof[-1] < prof[26]   # max at the centre
    assert prof[0] == pytest.approx(prof[-1], rel=1e-6)
    assert prof[0] < 0.25 * prof.max()


def test_wall_shear_derivative_is_exact_for_quadratic():
    """-2u1+3u2-u3 must be the exact wall derivative of any quadratic profile."""
    y = np.array([1.0, 2.0, 3.0])
    a, b, c = 0.7, -1.3, 2.1
    u = a + b * y + c * y * y
    exact = b + 2 * c * 0.5                     # dy/dx at the wall y = 1/2
    got = wall_shear_from_near_wall(np.array([u[0]]), np.array([u[1]]), np.array([u[2]]))
    assert float(got[0]) == pytest.approx(exact, rel=1e-12)
    # linear profile (plane Couette): derivative is the slope everywhere
    lin = 0.5 + 2.0 * y
    got_lin = wall_shear_from_near_wall(np.array([lin[0]]), np.array([lin[1]]), np.array([lin[2]]))
    assert float(got_lin[0]) == pytest.approx(2.0, rel=1e-12)


def test_zero_crossing_interpolates_between_columns():
    v = np.array([-1.0, -0.5, 0.5, 1.0])
    assert zero_crossing(v) == pytest.approx(1.5)
    assert math.isnan(zero_crossing(np.array([-1.0, -0.5])))
    # the 3-point reattachment formula subtracts the half-cell step face
    tau = np.array([0.0, 0.0, -2.0, -1.0, 1.0, 3.0])
    assert reattachment_from_wall_shear(tau, x_step=2, s=2) == pytest.approx((3.5 - 1.5) / 2.0)


def test_short_run_conserves_mass_flux_and_returns_finite_observable():
    cfg = ArmalyConfig(re=100.0, hi=10, s=9, u_bar=0.1, l_over_h=8.0, up_over_hi=3.0,
                       n_steps=2500, avg_from=0.5, avg_stride=100, check_interval=500)
    out = run_armaly_case(cfg)
    res = out["result"]
    assert res["n_averaged_samples"] >= 3
    assert res["mass_flux_rel_error"] < 0.10     # transient spin-up: flux imbalance decays toward 0
    assert res["rho_min"] > 0.9 and res["rho_max"] < 1.1
    assert math.isfinite(res["x1_over_s"])
    assert 0.5 < res["x1_over_s"] < 8.0


@pytest.mark.skipif(not (ART / "g2_re100" / "result.json").is_file(),
                    reason="pool artefacts not generated yet")
def test_committed_benchmark_meets_reference_tolerance():
    """Grade the committed artefacts exactly like a benchmark table would."""
    tol = REFS["adopted_numeric_reference"]["tolerance_pct"]
    points = REFS["adopted_numeric_reference"]["points"]
    checked = 0
    for re, ref in points.items():
        path = ART / f"g2_re{re}" / "result.json"
        if not path.is_file():
            continue
        got = json.loads(path.read_text())["result"]["x1_over_s"]
        err = 100.0 * (got - ref["x1_over_s"]) / ref["x1_over_s"]
        assert abs(err) <= tol, f"Re={re}: got {got:.3f} vs ref {ref['x1_over_s']} -> {err:+.1f}%"
        checked += 1
    assert checked >= 1
