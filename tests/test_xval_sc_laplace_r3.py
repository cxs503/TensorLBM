"""R3 guards for XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION.

R2 left the pre-registered P1..P5 numerical legs NOT_RUN because a single
monolithic job never finished in its queue window.  R3 keeps the protocol and
runs the legs in splitting pieces.  These tests lock the *bookkeeping* that
makes such a split run trustworthy — fast, no solver in the loop (except one
400-step 2D smoke):

1. the pre-registered tolerances were not relaxed after the fact, and
   ``run_r3_split`` reads the thresholds from ``run_r2`` (single source of
   truth) rather than keeping a private copy;
2. every verdict stored in ``artifacts/r3_summary.json`` can be *recomputed*
   from the raw numbers in the same file (report/data drift guard);
3. a missing leg is reported as ``NOT_RUN``, never as a pass — tested by hiding
   the 3D parts from the merger and checking it does not invent a verdict;
4. the 2D numbers themselves do satisfy the owner standards (<3% for σ against
   the library-locked 0.0561, coexistence vs the Maxwell construction).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
VERIFY = HERE / "xval" / "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION" / "verify_r2"
SUMMARY = VERIFY / "artifacts" / "r3_summary.json"
PREDECLARED = VERIFY / "predeclared.json"

sys.path.insert(0, str(VERIFY))
sys.path.insert(0, str(HERE / "src"))

pytest.importorskip("torch")

LIB_SIGMA_2D = 0.0561  # benchmarks/verified/laplace_droplet locked value (τ=1)


def _summary() -> dict:
    if not SUMMARY.exists():
        pytest.skip(f"missing {SUMMARY}; run `run_r3_split.py --merge`")
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. protocol integrity
# ---------------------------------------------------------------------------


def test_thresholds_and_protocol_match_predeclaration():
    import run_r2 as R2

    pre = json.loads(PREDECLARED.read_text(encoding="utf-8"))
    assert pre["radii"] == R2.RADII
    assert pre["G_lib"] == R2.G_LIB and pre["G_eff"] == R2.G_EFF
    assert pre["tau"] == R2.TAU
    assert pre["P1_tol"] == R2.THRESHOLDS["tolerance"] == 0.15
    assert pre["P2_mech_balance_tol"] == R2.THRESHOLDS["mech_balance"]
    # the split executor must not carry its own thresholds
    import run_r3_split as R3

    assert R3.R2 is R2
    assert not hasattr(R3, "THRESHOLDS")


# ---------------------------------------------------------------------------
# 2. verdicts recomputable from the stored numbers
# ---------------------------------------------------------------------------


def test_verdicts_are_recomputable_from_raw_numbers():
    d = _summary()
    tol = 0.15
    v = d["verdicts"]
    cd, flat, fit, mw = d.get("cross_dim"), d["flat"], d["fit"], d["maxwell"]

    if cd is not None:
        assert v["P1_cross_dim"] is (abs(cd["ratio"] - 1.0) < tol)
        assert v["P1_cross_dim_tight3pct"] is (abs(cd["ratio"] - 1.0) < 0.03)
        assert cd["ratio"] == pytest.approx(cd["sigma_3d"] / cd["sigma_2d"], rel=1e-12)
    else:
        assert v["P1_cross_dim"] == "NOT_RUN"

    for dim in ("2", "3"):
        f = flat.get(dim)
        if f is None:
            assert v[f"P2_mech_balance_{dim}d"] == "NOT_RUN"
            assert v[f"P5_coexist_vs_maxwell_{dim}d"] == "NOT_RUN"
            assert v[f"P4_sigmaK_vs_lap_{dim}d"] == "NOT_RUN"
        else:
            assert v[f"P2_mech_balance_{dim}d"] is (f["mech_gap"] < 1e-3)
            ok = (
                abs(f["rho_l"] - mw["rho_l"]) / mw["rho_l"] < tol
                and abs(f["rho_v"] - mw["rho_v"]) / mw["rho_v"] < tol
            )
            assert v[f"P5_coexist_vs_maxwell_{dim}d"] is ok
            s = fit.get(dim)
            if s is not None:
                sk = f["sigma_K_tanh"]
                assert v[f"P4_sigmaK_vs_lap_{dim}d"] is (abs(sk - s["sigma_mean_of_radii"]) / sk < tol)
        s = fit.get(dim)
        if s is None:
            assert v[f"P3_radius_consistency_{dim}d"] == "NOT_RUN"
        else:
            assert v[f"P3_radius_consistency_{dim}d"] is (s["max_dev"] < tol)


def test_every_verdict_is_bool_or_not_run():
    d = _summary()
    assert d["verdicts"], "no verdicts at all"
    for k, val in d["verdicts"].items():
        assert isinstance(val, bool) or val == "NOT_RUN", (k, val)


# ---------------------------------------------------------------------------
# 3. anti-fabrication: hiding a part must yield NOT_RUN, not a pass
# ---------------------------------------------------------------------------


def test_merger_marks_missing_parts_not_run(tmp_path, monkeypatch):
    import run_r3_split as R3

    real_load = R3._load
    out = tmp_path / "r3_summary_hidden.json"
    monkeypatch.setattr(R3, "SUMMARY", out)
    monkeypatch.setattr(R3, "_load", lambda leg, dim: None if dim == 3 else real_load(leg, dim))

    d = R3.merge()
    v = d["verdicts"]
    assert d["cross_dim"] is None
    assert v["P1_cross_dim"] == "NOT_RUN"
    for key in ("P2_mech_balance_3d", "P5_coexist_vs_maxwell_3d", "P3_radius_consistency_3d",
                "P4_sigmaK_vs_lap_3d"):
        assert v[key] == "NOT_RUN", key
    assert d["fit"]["3"] is None and d["flat"]["3"] is None
    # the 2D legs still stand on their own
    if real_load("droplet", 2) is not None:
        assert d["fit"]["2"] is not None


# ---------------------------------------------------------------------------
# 4. the 2D numbers meet the owner standards
# ---------------------------------------------------------------------------


def test_two_d_sigma_meets_library_locked_value():
    d = _summary()
    s = d["fit"].get("2")
    if s is None:
        pytest.skip("2D droplet leg not present")
    dev = abs(s["sigma_through_origin"] - LIB_SIGMA_2D) / LIB_SIGMA_2D
    assert dev < 0.03, f"σ_2D = {s['sigma_through_origin']:.5f} vs {LIB_SIGMA_2D} → {100 * dev:.2f}%"
    assert s["max_dev"] < 0.03  # radius consistency is tight here as well


def test_two_d_coexistence_matches_maxwell():
    d = _summary()
    f = d["flat"].get("2")
    if f is None:
        pytest.skip("2D flat leg not present")
    mw = d["maxwell"]
    assert abs(f["rho_l"] - mw["rho_l"]) / mw["rho_l"] < 0.03
    assert abs(f["rho_v"] - mw["rho_v"]) / mw["rho_v"] < 0.03
    assert f["mech_gap"] < 1e-3


# ---------------------------------------------------------------------------
# 5. cheap physics smoke: the droplet still has a Laplace pressure
# ---------------------------------------------------------------------------


def test_short_droplet_has_positive_laplace_pressure():
    import run_r2 as R2

    row = R2.run_droplet_case((2, 10.0, 600, 300, "cpu"))
    assert math.isfinite(row["dp"]) and row["dp"] > 0.0
    assert row["rho_in"] > row["rho_out"]
    assert row["mass_drift"] < 1e-3
    # σ_i = Δp·R_eq must sit near the calibrated value already after 600 steps
    assert 0.035 < row["sigma"] < 0.080, row["sigma"]