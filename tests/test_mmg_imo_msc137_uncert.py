"""Tests for the UNCERT (GCI / error bar) component of mission m4800.

Layer 1 verifies the estimator on synthetic series with a known observed order
so the machinery is checked before its output is trusted; layer 2 checks the
committed artifacts: every claimed uncertainty is either a number with an
observed order or an explicit fallback_reason.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "benchmarks" / "verified" / "mmg_imo_msc137_uncert" / "run_uncert.py"
_ART = _REPO / "artifacts" / "mmg_imo_uncert"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("run_uncert_m4800", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def payload():
    f = _ART / "mmg_imo_msc137_uncert.json"
    if not f.exists():
        pytest.skip("run benchmarks/verified/mmg_imo_msc137_uncert/run_uncert.py first")
    return json.loads(f.read_text(encoding="utf-8"))


# --------------------------- estimator unit tests -------------------------- #
def test_richardson_recovers_known_order(mod):
    ref, c = 1.0, 1e-4
    g = mod.richardson_gci(ref + c, ref + 16 * c, ref + 256 * c, r=2.0, fs=1.25)
    assert g["applicable"] is True
    assert abs(g["p"] - 4.0) < 1e-9
    assert abs(g["f_extrapolated"] - ref) < 1e-12
    assert math.isclose(g["gci_u95_pct"], 1.25 * 15 * c / (16 - 1) * 100, rel_tol=1e-9)


def test_roundoff_plateau_withheld(mod):
    g = mod.richardson_gci(1.0, 1.0 + 1e-15, 1.0 + 3e-15)
    assert g["applicable"] is False and "double-precision" in g["fallback_reason"]


def test_non_monotone_withheld(mod):
    g = mod.richardson_gci(1.0, 1.001, 0.999)
    assert g["applicable"] is False and "non-monotonic" in g["fallback_reason"]


def test_two_levels_do_not_produce_gci(mod):
    g = mod.gci_from_levels({0.5: 1.0, 0.25: 1.0 + 1e-6})
    assert g["applicable"] is False and "two-level" in g["fallback_reason"]


def test_gci_from_levels_scales_with_coarse_level(mod):
    lv = {0.5: 1.0 + 1e-4, 0.25: 1.0 + 1e-4 / 16, 0.125: 1.0 + 1e-4 / 256}
    g = mod.gci_from_levels(lv)
    assert g["applicable"] is True and abs(g["observed_order_p"] - 4.0) < 1e-6
    assert g["gci_u95_pct"] > 0.0


def test_sd_matches_statistics(mod):
    import statistics
    v = [1.0, 2.0, 4.0, 8.0, 3.0]
    assert math.isclose(mod._sd(v), statistics.stdev(v), rel_tol=1e-12)


# ------------------------------ artifact tests ----------------------------- #
def test_geometry_gate_reproduced(payload):
    r = payload["platform_kernel"]["geometry_gate_residual_pct"]
    assert 2.45 < r < 2.55 and r < 3.0     # reproduces #5200/#5204 (+2.50 %)


def test_every_uncertainty_is_defensible(payload):
    for scope, block in (("A", payload["A_timestep_refinement"]["gci"]),
                         ("C", payload["C_repo_linear_kernel_containment"]["gci"])):
        for key, g in block.items():
            if not g.get("applicable"):
                assert g.get("fallback_reason"), f"{scope}:{key} withheld without reason"
            else:
                assert g.get("observed_order_p", g.get("p")) is not None
                assert g["gci_u95_pct"] > 0


def test_transient_containment_holds(payload):
    """Where truncation dominates the U95 band must bound the true error."""
    g = payload["C_repo_linear_kernel_containment"]["gci"]["20deg:r_at_60s"]
    assert g["contained"] is True
    assert g["true_err_pct_at_finest"] <= g["gci_u95_pct"]


def test_steady_quantities_are_not_gci_controlled(payload):
    g = payload["C_repo_linear_kernel_containment"]["gci"]["20deg:r_steady"]
    assert g["contained"] is False
    assert g["method"] == "analytic bias (GCI withheld)"
    assert g["true_err_pct_at_finest"] > 3 * g["gci_u95_pct"]


def test_legacy_tactical_diameter_withheld(payload):
    g = payload["A_timestep_refinement"]["gci"]["35deg:legacy_tactical_diameter_over_L"]
    assert g["gci_applicable_screened"] is False and "window" in g["fallback_reason"]


def test_parametric_statistics(payload):
    mc = payload["B_parametric_monte_carlo"]
    assert mc["n_samples"] == 200
    for q, st in mc["statistics"].items():
        assert st["confidence_level"] == 0.95
        vals = [st["p2_5"], st["mean"], st["p97_5"]]
        if all(isinstance(v, (int, float)) and v == v for v in vals):
            assert st["p2_5"] <= st["mean"] <= st["p97_5"]
        else:  # quantity undefined on part of the sample: must be declared, not silent
            assert "statistics_note" in mc or q == "tactical_diameter_180_over_L"


def test_oat_table_covers_all_parameters(payload):
    oat = payload.get("B2_oat_parameter_levels") or []
    assert oat, "OAT parameter-level table missing"
    params = {r["parameter"] for r in oat}
    for want in ("cd_cross", "munk_factor", "f_alpha", "w_r", "radius_of_gyration_ratio"):
        assert want in params
        assert sum(1 for r in oat if r["parameter"] == want) == 2


def test_table1_csv_consistency(payload):
    rows = list(csv.DictReader((_ART / "table1_uncert.csv").open(encoding="utf-8")))
    assert len(rows) == len(payload["table1"])
    assert {float(r["dt_s"]) for r in rows} == {4.0, 2.0, 1.0, 0.5, 0.25}
    finest = [r for r in rows if float(r["dt_s"]) == 0.25]
    assert finest and sum(1 for r in finest if r["gci_u95_pct"]) >= 4


def test_figure_written():
    fig = _ART / "fig_uncert_vs_dt.png"
    if not fig.exists():
        pytest.skip("figure not generated yet")
    assert fig.stat().st_size > 20_000
