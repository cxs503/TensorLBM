"""Regression tests for the validation-matrix publication gate.

Two defects motivated these tests, both of the "the code agrees with itself"
class that a passing test suite happily hides:

1. ``runner.render_table`` published the **worst level of every kind** as the
   headline metric.  For ``ship_resistance_suboff`` the informational coarse
   level (-3.17%, explicitly not gated by the case pre-registration) was printed
   as ``cd_tot_err_pct -3.17% (tol 3%)`` next to a ``**passed**`` status — a
   published row that reads as a tolerance violation.  The gated number
   (L=80, -1.15%) never appeared.
2. Unresolvable metric paths leaked a ``+nan%`` into the published table
   (``channel_turb_re180``), which hides that the declared evidence path does
   not exist at all.

The tests below assert *properties of the published table* (a claim), and they
exercise the independent audit's detector on a deliberately bad row, so the
detector itself is proven to fire.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VM = REPO_ROOT / "benchmarks" / "validation_matrix"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, VM / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


criteria = _load("criteria")
runner = _load("runner")
independent = _load("independent_audit")


# --------------------------------------------------------------------------
# 1. per-level gating semantics
# --------------------------------------------------------------------------
def test_informational_level_cannot_fail_the_metric():
    spec = {
        "name": "cd_tot_err_pct",
        "kind": "error_pct",
        "tol_pct": 3.0,
        "tol_pct_by_level": [None, 3.0],
        "levels": ["L=48", "L=80"],
    }
    out = criteria._levels_of("cd_tot_err_pct", [-3.17, -1.15], spec)
    assert out["pass"] is True
    assert out["gated_levels"] == ["L=80"]
    assert out["informational_levels"] == ["L=48"]
    assert out["worst_gated_level"] == "L=80"
    assert out["worst_gated_err_pct"] == pytest.approx(-1.15)
    # the legacy all-level aggregate still spans the informational level
    assert out["max_abs_err_pct"] == pytest.approx(3.17)


def test_gated_aggregate_ignores_unresolved_nan():
    spec = {"name": "m", "kind": "error_pct", "tol_pct": 3.0, "levels": ["a", "b"]}
    out = criteria._levels_of("m", [float("nan"), 1.0], spec)
    assert out["pass"] is False  # a NaN level is missing evidence, never a pass
    assert out["unresolved_levels"] == ["a"]
    assert out["max_abs_err_pct_gated"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 2. published table
# --------------------------------------------------------------------------
def _synthetic_result(errors):
    verdict = criteria.judge_case(
        {"cd_tot_err_pct": errors},
        [
            {
                "name": "cd_tot_err_pct",
                "kind": "error_pct",
                "tol_pct": 3.0,
                "tol_pct_by_level": [None, 3.0],
            }
        ],
        [{"metric": "cd_tot_err_pct", "max_span_pct": 3.0, "require_monotone": True}],
        levels=["L=48", "L=80"],
    )
    return {
        "case": "ship_resistance_suboff",
        "slot": "ship",
        "artifact": "benchmarks/verified/suboff_re1000/result.json",
        "repro_cmd": "python benchmarks/verified/suboff_re1000/run.py",
        **verdict,
    }


def test_published_headline_uses_the_gated_level():
    assert _synthetic_result([-3.17, -1.15])["status"] == "passed"
    text = runner.render_table([_synthetic_result([-3.17, -1.15])])
    row = next(l for l in text.splitlines() if "ship_resistance_suboff" in l)
    assert "-1.15% (tol 3%)" in row
    assert "info L=48 -3.17%" in row
    # the informational level must not head the cell
    head = row.split("**passed**")[1].split("[")[0]
    assert "-3.17%" not in head


def test_unresolved_metric_is_labelled_not_nan():
    verdict = criteria.judge_case(
        {"uplus_rms_err_pct": [float("nan"), float("nan")]},
        [{"name": "uplus_rms_err_pct", "kind": "error_pct", "tol_pct": 3.0}],
        [{"metric": "uplus_rms_err_pct", "max_span_pct": 3.0}],
        levels=["coarse", "fine"],
    )
    rec = {
        "case": "channel_turb_re180",
        "slot": "channel",
        "artifact": "benchmarks/pending/channel_turb_3d/result.json",
        **verdict,
    }
    text = runner.render_table([rec])
    row = next(l for l in text.splitlines() if "channel_turb_re180" in l)
    assert "nan" not in row
    assert "unresolved" in row


def test_live_published_table_has_no_self_contradicting_pass():
    """Property over the real table: a passed row cannot quote |err| > tol."""
    table = (VM / "STATUS_MATRIX.md").read_text(encoding="utf-8")
    checked = 0
    for line in table.splitlines():
        m = independent.ROW_RE.match(line.strip())
        if not m:
            continue
        case_id, status, key_cell = m.group(2), m.group(3), m.group(4)
        if status != "passed":
            continue
        km = independent.KEY_METRIC_RE.match(key_cell)
        assert km, f"{case_id}: passed row without a parseable key metric: {key_cell!r}"
        err, tol = float(km.group(2)), float(km.group(3))
        assert abs(err) <= tol, (
            f"{case_id}: published passed row displays |err| {err:+.2f}% "
            f"> tol {tol:.2f}%"
        )
        checked += 1
    assert checked >= 6


# --------------------------------------------------------------------------
# 3. the detector actually fires on a bad row
# --------------------------------------------------------------------------
def test_audit_detects_a_contradicting_published_row(tmp_path):
    bad = tmp_path / "STATUS_MATRIX.md"
    bad.write_text(
        "\n".join(
            [
                "| # | slot | case | status | key metric (worst level) | convergence | evidence |",
                "|---|------|------|--------|--------------------------|-------------|----------|",
                "| 1 | ship | `ship_resistance_suboff` | **passed** | "
                "cd_tot_err_pct -3.17% (tol 3%) | cd_tot_err_pct span 2.02% | "
                "`benchmarks/verified/suboff_re1000/result.json` |",
            ]
        ),
        encoding="utf-8",
    )
    cases = {
        "ship_resistance_suboff": {
            "case": "ship_resistance_suboff",
            "status": "passed",
            "metrics": [
                {
                    "metric": "cd_tot_err_pct",
                    "worst_gated": {"err_pct": -1.15, "tol_pct": 3.0, "level": "L=80"},
                }
            ],
        }
    }
    report = independent.audit_table(bad, cases)
    assert report["pass"] is False
    msgs = " ".join(f["message"] for f in report["findings"])
    assert "CONTRADICTION" in msgs
    assert "value mismatch" in msgs


# --------------------------------------------------------------------------
# 4. independent verdicts agree with the official ones, on the live matrix
# --------------------------------------------------------------------------
def test_independent_audit_reproduces_official_statuses():
    report = independent.run()
    matrix = json.loads((VM / "matrix.json").read_text(encoding="utf-8"))
    official = {
        case["id"]: runner.evaluate_case(case)["status"] for case in matrix["cases"]
    }
    independent_status = {
        cid: rec["status"] for cid, rec in report["cases"].items()
    }
    assert independent_status == official
    assert report["registry_consistency"]["pass"] is True
    assert report["pass"] is True, report["findings"]