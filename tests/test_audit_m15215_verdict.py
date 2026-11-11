"""Guard rails for the m15273 board-audit verdict on m15215 (XVAL-LBM-TAYLORGREEN).

The audit verdict (`docs/audit/m15215/VERDICT.md` + `verdict_m15215.sql`) may not
drift away from the machine-recomputed numbers in
`docs/audit/m15215/audit_metrics.json`: the level, every PASS/FAIL and every
headline number are re-derived here from that JSON alone.

If the audited mission's worktree is present (``M15215_ARTIFACTS`` or the default
path), the raw timeseries are re-reduced from scratch and compared, so a silently
edited JSON is caught too.

Run: pytest -q tests/test_audit_m15215_verdict.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "docs" / "audit" / "m15215"
METRICS = AUDIT / "audit_metrics.json"
VERDICT_SQL = REPO / "verdict_m15215.sql"
DEFAULT_ARTIFACTS = Path("/root/tp-agent/work/m15215/benchmarks/xval_taylor_green_2d")

REQUIRED_EVIDENCE_KEYS = {
    "impl_files",
    "entry",
    "run_artifacts",
    "tests",
    "integration_callers",
    "benchmark",
    "repro_cmd",
    "external_anchor",
}


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(METRICS.read_text())


def test_audit_metrics_criteria_are_consistent(metrics):
    # naming: the mission/ledger slug is XVAL-LBM-TAYLORGREEN, the in-artifact
    # anchor (case_spec.json / result.json) is XVAL-TAYLORGREEN -> keep both pinned
    assert metrics["anchor"] == "XVAL-TAYLORGREEN"
    assert "XVAL-LBM-TAYLORGREEN" in VERDICT_SQL.read_text()
    assert metrics["n_tools"] == 2, "OpenFOAM axis is degraded -> only 2 tools"
    assert len(metrics["criteria"]) == 3
    for c in metrics["criteria"]:
        assert c["verdict"] == ("PASS" if c["got_pct"] < c["tol_pct"] else "FAIL"), c
    assert metrics["all_pass"] is True


def test_measured_numbers_are_inside_the_mission_tolerances(metrics):
    """Hard thresholds copied from the mission goal, not from the JSON's own verdicts."""
    m = metrics["metrics"]
    for tool in ("tensorlbm", "palabos"):
        assert abs(m[tool]["gamma_K_err_pct"]) < 8.0, (tool, m[tool]["gamma_K_err_pct"])
        assert abs(m[tool]["gamma_vel_err_pct"]) < 8.0, (tool, m[tool]["gamma_vel_err_pct"])
        assert m[tool]["phase_err_pct_max"] < 3.0
        assert m[tool]["r2_lnK"] > 0.9999
        assert m[tool]["rows"] == 99

    cross = (
        100.0
        * abs(m["tensorlbm"]["gamma_K"] - m["palabos"]["gamma_K"])
        / abs(m["palabos"]["gamma_K"])
    )
    assert cross < 5.0
    # two independent BGK D2Q9 codes must agree at round-off, not merely at 5%:
    assert metrics["cross_tool_abs_rel"]["K"] < 1e-10
    assert metrics["cross_tool_abs_rel"]["umax"] < 1e-10


def test_verdict_sql_declares_level_4_with_full_evidence(metrics):
    sql = VERDICT_SQL.read_text()
    assert "'TensorLBM'" in sql
    assert "'m15215 XVAL-LBM-TAYLORGREEN'" in sql
    assert "'board-audit'" in sql
    level = re.search(r"\)\s*VALUES\s*\(\s*'TensorLBM',\s*'[^']+',\s*(\d+),\s*\d+,\s*\$json\$",
                      sql)
    assert level, "unexpected INSERT shape"
    assert int(level.group(1)) == 15273, "evidence.mission_id must be the audit mission"
    m = re.search(r"\n\s{2}(\d),\n\s{2}\$json\$", sql)
    assert m and int(m.group(1)) == 4, "level must be 4 (benchmark validation)"

    payload = json.loads(sql.split("$json$")[1])
    assert REQUIRED_EVIDENCE_KEYS <= set(payload)
    # the JSON in the SQL must quote the same numbers as audit_metrics.json
    for tool in ("tensorlbm", "palabos"):
        assert payload["benchmark"]["auditor_recomputed"][f"gamma_K_{tool}"] == pytest.approx(
            metrics["metrics"][tool]["gamma_K"], rel=1e-6
        )
    assert payload["benchmark"]["auditor_recomputed"]["cross_tool_diff_pct"] == pytest.approx(
        metrics["criteria"][0]["got_pct"], rel=1e-6
    )
    # honest scoping: the degraded axis and the ledger gap must stay declared
    assert payload["benchmark"]["platform_ledger"].startswith("tensor_benchmark_cases: 0 rows")
    assert any("OpenFOAM" in s for s in payload["audit"]["not_run"])
    assert payload["audit"]["findings"], "severity-ranked findings required"


def test_raw_artifacts_still_reduce_to_the_same_numbers(tmp_path):
    """Re-derive everything from the audited mission's raw CSV when it is reachable."""
    base = Path(os.environ.get("M15215_ARTIFACTS", DEFAULT_ARTIFACTS))
    if not (base / "runs" / "palabos" / "timeseries.csv").exists():
        pytest.skip(f"m15215 raw artifacts not reachable at {base}")

    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(AUDIT / "recompute_m15215.py"),
         "--artifacts", str(base), "--out", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    fresh = json.loads((tmp_path / "audit_metrics.json").read_text())
    stored = json.loads(METRICS.read_text())
    assert fresh["all_pass"] is True
    for c_new, c_old in zip(fresh["criteria"], stored["criteria"]):
        assert c_new["got_pct"] == pytest.approx(c_old["got_pct"], rel=1e-6, abs=1e-18)
    for tool in ("tensorlbm", "palabos"):
        assert fresh["metrics"][tool]["gamma_K"] == pytest.approx(
            stored["metrics"][tool]["gamma_K"], rel=1e-9
        )