"""TDD coverage for strict, reproducible acoustic campaign artifact gates."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from tensorlbm.regression_gate import evaluate_acoustic_campaign_gate


def _write_campaign(root: Path, *, exit_code: str = "0", log: str | None = None) -> None:
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "case.log").write_text(
        log
        or "step 10/10\nmetric St=0.1000\nPASS — acoustic physics\n",
        encoding="utf-8",
    )
    with (root / "status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "exit_code", "log"])
        writer.writeheader()
        writer.writerow({"case": "case", "exit_code": exit_code, "log": "logs/case.log"})


def _spec() -> dict:
    return {
        "status_file": "status.csv",
        "cases": {
            "case": {
                "log": "logs/case.log",
                "required_metrics": [r"metric St=0\.1000"],
                "physics_pass": r"(?m)^PASS — acoustic physics$",
            }
        },
    }


def test_campaign_gate_requires_zero_exit_required_metric_and_terminal_physics_pass(tmp_path):
    _write_campaign(tmp_path)

    report = evaluate_acoustic_campaign_gate(tmp_path, _spec())

    assert report["pass"] is True
    row = report["cases"][0]
    assert row["completion"]["pass"] is True
    assert row["artifacts"]["pass"] is True
    assert row["metrics"]["pass"] is True
    assert row["physics"]["pass"] is True


def test_campaign_gate_rejects_exit_zero_without_terminal_physics_pass(tmp_path):
    _write_campaign(tmp_path, log="step 10/10\nmetric St=0.1000\n[PASS] preliminary\n")

    report = evaluate_acoustic_campaign_gate(tmp_path, _spec())

    assert report["pass"] is False
    assert report["cases"][0]["completion"]["pass"] is True
    assert report["cases"][0]["physics"]["pass"] is False


def test_campaign_gate_rejects_missing_metric_and_escaped_status_log(tmp_path):
    _write_campaign(tmp_path, log="PASS — acoustic physics\n")
    report = evaluate_acoustic_campaign_gate(tmp_path, _spec())
    assert report["pass"] is False
    assert report["cases"][0]["metrics"]["pass"] is False

    outside = tmp_path.parent / "escaped.log"
    outside.write_text("metric St=0.1000\nPASS — acoustic physics\n", encoding="utf-8")
    with (tmp_path / "status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "exit_code", "log"])
        writer.writeheader()
        writer.writerow({"case": "case", "exit_code": "0", "log": str(outside)})
    escaped = evaluate_acoustic_campaign_gate(tmp_path, _spec())
    assert escaped["pass"] is False
    assert "escapes artifacts root" in escaped["cases"][0]["errors"][0]


def test_checked_in_campaign_spec_reports_rossiter_eight_of_eight_and_tail_edge_three_of_eight():
    root = Path(__file__).parents[1] / "validation_logs" / "acoustic_sdaa_campaign_20260710T141500Z"
    spec = Path(__file__).parents[1] / "configs" / "acoustic_sdaa_campaign_20260710T141500Z_gate.json"

    report = evaluate_acoustic_campaign_gate(root, json.loads(spec.read_text(encoding="utf-8")))

    assert report["pass"] is True
    assert report["summary"] == {
        "rossiter": {"passed": 8, "total": 8},
        "tail_edge": {"passed": 3, "total": 8},
    }
    assert report["recommended_tail_edge_default"] == "te_u011"


def test_cli_dispatches_acoustic_status_manifest(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / "evaluate_benchmark_gate.py"
    spec = importlib.util.spec_from_file_location("evaluate_benchmark_gate", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.json"
    manifest.write_text(json.dumps(_spec()), encoding="utf-8")
    _write_campaign(tmp_path)
    monkeypatch.setattr("sys.argv", [str(script), "--artifacts", str(tmp_path), "--manifest", str(manifest), "--report", str(report)])
    assert module.main() == 0
    assert json.loads(report.read_text(encoding="utf-8"))["pass"] is True
