from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_dam_break import _compute_front_metrics, run_dam_break_benchmark


def test_compute_front_metrics_monotonic_and_finite() -> None:
    front = [(0.0, 1.0), (0.5, 1.6), (1.0, 2.1)]
    m = _compute_front_metrics(front)
    assert bool(m["monotonic_front"]) is True
    assert float(m["rmse_vs_martin_moyce"]) >= 0.0
    assert float(m["mae_vs_martin_moyce"]) >= 0.0


def test_compute_front_metrics_detects_non_monotonic() -> None:
    front = [(0.0, 1.0), (0.5, 1.4), (1.0, 1.3)]
    m = _compute_front_metrics(front)
    assert bool(m["monotonic_front"]) is False


def test_run_dam_break_benchmark_fast_single_model(tmp_path: Path) -> None:
    report = run_dam_break_benchmark(
        models=["cg"],
        fast=True,
        output_root=tmp_path,
        device="cpu",
    )
    assert "cases" in report
    assert "cg" in report["cases"]  # type: ignore[operator]
    case = report["cases"]["cg"]  # type: ignore[index]
    assert "metrics" in case
    assert "final" in case
    assert "run_dir" in case
    run_dir = Path(case["run_dir"])  # type: ignore[index]
    assert (run_dir / "front_position.csv").exists()
    report_path = tmp_path / "dam_break_benchmark_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "all_ok" in payload
