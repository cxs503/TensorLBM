"""Regression tests for FSI benchmark execution observability."""
from __future__ import annotations

import csv
import json
import math

import pytest
import torch

from tensorlbm.benchmark_observability import (
    BenchmarkReporter, FsiCheckpoint, assert_benchmark_tensor_device, resolve_benchmark_device,
)


def test_resolve_cpu_device_is_explicit_and_available():
    metadata = resolve_benchmark_device("cpu")

    assert metadata["requested"] == "cpu"
    assert metadata["resolved"] == "cpu"
    assert metadata["available"] is True
    assert metadata["allocation_device"] == "cpu"
    assert metadata["device_asserted"] is True


def test_device_assertion_rejects_wrong_tensor_device():
    with pytest.raises(RuntimeError, match="Device assertion failed"):
        assert_benchmark_tensor_device(torch.ones(1), torch.device("meta"), "state")


def test_reporter_persists_start_progress_and_failure(tmp_path):
    reporter = BenchmarkReporter(
        output_dir=tmp_path,
        benchmark="flag_flapping",
        requested_steps=10,
        device_metadata={"requested": "cpu", "resolved": "cpu", "available": True},
    )
    reporter.start()
    reporter.progress(step=2, elapsed_seconds=1.25, tip_y=0.1, tip_x=-0.2)
    reporter.finish(
        completed_steps=2,
        outcome="FAILED",
        numerical_failure="synthetic failure",
        metrics={"amplitude": 0.1, "pass": False},
    )

    with (tmp_path / "run_status.json").open() as fh:
        status = json.load(fh)
    assert status["state"] == "FAILED"
    assert status["requested_steps"] == 10
    assert status["completed_steps"] == 2
    assert status["numerical_failure"] == "synthetic failure"
    assert status["metrics"]["pass"] is False

    with (tmp_path / "progress.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["step"] == "2"


def test_reporter_serializes_nonfinite_metrics_as_null(tmp_path):
    reporter = BenchmarkReporter(tmp_path, "turek_hron", 1, {"requested": "cpu"})
    reporter.start()
    reporter.finish(1, "FAILED", None, {"frequency_ratio": math.nan, "pass": False})

    assert json.loads((tmp_path / "run_status.json").read_text())["metrics"]["frequency_ratio"] is None


def test_fsi_checkpoint_round_trip_is_atomic_and_cpu_portable(tmp_path):
    checkpoint = FsiCheckpoint(tmp_path / "fsi_state.pt")
    state = {"step": 7, "f": torch.arange(4, device="cpu"), "history": [0.25]}

    checkpoint.save(state)
    restored = checkpoint.load()

    assert restored is not None
    assert restored["step"] == 7
    assert restored["f"].device.type == "cpu"
    assert torch.equal(restored["f"], state["f"])
    assert restored["history"] == [0.25]


def test_reporter_resume_preserves_existing_progress(tmp_path):
    reporter = BenchmarkReporter(tmp_path, "flag_flapping", 10, {"requested": "cpu"})
    reporter.start()
    reporter.progress(3, 1.0, 0.1, 0.2)
    reporter.start(resume=True, completed_steps=3)
    reporter.progress(4, 2.0, 0.2, 0.3)

    with (tmp_path / "progress.csv").open(newline="") as fh:
        assert [row["step"] for row in csv.DictReader(fh)] == ["3", "4"]
