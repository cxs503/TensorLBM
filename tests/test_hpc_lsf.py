"""Tests for the LSF backend of the platform HPC scheduler.

Mirrors the style of ``tests/test_app_hpc.py``: the real module is imported
(only ``subprocess``/``shutil.which`` are mocked), so script generation,
bsub argument construction, output parsing, and status mapping are
exercised end-to-end without a cluster.

The ``bjobs -l`` fixtures reproduce output captured from the SWA/Sunway LSF
cluster (psn002) where this backend was validated live.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "app"))

from backend.services import hpc_scheduler  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: fake binaries
# ---------------------------------------------------------------------------

_LSF_BINS = {"bsub", "bjobs", "bkill"}

_PSN002_SUBMIT_OUTPUT = "Job <57245099> has been submitted to queue <q_sw_share>\n"

_PSN002_BJOBS_L_DONE = (
    "\nJob<57245099>, User<swbxyh>, Status<DONE>, Project<->, "
    "Queue<q_sw_share>, Command</bin/echo hello_from_au_probe; hostname>\n"
    "\nAug 20 12:32:17: Submitted from host <psn002>, "
    "CWD</home/export/online3/swbxyh/au_lsf_probe>.\n"
    "Aug 20 12:32:18: Started on mn <mn238>, using <0> nodes\n"
    "Aug 20 12:32:27: End!\n"
)


class FakeRun:
    """Records subprocess calls; answers with scripted stdout."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": list(argv), **kwargs})
        program = argv[0]
        if program == "bsub":
            return type("Result", (), {"returncode": 0, "stdout": _PSN002_SUBMIT_OUTPUT, "stderr": ""})()
        if program == "bjobs":
            return type("Result", (), {"returncode": 0, "stdout": _PSN002_BJOBS_L_DONE, "stderr": ""})()
        if program == "bkill":
            return type("Result", (), {"returncode": 0, "stdout": "Job <57245099> is being terminated", "stderr": ""})()
        raise AssertionError(f"unexpected program {program!r}")


@pytest.fixture
def fake_bins(monkeypatch, tmp_path):
    run = FakeRun()
    monkeypatch.setattr(hpc_scheduler.subprocess, "run", run)
    monkeypatch.setattr(
        hpc_scheduler.shutil, "which",
        lambda name: f"/usr/sw-mpp/bin/{name}" if name in _LSF_BINS else None,
    )
    monkeypatch.setenv("TENSORLBM_HPC_LOG_DIR", str(tmp_path / "hpc_logs"))
    monkeypatch.setenv("TENSORLBM_HPC_PARTITION", "q_sw_share")
    return run


# ---------------------------------------------------------------------------
# Script building
# ---------------------------------------------------------------------------

def test_build_lsf_script_has_directives_and_command(tmp_path) -> None:
    script = hpc_scheduler._build_lsf_script(
        "job42", "echo hi",
        queue="q_sw_share", cpus=4, log_dir=tmp_path,
    )
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#BSUB -J tensorlbm_job42" in lines
    assert "#BSUB -q q_sw_share" in lines
    assert "#BSUB -n 4" in lines
    assert any(line.startswith("#BSUB -o ") and line.endswith("job42.%J.out") for line in lines)
    assert "set -eu" in lines, "script body must be POSIX-sh compatible"
    assert "set -o pipefail 2>/dev/null || true" in lines
    assert lines[-1] == "echo hi"


def test_build_lsf_script_extra_options(tmp_path) -> None:
    script = hpc_scheduler._build_lsf_script(
        "job43", "true",
        queue="q", cpus=1, log_dir=tmp_path,
        extra_options=["-W 30"],
    )
    assert "#BSUB -W 30" in script.splitlines()


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def test_submit_lsf_builds_argv_and_parses_job_id(fake_bins) -> None:
    result = hpc_scheduler.submit_lsf("job44", "echo hello", queue="q_sw_share", cpus=1)
    assert result["backend"] == "lsf"
    assert result["status"] == "submitted"
    assert result["hpc_job_id"] == "57245099"
    assert result["queue"] == "q_sw_share"

    argv = fake_bins.calls[0]["argv"]
    assert argv[0] == "bsub"
    assert argv[1:8] == ["-q", "q_sw_share", "-n", "1", "-J", "tensorlbm_job44", "-o"]
    assert argv[-2] == "bash", "SWA LSF needs an explicit shell command word"
    assert Path(argv[-1]).name == "job44.lsf"

    # The staged script exists and carries the command.
    script_path = Path(result["script_path"])
    assert script_path.exists()
    assert "echo hello" in script_path.read_text()
    # Log path is a %J template expanded by bsub.
    assert result["log_path"].endswith("job44.%J.out")


def test_submit_lsf_shell_is_configurable_for_swa(fake_bins, monkeypatch) -> None:
    """SWA compute nodes lack /bin/bash; ops must be able to force /bin/sh."""
    monkeypatch.setenv("TENSORLBM_HPC_LSF_SHELL", "sh")
    hpc_scheduler.submit_lsf("job50", "true")
    argv = fake_bins.calls[0]["argv"]
    assert argv[-2] == "sh"


def test_submit_lsf_rejects_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr(hpc_scheduler.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="bsub not found"):
        hpc_scheduler.submit_lsf("job45", "true")


def test_submit_lsf_raises_on_bsub_failure(fake_bins, monkeypatch) -> None:
    def failing(argv, **kwargs):
        return type("Result", (), {
            "returncode": 1, "stdout": "", "stderr": "Job submit failed, Queue is closed",
        })()

    monkeypatch.setattr(hpc_scheduler.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="Queue is closed"):
        hpc_scheduler.submit_lsf("job46", "true")


def test_submit_lsf_unparseable_output(fake_bins, monkeypatch) -> None:
    def weird(argv, **kwargs):
        return type("Result", (), {"returncode": 0, "stdout": "silence", "stderr": ""})()

    monkeypatch.setattr(hpc_scheduler.subprocess, "run", weird)
    with pytest.raises(RuntimeError, match="parse bsub output"):
        hpc_scheduler.submit_lsf("job47", "true")


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PEND", "pending"),
        ("RUN", "running"),
        ("DONE", "done"),
        ("EXIT", "exit"),
        ("PSUSP", "suspended"),
        ("EXITING", "exiting"),  # transient SWA-LSF state observed on psn002
        ("UNKWN", "unknown"),
    ],
)
def test_query_lsf_status_maps_states(fake_bins, monkeypatch, raw, expected) -> None:
    def fake(argv, **kwargs):
        output = _PSN002_BJOBS_L_DONE.replace("Status<DONE>", f"Status<{raw}>")
        return type("Result", (), {"returncode": 0, "stdout": output, "stderr": ""})()

    monkeypatch.setattr(hpc_scheduler.subprocess, "run", fake)
    status = hpc_scheduler.query_lsf_status("57245099")
    assert status["state"] == expected
    assert status["raw_state"] == raw
    assert status["hpc_job_id"] == "57245099"


def test_query_lsf_status_uses_bjobs_l(fake_bins) -> None:
    hpc_scheduler.query_lsf_status("57245099")
    argv = fake_bins.calls[0]["argv"]
    assert argv[:2] == ["bjobs", "-l"]
    assert argv[2] == "57245099"


def test_query_lsf_status_job_not_found(fake_bins, monkeypatch) -> None:
    def none(argv, **kwargs):
        return type("Result", (), {
            "returncode": 0, "stdout": "", "stderr": "No match record found!",
        })()

    monkeypatch.setattr(hpc_scheduler.subprocess, "run", none)
    status = hpc_scheduler.query_lsf_status("1")
    assert status["state"] == "unknown"


def test_query_lsf_status_without_bjobs(monkeypatch) -> None:
    monkeypatch.setattr(hpc_scheduler.shutil, "which", lambda name: None)
    status = hpc_scheduler.query_lsf_status("1")
    assert status["state"] == "unknown"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

def test_cancel_lsf_invokes_bkill(fake_bins) -> None:
    result = hpc_scheduler.cancel_lsf("57245099")
    assert result["status"] == "cancelled"
    assert fake_bins.calls[0]["argv"] == ["bkill", "57245099"]


def test_cancel_lsf_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr(hpc_scheduler.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="bkill not found"):
        hpc_scheduler.cancel_lsf("1")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def test_submit_hpc_job_dispatches_lsf(fake_bins, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TENSORLBM_HPC_MODE", "lsf")
    result = hpc_scheduler.submit_hpc_job(
        "job48", str(tmp_path), solver_cmd="echo dispatched",
        partition="q_sw_share", cpus=1,
    )
    assert result["backend"] == "lsf"
    assert result["hpc_job_id"] == "57245099"
    assert result["solver_cmd"] == "echo dispatched"


def test_submit_hpc_job_unknown_mode(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TENSORLBM_HPC_MODE", "sge")
    with pytest.raises(ValueError, match="Unknown HPC mode"):
        hpc_scheduler.submit_hpc_job("job49", str(tmp_path))
