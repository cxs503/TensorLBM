"""Tests for :mod:`tensorlbm.apps.hpc` — the HPC full-stack service.

Verifies that :class:`HpcRunSpec` carries the expected defaults and that
:func:`submit_app_hpc` wraps ``AI4SApplication.produce_data`` as a SLURM job
via the platform HPC scheduler, plus that :func:`query_app_hpc` queries
status.  The scheduler is mocked (no real ``sbatch``/``sacct``), while the
SLURM *script* is built with the real
:func:`backend.services.hpc_scheduler._build_slurm_script` so the generated
directives and command are exercised end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "app"))

from backend.services import hpc_scheduler as real_scheduler  # noqa: E402

from tensorlbm.apps.base import AI4SApplication  # noqa: E402
from tensorlbm.apps.hpc import (  # noqa: E402
    HpcRunSpec,
    build_hpc_job_id,
    build_produce_data_cmd,
    query_app_hpc,
    submit_app_hpc,
)

# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


class FakeScheduler:
    """Drop-in mock of the platform HPC scheduler (no real submission)."""

    def __init__(self) -> None:
        self.submit_slurm_calls: list[dict[str, Any]] = []
        self.submit_pbs_calls: list[dict[str, Any]] = []
        self.scripts: list[str] = []

    def submit_slurm(self, job_id, cmd, **kwargs) -> dict[str, Any]:
        script = real_scheduler._build_slurm_script(
            job_id,
            cmd,
            partition=kwargs["partition"],
            nodes=kwargs["nodes"],
            cpus=kwargs["cpus"],
            mem=kwargs["mem"],
            walltime=kwargs["walltime"],
            log_dir=Path("/tmp/tensorlbm_hpc_test_logs"),
            extra_directives=kwargs.get("extra_directives"),
        )
        self.scripts.append(script)
        self.submit_slurm_calls.append({"job_id": job_id, "cmd": cmd, **kwargs})
        return {
            "hpc_job_id": "314159",
            "script_path": f"/tmp/{job_id}.sbatch",
            "status": "submitted",
            "backend": "slurm",
        }

    def submit_pbs(self, job_id, cmd, **kwargs) -> dict[str, Any]:
        self.submit_pbs_calls.append({"job_id": job_id, "cmd": cmd, **kwargs})
        return {
            "hpc_job_id": "271828.pbs",
            "status": "submitted",
            "backend": "pbs",
        }

    def query_slurm_status(self, hpc_job_id: str) -> dict[str, str]:
        return {"hpc_job_id": hpc_job_id, "state": "COMPLETED", "elapsed": "00:02:00"}


class _DummyApp(AI4SApplication):
    name = "demo_hpc_app"
    family = "demo"
    version = "1.0"

    def produce_data(self, cfg):
        raise NotImplementedError

    def build_model(self, arch):
        raise NotImplementedError

    def make_dataset(self, product):
        raise NotImplementedError

    def train(self, dataset, model, cfg):
        raise NotImplementedError

    def infer(self, model, sample):
        raise NotImplementedError


@pytest.fixture
def app() -> _DummyApp:
    return _DummyApp()


@pytest.fixture
def scheduler() -> FakeScheduler:
    return FakeScheduler()


# ---------------------------------------------------------------------------
# HpcRunSpec
# ---------------------------------------------------------------------------


def test_hpc_run_spec_defaults():
    spec = HpcRunSpec(app_name="demo_hpc_app")
    assert spec.app_name == "demo_hpc_app"
    assert spec.partition == "compute"
    assert spec.nodes == 1
    assert spec.cpus == 4
    assert spec.mem == "8G"
    assert spec.walltime == "02:00:00"
    assert spec.script_cmd is None
    assert spec.produce_cfg == {}
    assert spec.extra_directives == []
    assert spec.backend == "slurm"


def test_hpc_run_spec_custom():
    spec = HpcRunSpec(
        app_name="demo_hpc_app",
        partition="gpu",
        nodes=2,
        cpus=8,
        mem="16G",
        walltime="01:00:00",
        script_cmd="echo hi",
        extra_directives=["--gres=gpu:1"],
        backend="pbs",
    )
    assert (spec.partition, spec.nodes, spec.cpus, spec.mem, spec.walltime) == (
        "gpu",
        2,
        8,
        "16G",
        "01:00:00",
    )
    assert spec.script_cmd == "echo hi"
    assert spec.backend == "pbs"


# ---------------------------------------------------------------------------
# Command / id helpers
# ---------------------------------------------------------------------------


def test_build_produce_data_cmd_mentions_app_and_produce_data():
    cmd = build_produce_data_cmd("demo_hpc_app", {"nx": 32, "seed": 7})
    assert "produce_data" in cmd
    assert "demo_hpc_app" in cmd
    assert '"nx": 32' in cmd or "'nx': 32" in cmd
    assert cmd.startswith("python -c ")


def test_build_hpc_job_id_shape():
    jid = build_hpc_job_id("demo_hpc_app")
    assert jid.startswith("demo_hpc_app_hpc_")
    assert len(jid) == len("demo_hpc_app_hpc_") + 8


# ---------------------------------------------------------------------------
# submit_app_hpc (SLURM)
# ---------------------------------------------------------------------------


def test_submit_app_hpc_builds_slurm_script(app, scheduler):
    spec = HpcRunSpec(
        app_name="demo_hpc_app",
        partition="compute",
        nodes=2,
        cpus=8,
        mem="16G",
        walltime="01:30:00",
        produce_cfg={"nx": 64, "n_steps": 100},
        extra_directives=["--gres=gpu:1"],
    )
    result = submit_app_hpc(app, spec, scheduler=scheduler, job_id="demo_job_1")

    # no real submission: result came from the fake scheduler
    assert result["hpc_job_id"] == "314159"
    assert result["job_id"] == "demo_job_1"
    assert result["app_name"] == "demo_hpc_app"
    assert "produce_data" in result["script_cmd"]

    # exactly one SLURM submission with the expected resource request
    assert len(scheduler.submit_slurm_calls) == 1
    call = scheduler.submit_slurm_calls[0]
    assert call["job_id"] == "demo_job_1"
    assert call["partition"] == "compute"
    assert call["nodes"] == 2
    assert call["cpus"] == 8
    assert call["mem"] == "16G"
    assert call["walltime"] == "01:30:00"
    assert call["extra_directives"] == ["--gres=gpu:1"]

    # the generated SLURM script carries the directives + produce_data command
    script = scheduler.scripts[0]
    assert "#!/bin/bash" in script
    assert "#SBATCH --job-name=tensorlbm_demo_job_1" in script
    assert "#SBATCH --partition=compute" in script
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --time=01:30:00" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "produce_data" in script


def test_submit_app_hpc_defaults_script_cmd_from_produce_cfg(app, scheduler):
    result = submit_app_hpc(app, HpcRunSpec(app_name="demo_hpc_app"), scheduler=scheduler)
    assert result["hpc_job_id"] == "314159"
    # job id auto-generated with the app-name prefix
    assert result["job_id"].startswith("demo_hpc_app_hpc_")
    # auto-generated produce_data command landed in the script
    assert "produce_data" in scheduler.scripts[0]


def test_submit_app_hpc_uses_explicit_script_cmd(app, scheduler):
    spec = HpcRunSpec(app_name="demo_hpc_app", script_cmd="mpirun -n 4 ./solver")
    submit_app_hpc(app, spec, scheduler=scheduler, job_id="j")
    assert scheduler.submit_slurm_calls[0]["cmd"] == "mpirun -n 4 ./solver"
    assert "mpirun" in scheduler.scripts[0]
    assert "produce_data" not in scheduler.scripts[0]


def test_submit_app_hpc_pbs_backend(app, scheduler):
    spec = HpcRunSpec(app_name="demo_hpc_app", backend="pbs", partition="batch")
    result = submit_app_hpc(app, spec, scheduler=scheduler, job_id="pbs_job")
    assert result["backend"] == "pbs"
    assert len(scheduler.submit_pbs_calls) == 1
    assert len(scheduler.submit_slurm_calls) == 0
    assert scheduler.submit_pbs_calls[0]["queue"] == "batch"


def test_submit_app_hpc_validation_errors(app, scheduler):
    class _Nameless:
        name = ""

    with pytest.raises(ValueError):
        submit_app_hpc(_Nameless(), HpcRunSpec(), scheduler=scheduler)  # empty app_name
    with pytest.raises(ValueError):
        submit_app_hpc(
            app,
            HpcRunSpec(app_name="demo_hpc_app", nodes=0),
            scheduler=scheduler,
        )
    with pytest.raises(ValueError):
        submit_app_hpc(
            app,
            HpcRunSpec(app_name="demo_hpc_app", backend="torque"),
            scheduler=scheduler,
        )


# ---------------------------------------------------------------------------
# query_app_hpc
# ---------------------------------------------------------------------------


def test_query_app_hpc_returns_status(scheduler):
    status = query_app_hpc("314159", scheduler=scheduler)
    assert status == {"hpc_job_id": "314159", "state": "COMPLETED", "elapsed": "00:02:00"}
