"""HPC cluster job scheduler integration for TensorLBM platform.

Provides SLURM and PBS/Torque job-submission wrappers so that simulation
jobs can be dispatched to a cluster instead of running on the local thread
pool.  Analogous to the HPC integration in PowerFlow and XFlow.

Configuration
-------------
Set the environment variable ``TENSORLBM_HPC_MODE`` to one of:

* ``none``  – (default) local thread pool only, no HPC submission.
* ``slurm`` – submit via SLURM ``sbatch``.
* ``pbs``   – submit via PBS/Torque ``qsub``.

Additional optional environment variables:

* ``TENSORLBM_HPC_PARTITION``  – default SLURM partition or PBS queue.
* ``TENSORLBM_HPC_NODES``      – default node count (default 1).
* ``TENSORLBM_HPC_CPUS``       – CPUs per task (default 4).
* ``TENSORLBM_HPC_MEM``        – memory per node (default "8G").
* ``TENSORLBM_HPC_WALLTIME``   – default walltime (default "02:00:00").
* ``TENSORLBM_HPC_LOG_DIR``    – directory for scheduler stdout/stderr logs.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any

logger = logging.getLogger("tensorlbm.hpc_scheduler")

# ---------------------------------------------------------------------------
# Environment / defaults
# ---------------------------------------------------------------------------

def hpc_mode() -> str:
    return os.environ.get("TENSORLBM_HPC_MODE", "none").lower()


def _default_partition() -> str:
    return os.environ.get("TENSORLBM_HPC_PARTITION", "compute")


def _default_nodes() -> int:
    return int(os.environ.get("TENSORLBM_HPC_NODES", "1"))


def _default_cpus() -> int:
    return int(os.environ.get("TENSORLBM_HPC_CPUS", "4"))


def _default_mem() -> str:
    return os.environ.get("TENSORLBM_HPC_MEM", "8G")


def _default_walltime() -> str:
    return os.environ.get("TENSORLBM_HPC_WALLTIME", "02:00:00")


def _log_dir() -> Path:
    return Path(os.environ.get("TENSORLBM_HPC_LOG_DIR", "/tmp/tensorlbm_hpc_logs"))


# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------

def _build_slurm_script(
    job_id: str,
    cmd: str,
    *,
    partition: str,
    nodes: int,
    cpus: int,
    mem: str,
    walltime: str,
    log_dir: Path,
    extra_directives: list[str] | None = None,
) -> str:
    """Build a SLURM batch script string."""
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=tensorlbm_{job_id}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --output={log_dir}/{job_id}.out",
        f"#SBATCH --error={log_dir}/{job_id}.err",
    ]
    if extra_directives:
        lines.extend(f"#SBATCH {d}" for d in extra_directives)
    lines += [
        "",
        "set -euo pipefail",
        "",
        cmd,
    ]
    return "\n".join(lines) + "\n"


def submit_slurm(
    job_id: str,
    cmd: str,
    *,
    partition: str | None = None,
    nodes: int | None = None,
    cpus: int | None = None,
    mem: str | None = None,
    walltime: str | None = None,
    extra_directives: list[str] | None = None,
) -> dict[str, Any]:
    """Submit a shell command to SLURM via ``sbatch``.

    Args:
        job_id:    Platform job identifier (used for naming).
        cmd:       Shell command to run on the cluster node.
        partition: SLURM partition (default: $TENSORLBM_HPC_PARTITION).
        nodes:     Number of nodes (default: $TENSORLBM_HPC_NODES).
        cpus:      CPUs per task (default: $TENSORLBM_HPC_CPUS).
        mem:       Memory per node (default: $TENSORLBM_HPC_MEM).
        walltime:  Walltime limit (default: $TENSORLBM_HPC_WALLTIME).
        extra_directives: Additional ``#SBATCH`` lines (without the prefix).

    Returns:
        Dictionary with ``hpc_job_id`` (SLURM job number), ``script_path``,
        ``status`` (``submitted``), and ``backend`` (``slurm``).

    Raises:
        RuntimeError: If sbatch is not found or returns non-zero.
    """
    if shutil.which("sbatch") is None:
        raise RuntimeError(
            "sbatch not found on PATH. "
            "Ensure SLURM is installed or set TENSORLBM_HPC_MODE=none."
        )

    script = _build_slurm_script(
        job_id, cmd,
        partition=partition or _default_partition(),
        nodes=nodes or _default_nodes(),
        cpus=cpus or _default_cpus(),
        mem=mem or _default_mem(),
        walltime=walltime or _default_walltime(),
        log_dir=_log_dir(),
        extra_directives=extra_directives,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sbatch", delete=False, prefix=f"tensorlbm_{job_id}_",
    ) as fh:
        fh.write(script)
        script_path = fh.name

    result = subprocess.run(
        ["sbatch", script_path],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr}")

    # sbatch prints "Submitted batch job 12345"
    hpc_job_id = result.stdout.strip().split()[-1]
    logger.info("SLURM job submitted: platform_id=%s slurm_id=%s", job_id, hpc_job_id)

    return {
        "hpc_job_id": hpc_job_id,
        "script_path": script_path,
        "status": "submitted",
        "backend": "slurm",
        "partition": partition or _default_partition(),
        "nodes": nodes or _default_nodes(),
        "cpus": cpus or _default_cpus(),
        "mem": mem or _default_mem(),
        "walltime": walltime or _default_walltime(),
    }


def query_slurm_status(hpc_job_id: str) -> dict[str, str]:
    """Query SLURM job status via ``sacct``.

    Args:
        hpc_job_id: SLURM job ID.

    Returns:
        Dictionary with ``hpc_job_id``, ``state``, ``elapsed``.
    """
    if shutil.which("sacct") is None:
        return {"hpc_job_id": hpc_job_id, "state": "unknown", "elapsed": "n/a"}

    result = subprocess.run(
        ["sacct", "-j", hpc_job_id, "--format=JobID,State,Elapsed", "--noheader", "--parsable2"],
        capture_output=True, text=True, check=False,
    )
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.startswith(hpc_job_id + "|")]
    if not lines:
        return {"hpc_job_id": hpc_job_id, "state": "unknown", "elapsed": "n/a"}
    parts = lines[0].split("|")
    return {
        "hpc_job_id": hpc_job_id,
        "state": parts[1] if len(parts) > 1 else "unknown",
        "elapsed": parts[2] if len(parts) > 2 else "n/a",
    }


# ---------------------------------------------------------------------------
# PBS/Torque submission
# ---------------------------------------------------------------------------

def _build_pbs_script(
    job_id: str,
    cmd: str,
    *,
    queue: str,
    nodes: int,
    cpus: int,
    mem: str,
    walltime: str,
    log_dir: Path,
) -> str:
    """Build a PBS batch script string."""
    log_dir.mkdir(parents=True, exist_ok=True)
    return textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -N tensorlbm_{job_id}
        #PBS -q {queue}
        #PBS -l nodes={nodes}:ppn={cpus}
        #PBS -l mem={mem}
        #PBS -l walltime={walltime}
        #PBS -o {log_dir}/{job_id}.out
        #PBS -e {log_dir}/{job_id}.err

        set -euo pipefail

        {cmd}
    """)


def submit_pbs(
    job_id: str,
    cmd: str,
    *,
    queue: str | None = None,
    nodes: int | None = None,
    cpus: int | None = None,
    mem: str | None = None,
    walltime: str | None = None,
) -> dict[str, Any]:
    """Submit a shell command to PBS/Torque via ``qsub``.

    Args:
        job_id:   Platform job identifier.
        cmd:      Shell command to run.
        queue:    PBS queue name.
        nodes:    Number of nodes.
        cpus:     Processors per node.
        mem:      Memory per node.
        walltime: Walltime limit.

    Returns:
        Dictionary with ``hpc_job_id``, ``status``, ``backend`` (``pbs``).

    Raises:
        RuntimeError: If qsub is not found or fails.
    """
    if shutil.which("qsub") is None:
        raise RuntimeError(
            "qsub not found on PATH. "
            "Ensure PBS/Torque is installed or set TENSORLBM_HPC_MODE=none."
        )

    script = _build_pbs_script(
        job_id, cmd,
        queue=queue or _default_partition(),
        nodes=nodes or _default_nodes(),
        cpus=cpus or _default_cpus(),
        mem=mem or _default_mem(),
        walltime=walltime or _default_walltime(),
        log_dir=_log_dir(),
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pbs", delete=False, prefix=f"tensorlbm_{job_id}_",
    ) as fh:
        fh.write(script)
        script_path = fh.name

    result = subprocess.run(
        ["qsub", script_path],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"qsub failed:\n{result.stderr}")

    hpc_job_id = result.stdout.strip()
    logger.info("PBS job submitted: platform_id=%s pbs_id=%s", job_id, hpc_job_id)

    return {
        "hpc_job_id": hpc_job_id,
        "script_path": script_path,
        "status": "submitted",
        "backend": "pbs",
        "queue": queue or _default_partition(),
        "nodes": nodes or _default_nodes(),
        "cpus": cpus or _default_cpus(),
        "mem": mem or _default_mem(),
        "walltime": walltime or _default_walltime(),
    }


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------

def submit_hpc_job(
    job_id: str,
    output_dir: str,
    solver_cmd: str | None = None,
    *,
    partition: str | None = None,
    nodes: int | None = None,
    cpus: int | None = None,
    mem: str | None = None,
    walltime: str | None = None,
    extra_slurm_directives: list[str] | None = None,
) -> dict[str, Any]:
    """Dispatch a TensorLBM job to the configured HPC scheduler.

    Constructs a command that re-runs the platform API action for the given
    job in a cluster environment, or uses a user-supplied *solver_cmd*.

    Args:
        job_id:      Platform job ID.
        output_dir:  Job output directory (for logging / job metadata).
        solver_cmd:  Shell command to run on the cluster.  If None, a
                     placeholder ``echo`` command is used (useful for testing).
        partition:   Scheduler partition/queue.
        nodes, cpus, mem, walltime: Resource requests.
        extra_slurm_directives: Extra ``#SBATCH`` directives (SLURM only).

    Returns:
        Submission result dictionary from ``submit_slurm`` or ``submit_pbs``.

    Raises:
        ValueError: If HPC mode is ``none``.
        RuntimeError: If the scheduler binary is not found.
    """
    mode = hpc_mode()
    if mode == "none":
        raise ValueError(
            "HPC mode is disabled. Set TENSORLBM_HPC_MODE=slurm or pbs to enable."
        )

    if solver_cmd is None:
        solver_cmd = (
            f"echo 'TensorLBM HPC job {job_id}' && "
            f"ls '{output_dir}'"
        )

    if mode == "slurm":
        return submit_slurm(
            job_id, solver_cmd,
            partition=partition, nodes=nodes, cpus=cpus,
            mem=mem, walltime=walltime,
            extra_directives=extra_slurm_directives,
        )
    elif mode == "pbs":
        return submit_pbs(
            job_id, solver_cmd,
            queue=partition, nodes=nodes, cpus=cpus,
            mem=mem, walltime=walltime,
        )
    else:
        raise ValueError(f"Unknown HPC mode: {mode!r}. Choose from: none, slurm, pbs.")
